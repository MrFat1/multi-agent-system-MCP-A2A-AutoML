"""
Orchestrator Agent
==================
Coordina el pipeline ML completo delegando tareas a los agentes especializados
via protocolo A2A. No llama a ningún MCP Server directamente.

Flujo:
  1. Recibe dataset_path del usuario
  2. Delega al Data Agent → obtiene reporte EDA + rutas train/test
  3. Delega al ML Agent   → obtiene run_id + model_path + métricas
  4. Delega al Eval Agent → obtiene report_path + production_path
  5. Si métricas insuficientes → retry al ML Agent (max MAX_RETRIES)
  6. Devuelve resultado final al usuario

Puertos por defecto:
  Orchestrator : 8000
  Data Agent   : 8001
  ML Agent     : 8002
  Eval Agent   : 8003
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import httpx
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Message,
    Part,
    Role,
    TextPart,
    TransportProtocol,
)

from A2A.A2AServer import A2AServer

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# Configuración de agentes conocidos
# ─────────────────────────────────────────────────────────

AGENT_URLS = {
    "DataAgent": "http://localhost:8001",
    "MLAgent":   "http://localhost:8002",
    "EvalAgent": "http://localhost:8003",
}

MAX_RETRIES = 2        # intentos máximos de reentrenamiento si métricas son insuficientes
HTTP_TIMEOUT = 300.0   # segundos — los agentes pueden tardar en entrenar
ERROR_PREFIX = "AGENT_ERROR:"

class AgentCallError(Exception):
    """Error al llamar a un agente especializado."""
    def __init__(self, agent_name: str, reason: str):
        self.agent_name = agent_name
        self.reason     = reason
        super().__init__(f"{agent_name}: {reason}")


class OrchestratorAgent(A2AServer):
    """
    Orquestador del pipeline ML.

    Coordina Data Agent → ML Agent → Eval Agent via A2A.
    Gestiona el loop de retry si las métricas del ML Agent no son suficientes.
    """

    def __init__(self, host: str = "localhost", port: int = 8000):
        super().__init__(host=host, port=port)
        self._http = httpx.AsyncClient(timeout=HTTP_TIMEOUT)

    # ─────────────────────────────────────────────────────
    # Identidad A2AServer
    # ─────────────────────────────────────────────────────

    @property
    def agent_name(self) -> str:
        return "OrchestratorAgent"

    def build_agent_card(self) -> AgentCard:
        return AgentCard(
            name=self.agent_name,
            description=(
                "Orquestador del pipeline ML automatizado. "
                "Coordina Data Agent, ML Agent y Eval Agent para entrenar "
                "el mejor modelo posible y entregar un reporte final."
            ),
            url=f"http://{self.host}:{self.port}/",
            version="1.0.0",
            protocol_version="0.2.5",
            capabilities=AgentCapabilities(
                streaming=False,
                pushNotifications=False,
            ),
            preferred_transport=TransportProtocol.jsonrpc,
            defaultInputModes=["text/plain"],
            defaultOutputModes=["text/plain"],
            skills=[
                AgentSkill(
                    id="ml_pipeline",
                    name="ML Pipeline Orchestration",
                    description="Ejecuta el pipeline ML completo: datos → modelo → evaluación → reporte.",
                    tags=["orchestrator", "pipeline", "machine learning", "automation"],
                    examples=["Entrena un modelo ML con el dataset /data/titanic.csv"],
                )
            ],
        )

    # ─────────────────────────────────────────────────────
    # process_message — lógica principal del orquestador
    # ─────────────────────────────────────────────────────

    async def process_message(self, message: str, context: dict[str, Any]) -> str:
        """
        Punto de entrada del pipeline ML.

        Espera recibir una ruta a un dataset en el mensaje.
        Coordina los 3 agentes en secuencia y gestiona reintentos.
        """
        context_id = context.get("context_id", str(uuid.uuid4()))
        logger.info(f"[{self.agent_name}] Iniciando pipeline para: '{message[:80]}'")

        try:

            # ── Paso 1: Data Agent ─────────────────────────────────────────
            logger.info(f"[{self.agent_name}] → Delegando al Data Agent...")
            data_report = await self._call_agent(
                agent_name="DataAgent",
                message=f"Prepara el dataset para el pipeline ML: {message}",
                context_id=context_id,
            )
            logger.info(f"[{self.agent_name}] ✓ Data Agent completado.")

            # Verificar si el Data Agent encontró errores críticos
            if self._contains_critical_error(data_report):
                return (
                    f"❌ El pipeline se ha detenido en la fase de datos.\n\n"
                    f"El Data Agent detectó problemas críticos en el dataset:\n\n"
                    f"{data_report}"
                )

            # ── Paso 2: ML Agent (con retry) ────────────────────────────────
            ml_report = None
            retry_context = ""

            for attempt in range(1, MAX_RETRIES + 1):
                logger.info(f"[{self.agent_name}] → Delegando al ML Agent (intento {attempt}/{MAX_RETRIES})...")

                ml_report = await self._call_agent(
                    agent_name="MLAgent",
                    message=(
                        f"Entrena el mejor modelo ML posible con los siguientes datos:\n\n"
                        f"{data_report}{retry_context}"
                    ),
                    context_id=context_id,
                )
                logger.info(f"[{self.agent_name}] ✓ ML Agent intento {attempt} completado.")

                # Verificar si las métricas son aceptables
                if not self._needs_retry(ml_report) or attempt == MAX_RETRIES:
                    break

                logger.warning(
                    f"[{self.agent_name}] Métricas insuficientes en intento {attempt}. "
                    "Solicitando modelo alternativo..."
                )
                retry_context = (
                    f"\n\n## RETRY CONTEXT (intento {attempt + 1})\n"
                    f"El intento anterior produjo métricas insuficientes:\n"
                    f"{ml_report}\n"
                    f"Por favor, prueba un modelo diferente o ajusta los hiperparámetros."
                )

            # ── Paso 3: Eval Agent ──────────────────────────────────────────

            logger.info(f"[{self.agent_name}] → Delegando al Eval Agent...")
            eval_report = await self._call_agent(
                agent_name="EvalAgent",
                message=(
                    f"Evalúa el modelo entrenado y genera el reporte final.\n\n"
                    f"## DATA AGENT CONTEXT\n{data_report}\n\n"
                    f"## ML AGENT CONTEXT\n{ml_report}"
                ),
                context_id=context_id,
            )
            logger.info(f"[{self.agent_name}] ✓ Eval Agent completado.")

            # ── Respuesta final al usuario ───────────────────────────────────
            return (
                f"✅ Pipeline ML completado.\n\n"
                f"{'=' * 60}\n"
                f"{eval_report}\n"
                f"{'=' * 60}\n\n"
                f"**Pipeline summary:**\n"
                f"- Data preparation: ✓\n"
                f"- Model training: ✓\n"
                f"- Evaluation & report: ✓\n"
            )

        except AgentCallError as e:
            # Formato que main.py sabe parsear para mostrar el error
            logger.error(f"[{self.agent_name}] Fallo en {e.agent_name}: {e.reason}")
            return f"{ERROR_PREFIX}{e.agent_name}|{e.reason}"
 
        except Exception as e:
            logger.error(f"[{self.agent_name}] Error inesperado: {e}", exc_info=True)
            return f"{ERROR_PREFIX}OrchestratorAgent|Error inesperado: {e}"

    # ─────────────────────────────────────────────────────
    # Lógica de retry
    # ─────────────────────────────────────────────────────

    def _needs_retry(self, ml_report: str) -> bool:
        import re
        report_lower = ml_report.lower()
 
        if "error" in report_lower and "train_model" in report_lower:
            return True
 
        for match in re.findall(r"f1[_\s\w]*:\s*(0\.\d+)", report_lower):
            if float(match) < 0.4:
                return True
 
        for match in re.findall(r"r2[_\s\w]*:\s*(-?\d+\.\d+)", report_lower):
            if float(match) < 0.0:
                return True
 
        return False

    def _contains_critical_error(self, data_report: str) -> bool:
        """Detecta si el Data Agent reportó un error crítico que impide continuar."""
        keywords = ["can_train: false", "can_train=false", "critical", "detenido", "bloqueado"]
        return any(kw in data_report.lower() for kw in keywords)

    # ─────────────────────────────────────────────────────
    # Comunicación A2A con agentes
    # ─────────────────────────────────────────────────────

    async def _call_agent(
        self,
        agent_name: str,
        message: str,
        context_id: str,
    ) -> str:
        """
        Envía un mensaje a un agente via JSON-RPC A2A y devuelve su respuesta.
        """
        base_url = AGENT_URLS.get(agent_name)
        if not base_url:
            raise ValueError(f"Agente desconocido: {agent_name}")

        rpc_url = f"{base_url}/rpc"
        task_id = str(uuid.uuid4())

        a2a_message = Message(
            messageId=str(uuid.uuid4()),
            role=Role.user,
            parts=[Part(root=TextPart(text=message))],
            contextId=context_id,
            taskId=task_id,
        )

        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "message/send",
            "params": {
                "id": task_id,
                "contextId": context_id,
                "message": a2a_message.model_dump(exclude_none=True),
            },
        }

        try:
            resp = await self._http.post(
                rpc_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()

            if "error" in data:
                raise AgentCallError(agent_name, str(data["error"]))
            
            text = self._extract_text(data.get("result", {}))

            if text.startswith("Error procesando la tarea"):
                raise AgentCallError(agent_name, text)
            
            return text

        except httpx.TimeoutException:
            raise AgentCallError(agent_name, f"Timeout tras {HTTP_TIMEOUT}s sin respuesta.")
        except httpx.HTTPStatusError as e:
            raise AgentCallError(agent_name, f"HTTP {e.response.status_code}")
        except AgentCallError:
            raise
        except Exception as e:
            raise AgentCallError(agent_name, str(e))

    @staticmethod
    def _extract_text(result: dict[str, Any]) -> str:
        """Extrae el texto plano de una respuesta A2A."""
        for part in result.get("parts", []):
            text = part.get("text") or part.get("root", {}).get("text")
            if text:
                return text
        return str(result)

    async def close(self):
        await self._http.aclose()


if __name__ == "__main__":
    agent = OrchestratorAgent(host="localhost", port=8000)
    agent.run()