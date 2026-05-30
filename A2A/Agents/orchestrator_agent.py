"""
Orchestrator Agent
==================
Orquestador LLM del pipeline ML automatizado.
Usa Google ADK con RemoteA2aAgent para coordinar los 3 agentes especializados
via protocolo A2A. El razonamiento y la coordinación los ejecuta un LLM Gemini.

Flujo delegado al LLM:
  1. Recibe dataset_path del usuario
  2. Llama a data_agent  → reporte EDA + rutas train/test
  3. Llama a ml_agent    → run_id + model_path + métricas
  4. Llama a eval_agent  → report_path + production_path
  5. Devuelve resumen final al usuario

Puertos por defecto:
  Orchestrator : 8000
  Data Agent   : 8001
  ML Agent     : 8002
  Eval Agent   : 8003
"""

from __future__ import annotations

import json
from typing import Any

from google.adk.agents import Agent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.agent_tool import AgentTool
from google.genai import types as genai_types
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    TransportProtocol,
)

from A2A.A2AServer import A2AServer
from utils.logger import get_logger, get_agent_logger

from dotenv import load_dotenv
load_dotenv()

logger = get_logger(__name__)
agent_logger = get_agent_logger()

GEMINI_MODEL = "gemini-2.5-flash"

ORCHESTRATOR_INSTRUCTION = """
Eres el orquestador de un pipeline de Machine Learning automatizado.
Tienes acceso a tres agentes especializados como herramientas: data_agent, ml_agent y eval_agent.
Tu trabajo es coordinarlos en orden para completar el pipeline.

## Flujo OBLIGATORIO — sigue siempre este orden:

### Paso 1 — data_agent
Llama a data_agent pasándole exactamente la ruta del dataset recibida.
Espera su reporte completo antes de continuar.
Si el reporte contiene "can_train: false" o un error crítico de archivo no encontrado,
detén el pipeline inmediatamente y comunica el error al usuario con claridad.

### Paso 2 — ml_agent
Pasa el reporte COMPLETO del data_agent al ml_agent como contexto.
Espera su reporte con modelo entrenado y métricas.
Si las métricas son claramente insuficientes (F1 < 0.4 en clasificación, o R² < 0 en regresión),
vuelve a llamar al ml_agent indicándole que pruebe un modelo o configuración diferente.
Máximo 2 intentos en total.

### Paso 3 — eval_agent
Pasa los reportes COMPLETOS del data_agent y ml_agent al eval_agent.
Espera el reporte final con métricas detalladas, ruta del modelo en producción y ruta del reporte Markdown.

### Paso 4 — Reporte final del pipeline

Genera tú mismo un reporte resumido en Markdown con el siguiente formato:

---
## Pipeline ML — Reporte Final

**Dataset:** <ruta del dataset>
**Fecha:** <fecha actual>

### Artefactos
<dónde encontrar el reporte y el modelo.>

### Resumen del pipeline
- **Datos:** <resumen en 1 frase: filas, columnas, target, problemas detectados si los hubo>
- **Modelo:** <por qué se eligió este modelo, si hubo HPO, si hubo reintentos y por qué>
- **Evaluación:** <si el modelo es bueno, advertencias de overfitting si las hay>
---

No reenvíes el reporte completo del eval_agent. Sintetiza la información más relevante
de los tres reportes (data, ml, eval) en este formato compacto.

## Reglas estrictas:
- Sigue siempre el orden: data_agent → ml_agent → eval_agent. Nunca lo alteres.
- Pasa siempre el output COMPLETO de cada agente al siguiente. No lo resumas ni lo recortes.
- Si cualquier agente devuelve un error técnico, detén el pipeline e informa al usuario con el nombre del agente que falló y el motivo.
- Nunca inventes resultados ni métricas. Usa exactamente lo que devuelve cada agente.
- No llames a ningún agente más de una vez salvo el caso de retry del ml_agent descrito arriba.
"""


class OrchestratorAgent(A2AServer):
    """
    Orquestador LLM del pipeline ML.

    Usa Google ADK con RemoteA2aAgent para coordinar los 3 agentes especializados.
    El razonamiento y la orquestación los realiza un LLM Gemini en lugar de
    lógica Python hardcodeada.
    """

    agent_name = "OrchestratorAgent"

    def __init__(self, host: str = "localhost", port: int = 8000):
        super().__init__(host=host, port=port)
        self._runner: Runner | None = None
        self._session_service = InMemorySessionService()

    # ─────────────────────────────────────────────────────
    # Identidad A2AServer
    # ─────────────────────────────────────────────────────

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
    # Inicialización del runner ADK (lazy)
    # ─────────────────────────────────────────────────────

    async def _get_runner(self) -> Runner:
        if self._runner is not None:
            return self._runner

        logger.info(f"[{self.agent_name}] Inicializando RemoteA2aAgents y Runner ADK...")

        data_agent = RemoteA2aAgent(
            name="data_agent",
            description="Prepara el dataset: EDA, limpieza, preprocesado y split. Devuelve un reporte estructurado con rutas train/test, tipo de tarea y recomendaciones para el modelo.",
            agent_card=f"http://localhost:8001{AGENT_CARD_WELL_KNOWN_PATH}",
        )
        ml_agent = RemoteA2aAgent(
            name="ml_agent",
            description="Selecciona el modelo ML más apropiado, realiza HPO opcional, entrena y registra el experimento. Devuelve run_id, model_path y métricas.",
            agent_card=f"http://localhost:8002{AGENT_CARD_WELL_KNOWN_PATH}",
        )
        eval_agent = RemoteA2aAgent(
            name="eval_agent",
            description="Evalúa el modelo entrenado, compara con runs anteriores, guarda el mejor en producción y genera el reporte Markdown final.",
            agent_card=f"http://localhost:8003{AGENT_CARD_WELL_KNOWN_PATH}",
        )

        agent = Agent(
            model=GEMINI_MODEL,
            name=self.agent_name,
            instruction=ORCHESTRATOR_INSTRUCTION,
            tools=[
                AgentTool(agent=data_agent),
                AgentTool(agent=ml_agent),
                AgentTool(agent=eval_agent),
            ],
        )

        self._runner = Runner(
            agent=agent,
            app_name=self.agent_name,
            session_service=self._session_service,
        )

        logger.info(f"[{self.agent_name}] Runner ADK inicializado.")
        return self._runner

    # ─────────────────────────────────────────────────────
    # process_message — loop ReAct ADK
    # ─────────────────────────────────────────────────────

    async def process_message(self, message: str, context: dict[str, Any]) -> str:
        runner  = await self._get_runner()
        task_id = context.get("task_id", "default")

        session = await self._session_service.create_session(
            app_name=self.agent_name,
            user_id="pipeline",
            session_id=task_id,
        )

        user_content = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=message)],
        )

        final_response = ""

        async for event in runner.run_async(
            user_id="pipeline",
            session_id=session.id,
            new_message=user_content,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:

                    if hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        agent_logger.info(
                            f"AGENT: {self.agent_name}\n"
                            f"TOOL CALL: {fc.name}\n"
                            f"PARAMS:\n{json.dumps(dict(fc.args), indent=2, ensure_ascii=False, default=str)}"
                        )

                    elif hasattr(part, "function_response") and part.function_response:
                        fr = part.function_response
                        response_str = json.dumps(
                            fr.response, indent=2, ensure_ascii=False, default=str
                        ) if isinstance(fr.response, dict) else str(fr.response)

                        try:
                            parsed = json.loads(response_str)
                            pretty_response = json.dumps(parsed, indent=2, ensure_ascii=False)
                        except Exception:
                            pretty_response = response_str

                        agent_logger.info(
                            f"AGENT: {self.agent_name}\n"
                            f"TOOL RESPONSE: {fr.name}\n"
                            f"RESULT:\n{pretty_response}"
                        )

            if event.is_final_response():
                if event.content and event.content.parts:
                    final_response = "".join(
                        part.text
                        for part in event.content.parts
                        if hasattr(part, "text") and part.text
                    )

                try:
                    parsed = json.loads(final_response)
                    pretty_response = json.dumps(parsed, indent=2, ensure_ascii=False)
                except Exception:
                    pretty_response = final_response
                agent_logger.info(
                    f"AGENT: {self.agent_name}\n"
                    f"INPUT:\n{message[:500]}{'...' if len(message) > 500 else ''}\n"
                    f"FINAL RESPONSE:\n{pretty_response}"
                )
                break

        if not final_response:
            logger.warning(f"[{self.agent_name}] No se obtuvo respuesta final del runner.")
            final_response = "El agente no produjo respuesta. Revisa los logs del runner ADK."

        logger.info(f"[{self.agent_name}] Respuesta generada ({len(final_response)} chars).")
        return final_response


if __name__ == "__main__":
    agent = OrchestratorAgent(host="localhost", port=8000)
    agent.run()
