"""
Clase base reutilizable para todos los agentes del sistema.
Hereda de esta clase para crear nuevos agentes.
"""

# ===============================================================================
# Clase base reutilizable para todos los agentes del sistema.
# Hereda de esta clase para crear nuevos agentes.
# Cada agente será un servidor A2A por lo que heredarán de esta clase.
#
# Cada agente hereda de esta clase y obtiene:
#  - Servidor HTTP (FastAPI + uvicorn)
#  - Endpoints A2A estándar (AgentCard, JSON-RPC)
#  - Gestión de tareas en memoria
#  - Extracción de texto de mensajes A2A

#Para crear un agente nuevo:
#  1. Hereda de A2AServer
#  2. Implementa agent_name, build_agent_card(), process_message()
#  3. Llama a run()
# ===============================================================================

import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from a2a.utils.constants import PREV_AGENT_CARD_WELL_KNOWN_PATH, DEFAULT_RPC_URL
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.types import (
    AgentCard,
    SendMessageResponse,
    Artifact,
    Part,
    Message,
    Role,
    Task,
    TaskStatus,
    TaskState,
    TextPart
)

from utils.logger import get_logger
logger = get_logger(__name__)


class A2AServer(ABC):
    """
    Servidor A2A base. Todos los agentes heredan de esta clase.

    Implementa el protocolo A2A (JSON-RPC sobre HTTP):
      GET  /.well-known/agent-card.json  → AgentCard (descubrimiento)
      POST /rpc                     → handler JSON-RPC (tasks/send, tasks/get, tasks/cancel)
      GET  /health                  → health check
    """

    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self._tasks: Dict[str, Task] = {}
        self.app = FastAPI(title=self.agent_name, version="1.0.0")
        self._setup_routes()

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Nombre único del agente."""
        ...

    @abstractmethod
    def build_agent_card(self) -> AgentCard:
        """
        Define la tarjeta de identidad del agente.
        Este JSON se sirve en /.well-known/agent.json para descubrimiento.
        """
        ...

    @abstractmethod
    async def process_message(self, message: str, context: Dict[str, Any]) -> str:
        """
        Lógica principal del agente.
        Recibe el mensaje del usuario y devuelve la respuesta.
        """
        ...

    # ─────────────────────────────────────────────────────
    # RUTAS HTTP (Protocolo A2A)
    # ─────────────────────────────────────────────────────
    def _setup_routes(self):
        """Registra los endpoints estándar"""

        @self.app.get(PREV_AGENT_CARD_WELL_KNOWN_PATH)
        async def get_agent_card():
            """Endpoint de descubrimiento: devuelve el AgentCard del agente"""
            return self.build_agent_card().model_dump(exclude_none=True)

        @self.app.get("/health")
        async def health_check():
            return {"status": "ok", "agent": self.agent_name}

        @self.app.post(DEFAULT_RPC_URL)
        async def handle_jsonrpc(request: Request):
            """Punto de entrada principal: despacha métodos JSON-RPC A2A."""
            body = await request.json()
            method = body.get("method", "")
            params = body.get("params", {})
            req_id = body.get("id", str(uuid.uuid4()))

            try:
                if method == "message/send":
                    result = await self._handle_task_send(params)
                elif method == "tasks/get":
                    result = await self._handle_task_get(params)
                elif method == "tasks/cancel":
                    result = await self._handle_task_cancel(params)
                else:
                    return JSONResponse(content={
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32601, "message": f"Método no encontrado: {method}"},
                    })
                
                return JSONResponse(content={
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": result,
                })
            
            except Exception as e:
                logger.error(f"[{self.agent_name}] Error en {method}: {e}", exc_info=True)
                return JSONResponse(status_code=500, content={
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32603, "message": str(e)},
                })

    # ─────────────────────────────────────────────────────
    # HANDLERS DE TAREAS A2A
    # ─────────────────────────────────────────────────────
    async def _handle_task_send(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Maneja tasks/send: extrae el texto, ejecuta el agente, devuelve la respuesta.
        """

        task_id = params.get("id", str(uuid.uuid4()))
        context_id = params.get("contextId") or message_data.get("contextId") or str(uuid.uuid4())
        message_data = params.get("message", {})

        # Extraer texto del mensaje
        user_text = self._extract_text(message_data)
        logger.info(f"[{self.agent_name}] message/send task={task_id}: '{user_text[:80]}'")
        
        # Crear y almacenar tarea
        task = Task(
            id=task_id,
            context_id=context_id,
            #agent_name=self.agent_name,
            # message=f"Created task for {self.agent_name}", Tiene que ser un objeto Message, no un String
            status=TaskStatus(
                state=TaskState.working,
                timestamp=_now()
            ),
            history=[
                Message(
                    messageId=str(uuid.uuid4()),
                    role=Role.user,
                    parts=[Part(root=TextPart(text=user_text))],
                    taskId=task_id,
                    contextId=context_id,
                )
            ],
        )
        self._tasks[task_id] = task

        # Ejecutar lógica del agente
        try:
            response_text = await self.process_message(
                message=user_text,
                context={"task_id": task_id, "context_id": context_id},
            )

            # Actualizar tarea con resultado
            task.status = TaskStatus(state=TaskState.completed, timestamp=_now())

            #task.history.append(Message(role="agent", content=response_text))
            #task.artifacts = [Artifact(type="text", content=response_text)]

        except Exception as e:
            logger.error(f"[{self.agent_name}] Error en tarea {task_id}: {e}")
            task.status = TaskStatus(state=TaskState.failed, timestamp=_now())
            response_text = f"Error procesando la tarea: {e}"

        # Añadir respuesta al historial
        response_msg = Message(
            messageId=str(uuid.uuid4()),
            role=Role.agent,
            parts=[Part(root=TextPart(text=response_text))],
            taskId=task_id,
            contextId=context_id,
        )
        task.history.append(response_msg)

        # Devolver respuesta en formato A2A
        return response_msg.model_dump(exclude_none=True)

        # Formato de respuesta A2A estándar
        """return {
            "id": task_id,
            "sessionId": context_id,
            "status": {"state": task.status.state},
            "artifacts": [
                {
                    "parts": [{"type": "text", "text": response_text}],
                    "index": 0,
                }
            ],
            "messages": [
                {
                    "role": msg.role,
                    "parts": [{"type": "text", "text": msg.parts}],
                }
                for msg in task.history
            ],
        }"""

    async def _handle_task_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Recupera el estado de una tarea por su ID."""

        task_id = params.get("id")
        task = self._tasks.get(task_id)

        if not task:
            raise ValueError(f"Tarea no encontrada: {task_id}")
        
        return task.model_dump(exclude_none=True)

        """return {
            "id": task.id,
            "status": {"state": task.status},
            "artifacts": [
                {"parts": [{"type": "text", "text": a.parts}]}
                for a in task.artifacts
            ],
        }"""

    async def _handle_task_cancel(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Cancela una tarea en curso."""

        task_id = params.get("id")
        if task_id in self._tasks:
            self._tasks[task_id].status = TaskStatus(
                state=TaskState.canceled, timestamp=_now()
            )
        return {"id": task_id, "status": {"state": TaskState.canceled}}

    # ─────────────────────────────────────────────────────
    # UTILIDADES
    # ─────────────────────────────────────────────────────
    @staticmethod
    def _extract_text_from_context(context: RequestContext) -> str:
        """Extrae el texto plano del RequestContext del SDK."""
        try:
            if context.message and context.message.parts:
                texts = []
                for part in context.message.parts:
                    # Part es RootModel[TextPart | FilePart | DataPart]
                    if isinstance(part.root, TextPart):
                        texts.append(part.root.text)
                return " ".join(texts) if texts else ""
        except Exception:
            pass
        return ""
    
    @staticmethod
    def _extract_text(message_data: Dict[str, Any]) -> str:
        """Extrae el texto plano de un mensaje A2A (dict con parts)."""
        texts = []
        for part in message_data.get("parts", []):
            # Soporta tanto {"text": "..."} como {"root": {"text": "..."}}
            text = part.get("text") or part.get("root", {}).get("text", "")
            if text:
                texts.append(text)
        return " ".join(texts)
    
    def run(self):
        logger.info(
            f"Iniciando agente A2A '{self.agent_name}' en {self.host}:{self.port}\n"
            f"  Servidor A2A: http://{self.host}:{self.port}/\n"
            f"  AgentCard:    http://{self.host}:{self.port}/.well-known/agent.json"
        )
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")

def _now() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
