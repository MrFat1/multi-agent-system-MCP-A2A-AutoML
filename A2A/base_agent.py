"""
Base Agent
==========
Extiende A2AServer añadiendo el loop ReAct de Google ADK con tools MCP via SSE.

Cada agente especializado hereda de esta clase e implementa:
  - agent_name       → nombre único
  - agent_instruction → system prompt del agente
  - mcp_server_url   → URL del servidor MCP que usa este agente
  - build_agent_card → identidad A2A

El loop ReAct lo gestiona ADK internamente: el LLM decide qué tool llamar,
el MCPToolset la ejecuta, y el resultado vuelve al LLM hasta que decide
que tiene suficiente información para responder.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, McpToolset
from google.adk.tools.mcp_tool import SseConnectionParams
from google.genai import types as genai_types

from A2A.A2AServer import A2AServer
from a2a.types import AgentCard

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash"

from dotenv import load_dotenv
load_dotenv()

class BaseMLAgent(A2AServer):
    """
    Agente base para el pipeline ML.

    Añade sobre A2AServer:
      - Conexión al servidor MCP via SSE (MCPToolset)
      - Loop ReAct gestionado por ADK (Runner + Agent)
      - Session service en memoria (stateless entre tareas)

    Los agentes hijos solo necesitan definir:
      - agent_name        → str
      - agent_instruction → str  (system prompt)
      - mcp_server_url    → str  (URL del MCP Server SSE)
      - build_agent_card  → AgentCard
    """

    def __init__(self, host: str = "localhost", port: int = 8000):
        super().__init__(host=host, port=port)
        self._runner: Runner | None = None
        self._session_service = InMemorySessionService()

    # ─────────────────────────────────────────────────────
    # Propiedades abstractas que cada agente debe definir
    # ─────────────────────────────────────────────────────

    @property
    @abstractmethod
    def agent_instruction(self) -> str:
        """System prompt del agente. Define su rol, flujo de trabajo y restricciones."""
        ...

    @property
    @abstractmethod
    def mcp_server_url(self) -> str:
        """URL SSE del servidor MCP que usa este agente. Ej: 'http://localhost:9001/sse'"""
        ...

    # ─────────────────────────────────────────────────────
    # Inicialización del runner ADK (lazy)
    # ─────────────────────────────────────────────────────

    async def _get_runner(self) -> Runner:
        """
        Construye el Runner ADK con las tools MCP la primera vez que se necesita.
        Lazy para evitar conexiones en el arranque si el MCP server aún no está listo.
        """
        if self._runner is not None:
            return self._runner

        logger.info(f"[{self.agent_name}] Conectando a MCP server: {self.mcp_server_url}")

        toolset = McpToolset(
            connection_params=SseConnectionParams(url=self.mcp_server_url)
        )

        agent = Agent(
            model=GEMINI_MODEL,
            name=self.agent_name,
            instruction=self.agent_instruction,
            tools=[toolset],
        )

        self._runner = Runner(
            agent=agent,
            app_name=agent.name,
            session_service=self._session_service,
        )

        logger.info(f"[{self.agent_name}] Runner ADK inicializado.")
        return self._runner

    # ─────────────────────────────────────────────────────
    # Loop ReAct — process_message
    # ─────────────────────────────────────────────────────

    async def process_message(self, message: str, context: dict[str, Any]) -> str:
        """
        Ejecuta el loop ReAct de ADK para procesar un mensaje.

        ADK gestiona internamente:
          1. LLM recibe el mensaje + system prompt
          2. LLM decide qué tool MCP llamar
          3. MCPToolset ejecuta la tool contra el servidor MCP
          4. Resultado vuelve al LLM
          5. Repite hasta que el LLM produce una respuesta final

        Devuelve el texto de la última respuesta del agente.
        """
        runner  = await self._get_runner()
        task_id = context.get("task_id", "default")

        # Crear sesión ADK para esta tarea (aislada por task_id)
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

        # Iterar sobre los eventos del runner hasta obtener la respuesta final
        async for event in runner.run_async(
            user_id="pipeline",
            session_id=session.id,
            new_message=user_content,
        ):
            # ADK emite eventos de distintos tipos; nos interesa la respuesta final
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_response = "".join(
                        part.text
                        for part in event.content.parts
                        if hasattr(part, "text") and part.text
                    )
                break

        if not final_response:
            logger.warning(f"[{self.agent_name}] No se obtuvo respuesta final del runner.")
            final_response = "El agente no produjo respuesta. Revisa los logs del servidor MCP."

        logger.info(f"[{self.agent_name}] Respuesta generada ({len(final_response)} chars).")
        return final_response