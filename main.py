"""
main.py
=======
Punto de entrada del sistema multi-agente ML.

Arranca los 4 agentes en procesos separados y lanza
una interfaz de línea de comandos para enviar peticiones al Orchestrator.

Uso:
    python main.py --dataset /ruta/al/dataset.csv

O para arrancar solo los agentes (sin CLI):
    python main.py --serve-only
"""

from __future__ import annotations

import argparse
import asyncio
import multiprocessing
import sys
import time
from typing import Optional
from pathlib import Path

import httpx

from utils.logger import get_logger
logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────
# Configuración de puertos
# ─────────────────────────────────────────────────────────

AGENTS = {
    "OrchestratorAgent": {"module": "A2A.Agents.orchestrator_agent", "class": "OrchestratorAgent", "port": 8000},
    "DataAgent":         {"module": "A2A.Agents.data_agent",         "class": "DataAgent",         "port": 8001},
    "MLAgent":           {"module": "A2A.Agents.ml_agent",           "class": "MLAgent",           "port": 8002},
    "EvalAgent":         {"module": "A2A.Agents.eval_agent",         "class": "EvalAgent",         "port": 8003},
}

MCP_SERVERS = {
    "DataMCPServer": "http://localhost:9001/",
    "MLMCPServer":   "http://localhost:9002/",
    "EvalMCPServer": "http://localhost:9003/",
}
 
# Prefijo que el Orchestrator añade cuando un agente falla
AGENT_ERROR_PREFIX = "AGENT_ERROR:"

# ─────────────────────────────────────────────────────────
# Arranque de agentes en subprocesos
# ─────────────────────────────────────────────────────────

def _run_agent(module_name: str, class_name: str, port: int):
    """Función ejecutada en cada subproceso para inicializar un agente."""
    import importlib
    from dotenv import load_dotenv
    load_dotenv()
    module = importlib.import_module(module_name)
    agent_class = getattr(module, class_name)
    agent = agent_class(host="localhost", port=port)
    agent.run()


def start_agents() -> list[multiprocessing.Process]:
    """Arranca todos los agentes en subprocesos independientes."""
    processes = []
    for name, config in AGENTS.items():
        p = multiprocessing.Process(
            target=_run_agent,
            args=(config["module"], config["class"], config["port"]),
            name=name,
            daemon=True,
        )
        p.start()
        logger.info(f"Agente '{name}' arrancado en puerto {config['port']} (PID {p.pid})")
        processes.append(p)
    return processes

async def check_mcp_servers() -> bool:
    """Verifica que los servidores MCP estén corriendo antes de arrancar los agentes."""
    logger.info("Verificando servidores MCP...")
    async with httpx.AsyncClient(timeout=3.0) as client:
        all_ok = True
        for name, url in MCP_SERVERS.items():
            try:
                resp = await client.get(url)
                # Cualquier respuesta HTTP (200, 404, 405...) significa que el servidor está vivo
                logger.info(f"  {name} disponible (status {resp.status_code})")
            except httpx.ConnectError:
                logger.error(f"  {name} no disponible en {url} — ¿está arrancado?")
                all_ok = False
            except Exception as e:
                logger.error(f"  {name} error inesperado: {e}")
                all_ok = False

    if not all_ok:
        logger.error(
            "\nArranque los servidores MCP antes de ejecutar main.py:\n"
            "  python MCP/data_mcp_server.py\n"
            "  python MCP/ml_mcp_server.py\n"
            "  python MCP/eval_mcp_server.py\n"
        )
    return all_ok

async def wait_for_agents(timeout: float = 30.0) -> bool:
    """Espera a que todos los agentes estén listos respondiendo en /health."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        deadline = time.time() + timeout
        pending = {name: f"http://localhost:{cfg['port']}/health" for name, cfg in AGENTS.items()}

        while pending and time.time() < deadline:
            await asyncio.sleep(1.0)
            ready = []
            for name, url in pending.items():
                try:
                    logger.info(f"Probando conexión con {name} en la url: {url}")
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        logger.info(f" {name} listo")
                        ready.append(name)
                except Exception:
                    pass
            for name in ready:
                pending.pop(name)

        if pending:
            logger.error(f"Agentes no disponibles tras {timeout}s: {list(pending.keys())}")
            return False

    return True


# ─────────────────────────────────────────────────────────
# Cliente CLI para enviar peticiones al Orchestrator
# ─────────────────────────────────────────────────────────

class PipelineError(Exception):
    """Error controlado: un agente del pipeline ha fallado."""
    def __init__(self, agent: str, message: str):
        self.agent   = agent
        self.message = message
        super().__init__(f"[{agent}] {message}")

async def run_pipeline(dataset_path: str) -> str:
    """Envía el dataset al Orchestrator y devuelve el resultado."""
    import uuid
    from a2a.types import Message, Part, Role, TextPart

    orchestrator_url = "http://localhost:8000/rpc"
    context_id = str(uuid.uuid4())
    task_id    = str(uuid.uuid4())

    message = Message(
        messageId=str(uuid.uuid4()),
        role=Role.user,
        parts=[Part(root=TextPart(text=dataset_path))],
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
            "message": message.model_dump(exclude_none=True),
        },
    }

    logger.info(f"Enviando dataset al Orchestrator: {dataset_path}")
    async with httpx.AsyncClient(timeout=600.0) as client:
        resp = await client.post(
            orchestrator_url,
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()

    if "error" in data:
        raise PipelineError("OrchestratorAgent", str(data["error"]))

    result = data.get("result", {})
    for part in result.get("parts", []):
        text = part.get("text") or part.get("root", {}).get("text")
        if text:
            if text.startswith(AGENT_ERROR_PREFIX):
                            payload_str = text[len(AGENT_ERROR_PREFIX):]
                            agent_name, _, error_msg = payload_str.partition("|")
                            raise PipelineError(agent_name.strip(), error_msg.strip())
            return text
    return str(result)


# ─────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────

async def main(dataset_path: Optional[str]):

    path = Path(dataset_path)
    if not path.exists():
        raise PipelineError("main", f"El archivo no existe: {dataset_path}")
    if not path.is_file():
        raise PipelineError("main", f"La ruta no apunta a un archivo: {dataset_path}")
    if path.suffix.lower() not in (".csv", ".parquet", ".json", ".jsonl", ".xlsx", ".xls", ".feather", ".orc"):
        raise PipelineError("main", f"Formato no soportado: '{path.suffix}'. Usa CSV")

    # Verificar MCP servers primero
    mcp_ok = await check_mcp_servers()
    if not mcp_ok:
        sys.exit(1)

    # Arrancar agentes
    logger.info("Arrancando agentes...")
    processes = start_agents()

    # Esperar a que estén listos
    logger.info("Esperando a que los agentes estén disponibles...")
    ready = await wait_for_agents(timeout=120.0)
    if not ready:
        logger.error("No se pudieron arrancar todos los agentes. Abortando.")
        for p in processes:
            p.terminate()
        sys.exit(1)

    logger.info("Todos los agentes están listos.\n")

    if dataset_path:
        try:
            result = await run_pipeline(dataset_path)
            print("\n" + "=" * 60)
            print("RESULTADO DEL PIPELINE ML")
            print("=" * 60)
            print(result)
            print("=" * 60)
        except PipelineError as e:
            print("\n" + "=" * 60)
            print("PIPELINE INTERRUMPIDO")
            print("=" * 60)
            print(f"Agente:  {e.agent}")
            print(f"Motivo:  {e.message}")
            print("=" * 60)
            sys.exit(1)
        except httpx.HTTPStatusError as e:
            logger.error(f"Error HTTP comunicándose con el Orquestador: {e}")
            sys.exit(1)
    else:
        # Modo interactivo
        logger.info("Modo interactivo. Escribe la ruta al dataset o 'exit' para salir.")
        while True:
            try:
                user_input = input("\nDataset path: ").strip()
                if user_input.lower() in ("exit", "quit", "q"):
                    break
                if not user_input:
                    continue
                result = await run_pipeline(user_input)
                print("\n" + result)
            except KeyboardInterrupt:
                break

    # Cleanup
    for p in processes:
        p.terminate()
    logger.info("Sistema detenido.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline ML Multi-Agente")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Ruta al dataset a procesar. Si no se indica, modo interactivo.",
    )
    args = parser.parse_args()

    multiprocessing.set_start_method("spawn", force=True)
    asyncio.run(main(dataset_path=args.dataset))