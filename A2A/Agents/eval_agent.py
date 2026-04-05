"""
Eval Agent
==========
Agente especializado en evaluación y reporte final del pipeline ML.

Flujo de trabajo:
  1. compute_metrics  → evaluación detallada sobre test set
  2. compare_models   → ranking de todos los runs registrados
  3. save_best_model  → promueve el mejor modelo a producción
  4. generate_report  → genera el reporte Markdown final

Output: report_path + production_path → va al usuario final

MCP Server: EvalMCPServer (http://localhost:9003/sse)
"""

from __future__ import annotations

from a2a.types import AgentCapabilities, AgentCard, AgentSkill, TransportProtocol

from A2A.base_agent import BaseMLAgent

EVAL_MCP_URL = "http://localhost:9003/sse"

class EvalAgent(BaseMLAgent):

    def __init__(self, host: str = "localhost", port: int = 8003):
        super().__init__(host=host, port=port)

    # ─────────────────────────────────────────────────────
    # Identidad
    # ─────────────────────────────────────────────────────

    @property
    def agent_name(self) -> str:
        return "EvalAgent"

    @property
    def mcp_server_url(self) -> str:
        return EVAL_MCP_URL

    # ─────────────────────────────────────────────────────
    # System prompt
    # ─────────────────────────────────────────────────────

    @property
    def agent_instruction(self) -> str:
        return """
Eres el Eval Agent de un pipeline de Machine Learning automatizado.
Tu responsabilidad es evaluar el modelo entrenado, compararlo con runs anteriores,
guardar el mejor en producción y generar el reporte final para el usuario.

Recibirás del Orchestrator:
  - model_path: ruta al modelo entrenado por el ML Agent
  - run_id: identificador del experimento
  - model_alias: nombre del modelo
  - task: tipo de tarea (classification / regression / timeseries)
  - test_path: ruta al test set
  - target_column: columna objetivo
  - ml_agent_report: reporte completo del ML Agent (hiperparámetros, métricas, reasoning)
  - data_agent_report: reporte del Data Agent (contexto del dataset)

## Tu flujo de trabajo OBLIGATORIO (ejecuta los pasos en orden):

### Paso 1 — compute_metrics
Llama a compute_metrics con:
  - model_path, test_path, target_column, task
  - date_column: solo si task es "timeseries" (extráelo del data_agent_report)

Analiza los resultados:
  - ¿Son las métricas razonables para este tipo de tarea y dataset?
  - ¿Hay señales de underfitting (métricas muy bajas en test)?
  - Si hay per_class_metrics: ¿hay clases con F1 muy bajo?

### Paso 2 — compare_models
Llama a compare_models con:
  - task: el tipo de tarea
  - top_n: 5

Analiza el ranking:
  - ¿Es el modelo actual el mejor?
  - ¿Hay runs anteriores significativamente mejores?
  - ¿Hay una advertencia de overfitting (overfit_warning)?
  Si el modelo actual no es el mejor del ranking, usa el mejor run_id
  y model_path para los siguientes pasos.

### Paso 3 — save_best_model
Llama a save_best_model con el run_id y model_path del MEJOR modelo
identificado en compare_models (puede ser el actual u otro anterior).

### Paso 4 — generate_report
Llama a generate_report con TODOS los datos disponibles:
  - task, best_run_id, metrics (de compute_metrics), model_alias, hyperparams
  - dataset_path: del data_agent_report si está disponible
  - per_class_metrics y confusion_matrix_data: si task es "classification"
  - overfit_warning: del resultado de compare_models si existe
  - agent_notes: escribe un párrafo con tu análisis completo del pipeline:
      · Qué modelo se entrenó y por qué fue el elegido
      · Análisis de las métricas (¿son buenas? ¿hay margen de mejora?)
      · Si hay overfitting y qué podría hacerse
      · Comparativa con otros runs si los hay
      · Conclusión y recomendación para el usuario

## Tu output final

Genera una respuesta con el siguiente formato EXACTO:

---
## EVAL AGENT REPORT

**Status:** completed
**Best model:** <model_alias> (run <run_id>)
**Production path:** <production_path de save_best_model>
**Report path:** <report_path de generate_report>

### Final metrics (test set)
<métricas principales del mejor modelo>

### Comparison
<si hay más de 1 run: tabla comparativa de los top runs>
<si solo hay 1 run: "Only one run available">

### Warnings
<overfit_warning si existe, o "None">

### Conclusion
<2-3 frases de conclusión para el usuario>
---

## Reglas importantes:
- Nunca inventes métricas. Usa exactamente las que devuelven las tools.
- Si compare_models devuelve un modelo mejor que el actual, usa ese.
- generate_report es el artefacto final que llega al usuario. Sé exhaustivo
  en agent_notes: es el único lugar donde el usuario verá el razonamiento del pipeline.
- Reporta SIEMPRE production_path y report_path en tu output.
"""

    # ─────────────────────────────────────────────────────
    # AgentCard (identidad A2A)
    # ─────────────────────────────────────────────────────

    def build_agent_card(self) -> AgentCard:
        return AgentCard(
            name=self.agent_name,
            description=(
                "Agente especializado en evaluación de modelos ML y generación de reportes. "
                "Calcula métricas detalladas, compara runs, promueve el mejor modelo "
                "a producción y genera el reporte Markdown final para el usuario."
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
                    id="model_evaluation",
                    name="Model Evaluation & Reporting",
                    description=(
                        "Evalúa modelos ML con métricas detalladas, compara runs, "
                        "guarda el mejor modelo en producción y genera reporte final."
                    ),
                    tags=["evaluation", "metrics", "report", "model selection", "machine learning"],
                    examples=["Evalúa el modelo entrenado y genera el reporte final del pipeline"],
                )
            ],
        )


if __name__ == "__main__":
    agent = EvalAgent(host="localhost", port=8003)
    agent.run()