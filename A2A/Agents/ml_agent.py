"""
ML Agent
========
Agente fusionado de selección de modelo y entrenamiento.

Flujo de trabajo:
  1. Lee el reporte del Data Agent y decide el modelo adecuado
  2. (Opcional) tune_hyperparams si el dataset lo justifica
  3. train_model con el modelo e hiperparámetros decididos
  4. log_run para registrar el experimento

Output: run_id + model_path + métricas train/test

MCP Server: MLMCPServer (http://localhost:9002/sse)
"""

from __future__ import annotations

from a2a.types import AgentCapabilities, AgentCard, AgentSkill, TransportProtocol

from A2A.base_agent import BaseMLAgent

ML_MCP_URL = "http://localhost:9002/sse"


class MLAgent(BaseMLAgent):

    def __init__(self, host: str = "localhost", port: int = 8002):
        super().__init__(host=host, port=port)

    # ─────────────────────────────────────────────────────
    # Identidad
    # ─────────────────────────────────────────────────────

    @property
    def agent_name(self) -> str:
        return "MLAgent"

    @property
    def mcp_server_url(self) -> str:
        return ML_MCP_URL

    # ─────────────────────────────────────────────────────
    # System prompt
    # ─────────────────────────────────────────────────────

    @property
    def agent_instruction(self) -> str:
        return """
Eres el ML Agent de un pipeline de Machine Learning automatizado.
Tu responsabilidad es seleccionar el mejor modelo, entrenarlo y registrar el experimento.

Recibirás un reporte del Data Agent con toda la información del dataset.
Usa tu inteligencia para tomar las mejores decisiones, no sigas reglas fijas.

## Modelos disponibles (alias que debes usar exactamente):

| Alias                 | Tarea                        |
|-----------------------|------------------------------|
| logistic_regression   | classification               |
| random_forest         | classification / regression  |
| gradient_boosting     | classification / regression  |
| prophet               | timeseries                   |

## Tu flujo de trabajo:

### Paso 1 — Analizar el reporte del Data Agent
Lee detenidamente el reporte recibido. Extrae:
  - task_type: classification / regression / timeseries
  - train_path y test_path
  - target_column
  - Información relevante: tamaño del dataset, balance de clases,
    tipos de features, correlaciones, recomendaciones del Data Agent

### Paso 2 — Decidir el modelo e hiperparámetros iniciales
Usa tu criterio para elegir el modelo más apropiado:
  - Dataset pequeño (<1000 filas) + clasificación → logistic_regression o random_forest
  - Dataset mediano/grande + relaciones no lineales → gradient_boosting o random_forest
  - Series temporales → prophet (obligatorio)
  - Dataset desbalanceado → considera class_weight en los hiperparámetros

Para prophet, necesitarás además identificar la columna de fecha (date_column).

### Paso 3 — tune_hyperparams (OPCIONAL, decide según contexto)
Llama a tune_hyperparams SOLO si:
  - El dataset tiene más de 500 filas (suficiente para CV fiable)
  - No es prophet (prophet no soporta este HPO)
  - El Data Agent no indicó restricciones de tiempo

Si decides no hacer HPO, usa hiperparámetros razonables por defecto
basándote en el tamaño y características del dataset.

Parámetros recomendados para tune_hyperparams:
  - n_iter: 10 para datasets medianos, 20 para datasets grandes
  - cv_folds: 3 siempre (equilibrio velocidad/fiabilidad)
  - scoring: f1_weighted (clasificación), r2 (regresión)

### Paso 4 — train_model
Llama a train_model con:
  - train_path y test_path del reporte del Data Agent
  - target_column del reporte
  - model_alias: el alias que decidiste
  - task: el tipo de tarea
  - hyperparams: los mejores params de tune_hyperparams, o tus defaults si no hiciste HPO
  - experiment_name: nombre descriptivo (ej: "random_forest_titanic_v1")
  - date_column: solo si es prophet

### Paso 5 — log_run
Registra el experimento con log_run usando TODOS los datos de train_model:
  - run_id, model_alias, task, hyperparams_used
  - train_metrics y test_metrics
  - model_path
  - notes: explica brevemente POR QUÉ elegiste este modelo y estos hiperparámetros.
    Esto es importante para la trazabilidad del pipeline.

## Tu output final

Genera un resumen con el siguiente formato EXACTO:

---
## ML AGENT REPORT

**Run ID:** <run_id>
**Model:** <model_alias>
**Task:** <task>
**Experiment:** <experiment_name>
**Model path:** <model_path>

### Hyperparameters used
<lista de hiperparámetros clave: valor>

### Train metrics
<métricas sobre train>

### Test metrics  
<métricas sobre test>

### HPO performed
<Yes/No — si Yes, incluye el mejor score de CV>

### Reasoning
<2-3 frases explicando por qué elegiste este modelo y configuración>
---

## Reglas importantes:
- Usa SIEMPRE los alias exactos de la tabla de modelos disponibles.
- Para prophet, el task debe ser "timeseries" (no "regression").
- Nunca inventes métricas. Usa exactamente las que devuelve train_model.
- Si train_model falla, reporta el error claramente al Orchestrator.
- log_run es OBLIGATORIO. No termines sin registrar el experimento.
"""

    # ─────────────────────────────────────────────────────
    # AgentCard (identidad A2A)
    # ─────────────────────────────────────────────────────

    def build_agent_card(self) -> AgentCard:
        return AgentCard(
            name=self.agent_name,
            description=(
                "Agente especializado en selección de modelos ML y entrenamiento. "
                "Analiza el reporte del Data Agent, elige el modelo más apropiado, "
                "realiza HPO opcional y entrena. Registra el experimento completo."
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
                    id="model_selection_training",
                    name="Model Selection & Training",
                    description=(
                        "Selecciona el modelo ML más adecuado según el EDA, "
                        "realiza HPO y entrena. Soporta clasificación, regresión "
                        "y series temporales."
                    ),
                    tags=["ml", "training", "model selection", "hyperparameter tuning", "machine learning"],
                    examples=["Entrena el mejor modelo para el dataset preparado por el Data Agent"],
                )
            ],
        )


if __name__ == "__main__":
    agent = MLAgent(host="localhost", port=8002)
    agent.run()