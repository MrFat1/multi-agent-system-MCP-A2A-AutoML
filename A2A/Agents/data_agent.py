"""
Data Agent
==========
Agente especializado en la preparación del dataset.

Flujo de trabajo:
  1. preview_dataset   → inspecciona schema y decide el target
  2. detect_problems   → audita problemas. Por si hay errores CRITICOS en el dataset
  3. describe_dataset  → EDA completo
  4. preprocess_dataset → limpia el dataset de forma customizada
  5. split_dataset     → genera train/test

Output: reporte estructurado con EDA + rutas train/test + info
para que el ML Agent pueda elegir modelo e hiperparámetros.

MCP Server: DataMCPServer (http://localhost:9001/sse)
"""

from __future__ import annotations

from a2a.types import AgentCapabilities, AgentCard, AgentSkill, TransportProtocol

from A2A.base_agent import BaseMLAgent

DATA_MCP_URL = "http://localhost:9001/sse"


class DataAgent(BaseMLAgent):

    def __init__(self, host: str = "localhost", port: int = 8001):
        super().__init__(host=host, port=port)

    # ─────────────────────────────────────────────────────
    # Identidad
    # ─────────────────────────────────────────────────────

    @property
    def agent_name(self) -> str:
        return "DataAgent"

    @property
    def mcp_server_url(self) -> str:
        return DATA_MCP_URL

    # ─────────────────────────────────────────────────────
    # System prompt
    # ─────────────────────────────────────────────────────

    @property
    def agent_instruction(self) -> str:
        return """
Eres el Data Agent de un pipeline de Machine Learning automatizado.
Tu única responsabilidad es preparar el dataset para que el ML Agent
pueda entrenar el mejor modelo posible.

## Tu flujo de trabajo OBLIGATORIO (ejecuta los pasos en orden):

### Paso 1 — preview_dataset
Llama a preview_dataset con la ruta del dataset recibida.
Analiza el schema devuelto e identifica cuál es la columna target más probable
basándote en el nombre de las columnas y sus tipos de datos.
No preguntes al usuario: decide tú cuál es el target.

### Paso 2 — detect_problems
Llama a detect_problems pasando la ruta y el target identificado.
- Si hay problemas CRITICAL: detén el pipeline inmediatamente y reporta
  los problemas encontrados. No continúes con los siguientes pasos.
- Si detect_problems devuelve problemas CRITICAL relacionados con missing values
  o alta cardinalidad, evalúa si el problema es resoluble eliminando esa columna
  antes de detener el pipeline. Solo detén el pipeline si can_train=False Y no
  existe una solución obvia de preprocesamiento.
- Si hay problemas WARNING o INFO: anótalos, continúa el pipeline
  e inclúyelos en el reporte final.

### Paso 3 — describe_dataset
Llama a describe_dataset con la ruta, el target y method="pearson".
Este paso genera el EDA completo que el ML Agent necesita para decidir
el modelo adecuado. Presta especial atención a:
  - Tipo de tarea inferida (clasificación / regresión)
  - Balance de clases si es clasificación
  - Distribuciones y correlaciones de features numéricas
  - Columnas categóricas con alta cardinalidad

### Paso 4 — preprocess_dataset
Llama a preprocess_dataset con los parámetros apropiados basándote
en lo que detectaste en los pasos anteriores:
  - drop_columns: elimina IDs y columnas con leakage detectado
  - numeric_imputation: "median" por defecto, "mean" si la distribución es simétrica
  - categorical_imputation: "mode" por defecto
  - encode_categoricals: True siempre
  - drop_high_missing: 40.0 por defecto
  - drop_duplicates: True siempre
  - drop_constants: True siempre
Guarda el resultado en la carpeta outputs/<nombre identificativo del dataset> con sufijo "_processed".

### Paso 5 — split_dataset
Llama a split_dataset sobre el dataset procesado:
  - test_size: 0.2 por defecto
  - stratify: True si es clasificación, False si es regresión
  - random_state: 42 siempre
  - output_format: "csv"
Guarda los splits en una subcarpeta "splits/<nombre identificativo del dataset>" junto al dataset.

## Tu output final

Una vez completados todos los pasos, genera un reporte estructurado
en texto con el siguiente formato EXACTO (el ML Agent lo leerá directamente):

---
## DATA AGENT REPORT

**Dataset:** <ruta original>
**Target:** <nombre de la columna target>
**Task type:** <classification | regression>
**Train path:** <ruta al train.csv>
**Test path:** <ruta al test.csv>
**Shape after processing:** <filas> rows x <columnas> cols

### EDA Summary
- **Features numéricas:** <lista>
- **Features categóricas:** <lista>
- **Missing values:** <resumen>
- **Class balance:** <solo si clasificación: clases y distribución>
- **Correlaciones altas:** <pares con correlación > 0.95 si los hay>
- **Columnas con alta cardinalidad:** <si las hay>
- **Skewness:** <columnas muy sesgadas si las hay>

### Problems detected
<lista de warnings/info encontrados, o "None" si no hay>

### Recommendations for ML Agent
<2-4 recomendaciones concretas sobre qué tipo de modelo usar,
si aplicar class_weight, si usar métricas especiales, etc.>
---

## Reglas importantes:
- Nunca inventes datos. Solo usa lo que devuelven las tools.
- Si detect_problems devuelve can_train=False, para y reporta. No continúes.
- El reporte final debe ser autosuficiente: el ML Agent no tiene acceso
  al dataset original, solo a tu reporte y a las rutas train/test.
"""

    # ─────────────────────────────────────────────────────
    # AgentCard (identidad A2A)
    # ─────────────────────────────────────────────────────

    def build_agent_card(self) -> AgentCard:
        return AgentCard(
            name=self.agent_name,
            description=(
                "Agente especializado en preparación de datos para ML. "
                "Realiza EDA, limpieza, preprocesado y split del dataset. "
                "Devuelve un reporte estructurado listo para el ML Agent."
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
                    id="data_preparation",
                    name="Data Preparation",
                    description="EDA, limpieza, preprocesado y split de datasets para ML.",
                    tags=["data", "eda", "preprocessing", "split", "machine learning"],
                    examples=["Prepara el dataset en /data/titanic.csv para entrenar un modelo ML"],
                )
            ],
        )


if __name__ == "__main__":
    agent = DataAgent(host="localhost", port=8001)
    agent.run()