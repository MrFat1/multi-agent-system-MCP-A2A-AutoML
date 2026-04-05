"""
Data MCP Server
===============
Servidor MCP para el pipeline de Machine Learning.
Gestiona la carga, exploración, diagnóstico y preparación de datasets.

Diseño: stateless — cada tool recibe los datos como argumento y devuelve
un resultado serializable (dict/JSON). El estado lo gestiona el agente orquestador.

Fuentes soportadas: CSV, Parquet, JSON, Excel (.xlsx/.xls), Feather, ORC.

Uso:
    fastmcp run data_mcp_server.py
    fastmcp dev data_mcp_server.py   # con inspector MCP
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Annotated, Any, Literal, Optional
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP
from pydantic import Field
import sys

warnings.filterwarnings("ignore")

import logging
import uvicorn

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# ---------------------------------------------------------------------------
# Servidor
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="DataMCPServer",
    instructions=(
        "Servidor MCP para operaciones de datos en pipelines de ML. "
        "Todas las tools son stateless: reciben la ruta al dataset y devuelven "
        "resultados serializables listos para ser consumidos por agentes LLM."
    ),
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".csv": "csv",
    ".parquet": "parquet",
    ".json": "json",
    ".jsonl": "json",
    ".xlsx": "excel",
    ".xls": "excel",
}

MAX_SAMPLE_ROWS = 5          # filas de muestra devueltas al agente
HIGH_CARDINALITY_THRESHOLD = 50   # categorías únicas para flag de alta cardinalidad
CORRELATION_THRESHOLD = 0.95  # umbral para alertar correlaciones altas
SKEWNESS_THRESHOLD = 1.0      # |skew| > threshold → flag distribución sesgada
MISSING_WARN_PCT = 5.0        # % missing para warning
MISSING_CRITICAL_PCT = 40.0  # % missing para error crítico


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _load_df(dataset_path: str, separator: str = ",") -> pd.DataFrame:
    """Carga un DataFrame desde una ruta local según su extensión."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {dataset_path}")

    ext = path.suffix.lower()
    fmt = SUPPORTED_EXTENSIONS.get(ext)
    if fmt is None:
        raise ValueError(
            f"Extensión '{ext}' no soportada. "
            f"Formatos admitidos: {list(SUPPORTED_EXTENSIONS.keys())}"
        )

    print(f">>> Cargando con formato: {fmt}, extensión: {ext}", file=sys.stderr, flush=True)

    loaders = {
        "csv":     lambda p: pd.read_csv(p, sep=separator),
        "parquet": lambda p: pd.read_parquet(p),
        "json":    lambda p: pd.read_json(p, lines=ext == ".jsonl"),
        "excel":   lambda p: pd.read_excel(p),
    }
    return loaders[fmt](path)


def _dtype_category(dtype: np.dtype) -> str:
    """Clasifica un dtype de pandas en una categoría semántica."""
    kind = dtype.kind
    return {
        "i": "integer",
        "u": "unsigned_integer",
        "f": "float",
        "b": "boolean",
        "O": "object/string",
        "U": "unicode_string",
        "M": "datetime",
        "m": "timedelta",
        "c": "complex",
    }.get(kind, "other")


def _safe_float(value: Any) -> Any:
    """Convierte a float Python nativo; NaN/Inf → None."""
    try:
        v = float(value)
        return None if (math.isnan(v) or math.isinf(v)) else v
    except (TypeError, ValueError):
        return None


def _series_stats(series: pd.Series) -> dict:
    """Estadísticas descriptivas para una columna numérica."""
    desc = series.describe()
    return {
        "mean":   _safe_float(desc.get("mean")),
        "std":    _safe_float(desc.get("std")),
        "min":    _safe_float(desc.get("min")),
        "p25":    _safe_float(desc.get("25%")),
        "median": _safe_float(desc.get("50%")),
        "p75":    _safe_float(desc.get("75%")),
        "max":    _safe_float(desc.get("max")),
        "skewness": _safe_float(series.skew()),
        "kurtosis": _safe_float(series.kurt()),
    }


def _validate_dataset_file(dataset_path: str) -> Path:
    """Verifica que el dataset exista, sea un archivo y tenga un formato soportado."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {dataset_path}. Los datasets suelen encontrarse en data/<nombre.extension>")
    if not path.is_file():
        raise ValueError(f"La ruta no es un archivo: {dataset_path}.")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Extensión '{ext}' no soportada. "
            f"Formatos admitidos: {list(SUPPORTED_EXTENSIONS.keys())}"
        )

    return path


def _build_schema(df: pd.DataFrame) -> dict[str, Any]:
    """Construye el schema básico del DataFrame."""
    return {
        col: {
            "dtype": str(dtype),
            "dtype_category": _dtype_category(dtype),
            "missing_pct": round(df[col].isnull().mean() * 100, 2),
            "unique_values": int(df[col].nunique()),
        }
        for col, dtype in df.dtypes.items()
    }


# ---------------------------------------------------------------------------
# TOOL 1 — preview_dataset
# ---------------------------------------------------------------------------

@mcp.tool()
def preview_dataset(
    dataset_path: Annotated[
        str,
        Field(description="Ruta absoluta o relativa al archivo del dataset.")
    ],
    separator: Annotated[
        str,
        Field(description="Separador para archivos CSV (por defecto ',').")
    ] = ",",
) -> dict:
    """
    Verifica que el dataset exista y sea legible, y devuelve el schema básico.

    Esta tool está pensada para obtener información inicial del dataset sin
    realizar un análisis completo ni devolver muestras de filas.
    """
    _validate_dataset_file(dataset_path)
    df = _load_df(dataset_path, separator)

    return {
        "status": "schema_ready",
        "dataset_path": str(dataset_path),
        "shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1e6, 3),
        "schema": _build_schema(df),
        "has_missing": bool(df.isnull().any().any()),
        "total_missing": int(df.isnull().sum().sum()),
    }


# ---------------------------------------------------------------------------
# TOOL 2 — describe_dataset  (EDA para LLM)
# ---------------------------------------------------------------------------

@mcp.tool()
def describe_dataset(
    dataset_path: Annotated[str, Field(description="Ruta al archivo del dataset.")],
    separator: Annotated[str, Field(description="Separador CSV.")] = ",",
    target_column: Annotated[
        Optional[str],
        Field(description="Columna objetivo para análisis adicional.")
    ] = None,
) -> dict:
    """
    Análisis Exploratorio de Datos (EDA) orientado a LLMs.

    Devuelve un reporte estructurado con toda la información necesaria para
    que un agente LLM decida: tipo de tarea, modelos candidatos, estrategia
    de preprocesado y posibles problemas antes del entrenamiento.

    Incluye:
    - Estadísticas por columna (numéricas y categóricas)
    - Distribuciones, outliers y skewness
    - Matriz de correlación (numéricas)
    - Análisis de la variable objetivo (balance de clases, distribución)
    - Recomendaciones automáticas para el agente
    """
    df = _load_df(dataset_path, separator)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()

    # ── Análisis por columna ───────────────────────────────────────────────
    columns_analysis: dict[str, Any] = {}

    for col in df.columns:
        series = df[col]
        missing_count = int(series.isnull().sum())
        missing_pct = round(missing_count / len(df) * 100, 2)
        col_info: dict[str, Any] = {
            "dtype": str(series.dtype),
            "dtype_category": _dtype_category(series.dtype),
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "unique_values": int(series.nunique()),
            "unique_pct": round(series.nunique() / len(df) * 100, 2),
        }

        if col in numeric_cols:
            col_info["statistics"] = _series_stats(series.dropna())
            # Detección de outliers con IQR
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            outliers = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
            col_info["outliers_iqr"] = outliers
            col_info["outliers_pct"] = round(outliers / len(df) * 100, 2)

        elif col in categorical_cols:
            value_counts = series.value_counts()
            col_info["top_values"] = {
                str(k): int(v)
                for k, v in value_counts.head(10).items()
            }
            col_info["high_cardinality"] = (
                series.nunique() > HIGH_CARDINALITY_THRESHOLD
            )

        columns_analysis[col] = col_info

    # ── Correlaciones ─────────────────────────────────────────────────────
    correlation_matrix: dict = {}
    high_correlations: list[dict] = []
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().round(4)
        correlation_matrix = json.loads(corr.to_json())

        # Pares con correlación muy alta (posible multicolinealidad)
        for i, c1 in enumerate(numeric_cols):
            for c2 in numeric_cols[i + 1:]:
                val = corr.loc[c1, c2]
                if abs(val) >= CORRELATION_THRESHOLD:
                    high_correlations.append({
                        "col_a": c1, "col_b": c2,
                        "correlation": round(float(val), 4),
                    })

    # ── Análisis de la variable objetivo ─────────────────────────────────
    target_analysis: dict = {}
    if target_column and target_column in df.columns:
        target = df[target_column]
        vc = target.value_counts()
        if target_column in categorical_cols or target.nunique() <= 20:
            # Clasificación
            class_distribution = {str(k): int(v) for k, v in vc.items()}
            majority = int(vc.iloc[0])
            minority = int(vc.iloc[-1])
            imbalance_ratio = round(majority / minority, 2) if minority > 0 else None
            target_analysis = {
                "task_type": "classification",
                "num_classes": int(target.nunique()),
                "class_distribution": class_distribution,
                "imbalance_ratio": imbalance_ratio,
                "is_imbalanced": imbalance_ratio is not None and imbalance_ratio > 3,
            }
        else:
            # Regresión
            target_analysis = {
                "task_type": "regression",
                "statistics": _series_stats(target.dropna()),
                "skewed": abs(_safe_float(target.skew()) or 0) > SKEWNESS_THRESHOLD,
            }

    # ── Recomendaciones automáticas para el agente ────────────────────────
    recommendations: list[str] = []

    high_missing = [
        col for col, info in columns_analysis.items()
        if info["missing_pct"] > MISSING_CRITICAL_PCT
    ]
    if high_missing:
        recommendations.append(
            f"Columnas con >40% de valores faltantes (considerar eliminar o imputar con cautela): {high_missing}"
        )

    high_card = [
        col for col, info in columns_analysis.items()
        if info.get("high_cardinality", False)
    ]
    if high_card:
        recommendations.append(
            f"Alta cardinalidad en columnas categóricas (usar embeddings o target encoding): {high_card}"
        )

    if high_correlations:
        recommendations.append(
            f"Posible multicolinealidad detectada en {len(high_correlations)} par(es). "
            "Considerar PCA o eliminación de features redundantes."
        )

    skewed_cols = [
        col for col in numeric_cols
        if abs(_safe_float(df[col].skew()) or 0) > SKEWNESS_THRESHOLD
    ]
    if skewed_cols:
        recommendations.append(
            f"Columnas con distribución muy sesgada (considerar log/sqrt transform): {skewed_cols}"
        )

    if target_analysis.get("is_imbalanced"):
        recommendations.append(
            f"Dataset desbalanceado (ratio {target_analysis['imbalance_ratio']}:1). "
            "Considerar SMOTE, class_weight o métricas como F1/AUC-ROC en lugar de accuracy."
        )

    return {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "column_types": {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "datetime": datetime_cols,
            "boolean": bool_cols,
        },
        "columns_analysis": columns_analysis,
        "correlation": {
            "method": "pearson",
            "matrix": correlation_matrix,
            "high_correlation_pairs": high_correlations,
        },
        "target_analysis": target_analysis,
        "recommendations": recommendations,
    }


# ---------------------------------------------------------------------------
# TOOL 3 — detect_problems
# ---------------------------------------------------------------------------

@mcp.tool()
def detect_problems(
    dataset_path: Annotated[str, Field(description="Ruta al archivo del dataset.")],
    separator: Annotated[str, Field(description="Separador CSV.")] = ",",
    target_column: Annotated[
        Optional[str],
        Field(description="Columna objetivo para detectar leakage.")
    ] = None,
) -> dict:
    """
    Detecta problemas en el dataset que podrían comprometer el entrenamiento ML.

    Categorías de problemas:
    - CRITICAL: bloquean el entrenamiento o garantizan resultados erróneos.
    - WARNING: degradan el rendimiento o la fiabilidad del modelo.
    - INFO: observaciones a tener en cuenta durante el diseño.

    Detecta: valores faltantes, duplicados, constantes, leakage, tipos
    incorrectos, outliers extremos, target en features, datasets vacíos, etc.
    """
    df = _load_df(dataset_path, separator)

    problems: list[dict[str, Any]] = []

    def add(severity: str, code: str, message: str, details: Any = None):
        problems.append({
            "severity": severity,    # CRITICAL | WARNING | INFO
            "code": code,
            "message": message,
            "details": details,
        })

    # ── Dataset vacío ────────────────────────────────────────────────────
    if df.empty:
        add("CRITICAL", "EMPTY_DATASET", "El dataset está vacío (0 filas).")
        return {"total_problems": 1, "problems": problems, "can_train": False}

    if len(df) < 50:
        add("WARNING", "VERY_SMALL_DATASET",
            f"El dataset tiene solo {len(df)} filas. Los modelos pueden no generalizar.",
            {"rows": len(df)})

    # ── Columnas constantes ──────────────────────────────────────────────
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        add("WARNING", "CONSTANT_COLUMNS",
            "Columnas con un único valor (zero-variance). No aportan información.",
            {"columns": constant_cols})

    # ── Filas duplicadas ────────────────────────────────────────────────
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        dup_pct = round(dup_count / len(df) * 100, 2)
        severity = "CRITICAL" if dup_pct > 20 else "WARNING"
        add(severity, "DUPLICATE_ROWS",
            f"{dup_count} filas duplicadas ({dup_pct}% del dataset).",
            {"duplicate_rows": dup_count, "duplicate_pct": dup_pct})

    # ── Valores faltantes ────────────────────────────────────────────────
    missing = df.isnull().sum()
    for col, count in missing[missing > 0].items():
        pct = round(count / len(df) * 100, 2)
        severity = "CRITICAL" if pct >= MISSING_CRITICAL_PCT else "WARNING"
        add(severity, "MISSING_VALUES",
            f"'{col}' tiene {count} valores faltantes ({pct}%).",
            {"column": col, "missing_count": int(count), "missing_pct": pct})

    # ── Columnas con tipo object que podrían ser numéricas ───────────────
    for col in df.select_dtypes(include="object").columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        valid_pct = converted.notna().mean()
        if valid_pct > 0.9:
            add("WARNING", "WRONG_DTYPE",
                f"'{col}' es tipo object pero el 90%+ de sus valores son numéricos.",
                {"column": col, "numeric_pct": round(valid_pct * 100, 1)})

    # ── Outliers extremos (Z-score > 5) ──────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        series = df[col].dropna()
        if series.std() == 0:
            continue
        z_scores = np.abs((series - series.mean()) / series.std())
        extreme = int((z_scores > 5).sum())
        if extreme > 0:
            add("WARNING", "EXTREME_OUTLIERS",
                f"'{col}' tiene {extreme} valores con Z-score > 5 (outliers extremos).",
                {"column": col, "extreme_outliers": extreme})

    # ── Alta cardinalidad en categóricas ─────────────────────────────────
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if col == target_column:
            continue
        nunique = df[col].nunique()
        if nunique > HIGH_CARDINALITY_THRESHOLD:
            add("INFO", "HIGH_CARDINALITY",
                f"'{col}' tiene {nunique} valores únicos (alta cardinalidad).",
                {"column": col, "unique_count": int(nunique)})

    # ── Columnas de ID (probable leakage inocente pero ruido) ────────────
    id_like = [
        col for col in df.columns
        if df[col].nunique() == len(df) and col.lower() in ("id", "index", "uuid", "key")
    ]
    if id_like:
        add("INFO", "LIKELY_ID_COLUMNS",
            "Posibles columnas ID detectadas. Deben excluirse del entrenamiento.",
            {"columns": id_like})

    # ── Target leakage: correlación perfecta feature→target ──────────────
    if target_column and target_column in df.columns:
        target = df[target_column]
        leakage_candidates = []
        for col in numeric_cols:
            if col == target_column:
                continue
            try:
                corr = abs(df[col].corr(target.astype(float) if target.dtype == bool else target))
                if _safe_float(corr) is not None and corr >= 0.99:
                    leakage_candidates.append({"column": col, "correlation": round(float(corr), 4)})
            except Exception:
                pass
        if leakage_candidates:
            add("CRITICAL", "TARGET_LEAKAGE",
                "Columnas con correlación ≥0.99 con el target. Posible data leakage.",
                {"candidates": leakage_candidates})

    # ── Target en las features ────────────────────────────────────────────
    if target_column and target_column in df.columns:
        target_name_lower = target_column.lower()
        suspicious = [
            col for col in df.columns
            if col != target_column and target_name_lower in col.lower()
        ]
        if suspicious:
            add("WARNING", "TARGET_IN_FEATURES",
                "Hay columnas cuyo nombre contiene el nombre del target. Revisar si son variantes del target.",
                {"columns": suspicious})

    # ── Distribución de clases (para target categórico) ──────────────────
    if target_column and target_column in df.columns:
        target = df[target_column]
        if target.nunique() <= 30:
            vc = target.value_counts()
            if len(vc) >= 2:
                ratio = vc.iloc[0] / vc.iloc[-1]
                if ratio > 10:
                    add("WARNING", "CLASS_IMBALANCE",
                        f"Desbalanceo severo en el target (ratio {ratio:.1f}:1).",
                        {"majority_class": str(vc.index[0]),
                         "minority_class": str(vc.index[-1]),
                         "ratio": round(float(ratio), 2)})

    # ── Resumen ──────────────────────────────────────────────────────────
    criticals = [p for p in problems if p["severity"] == "CRITICAL"]
    can_train = len(criticals) == 0

    return {
        "total_problems": len(problems),
        "critical": len(criticals),
        "warnings": len([p for p in problems if p["severity"] == "WARNING"]),
        "info": len([p for p in problems if p["severity"] == "INFO"]),
        "can_train": can_train,
        "blocking_reason": (
            None if can_train
            else [p["message"] for p in criticals]
        ),
        "problems": problems,
    }


# ---------------------------------------------------------------------------
# TOOL 4 — preprocess_dataset
# ---------------------------------------------------------------------------

@mcp.tool()
def preprocess_dataset(
    dataset_path: Annotated[str, Field(description="Ruta al archivo del dataset.")],
    output_path: Annotated[str, Field(description="Ruta donde guardar el dataset preprocesado.")],
    separator: Annotated[str, Field(description="Separador CSV.")] = ",",
    target_column: Annotated[
        Optional[str],
        Field(description="Columna objetivo. Se excluye del preprocesado de features.")
    ] = None,
    drop_columns: Annotated[
        Optional[list[str]],
        Field(description="Columnas a eliminar explícitamente (e.g., IDs, leakage).")
    ] = None,
    numeric_imputation: Annotated[
        Literal["mean", "median", "zero", "none"],
        Field(description="Estrategia de imputación para columnas numéricas.")
    ] = "median",
    categorical_imputation: Annotated[
        Literal["mode", "unknown", "none"],
        Field(description="Estrategia de imputación para columnas categóricas.")
    ] = "mode",
    encode_categoricals: Annotated[
        bool,
        Field(description="Si True, aplica one-hot encoding a columnas categóricas de baja cardinalidad.")
    ] = True,
    drop_duplicates: Annotated[
        bool,
        Field(description="Si True, elimina filas duplicadas.")
    ] = True,
    drop_constants: Annotated[
        bool,
        Field(description="Si True, elimina columnas de varianza cero.")
    ] = True,
) -> dict:
    """
    Aplica un preprocesado básico configurable al dataset.

    El agente LLM debe decidir los parámetros en base al EDA y a los problemas
    detectados. El resultado se guarda en output_path y se devuelven metadatos
    del proceso para que el agente pueda confirmar las transformaciones.

    No aplica scaling (esto se hace en el pipeline de sklearn del ML Server
    para evitar leakage entre train y test).
    """
    df = _load_df(dataset_path, separator)
    report: dict[str, Any] = {"steps": []}

    def log(step: str, details: Any = None):
        report["steps"].append({"step": step, "details": details})

    original_shape = df.shape

    # ── Eliminar columnas explícitas ─────────────────────────────────────
    if drop_columns:
        existing = [c for c in drop_columns if c in df.columns]
        df.drop(columns=existing, inplace=True)
        log("drop_columns", {"dropped": existing})

    # ── Eliminar columnas constantes ─────────────────────────────────────
    if drop_constants:
        const_cols = [c for c in df.columns if df[c].nunique() <= 1 and c != target_column]
        df.drop(columns=const_cols, inplace=True)
        log("drop_constants", {"dropped": const_cols})

    # ── Eliminar duplicados ──────────────────────────────────────────────
    if drop_duplicates:
        before = len(df)
        df.drop_duplicates(inplace=True)
        removed = before - len(df)
        log("drop_duplicates", {"removed_rows": removed})

    # ── Separar target para no procesarlo ────────────────────────────────
    target_series = None
    if target_column and target_column in df.columns:
        target_series = df.pop(target_column)

    # ── Imputación numérica ───────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_imputation != "none" and numeric_cols:
        for col in numeric_cols:
            if df[col].isnull().any():
                fill_val = {
                    "mean":   df[col].mean(),
                    "median": df[col].median(),
                    "zero":   0,
                }[numeric_imputation]
                df[col].fillna(fill_val, inplace=True)
        log("numeric_imputation", {"strategy": numeric_imputation, "columns": numeric_cols})

    # ── Imputación categórica ─────────────────────────────────────────────
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_imputation != "none" and cat_cols:
        for col in cat_cols:
            if df[col].isnull().any():
                fill_val = (
                    df[col].mode().iloc[0]
                    if categorical_imputation == "mode" and not df[col].mode().empty
                    else "unknown"
                )
                df[col].fillna(fill_val, inplace=True)
        log("categorical_imputation", {"strategy": categorical_imputation, "columns": cat_cols})

    # ── One-hot encoding ──────────────────────────────────────────────────
    encoded_cols: list[str] = []
    if encode_categoricals:
        low_card = [c for c in cat_cols if df[c].nunique() <= HIGH_CARDINALITY_THRESHOLD]
        if low_card:
            df = pd.get_dummies(df, columns=low_card, drop_first=False)
            encoded_cols = low_card
            log("one_hot_encoding", {"encoded_columns": encoded_cols})

    # ── Reunir target ─────────────────────────────────────────────────────
    if target_series is not None:
        df[target_column] = target_series.values

    # ── Guardar ───────────────────────────────────────────────────────────
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ext = output.suffix.lower()
    if ext == ".parquet":
        df.to_parquet(output, index=False)
    elif ext in (".csv", ""):
        df.to_csv(output, index=False)
    else:
        df.to_csv(output, index=False)

    log("save", {"output_path": str(output)})

    return {
        "status": "success",
        "original_shape": {"rows": original_shape[0], "columns": original_shape[1]},
        "final_shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "output_path": str(output),
        "columns_after": list(df.columns),
        "processing_report": report,
    }


# ---------------------------------------------------------------------------
# TOOL 5 — split_dataset
# ---------------------------------------------------------------------------

@mcp.tool()
def split_dataset(
    dataset_path: Annotated[str, Field(description="Ruta al dataset (ya preprocesado).")],
    output_dir: Annotated[str, Field(description="Directorio donde guardar train.csv y test.csv.")],
    separator: Annotated[str, Field(description="Separador CSV.")] = ",",
    target_column: Annotated[
        Optional[str],
        Field(description="Columna objetivo.")
    ] = None,
    test_size: Annotated[
        float,
        Field(description="Proporción del test set (0.0–1.0).", ge=0.05, le=0.5)
    ] = 0.2,
    random_state: Annotated[
        int,
        Field(description="Semilla para reproducibilidad.")
    ] = 42,
) -> dict:
    """
    Divide el dataset en train/test.

    El agente controla los parámetros esenciales del split para permitir
    experimentación sistemática y salida reproducible.
    """

    try:

        df = _load_df(dataset_path, separator)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── Train / Test split ────────────────────────────────────────────────
        df_train, df_test = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
        )

        splits: dict[str, Any] = {
            "train": {"rows": len(df_train), "columns": len(df_train.columns)},
            "test": {"rows": len(df_test), "columns": len(df_test.columns)},
        }

        # ── Guardar splits ────────────────────────────────────────────────────
        def save(split_df: pd.DataFrame, name: str) -> str:
            p = out_dir / f"{name}.csv"
            split_df.to_csv(p, index=False)
            return str(p)

        paths: dict[str, str] = {
            "train": save(df_train, "train"),
            "test": save(df_test, "test"),
        }

        # ── Distribución del target por split ────────────────────────────────
        target_distribution: dict = {}
        if target_column and target_column in df.columns:
            for name, split_df in [("train", df_train), ("test", df_test)]:
                if split_df[target_column].nunique() <= 50:
                    vc = split_df[target_column].value_counts(normalize=True).round(4)
                    target_distribution[name] = {str(k): float(v) for k, v in vc.items()}

        return {
            "status": "success",
            "original_rows": len(df),
            "splits": splits,
            "paths": paths,
            "random_state": random_state,
            "target_distribution_pct": target_distribution,
        }
    
    except Exception as e:
            print(f"Error en split_dataset: {str(e)}", file=sys.stderr)
            return {"status": "error", "message": str(e)}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.info("Starting Data MCP server at port 9001...")
    #mcp.run(transport="sse")
    #from mcp.server.fastmcp import FastMCP
    #logging.info([m for m in dir(FastMCP) if not m.startswith('_')])
    uvicorn.run(mcp.sse_app(), host="localhost", port=9001)