"""
ML MCP Server
=============
Servidor MCP para entrenamiento y gestión de experimentos en el pipeline ML.

Responsabilidades:
  - Entrenar modelos a partir de un alias y unos hiperparámetros decididos por el agente.
  - Buscar hiperparámetros óptimos (HPO) para un modelo dado.
  - Registrar experimentos en disco (log_run).
  - Comparar runs registrados para seleccionar el mejor modelo (compare_runs).

Diseño: stateless — cada tool recibe rutas y parámetros explícitos.
El estado del pipeline (qué modelo usar, qué métricas son aceptables) lo decide
el agente LLM, no este servidor.

Modelos soportados (alias → clase):
  Clasificación / Regresión:
    - linear_regression    → LinearRegression
    - logistic_regression  → LogisticRegression
    - random_forest        → RandomForestClassifier / RandomForestRegressor
    - xgboost              → XGBClassifier/XGBRegressor

Uso:
    python  MCP/ml_mcp_server.py
    mcp dev MCP/ml_mcp_server.py   # con inspector MCP
"""

from __future__ import annotations

import json
import sys
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

import joblib
import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    mean_squared_error, r2_score, roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import uvicorn

from utils.logger import get_logger
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Servidor
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="MLMCPServer",
    instructions=(
        "Servidor MCP para entrenamiento y registro de experimentos ML. "
        "El agente decide el modelo (alias) e hiperparámetros; este servidor "
        "ejecuta el entrenamiento y persiste los artefactos y métricas. "
        "Usa 'train_model' para entrenar, 'tune_hyperparams' para HPO, "
        "'log_run' para registrar un experimento y 'compare_runs' para "
        "seleccionar el mejor modelo entre varios runs."
    ),
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

EXPERIMENTS_DIR = Path("experiments")
MODELS_DIR = Path("models")

# Modelos soportados por tarea
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "logistic_regression": {
        "classification": LogisticRegression,
        "regression": None,
        "default_params": {"max_iter": 1000, "random_state": 42},
    },
    "linear_regression": {
        "classification": None,
        "regression": LinearRegression,
        "default_params": {"fit_intercept": True},
    },
    "random_forest": {
        "classification": RandomForestClassifier,
        "regression": RandomForestRegressor,
        "default_params": {"n_estimators": 100, "random_state": 42},
    },
    "xgboost": {
        "classification": XGBClassifier,
        "regression": XGBRegressor,
        "default_params": {"n_estimators": 100, "random_state": 42, "verbosity": 0},
    },
}

# Espacios de búsqueda para HPO por alias
HPO_SEARCH_SPACES: dict[str, dict[str, Any]] = {
    "logistic_regression": {
        "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "model__solver": ["lbfgs", "liblinear"],
        "model__penalty": ["l2"],
    },
    "linear_regression": {
        "model__fit_intercept": [True, False],
        "model__positive": [False, True],
    },
    "random_forest": {
        "model__n_estimators": [50, 100, 200, 300],
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    },
    "xgboost": {
        "model__n_estimators": [50, 100, 200, 300],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__max_depth": [3, 5, 7, 9],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
        "model__min_child_weight": [1, 3, 5],
    },
}


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _load_splits(
    train_path: str,
    test_path: str,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Carga train/test y separa features de target."""
    df_train = pd.read_csv(train_path) if train_path.endswith(".csv") else pd.read_parquet(train_path)
    df_test  = pd.read_csv(test_path)  if test_path.endswith(".csv")  else pd.read_parquet(test_path)

    if target_column not in df_train.columns:
        raise ValueError(f"Columna target '{target_column}' no encontrada en train.")

    X_train = df_train.drop(columns=[target_column])
    y_train = df_train[target_column]
    X_test  = df_test.drop(columns=[target_column])
    y_test  = df_test[target_column]

    return X_train, y_train, X_test, y_test


def _build_sklearn_pipeline(
    model_alias: str,
    task: Literal["classification", "regression"],
    hyperparams: dict[str, Any],
    X: pd.DataFrame,
) -> Pipeline:
    """
    Construye un pipeline sklearn completo:
    preprocessing (imputer + scaler/encoder) + estimador.
    El agente nunca ve esta lógica.
    """
    registry_entry = MODEL_REGISTRY[model_alias]
    model_class = registry_entry.get(task)
    if model_class is None:
        raise ValueError(
            f"El modelo '{model_alias}' no soporta la tarea '{task}'."
        )

    # Separar columnas numéricas y categóricas
    numeric_cols    = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Preprocessor numérico: imputer + scaler
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    # Preprocessor categórico: imputer + one-hot
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # Combinar defaults con hiperparámetros del agente
    defaults = MODEL_REGISTRY[model_alias]["default_params"].copy()
    defaults.update(hyperparams)

    estimator = model_class(**defaults)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])


def _compute_classification_metrics(y_true: pd.Series, y_pred: np.ndarray, y_prob: Optional[np.ndarray]) -> dict:
    """Métricas para clasificación."""
    metrics: dict[str, Any] = {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro":  round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
    }
    if y_prob is not None:
        try:
            n_classes = len(np.unique(y_true))
            if n_classes == 2:
                auc = roc_auc_score(y_true, y_prob[:, 1])
            else:
                auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            metrics["roc_auc"] = round(float(auc), 4)
        except Exception:
            pass
    return metrics


def _compute_regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Métricas para regresión."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": round(float(np.sqrt(mse)), 4),
        "mae":  round(float(mean_absolute_error(y_true, y_pred)), 4),
        "r2":   round(float(r2_score(y_true, y_pred)), 4),
    }


def _save_experiment(run: dict) -> Path:
    """Persiste un run en experiments/<run_id>.json"""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    path = EXPERIMENTS_DIR / f"{run['run_id']}.json"
    path.write_text(json.dumps(run, indent=2, default=str))
    return path


def _verdict(test_metrics: dict, task: str) -> dict:
    if task == "classification":
        metric, val = "f1_weighted", test_metrics.get("f1_weighted", 0)
    else:  # regression
        metric, val = "r2", test_metrics.get("r2", 0)
    thresholds = {"good": 0.85, "acceptable": 0.70}
    status = "good" if val >= thresholds["good"] else "acceptable" if val >= thresholds["acceptable"] else "poor"
    return {"metric": metric, "value": val, "status": status}

def _overfit_check(train_metrics: dict, test_metrics: dict, task: str) -> dict:
    metric = "f1_weighted" if task == "classification" else "r2"

    train_val = train_metrics.get(metric, 0)
    test_val  = test_metrics.get(metric, 0)
    gap = abs(train_val - test_val)

    # Para rmse, gap relativo es más útil
    overfit = gap > 0.15
    return {
        "metric": metric,
        "train_value": train_val,
        "test_value": test_val,
        "gap": round(gap, 4),
        "overfit_detected": overfit,
    }

def _recommendation(verdict: dict, overfit: dict, model_alias: str) -> str:
    if verdict["status"] == "good" and not overfit["overfit_detected"]:
        return "Model is ready. Proceed to log_run and hand off to Eval Agent."
    elif overfit["overfit_detected"]:
        return f"Overfitting detected (gap={overfit['gap']}). Try increasing regularization or reducing complexity before proceeding."
    elif verdict["status"] == "acceptable":
        return "Acceptable performance. Consider trying another model_alias before handing off to Eval Agent."
    else:
        return f"Poor performance ({verdict['metric']}={verdict['value']}). Discard and try a different model_alias or hyperparams."

# ---------------------------------------------------------------------------
# TOOL 1 — train_model
# ---------------------------------------------------------------------------

@mcp.tool()
def train_model(
    train_path: Annotated[str, Field(description="Ruta al archivo train (CSV o Parquet).")],
    test_path: Annotated[str, Field(description="Ruta al archivo test (CSV o Parquet).")],
    target_column: Annotated[str, Field(description="Nombre de la columna objetivo.")],
    model_alias: Annotated[
        Literal["logistic_regression", "linear_regression", "random_forest", "xgboost"],
        Field(description="Alias del modelo a entrenar.")
    ],
    task: Annotated[
        Literal["classification", "regression"],
        Field(description="Tipo de tarea ML.")
    ],
    hyperparams: Annotated[
        dict[str, Any],
        Field(description="Hiperparámetros decididos por el agente. Puede ser {} para usar defaults.")
    ] = {},
    experiment_name: Annotated[
        Optional[str],
        Field(description="Nombre descriptivo del experimento. Opcional.")
    ] = None,
    output_dir: Annotated[
        str,
        Field(description="Directorio donde guardar el modelo entrenado.")
    ] = "models",
) -> dict:
    """
    Entrena un modelo completo a partir de un alias y unos hiperparámetros.

    Internamente construye el pipeline de preprocessing + estimador,
    entrena sobre el train set y evalúa sobre el test set.
    El agente solo decide el alias y los hiperparámetros; toda la
    complejidad de construcción del pipeline es transparente para él.

    Devuelve métricas de train y test, run_id para log_run, y ruta al artefacto.
    """
    MODELS_DIR_ = Path(output_dir)
    MODELS_DIR_.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())[:8]

    # ── Sklearn (clasificación / regresión) ───────────────────────────────
    X_train, y_train, X_test, y_test = _load_splits(train_path, test_path, target_column)

    # Label encoding para clasificación con target string
    le = None
    if task == "classification" and y_train.dtype == object:
        le = LabelEncoder()
        y_train = pd.Series(le.fit_transform(y_train))
        y_test  = pd.Series(le.transform(y_test))

    pipeline = _build_sklearn_pipeline(model_alias, task, hyperparams, X_train)
    pipeline.fit(X_train, y_train)

    # Métricas train
    y_pred_train = pipeline.predict(X_train)
    if task == "classification":
        y_prob_train = pipeline.predict_proba(X_train) if hasattr(pipeline, "predict_proba") else None
        train_metrics = _compute_classification_metrics(y_train, y_pred_train, y_prob_train)
    else:
        train_metrics = _compute_regression_metrics(y_train, y_pred_train)

    # Métricas test
    y_pred_test = pipeline.predict(X_test)
    if task == "classification":
        y_prob_test = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None
        test_metrics = _compute_classification_metrics(y_test, y_pred_test, y_prob_test)
    else:
        test_metrics = _compute_regression_metrics(y_test, y_pred_test)

    model_path = MODELS_DIR_ / f"{model_alias}_{run_id}.pkl"
    joblib.dump(pipeline, model_path)

    verdict = _verdict(test_metrics, task)
    overfit = _overfit_check(train_metrics, test_metrics, task)

    return {
        "run_id": run_id,
        "model_alias": model_alias,
        "task": task,
        "hyperparams_used": {
            **MODEL_REGISTRY[model_alias]["default_params"],
            **hyperparams,
        },
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "model_path": str(model_path),
        "dataset_path": str(Path(train_path).parent),
        "experiment_name": experiment_name or f"{model_alias}_{run_id}",
        "label_encoder_classes": le.classes_.tolist() if le else None,
        "verdict": verdict,
        "overfit_check": overfit,
        "recommendation": _recommendation(verdict, overfit, model_alias),
    }


# ---------------------------------------------------------------------------
# TOOL 2 — tune_hyperparams
# ---------------------------------------------------------------------------

@mcp.tool()
def tune_hyperparams(
    train_path: Annotated[str, Field(description="Ruta al archivo train (CSV o Parquet).")],
    target_column: Annotated[str, Field(description="Nombre de la columna objetivo.")],
    model_alias: Annotated[
        Literal["logistic_regression", "linear_regression", "random_forest", "xgboost"],
        Field(description="Alias del modelo. Prophet no soporta HPO con este método.")
    ],
    task: Annotated[
        Literal["classification", "regression"],
        Field(description="Tipo de tarea ML.")
    ],
    n_iter: Annotated[
        int,
        Field(description="Número de combinaciones a probar en RandomizedSearchCV.", ge=5, le=50)
    ] = 10,
    cv_folds: Annotated[
        int,
        Field(description="Número de folds para cross-validation.", ge=2, le=10)
    ] = 3,
    scoring: Annotated[
        Optional[str],
        Field(description=(
            "Métrica de scoring. Si None, se usa f1_weighted para clasificación y r2 para regresión. "
            "Ejemplos: 'f1_weighted', 'roc_auc', 'r2', 'neg_mean_squared_error'."
        ))
    ] = None,
    custom_search_space: Annotated[
        Optional[dict[str, list[Any]]],
        Field(description=(
            "Espacio de búsqueda personalizado. Si None, se usa el espacio predefinido para el modelo. "
            "Formato: {'param_name': [val1, val2, ...]}. "
            "Nota: los parámetros deben usar el prefijo 'model__' (e.g. 'model__n_estimators')."
        ))
    ] = None,
) -> dict:
    """
    Busca los mejores hiperparámetros para un modelo mediante RandomizedSearchCV.

    El agente recibe los mejores hiperparámetros encontrados y decide
    si los acepta antes de llamar a train_model. No entrena el modelo final,
    solo devuelve la configuración óptima encontrada.

    Prophet no está soportado aquí ya que su HPO requiere un enfoque diferente.
    """
    X_train, y_train, _, _ = _load_splits(
        train_path,
        train_path,  # no necesitamos test para HPO
        target_column,
    )

    # Label encoding si hace falta
    if task == "classification" and y_train.dtype == object:
        le = LabelEncoder()
        y_train = pd.Series(le.fit_transform(y_train))

    pipeline = _build_sklearn_pipeline(model_alias, task, {}, X_train)

    # Scoring por defecto
    if scoring is None:
        scoring = "f1_weighted" if task == "classification" else "r2"

    # Espacio de búsqueda
    search_space = custom_search_space or HPO_SEARCH_SPACES[model_alias]

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=search_space,
        n_iter=n_iter,
        cv=cv_folds,
        scoring=scoring,
        n_jobs=1,
        random_state=42,
        refit=False,  # no reentrenamos aquí
        error_score="raise",
        verbose=2
    )
    search.fit(X_train, y_train)

    best_params_raw = search.best_params_
    # Eliminar prefijo "model__" para devolver params limpios al agente
    best_params_clean = {
        k.replace("model__", ""): v
        for k, v in best_params_raw.items()
    }

    return {
        "model_alias": model_alias,
        "task": task,
        "best_params": best_params_clean,
        "best_score": round(float(search.best_score_), 4),
        "scoring_metric": scoring,
        "n_iter": n_iter,
        "cv_folds": cv_folds,
        "total_models_trained": n_iter * cv_folds,
        "dataset_rows": len(X_train),
        "search_space_used": search_space,
        "recommendation": (
            f"Usa estos parámetros en train_model para obtener "
            f"el mejor {scoring} estimado ({round(float(search.best_score_), 4)})."
        ),
    }


# ---------------------------------------------------------------------------
# TOOL 3 — log_run
# ---------------------------------------------------------------------------

@mcp.tool()
def log_run(
    run_id: Annotated[str, Field(description="ID del run devuelto por train_model.")],
    model_alias: Annotated[str, Field(description="Alias del modelo entrenado.")],
    task: Annotated[str, Field(description="Tipo de tarea (classification/regression).")],
    hyperparams: Annotated[dict[str, Any], Field(description="Hiperparámetros usados en el entrenamiento.")],
    train_metrics: Annotated[dict[str, float], Field(description="Métricas sobre el train set.")],
    test_metrics: Annotated[dict[str, float], Field(description="Métricas sobre el test set.")],
    model_path: Annotated[str, Field(description="Ruta al artefacto del modelo guardado.")],
    experiment_name: Annotated[
        Optional[str],
        Field(description="Nombre descriptivo del experimento.")
    ] = None,
    notes: Annotated[
        Optional[str],
        Field(description="Notas libres del agente sobre este run (por qué eligió este modelo, etc.).")
    ] = None,
    dataset_path: Annotated[
        Optional[str],
        Field(description="Ruta al dataset usado. Útil para trazabilidad.")
    ] = None,
) -> dict:
    """
    Registra un experimento completo en disco (experiments/<run_id>.json).

    El agente llama a esta tool después de train_model para persistir
    los resultados. Los campos notes y experiment_name permiten al agente
    documentar su razonamiento (por qué eligió ese modelo, qué observó).
    """
    run: dict[str, Any] = {
        "run_id": run_id,
        "experiment_name": experiment_name or f"{model_alias}_{run_id}",
        "model_alias": model_alias,
        "task": task,
        "hyperparams": hyperparams,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "model_path": str(model_path),
        "dataset_path": str(dataset_path) if dataset_path else None,
        "notes": notes,
        "timestamp": datetime.now().isoformat(),
    }

    path = _save_experiment(run)

    return {
        "status": "logged",
        "run_id": run_id,
        "experiment_path": str(path),
        "experiment_name": run["experiment_name"],
    }

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting ML MCP server at port 9002...")
    #mcp.run(transport="sse", port=9002)
    uvicorn.run(mcp.sse_app(), host="localhost", port=9002)