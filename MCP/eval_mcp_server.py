"""
Eval MCP Server
===============
Servidor MCP para evaluación y reporte final del pipeline ML.

Responsabilidades:
  - Calcular métricas detalladas de un modelo sobre el test set.
  - Comparar varios modelos registrados y seleccionar el mejor.
  - Guardar el mejor modelo en un directorio de producción.
  - Generar un reporte final en Markdown.

Diseño: stateless — cada tool recibe rutas y parámetros explícitos.

Uso:
    python  MCP/eval_mcp_server.py
    mcp dev MCP/eval_mcp_server.py   # con inspector MCP
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

import uvicorn

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.logger import get_logger
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Servidor
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="EvalMCPServer",
    instructions=(
        "Servidor MCP para evaluación y reporte final de modelos ML. "
        "Flujo típico: compute_metrics → compare_models → save_best_model → generate_report."
    ),
)

EXPERIMENTS_DIR = Path("experiments")
REPORTS_DIR     = Path("reports")
PRODUCTION_DIR  = Path("models/production")


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _load_test(test_path: str, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(test_path) if test_path.endswith(".csv") else pd.read_parquet(test_path)
    if target_column not in df.columns:
        raise ValueError(f"Columna target '{target_column}' no encontrada en {test_path}.")
    return df.drop(columns=[target_column]), df[target_column]


def _load_all_runs() -> list[dict]:
    if not EXPERIMENTS_DIR.exists():
        return []
    runs = []
    for p in sorted(EXPERIMENTS_DIR.glob("*.json")):
        try:
            runs.append(json.loads(p.read_text()))
        except Exception:
            pass
    return runs


def _same_dataset(run_dataset_path: str | None, target: str) -> bool:
    """True if both paths refer to the same dataset file.

    Uses Path equality which normalises separators and removes redundant
    components (e.g. './', duplicate slashes) without hitting the filesystem.
    """
    if run_dataset_path is None:
        return False
    return Path(run_dataset_path) == Path(target)


def _format_metrics_table(metrics: dict[str, Any]) -> str:
    """Convierte un dict de métricas en tabla Markdown."""
    lines = ["| Métrica | Valor |", "|---------|-------|"]
    for k, v in metrics.items():
        val = f"{v:.4f}" if isinstance(v, float) else str(v)
        lines.append(f"| {k} | {val} |")
    return "\n".join(lines)

def _save_confusion_matrix_plot(cm, labels, run_id, output_dir="reports/plots") -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {run_id}")
    
    path = Path(output_dir) / f"confusion_matrix_{run_id}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return str(path)


def _save_regression_plot(y_true, y_pred, run_id, output_dir="reports/plots") -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Actual vs Predicted
    axes[0].scatter(y_true, y_pred, alpha=0.5)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title("Actual vs Predicted")

    # Residuals
    residuals = y_true - y_pred
    axes[1].hist(residuals, bins=30, edgecolor="black")
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Residual")
    axes[1].set_title("Residuals Distribution")

    fig.suptitle(f"Regression Diagnostics — {run_id}")
    path = Path(output_dir) / f"regression_plot_{run_id}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return str(path)

# ---------------------------------------------------------------------------
# TOOL 1 — compute_metrics
# ---------------------------------------------------------------------------

@mcp.tool()
def compute_metrics(
    model_path: Annotated[str, Field(description="Ruta al modelo serializado (.pkl).")],
    test_path: Annotated[str, Field(description="Ruta al archivo test (CSV o Parquet).")],
    target_column: Annotated[str, Field(description="Nombre de la columna objetivo.")],
    task: Annotated[
        Literal["classification", "regression"],
        Field(description="Tipo de tarea ML.")
    ],
    run_id: Annotated[Optional[str], Field(description="run_id del experimento, para nombrar los plots.")] = None,
    plots_dir: Annotated[str, Field(description="Directorio donde guardar los plots.")] = "reports/plots",
) -> dict:
    """
    Calcula métricas detalladas de un modelo sobre el test set.

    Clasificación: accuracy, f1 (macro/weighted), roc_auc, matriz de confusión
    y classification_report completo por clase.
    Regresión: RMSE, MAE, R², MAPE.

    Devuelve también un diagnóstico de overfitting si se encuentran
    las métricas de train en el experimento registrado.
    """
    model = joblib.load(model_path)

    # ── Sklearn ───────────────────────────────────────────────────────────
    X_test, y_test = _load_test(test_path, target_column)
    y_pred = model.predict(X_test)

    if task == "classification":
        y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        # roc_auc
        roc_auc = None
        if y_prob is not None:
            try:
                n_classes = len(np.unique(y_test))
                roc_auc = float(
                    roc_auc_score(y_test, y_prob[:, 1])
                    if n_classes == 2
                    else roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
                )
            except Exception:
                pass

        # Confusion matrix serializable
        cm = confusion_matrix(y_test, y_pred)
        labels = sorted(y_test.unique().tolist())

        # Classification report por clase
        cr = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        per_class = {
            str(cls): {
                "precision": round(cr[str(cls)]["precision"], 4),
                "recall":    round(cr[str(cls)]["recall"], 4),
                "f1":        round(cr[str(cls)]["f1-score"], 4),
                "support":   int(cr[str(cls)]["support"]),
            }
            for cls in labels
            if str(cls) in cr
        }

        metrics = {
            "accuracy":     round(float(accuracy_score(y_test, y_pred)), 4),
            "f1_macro":     round(float(f1_score(y_test, y_pred, average="macro",    zero_division=0)), 4),
            "f1_weighted":  round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        }
        if roc_auc is not None:
            metrics["roc_auc"] = round(roc_auc, 4)

        plot_path = _save_confusion_matrix_plot(cm, labels, run_id or "unknown", plots_dir)

        return {
            "task": "classification",
            "model_path": model_path,
            "metrics": metrics,
            "per_class_metrics": per_class,
            "confusion_matrix": {
                "labels": [str(l) for l in labels],
                "matrix": cm.tolist(),
                "plot_path": plot_path,
            },
        }

    else:  # regression
        mse  = mean_squared_error(y_test, y_pred)
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)
        mape = float(np.mean(np.abs((y_test.values - y_pred) / np.where(y_test.values == 0, 1e-9, y_test.values))) * 100)

        plot_path = _save_regression_plot(y_test, y_pred, run_id or "unknown", plots_dir)

        return {
            "task": "regression",
            "model_path": model_path,
            "metrics": {
                "rmse":     round(float(np.sqrt(mse)), 4),
                "mae":      round(mae, 4),
                "r2":       round(r2, 4),
                "mape_pct": round(mape, 4),
                "plot_path": plot_path,
            },
        }


# ---------------------------------------------------------------------------
# TOOL 2 — compare_models
# ---------------------------------------------------------------------------

@mcp.tool()
def compare_models(
    run_id: Annotated[str, Field(description="run_id del experimento actual (devuelto por log_run). El servidor lo usa para determinar el dataset y filtrar runs comparables.")],
    task: Annotated[
        Literal["classification", "regression"],
        Field(description="Tipo de tarea para seleccionar la métrica de comparación.")
    ],
    primary_metric: Annotated[
        Optional[str],
        Field(description=(
            "Métrica principal de comparación. Si None se usa: "
            "f1_weighted (clasificación), r2 (regresión)."
        ))
    ] = None,
    top_n: Annotated[
        int,
        Field(description="Número máximo de modelos a mostrar en el ranking.", ge=1, le=20)
    ] = 5,
) -> dict:
    """
    Lee los experimentos registrados y devuelve un ranking comparativo.

    Usa run_id para localizar el dataset del experimento actual y filtra
    automáticamente los runs al mismo dataset, garantizando que las métricas
    sean comparables. El agente solo necesita pasar su run_id.

    Devuelve el run_id y model_path del ganador para que el agente pueda
    usarlos directamente en save_best_model y generate_report.
    """
    all_runs = _load_all_runs()
    if not all_runs:
        return {
            "status": "no_runs",
            "message": "No hay experimentos en experiments/. Entrena al menos un modelo primero.",
            "ranking": [],
            "best_run": None,
        }

    # Resolve dataset from the current run
    current_run = next((r for r in all_runs if r["run_id"] == run_id), None)
    dataset_path = current_run.get("dataset_path") if current_run else None

    dataset_warning = None
    if dataset_path is None:
        runs = all_runs
        dataset_warning = (
            f"El run '{run_id}' no tiene dataset_path registrado: se comparan todos "
            "los runs. Las métricas pueden no ser comparables entre datasets distintos."
        )
    else:
        runs = [r for r in all_runs if _same_dataset(r.get("dataset_path"), dataset_path)]

    lower_is_better = {"rmse", "mae", "mape_pct"}
    default_metrics = {
        "classification": "f1_weighted",
        "regression":     "r2",
    }
    metric = primary_metric or default_metrics[task]

    ranked = []
    for run in runs:
        val = run.get("test_metrics", {}).get(metric)
        if val is None:
            continue
        ranked.append({
            "run_id":          run["run_id"],
            "experiment_name": run.get("experiment_name"),
            "model_alias":     run.get("model_alias"),
            "hyperparams":     run.get("hyperparams", {}),
            "test_metrics":    run.get("test_metrics", {}),
            "train_metrics":   run.get("train_metrics", {}),
            "model_path":      run.get("model_path"),
            "timestamp":       run.get("timestamp"),
            "notes":           run.get("notes"),
            "_sort_val":       float(val),
        })

    ranked.sort(key=lambda x: x["_sort_val"], reverse=(metric not in lower_is_better))
    for r in ranked:
        r.pop("_sort_val")
    ranked = ranked[:top_n]

    best = ranked[0] if ranked else None

    # Detección simple de overfitting en el mejor modelo
    overfit_warning = None
    if best and best.get("train_metrics") and best.get("test_metrics"):
        train_val = best["train_metrics"].get(metric)
        test_val  = best["test_metrics"].get(metric)
        if train_val is not None and test_val is not None:
            gap = abs(float(train_val) - float(test_val))
            threshold = 0.1
            if gap > threshold:
                overfit_warning = (
                    f"Posible overfitting: {metric} en train={train_val:.4f} "
                    f"vs test={test_val:.4f} (gap={gap:.4f} > {threshold})."
                )

    return {
        "status": "ok",
        "dataset_path": dataset_path,
        "total_runs_for_dataset": len(runs),
        "total_runs_all": len(all_runs),
        "primary_metric": metric,
        "lower_is_better": metric in lower_is_better,
        "best_run": {
            "run_id":      best["run_id"],
            "model_alias": best["model_alias"],
            "model_path":  best["model_path"],
            f"test_{metric}": best["test_metrics"].get(metric),
        } if best else None,
        "overfit_warning": overfit_warning,
        "dataset_warning": dataset_warning,
        "ranking": ranked,
        "recommendation": (
            f"Mejor modelo: '{best['model_alias']}' (run {best['run_id']}) "
            f"con {metric}={best['test_metrics'].get(metric):.4f} en test."
            if best else "No hay runs con la métrica solicitada."
        ),
    }


# ---------------------------------------------------------------------------
# TOOL 3 — save_best_model
# ---------------------------------------------------------------------------

@mcp.tool()
def save_best_model(
    model_path: Annotated[str, Field(description="Ruta actual del modelo a promover a producción.")],
    run_id: Annotated[str, Field(description="run_id del experimento ganador.")],
    model_alias: Annotated[str, Field(description="Alias del modelo (para nombrar el artefacto).")],
    production_dir: Annotated[
        str,
        Field(description="Directorio de producción donde guardar el modelo final.")
    ] = "models/production",
) -> dict:
    """
    Copia el mejor modelo a un directorio de producción con nombre estable.

    El archivo de destino siempre se llama 'best_model.pkl' para que otros
    sistemas puedan referenciarlo sin conocer el run_id. Además guarda un
    fichero 'best_model_metadata.json' con trazabilidad completa.
    """
    import shutil

    prod_dir = Path(production_dir)
    prod_dir.mkdir(parents=True, exist_ok=True)

    src = Path(model_path)
    if not src.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

    dst = prod_dir / "best_model.pkl"
    shutil.copy2(src, dst)

    # Metadatos de trazabilidad
    metadata = {
        "run_id":       run_id,
        "model_alias":  model_alias,
        "source_path":  str(src),
        "promoted_at":  datetime.now().isoformat(),
        "production_path": str(dst),
    }
    meta_path = prod_dir / "best_model_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    return {
        "status": "saved",
        "production_path": str(dst),
        "metadata_path":   str(meta_path),
        "run_id":          run_id,
        "model_alias":     model_alias,
    }


# ---------------------------------------------------------------------------
# TOOL 4 — generate_report
# ---------------------------------------------------------------------------

@mcp.tool()
def generate_report(
    task: Annotated[
        Literal["classification", "regression"],
        Field(description="Tipo de tarea ML.")
    ],
    best_run_id: Annotated[str, Field(description="run_id del mejor modelo seleccionado.")],
    metrics: Annotated[
        dict[str, Any],
        Field(description="Métricas del mejor modelo (salida de compute_metrics).")
    ],
    model_alias: Annotated[str, Field(description="Alias del modelo ganador.")],
    hyperparams: Annotated[dict[str, Any], Field(description="Hiperparámetros usados.")],
    dataset_path: Annotated[
        Optional[str],
        Field(description="Ruta al dataset original para incluir en el reporte.")
    ] = None,
    per_class_metrics: Annotated[
        Optional[dict[str, Any]],
        Field(description="Métricas por clase (solo clasificación, salida de compute_metrics).")
    ] = None,
    confusion_matrix_data: Annotated[
        Optional[dict[str, Any]],
        Field(description="Matriz de confusión serializada (salida de compute_metrics).")
    ] = None,
    overfit_warning: Annotated[
        Optional[str],
        Field(description="Aviso de overfitting detectado por compare_models, si existe.")
    ] = None,
    ranking: Annotated[
        Optional[list[dict[str, Any]]],
        Field(description=(
            "Lista de runs del ranking devuelta por compare_models. "
            "Si se pasa y contiene más de un run, se genera la tabla comparativa en el reporte. "
            "Cada entrada debe tener run_id, model_alias y test_metrics."
        ))
    ] = None,
    agent_notes: Annotated[
        Optional[str],
        Field(description="Notas del agente sobre el proceso: modelos probados, decisiones tomadas, etc.")
    ] = None,
    output_dir: Annotated[
        str,
        Field(description="Directorio donde guardar el reporte Markdown.")
    ] = "reports",
) -> dict:
    """
    Genera el reporte final del pipeline en formato Markdown.

    Incluye: resumen del experimento, métricas del mejor modelo,
    desglose por clase (clasificación), matriz de confusión textual,
    advertencias de overfitting y notas del agente.

    El agente debe llamar a esta tool como último paso del pipeline,
    pasando la información recopilada durante la ejecución.
    """
    REPORTS_DIR_ = Path(output_dir)
    REPORTS_DIR_.mkdir(parents=True, exist_ok=True)

    ts     = datetime.now()
    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
    fname  = f"report_{best_run_id}_{ts.strftime('%Y%m%d_%H%M%S')}.md"
    path   = REPORTS_DIR_ / fname

    lines: list[str] = []

    # ── Cabecera ──────────────────────────────────────────────────────────
    lines += [
        f"# Reporte ML — {model_alias}",
        "",
        f"**Fecha:** {ts_str}  ",
        f"**Run ID:** `{best_run_id}`  ",
        f"**Tarea:** {task}  ",
        f"**Modelo:** {model_alias}  ",
    ]
    if dataset_path:
        lines += [f"**Dataset:** `{dataset_path}`  "]
    lines += [""]

    # ── Hiperparámetros ───────────────────────────────────────────────────
    lines += ["## Hiperparámetros", ""]
    lines += ["| Parámetro | Valor |", "|-----------|-------|"]
    for k, v in hyperparams.items():
        lines.append(f"| {k} | {v} |")
    lines += [""]

    # ── Métricas principales ──────────────────────────────────────────────
    lines += ["## Métricas en test", ""]
    lines += [_format_metrics_table(metrics), ""]

    # ── Métricas por clase (clasificación) ────────────────────────────────
    if per_class_metrics:
        lines += ["## Métricas por clase", ""]
        lines += ["| Clase | Precision | Recall | F1 | Support |",
                  "|-------|-----------|--------|----|---------|"]
        for cls, m in per_class_metrics.items():
            lines.append(
                f"| {cls} | {m['precision']:.4f} | {m['recall']:.4f} "
                f"| {m['f1']:.4f} | {m['support']} |"
            )
        lines += [""]

    # ── Matriz de confusión ───────────────────────────────────────────────
    if confusion_matrix_data:
        labels = confusion_matrix_data.get("labels", [])
        matrix = confusion_matrix_data.get("matrix", [])
        if labels and matrix:
            lines += ["## Matriz de confusión", ""]
            header = "| | " + " | ".join(f"**{l}**" for l in labels) + " |"
            sep    = "|---|" + "---|" * len(labels)
            lines += [header, sep]
            for label, row in zip(labels, matrix):
                lines.append(f"| **{label}** | " + " | ".join(str(v) for v in row) + " |")
            lines += [""]

    # ── Advertencias ─────────────────────────────────────────────────────
    if overfit_warning:
        lines += ["## ⚠️ Advertencias", "", f"> {overfit_warning}", ""]

    # ── Notas del agente ──────────────────────────────────────────────────
    if agent_notes:
        lines += ["## Notas del agente", "", agent_notes, ""]

    # ── Comparativa de runs del mismo dataset ────────────────────────────
    if ranking and len(ranking) > 1:
        dataset_label = f" (dataset: `{dataset_path}`)" if dataset_path else ""
        lines += [f"## Comparativa de runs{dataset_label}", ""]
        metric_keys = list(ranking[0].get("test_metrics", metrics).keys())
        lines += ["| Run ID | Modelo | " + " | ".join(metric_keys) + " |",
                  "|--------|--------|" + "--------|" * len(metric_keys)]
        for run in ranking:
            tm = run.get("test_metrics", {})
            vals = " | ".join(
                f"{tm.get(k, 'N/A'):.4f}" if isinstance(tm.get(k), float) else str(tm.get(k, "N/A"))
                for k in metric_keys
            )
            marker = " ✅" if run["run_id"] == best_run_id else ""
            lines.append(f"| `{run['run_id']}`{marker} | {run.get('model_alias', '?')} | {vals} |")
        lines += [""]

    # ── Footer ────────────────────────────────────────────────────────────
    lines += [
        "---",
        f"*Generado automáticamente por EvalMCPServer · {ts_str}*",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "status":      "generated",
        "report_path": str(path),
        "run_id":      best_run_id,
        "model_alias": model_alias,
        "summary": {
            "task":       task,
            "model":      model_alias,
            "metrics":    metrics,
            "timestamp":  ts_str,
        },
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting Eval MCP server at port 9003...")
    #mcp.run(transport="sse", port=9003)
    uvicorn.run(mcp.sse_app(), host="localhost", port=9003)