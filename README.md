# Multi-Agent System (MCP + A2A) — AutoML

An end-to-end AutoML pipeline built on a multi-agent architecture. LLM-driven agents communicate via the **A2A protocol** and invoke tools exposed through **MCP servers** to automate the full machine learning workflow: data preparation, model training, hyperparameter optimization, evaluation, and report generation.

---

## Architecture Overview

```
User Input (CSV path)
       │
       ▼
OrchestratorAgent (port 8000)          ← pipeline director
       │
       ├─► DataAgent (port 8001)        ← EDA & preprocessing
       │         └── DataMCPServer (port 9001)   [preview, describe, detect, preprocess, split]
       │
       ├─► MLAgent (port 8002)          ← model selection & training
       │         └── MLMCPServer (port 9002)     [train, tune_hyperparams, log_run, compare_runs]
       │
       └─► EvalAgent (port 8003)        ← evaluation & reporting
                 └── EvalMCPServer (port 9003)   [compute_metrics, compare_models, save_best, generate_report]
```

---

## Key Technologies

| Technology | Role |
|---|---|
| [Google ADK](https://google.github.io/adk-docs/) | Agent framework — `RemoteA2aAgent`, `MCPToolset`, ReAct loop |
| [A2A Protocol](https://google.github.io/A2A/) | HTTP/JSON-RPC agent-to-agent communication |
| [MCP Protocol](https://modelcontextprotocol.io/) | SSE + JSON-RPC tool exposure to LLMs |
| [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) | HTTP server for all agents |
| [FastMCP](https://github.com/jlowin/fastmcp) | MCP server framework |
| scikit-learn, XGBoost | ML model implementations |
| pandas, NumPy, SciPy | Data processing |

---

## Project Structure

```
Multi-Agent_System_MCP-A2A_AutoML/
├── A2A/
│   ├── Agents/
│   │   ├── orchestrator_agent.py   # Pipeline coordinator (LLM decides flow)
│   │   ├── data_agent.py           # EDA, cleaning, train/test split
│   │   ├── ml_agent.py             # Model selection, training, HPO
│   │   └── eval_agent.py           # Metrics, model comparison, report
│   ├── A2AServer.py                # Base A2A HTTP server (FastAPI)
│   └── base_agent.py               # Base class with ReAct loop & MCP integration
├── MCP/
│   ├── data_mcp_server.py          # Data tools (port 9001)
│   ├── ml_mcp_server.py            # ML tools (port 9002)
│   └── eval_mcp_server.py          # Evaluation tools (port 9003)
├── utils/
│   └── logger.py                   # Dual-logger configuration
├── main.py                         # Entry point — spawns agents, runs CLI
├── data/                           # Input datasets
├── experiments/                    # Experiment metadata (.json)
├── reports/                        # Final Markdown reports
└── outputs/logs/                   # Runtime logs
```

---

## MCP Servers

### DataMCPServer — port 9001

| Tool | Description |
|---|---|
| `preview_dataset` | Quick schema & sample rows |
| `describe_dataset` | Full EDA with statistics, missing values, correlations, target distribution |
| `detect_problems` | Dataset audit — flags CRITICAL / WARNING / INFO issues |
| `preprocess_dataset` | Cleans data: imputation, encoding, deduplication |
| `split_dataset` | Stratified train/test split; saves CSVs to `splits/` |

### MLMCPServer — port 9002

| Tool | Description |
|---|---|
| `train_model` | Trains a model with given hyperparameters |
| `tune_hyperparams` | RandomizedSearchCV for hyperparameter optimization |
| `log_run` | Persists experiment metadata and metrics to `experiments/` |
| `compare_runs` | Ranks models by primary metric for a given dataset |

### EvalMCPServer — port 9003

| Tool | Description |
|---|---|
| `compute_metrics` | Calculates accuracy, F1, AUC-ROC, MAE, RMSE, R², etc. |
| `compare_models` | Compares run rankings; flags overfitting |
| `save_best_model` | Copies the winning model to `models/production/` |
| `generate_report` | Creates a final Markdown report with tables and analysis |

---

## Output Artifacts

| Artifact | Location |
|---|---|
| Train/test splits | `splits/<dataset_name>/train.csv`, `test.csv` |
| Trained models | `models/<run_id>.pkl` |
| Best model (production) | `models/production/<run_id>.pkl` |
| Experiment logs | `experiments/<run_id>.json` |
| Final report | `reports/report_<run_id>_<timestamp>.md` |
| Agent reasoning log | `outputs/logs/agents_v3.log` |
| Infrastructure log | `outputs/logs/ml_pipeline.log` |

---

## Prerequisites

- Python >= 3.11
- A [Google AI API key](https://aistudio.google.com/apikey) with access to Gemini models

---

## Installation

```bash
git clone <repo-url>
cd Multi-Agent_System_MCP-A2A_AutoML

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

---

## Running the System

### Step 1 — Start the MCP Servers

Open three separate terminals and run:

```bash
python MCP/data_mcp_server.py    # port 9001
python MCP/ml_mcp_server.py      # port 9002
python MCP/eval_mcp_server.py    # port 9003
```

### Step 2 — Start the Pipeline

```bash
# Provide a dataset directly
python main.py --dataset /path/to/dataset.csv
```

`main.py` will:
1. Verify all three MCP servers are reachable
2. Spawn the four agents (each in its own process)
3. Wait up to 120 seconds for all agents to pass their `/health` check
4. Send the dataset path to the OrchestratorAgent via JSON-RPC
5. Print the final Markdown report to stdout

---

## Logs

Two log files are maintained at runtime:

- `outputs/logs/ml_pipeline.log` — infrastructure events (startup, health checks, errors)
- `outputs/logs/agents.log` — full agent reasoning trace (tool calls, LLM responses, decisions)
