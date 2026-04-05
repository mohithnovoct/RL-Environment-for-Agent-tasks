---
title: Reinforcement Learning Agent Environment OpenEnv
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
python_version: "3.10"
---
# Data Cleaning Environment — OpenEnv

An **OpenEnv** environment that simulates real-world **data cleaning / data wrangling** tasks. An AI agent receives messy, real-world–style tabular data and must issue cleaning actions (fix types, handle nulls, remove duplicates, standardize formats, etc.) to transform it into a clean target state.

---

## Why Data Cleaning?

- **Real-world utility**: Data cleaning consumes 60–80% of a data professional's time. This environment provides a genuine, practical test for AI agents.
- **Novelty**: Unlike toy or game environments, data cleaning is a rich, under-explored domain for agent evaluation.
- **Rich reward signal**: Partial progress is natural — each column/row cleaned moves the quality score toward 1.0.
- **Difficulty scaling**: Easy tasks have 1–2 issue types; hard tasks combine all issue types with ambiguous transformations.

---

## Tasks

| Task | Difficulty | Columns | Issue Types | Max Steps | Pass Threshold |
|------|-----------|---------|-------------|-----------|----------------|
| `basic_cleanup` | Easy | 3 (name, age, email) | Nulls, whitespace | 10 | 80% |
| `type_and_format` | Medium | 5 (product, price, qty, date, category) | Nulls, type errors, date formats, casing | 15 | 70% |
| `full_pipeline` | Hard | 7 (employee, dept, salary, date, phone, score, status) | All issue types | 20 | 60% |

All tasks use **fixed seeds** for deterministic, reproducible scoring.

---

## Action Space

The agent sends a `DataCleaningAction` with:

| Field | Type | Description |
|-------|------|-------------|
| `operation` | `CleaningOperation` | Which cleaning operation to apply |
| `column` | `str` | Target column name (empty for row-level ops) |
| `params` | `Dict[str, Any]` | Operation-specific parameters |

### Available Operations

| Operation | Params | Description |
|-----------|--------|-------------|
| `fill_null` | `{"strategy": "mean"\|"median"\|"mode"}` or `{"value": <fill>}` | Fill missing values |
| `fix_type` | `{"target_type": "int"\|"float"\|"str"\|"datetime"}` | Convert column dtype |
| `remove_duplicates` | `{}` | Remove duplicate rows (no column needed) |
| `standardize_format` | `{"format": "%Y-%m-%d"}` or `{"decimal_places": 2}` | Standardize date/number format |
| `trim_whitespace` | `{}` | Strip leading/trailing whitespace |
| `fix_case` | `{"case": "lower"\|"upper"\|"title"}` | Standardize text casing |
| `remove_outliers` | `{"method": "iqr"\|"zscore", "threshold": 1.5}` | Remove statistical outliers |
| `drop_column` | `{}` | Drop a column entirely |

---

## Observation Space

The agent receives a `DataCleaningObservation` with:

| Field | Type | Description |
|-------|------|-------------|
| `data_preview` | `List[Dict]` | First 5 rows of the current dataset |
| `column_stats` | `Dict` | Per-column: dtype, null_count, unique_count, sample_values |
| `issues_detected` | `List[str]` | Remaining data quality issues |
| `data_quality_score` | `float` | Overall quality score [0.0–1.0] |
| `task_name` | `str` | Active task name |
| `step_number` | `int` | Current step number |
| `max_steps` | `int` | Maximum steps allowed |
| `done` | `bool` | Whether the episode is finished |
| `reward` | `float` | Reward for the last action (shaped: improvement delta) |

---

## Reward Function

The quality score is a **weighted average** of 4 components (25% each):

| Component | Measures |
|-----------|----------|
| **Null completeness** | Fraction of expected non-null cells that are filled |
| **Type correctness** | Fraction of columns with correct dtype |
| **Duplicate absence** | Penalizes excess rows vs. clean reference |
| **Format consistency** | How well values match clean reference format |

**Reward per step** = `new_quality_score - old_quality_score` (shaped reward signal).

**Episode ends** when `quality_score >= 0.95` **or** `step_count >= max_steps`.

---

## Project Structure

```
RL_ENV/
├── __init__.py                  # Package exports
├── models.py                    # Pydantic Action/Observation models
├── client.py                    # EnvClient subclass
├── data_generator.py            # Seeded messy data generation
├── inference.py                 # Baseline inference script
├── openenv.yaml                 # OpenEnv manifest
├── pyproject.toml               # Package config + deps
├── .dockerignore                # Docker exclusions
└── server/
    ├── __init__.py
    ├── data_cleaning_environment.py  # Core environment: reset/step/state
    ├── graders.py                    # 3 task graders (easy/medium/hard)
    ├── app.py                        # FastAPI app (create_app)
    ├── Dockerfile                    # Container build
    └── requirements.txt              # Server dependencies
```

---

## Setup

### Install Dependencies

```bash
pip install -e .
# or with uv:
uv sync
```

### Start the Server

```bash
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

### Test the API

```bash
# Reset environment (basic_cleanup task)
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "basic_cleanup"}'

# Take a cleaning step
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"operation": "fill_null", "column": "age", "params": {"strategy": "mean"}}'
```

---

## Docker

### Build

```bash
docker build -f server/Dockerfile -t data-cleaning-env .
```

### Run

```bash
docker run -p 8000:8000 data-cleaning-env
```

---

## Inference Script

The `inference.py` baseline runs all 3 tasks using an LLM (default: `Qwen/Qwen2.5-72B-Instruct`) and emits the required `[START]`/`[STEP]`/`[END]` stdout format.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `HF_TOKEN` | (required) | Hugging Face API key |
| `IMAGE_NAME` | (optional) | Docker image name |

### Run

```bash
export HF_TOKEN="your_hf_token"
python inference.py
```

### Expected Output Format

```
[START] task=basic_cleanup env=data_cleaning_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=fill_null('age') reward=0.05 done=false error=null
[STEP] step=2 action=trim_whitespace('name') reward=0.03 done=false error=null
...
[END] success=true steps=5 rewards=0.05,0.03,0.02,0.01,0.04
```

---

## Client Usage

### Async

```python
from models import DataCleaningAction
from client import DataCleaningEnv

async with DataCleaningEnv(base_url="http://localhost:8000") as env:
    result = await env.reset()
    result = await env.step(DataCleaningAction(
        operation="fill_null",
        column="age",
        params={"strategy": "mean"}
    ))
    print(result.observation.data_quality_score)
```

### Sync

```python
with DataCleaningEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset()
    result = env.step(DataCleaningAction(
        operation="trim_whitespace",
        column="name"
    ))
```

---

## License

© 2026. All rights reserved.
