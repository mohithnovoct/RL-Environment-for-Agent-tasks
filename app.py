"""
Data Cleaning Environment — Gradio App for Hugging Face Spaces
===============================================================

Interactive web interface for the Data Cleaning OpenEnv environment.
Lets users (or LLM agents) select a task, inspect messy data, apply
cleaning operations, and track quality score improvements in real-time.
"""

import json
import warnings
from typing import Any, Dict, List, Optional

import gradio as gr
import pandas as pd

warnings.filterwarnings("ignore")

# ── Local imports ────────────────────────────────────────────────────────────
from models import CleaningOperation, DataCleaningAction, DataCleaningObservation
from data_generator import TASK_CONFIGS
from server.data_cleaning_environment import (
    DataCleaningEnvironment,
    compute_quality_score,
    detect_issues,
    get_column_stats,
)
from server.graders import grade_task, PASS_THRESHOLDS

# ── Constants ────────────────────────────────────────────────────────────────
OPERATIONS = [op.value for op in CleaningOperation]
TASKS = list(TASK_CONFIGS.keys())

TASK_META = {
    name: {
        "difficulty": cfg.difficulty.capitalize(),
        "columns": list(cfg.columns.keys()),
        "col_types": cfg.columns,
        "issues": cfg.issue_types,
        "max_steps": cfg.max_steps,
        "description": cfg.description,
        "threshold": PASS_THRESHOLDS.get(name, 0.70),
    }
    for name, cfg in TASK_CONFIGS.items()
}

OPERATION_INFO = {
    "fill_null":           ("", "Fill Nulls",           "Fill missing values with a strategy or constant.",
                            '{"strategy": "mean"}',      'strategy: mean | median | mode — or value: <val>'),
    "fix_type":            ("", "Fix Type",             "Cast a column to the correct data type.",
                            '{"target_type": "float"}',  'target_type: int | float | str | datetime'),
    "remove_duplicates":   ("", "Remove Duplicates",   "Drop duplicate rows from the dataset.",
                            '{}',                         'No params needed. Operates on entire table.'),
    "standardize_format":  ("", "Standardize Format",   "Normalise date or numeric formats.",
                            '{"format": "%Y-%m-%d"}',    'format: strftime pattern — or decimal_places: int'),
    "trim_whitespace":     ("", "Trim Whitespace",     "Strip leading / trailing spaces from strings.",
                            '{}',                         'No params needed.'),
    "fix_case":            ("", "Fix Case",             "Standardise text casing in a column.",
                            '{"case": "title"}',         'case: lower | upper | title'),
    "remove_outliers":     ("", "Remove Outliers",      "Remove statistical outliers from numeric columns.",
                            '{"method": "iqr", "threshold": 1.5}',
                            'method: iqr | zscore — threshold: float'),
    "drop_column":         ("", "Drop Column",         "Remove a column entirely from the dataset.",
                            '{}',                         'No params needed.'),
}

DIFFICULTY_COLORS = {"Easy": "#22c55e", "Medium": "#f59e0b", "Hard": "#ef4444"}
DIFFICULTY_EMOJI  = {"Easy": "", "Medium": "", "Hard": ""}


# ── CSS ──────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
/* ── Layout ─────────────────────────────────────── */
.gradio-container { max-width: 1360px !important; }

/* ── Hero banner ────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 8px;
    color: white;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 30% 40%, rgba(59,130,246,0.12) 0%, transparent 60%),
                radial-gradient(circle at 80% 70%, rgba(16,185,129,0.10) 0%, transparent 50%);
    pointer-events: none;
}
.hero-banner h1 {
    margin: 0 0 6px 0 !important;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    position: relative;
}
.hero-banner p {
    margin: 0 !important;
    font-size: 0.95rem;
    color: #94a3b8;
    position: relative;
    line-height: 1.5;
}

/* ── Score gauge ─────────────────────────────────── */
.score-gauge {
    text-align: center;
    padding: 8px 0 4px;
}
.score-ring {
    display: inline-block;
    position: relative;
    width: 150px; height: 150px;
}
.score-ring svg { transform: rotate(-90deg); }
.score-ring .track {
    fill: none;
    stroke: #1e293b;
    stroke-width: 10;
}
.score-ring .fill {
    fill: none;
    stroke-width: 10;
    stroke-linecap: round;
    transition: stroke-dashoffset 0.8s cubic-bezier(.4,0,.2,1),
                stroke 0.5s ease;
}
.score-label {
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -1px;
}

/* ── Status badge ───────────────────────────────── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 16px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}
.pill-idle    { background: #1e293b; color: #64748b; border: 1px solid #334155; }
.pill-running { background: #1e3a5f; color: #60a5fa; border: 1px solid #2563eb; }
.pill-passed  { background: #14532d; color: #4ade80; border: 1px solid #22c55e; }
.pill-failed  { background: #450a0a; color: #f87171; border: 1px solid #ef4444; }
.pill-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
}
.pill-idle .pill-dot    { background: #64748b; }
.pill-running .pill-dot { background: #60a5fa; animation: pulse-dot 1.5s infinite; }
.pill-passed .pill-dot  { background: #4ade80; }
.pill-failed .pill-dot  { background: #f87171; }
@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.3; }
}

/* ── Task card ──────────────────────────────────── */
.task-card {
    background: linear-gradient(135deg, #f8fafc, #f1f5f9);
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 14px 18px;
    margin-top: 6px;
    font-size: 0.88rem;
    line-height: 1.55;
}
.task-card .tc-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
}
.task-card .tc-diff {
    font-size: 0.72rem;
    font-weight: 700;
    padding: 2px 10px;
    border-radius: 999px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.task-card .tc-desc {
    color: #475569;
    margin-bottom: 10px;
}
.task-card .tc-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}
.task-card .tc-chip {
    background: #e2e8f0;
    color: #334155;
    padding: 2px 10px;
    border-radius: 6px;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Operation hint ─────────────────────────────── */
.op-hint {
    background: #f0f9ff;
    border-left: 3px solid #3b82f6;
    padding: 8px 14px;
    border-radius: 0 8px 8px 0;
    font-size: 0.82rem;
    color: #334155;
    margin-top: 4px;
    line-height: 1.5;
}
.op-hint code {
    background: #dbeafe;
    padding: 1px 5px;
    border-radius: 4px;
    font-size: 0.78rem;
}

/* ── Step counter ───────────────────────────────── */
.step-bar {
    margin-top: 6px;
}
.step-track {
    height: 6px;
    background: #1e293b;
    border-radius: 3px;
    overflow: hidden;
    margin-bottom: 4px;
}
.step-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.5s ease;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
}
.step-text {
    font-size: 0.78rem;
    color: #64748b;
    text-align: right;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Issues list ────────────────────────────────── */
.issue-item {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 6px 0;
    border-bottom: 1px solid #f1f5f9;
    font-size: 0.85rem;
    line-height: 1.45;
}
.issue-item:last-child { border-bottom: none; }
.issue-icon {
    flex-shrink: 0;
    width: 20px; height: 20px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.7rem;
    margin-top: 1px;
}
.issue-warn  { background: #fef3c7; color: #d97706; }
.issue-ok    { background: #dcfce7; color: #16a34a; }

/* ── History table ──────────────────────────────── */
.history-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
    font-family: 'JetBrains Mono', monospace;
}
.history-table th {
    background: #f1f5f9;
    padding: 8px 10px;
    text-align: left;
    font-weight: 700;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #64748b;
    border-bottom: 2px solid #e2e8f0;
}
.history-table td {
    padding: 7px 10px;
    border-bottom: 1px solid #f1f5f9;
}
.history-table tr:hover td {
    background: #f8fafc;
}
.reward-pos { color: #16a34a; font-weight: 600; }
.reward-neg { color: #dc2626; font-weight: 600; }
.reward-zero { color: #94a3b8; }

/* ── Column stats ───────────────────────────────── */
.col-stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 8px;
    margin-top: 4px;
}
.col-stat-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 10px 14px;
}
.col-stat-name {
    font-weight: 700;
    font-size: 0.82rem;
    color: #1e293b;
    margin-bottom: 4px;
    font-family: 'JetBrains Mono', monospace;
}
.col-stat-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    color: #64748b;
    padding: 1px 0;
}
.col-stat-val {
    font-family: 'JetBrains Mono', monospace;
    color: #334155;
}
.null-bar-track {
    height: 4px;
    background: #e2e8f0;
    border-radius: 2px;
    margin-top: 4px;
    overflow: hidden;
}
.null-bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.4s ease;
}

/* ── Completion card ────────────────────────────── */
.completion-card {
    text-align: center;
    padding: 20px;
    border-radius: 14px;
    margin-top: 8px;
}
.completion-passed {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 1px solid #bbf7d0;
}
.completion-failed {
    background: linear-gradient(135deg, #fef2f2, #fee2e2);
    border: 1px solid #fecaca;
}
.completion-title {
    font-size: 1.3rem;
    font-weight: 800;
    margin-bottom: 4px;
}
.completion-sub {
    font-size: 0.9rem;
    color: #64748b;
}
"""


# ── Helper: SVG ring gauge ───────────────────────────────────────────────────

def build_score_gauge(score: float) -> str:
    """Render an SVG ring gauge with animated fill."""
    pct = int(score * 100)
    r = 60
    circ = 2 * 3.14159 * r
    offset = circ * (1 - score)

    if score >= 0.8:
        color = "#22c55e"
    elif score >= 0.5:
        color = "#f59e0b"
    else:
        color = "#ef4444"

    return f"""
    <div class="score-gauge">
      <div class="score-ring">
        <svg width="150" height="150" viewBox="0 0 150 150">
          <circle class="track" cx="75" cy="75" r="{r}" />
          <circle class="fill" cx="75" cy="75" r="{r}"
                  stroke="{color}"
                  stroke-dasharray="{circ:.1f}"
                  stroke-dashoffset="{offset:.1f}" />
        </svg>
        <span class="score-label" style="color:{color}">{pct}%</span>
      </div>
    </div>
    """


# ── Helper: Status pill ──────────────────────────────────────────────────────

def build_status_pill(status: str) -> str:
    return f"""<span class="status-pill pill-{status}">
        <span class="pill-dot"></span> {status}
    </span>"""


# ── Helper: Step progress bar ────────────────────────────────────────────────

def build_step_bar(step: int, max_steps: int) -> str:
    pct = min(step / max_steps * 100, 100) if max_steps > 0 else 0
    return f"""<div class="step-bar">
        <div class="step-track"><div class="step-fill" style="width:{pct:.0f}%"></div></div>
        <div class="step-text">{step} ∕ {max_steps} steps</div>
    </div>"""


# ── Helper: Task card ────────────────────────────────────────────────────────

def build_task_card(task_name: str) -> str:
    info = TASK_META[task_name]
    diff = info["difficulty"]
    color = DIFFICULTY_COLORS.get(diff, "#6b7280")
    emoji = DIFFICULTY_EMOJI.get(diff, "")

    chips = "".join(f'<span class="tc-chip">{c}</span>' for c in info["columns"])
    issue_chips = "".join(
        f'<span class="tc-chip" style="background:#fef3c7;color:#92400e">{i}</span>'
        for i in info["issues"]
    )
    return f"""<div class="task-card">
        <div class="tc-header">
            <span class="tc-diff" style="background:{color}22;color:{color}">{diff}</span>
            <span style="color:#94a3b8;font-size:0.78rem">
                Pass ≥ {int(info['threshold']*100)}% · Max {info['max_steps']} steps
            </span>
        </div>
        <div class="tc-desc">{info['description']}</div>
        <div style="font-size:0.78rem;font-weight:600;color:#64748b;margin-bottom:4px">COLUMNS</div>
        <div class="tc-meta">{chips}</div>
        <div style="font-size:0.78rem;font-weight:600;color:#64748b;margin:8px 0 4px">ISSUES INJECTED</div>
        <div class="tc-meta">{issue_chips}</div>
    </div>"""


# ── Helper: Operation hint ───────────────────────────────────────────────────

def build_op_hint(operation: str) -> str:
    icon, name, desc, _, hint = OPERATION_INFO.get(
        operation, ("", operation, "", "{}", "")
    )
    return f"""<div class="op-hint">
        <strong>{name}</strong> — {desc}<br/>
        <code>{hint}</code>
    </div>"""


# ── Helper: Issues HTML ──────────────────────────────────────────────────────

def build_issues_html(issues: List[str]) -> str:
    if not issues:
        return """<div class="issue-item">
            <span class="issue-icon issue-ok">✓</span>
            <span style="color:#16a34a;font-weight:600">No issues remaining — data is clean!</span>
        </div>"""
    items = []
    for iss in issues:
        items.append(f"""<div class="issue-item">
            <span class="issue-icon issue-warn">!</span>
            <span>{iss}</span>
        </div>""")
    return "".join(items)


# ── Helper: Column stats ─────────────────────────────────────────────────────

def build_column_stats_html(stats: Dict[str, Any]) -> str:
    if not stats:
        return '<div style="color:#94a3b8;text-align:center;padding:16px">Reset the environment to see column statistics.</div>'
    cards = []
    for col, info in stats.items():
        nulls = info.get("null_count", 0)
        total = info.get("total_count", 1)
        null_pct = nulls / total * 100 if total > 0 else 0
        null_color = "#ef4444" if null_pct > 10 else ("#f59e0b" if null_pct > 0 else "#22c55e")

        cards.append(f"""<div class="col-stat-card">
            <div class="col-stat-name">{col}</div>
            <div class="col-stat-row"><span>Type</span><span class="col-stat-val">{info.get('dtype','?')}</span></div>
            <div class="col-stat-row"><span>Nulls</span><span class="col-stat-val">{nulls}/{total}</span></div>
            <div class="col-stat-row"><span>Unique</span><span class="col-stat-val">{info.get('unique_count',0)}</span></div>
            <div class="null-bar-track"><div class="null-bar-fill" style="width:{null_pct:.0f}%;background:{null_color}"></div></div>
        </div>""")
    return f'<div class="col-stats-grid">{"".join(cards)}</div>'


# ── Helper: Step history table ───────────────────────────────────────────────

def build_history_html(history: List[Dict]) -> str:
    if not history:
        return '<div style="color:#94a3b8;text-align:center;padding:24px">Take a step to see the history here.</div>'

    rows = []
    for h in history:
        r = h["reward"]
        if r > 0:
            cls, sign = "reward-pos", "+"
        elif r < 0:
            cls, sign = "reward-neg", ""
        else:
            cls, sign = "reward-zero", ""
        icon = OPERATION_INFO.get(h["operation"], ("·",))[0]
        rows.append(f"""<tr>
            <td>{h['step']}</td>
            <td>{icon} {h['operation']}</td>
            <td>{h['column'] or '—'}</td>
            <td class="{cls}">{sign}{r:.4f}</td>
            <td>{h['score']:.3f}</td>
        </tr>""")
    return f"""<table class="history-table">
        <thead><tr><th>#</th><th>Operation</th><th>Column</th><th>Reward</th><th>Score</th></tr></thead>
        <tbody>{"".join(rows)}</tbody>
    </table>"""


# ── Helper: Completion card ──────────────────────────────────────────────────

def build_completion_html(passed: bool, score: float, steps: int, issues_left: int) -> str:
    if passed:
        return f"""<div class="completion-card completion-passed">
            <div class="completion-title">Task Passed!</div>
            <div class="completion-sub">Final score <strong>{int(score*100)}%</strong> in {steps} steps · {issues_left} issues remaining</div>
        </div>"""
    else:
        return f"""<div class="completion-card completion-failed">
            <div class="completion-title">Task Failed</div>
            <div class="completion-sub">Final score <strong>{int(score*100)}%</strong> in {steps} steps · {issues_left} issues remaining</div>
        </div>"""


# ── Callbacks ────────────────────────────────────────────────────────────────

def columns_for_task(task_name: str) -> List[str]:
    return list(TASK_CONFIGS[task_name].columns.keys())


def reset_environment(task_name: str, state: Dict):
    env = DataCleaningEnvironment(task_name=task_name)
    obs = env.reset()
    obs_data = obs.model_dump()
    score = obs.data_quality_score
    config = TASK_CONFIGS[task_name]

    state = {
        "env": env,
        "task_name": task_name,
        "obs_data": obs_data,
        "score": score,
        "step": 0,
        "history": [],
        "done": False,
        "passed": False,
    }
    cols = columns_for_task(task_name)
    default_op = OPERATIONS[0]

    return (
        state,
        pd.DataFrame(obs.data_preview),
        build_score_gauge(score),
        build_status_pill("running"),
        build_step_bar(0, config.max_steps),
        build_task_card(task_name),
        build_issues_html(obs.issues_detected),
        build_column_stats_html(obs_data.get("column_stats", {})),
        build_history_html([]),
        "",  # completion card hidden
        build_op_hint(default_op),
        gr.update(choices=cols, value=cols[0] if cols else ""),
        gr.update(interactive=True),
    )


def take_step(operation: str, column: str, params_json: str, state: Dict):
    if not state or "env" not in state:
        return (state, pd.DataFrame(), build_score_gauge(0),
                build_status_pill("idle"), build_step_bar(0, 1),
                build_issues_html([]), build_column_stats_html({}),
                build_history_html([]), "")

    if state.get("done"):
        # Already done — just return current state
        return (state, pd.DataFrame(state["obs_data"].get("data_preview", [])),
                build_score_gauge(state["score"]),
                build_status_pill("passed" if state.get("passed") else "failed"),
                build_step_bar(state["step"], TASK_CONFIGS[state["task_name"]].max_steps),
                build_issues_html(state["obs_data"].get("issues_detected", [])),
                build_column_stats_html(state["obs_data"].get("column_stats", {})),
                build_history_html(state["history"]),
                build_completion_html(state.get("passed", False), state["score"],
                                      state["step"], len(state["obs_data"].get("issues_detected", []))))

    env: DataCleaningEnvironment = state["env"]
    task_name = state["task_name"]
    config = TASK_CONFIGS[task_name]

    params = {}
    if params_json and params_json.strip():
        try:
            params = json.loads(params_json)
        except json.JSONDecodeError:
            pass

    try:
        action = DataCleaningAction(
            operation=CleaningOperation(operation),
            column=column,
            params=params,
        )
        obs = env.step(action)
    except Exception as e:
        return (state, pd.DataFrame(state["obs_data"].get("data_preview", [])),
                build_score_gauge(state["score"]),
                build_status_pill("running"),
                build_step_bar(state["step"], config.max_steps),
                build_issues_html([f"Action error: {e}"]),
                build_column_stats_html(state["obs_data"].get("column_stats", {})),
                build_history_html(state["history"]), "")

    obs_data = obs.model_dump()
    score = obs.data_quality_score
    reward = obs.reward if obs.reward is not None else 0.0
    step_num = state["step"] + 1
    done = obs.done

    state["obs_data"] = obs_data
    state["score"] = score
    state["step"] = step_num
    state["done"] = done
    state["history"].append({
        "step": step_num,
        "operation": operation,
        "column": column,
        "reward": reward,
        "score": score,
    })

    completion_html = ""
    if done:
        result = grade_task(task_name=task_name, final_quality_score=score,
                            steps_taken=step_num, issues_remaining=len(obs.issues_detected))
        state["passed"] = result.passed
        status = "passed" if result.passed else "failed"
        completion_html = build_completion_html(result.passed, score, step_num, len(obs.issues_detected))
    else:
        status = "running"

    return (
        state,
        pd.DataFrame(obs_data.get("data_preview", [])),
        build_score_gauge(score),
        build_status_pill(status),
        build_step_bar(step_num, config.max_steps),
        build_issues_html(obs.issues_detected),
        build_column_stats_html(obs_data.get("column_stats", {})),
        build_history_html(state["history"]),
        completion_html,
    )


def on_task_change(task_name: str):
    cols = columns_for_task(task_name)
    return (
        build_task_card(task_name),
        gr.update(choices=cols, value=cols[0] if cols else ""),
    )


def on_op_change(operation: str):
    _, _, _, default_params, _ = OPERATION_INFO.get(
        operation, ("", "", "", "{}", "")
    )
    return default_params, build_op_hint(operation)


# ── Build UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="Data Cleaning Environment — OpenEnv",
    css=CUSTOM_CSS,
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    ),
) as demo:

    env_state = gr.State({})

    # ── Hero banner ──────────────────────────────────────────────────────
    gr.HTML("""<div class="hero-banner">
        <h1>Data Cleaning Environment</h1>
        <p>An <strong>OpenEnv</strong> environment for tabular data cleaning.
           Select a task, inspect the messy data, and apply cleaning operations
           step-by-step to improve data quality to the target threshold.</p>
    </div>""")

    with gr.Row(equal_height=False):

        # ═══════════════ LEFT PANEL ═══════════════════════════════════════
        with gr.Column(scale=1, min_width=300):

            # ── Task selector ────────────────────────────────────────────
            with gr.Group():
                gr.Markdown("#### Task")
                task_dropdown = gr.Dropdown(
                    choices=TASKS, value=TASKS[0],
                    label="Select task", show_label=False,
                    interactive=True,
                )
                task_card_html = gr.HTML(value=build_task_card(TASKS[0]))
                reset_btn = gr.Button("Reset Environment",
                                      variant="primary", size="lg")

            # ── Score gauge ──────────────────────────────────────────────
            with gr.Group():
                gr.Markdown("#### Quality Score")
                score_gauge = gr.HTML(value=build_score_gauge(0.0))
                status_pill = gr.HTML(value=build_status_pill("idle"))
                step_bar = gr.HTML(value=build_step_bar(0, 10))

            # ── Completion banner ────────────────────────────────────────
            completion_banner = gr.HTML(value="")

        # ═══════════════ RIGHT PANEL ══════════════════════════════════════
        with gr.Column(scale=2):

            # ── Action controls ──────────────────────────────────────────
            with gr.Group():
                gr.Markdown("#### Cleaning Action")
                with gr.Row():
                    op_dropdown = gr.Dropdown(
                        choices=OPERATIONS, value=OPERATIONS[0],
                        label="Operation", interactive=True, scale=2,
                    )
                    col_dropdown = gr.Dropdown(
                        choices=columns_for_task(TASKS[0]),
                        value=columns_for_task(TASKS[0])[0],
                        label="Column", interactive=True, scale=2,
                    )
                op_hint = gr.HTML(value=build_op_hint(OPERATIONS[0]))
                with gr.Row():
                    params_box = gr.Textbox(
                        value='{"strategy": "mean"}',
                        label="Params (JSON)",
                        placeholder='e.g. {"strategy": "mean"}',
                        scale=3,
                    )
                    step_btn = gr.Button("Execute Step",
                                         variant="secondary",
                                         interactive=False,
                                         scale=1, size="lg")

            # ── Tabs: Preview / Stats / History ──────────────────────────
            with gr.Tabs():
                with gr.Tab("Data Preview"):
                    data_table = gr.Dataframe(
                        value=pd.DataFrame(),
                        interactive=False, wrap=True,
                    )
                with gr.Tab("Column Stats"):
                    col_stats_html = gr.HTML(
                        value=build_column_stats_html({})
                    )
                with gr.Tab("Issues"):
                    issues_html = gr.HTML(
                        value=build_issues_html([])
                    )
                with gr.Tab("Step History"):
                    history_html = gr.HTML(
                        value=build_history_html([])
                    )

    # ── Wiring ───────────────────────────────────────────────────────────

    task_dropdown.change(
        fn=on_task_change,
        inputs=[task_dropdown],
        outputs=[task_card_html, col_dropdown],
    )

    op_dropdown.change(
        fn=on_op_change,
        inputs=[op_dropdown],
        outputs=[params_box, op_hint],
    )

    reset_btn.click(
        fn=reset_environment,
        inputs=[task_dropdown, env_state],
        outputs=[
            env_state, data_table, score_gauge, status_pill, step_bar,
            task_card_html, issues_html, col_stats_html, history_html,
            completion_banner, op_hint, col_dropdown, step_btn,
        ],
    )

    step_btn.click(
        fn=take_step,
        inputs=[op_dropdown, col_dropdown, params_box, env_state],
        outputs=[
            env_state, data_table, score_gauge, status_pill, step_bar,
            issues_html, col_stats_html, history_html, completion_banner,
        ],
    )


# ── Launch ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
