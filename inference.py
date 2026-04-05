"""
Inference Script — Data Cleaning OpenEnv Environment
=====================================================

MANDATORY ENVIRONMENT VARIABLES:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    IMAGE_NAME     The Docker image name (if using from_docker_image).

Defaults:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment imports (local package)
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import CleaningOperation, DataCleaningAction, DataCleaningObservation
from data_generator import TASK_CONFIGS, generate_task_data
from server.data_cleaning_environment import (
    DataCleaningEnvironment,
    compute_quality_score,
    detect_issues,
    get_column_stats,
)
from server.graders import grade_task

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

BENCHMARK = "data_cleaning_env"
ALL_TASKS = ["basic_cleanup", "type_and_format", "full_pipeline"]
TEMPERATURE = 0.3
MAX_TOKENS = 512

# ---------------------------------------------------------------------------
# Stdout logging helpers (exact format required by the competition)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert data cleaning agent. You interact with a data cleaning
environment where you must fix quality issues in messy tabular data.

Each turn you receive:
- A preview of the current data (first 5 rows)
- Per-column statistics (dtype, null counts, unique counts)
- A list of detected data quality issues
- Your current quality score (0.0 to 1.0)

You must respond with a JSON object specifying a cleaning action:
{
    "operation": "<operation>",
    "column": "<column_name>",
    "params": {<operation-specific parameters>}
}

Available operations and their params:
- "fill_null": Fill missing values. params: {"strategy": "mean"|"median"|"mode"} or {"value": <fill_value>}
- "fix_type": Convert column type. params: {"target_type": "int"|"float"|"str"|"datetime"}
- "remove_duplicates": Remove duplicate rows. params: {} (column can be empty)
- "standardize_format": Standardize date/number formats. params: {"format": "%Y-%m-%d"} or {"decimal_places": 2}
- "trim_whitespace": Strip whitespace from strings. params: {}
- "fix_case": Standardize text casing. params: {"case": "lower"|"upper"|"title"}
- "remove_outliers": Remove statistical outliers. params: {"method": "iqr"|"zscore", "threshold": 1.5}
- "drop_column": Drop a column. params: {}

Strategy:
1. Start with the most impactful issues (duplicates, nulls)
2. Fix type errors before format issues
3. Trim whitespace before fixing case
4. Use remove_outliers carefully (it removes rows)

Respond with ONLY the JSON object, no explanation.
""").strip()


# ---------------------------------------------------------------------------
# Build observation prompt
# ---------------------------------------------------------------------------

def build_observation_prompt(obs_data: Dict[str, Any], step: int, last_reward: float) -> str:
    """Build a user prompt from the environment observation."""
    preview = obs_data.get("data_preview", [])
    stats = obs_data.get("column_stats", {})
    issues = obs_data.get("issues_detected", [])
    score = obs_data.get("data_quality_score", 0.0)
    max_steps = obs_data.get("max_steps", 20)
    task = obs_data.get("task_name", "unknown")

    # Format data preview as a table
    if preview:
        cols = list(preview[0].keys()) if preview else []
        header = " | ".join(cols)
        rows = []
        for row in preview[:5]:
            rows.append(" | ".join(str(row.get(c, "")) for c in cols))
        table = f"{header}\n" + "\n".join(rows)
    else:
        table = "(no data)"

    # Format issues
    issues_text = "\n".join(f"  - {issue}" for issue in issues) if issues else "  (none)"

    # Format column stats
    stats_text = ""
    for col_name, col_info in stats.items():
        stats_text += (
            f"  {col_name}: dtype={col_info.get('dtype', '?')}, "
            f"nulls={col_info.get('null_count', 0)}/{col_info.get('total_count', 0)}, "
            f"unique={col_info.get('unique_count', 0)}\n"
        )

    return textwrap.dedent(f"""
Task: {task}
Step: {step}/{max_steps}
Current Quality Score: {score:.4f}
Last Reward: {last_reward:.4f}

DATA PREVIEW:
{table}

COLUMN STATS:
{stats_text}
DETECTED ISSUES:
{issues_text}

Respond with a single JSON cleaning action.
    """).strip()


# ---------------------------------------------------------------------------
# Parse LLM response
# ---------------------------------------------------------------------------

def parse_action_from_response(text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON action from the LLM response text."""
    text = text.strip()

    # Try to extract JSON from code blocks
    if "```" in text:
        parts = text.split("```")
        for part in parts[1:]:
            # Remove language identifier
            lines = part.strip().split("\n")
            if lines[0].lower() in ("json", "python", ""):
                lines = lines[1:]
            candidate = "\n".join(lines).strip()
            if candidate.startswith("{"):
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue

    # Try parsing the whole text as JSON
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Try to find JSON within the text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Get model action
# ---------------------------------------------------------------------------

def get_model_action(
    client: OpenAI,
    obs_data: Dict[str, Any],
    step: int,
    last_reward: float,
) -> Dict[str, Any]:
    """Query the LLM for a cleaning action based on current observation."""
    user_prompt = build_observation_prompt(obs_data, step, last_reward)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        action = parse_action_from_response(text)

        if action and "operation" in action:
            return action

        # Fallback: if parsing fails, try a simple heuristic
        print(f"[DEBUG] Could not parse model response: {text[:200]}", flush=True)

    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)

    # Fallback action based on common issues in observation
    issues = obs_data.get("issues_detected", [])
    return _fallback_action(issues, obs_data)


def _fallback_action(issues: List[str], obs_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a simple heuristic action when the LLM fails."""
    stats = obs_data.get("column_stats", {})

    for issue in issues:
        if "duplicate" in issue.lower():
            return {"operation": "remove_duplicates", "column": "", "params": {}}

        if "null" in issue.lower():
            # Extract column name from issue text
            for col_name in stats:
                if col_name in issue:
                    col_info = stats[col_name]
                    dtype = col_info.get("dtype", "object")
                    if dtype in ("int64", "float64", "int32", "float32"):
                        return {"operation": "fill_null", "column": col_name, "params": {"strategy": "median"}}
                    else:
                        return {"operation": "fill_null", "column": col_name, "params": {"strategy": "mode"}}

        if "whitespace" in issue.lower():
            for col_name in stats:
                if col_name in issue:
                    return {"operation": "trim_whitespace", "column": col_name, "params": {}}

        if "non-numeric" in issue.lower():
            for col_name in stats:
                if col_name in issue:
                    return {"operation": "fix_type", "column": col_name, "params": {"target_type": "float"}}

        if "casing" in issue.lower() or "case" in issue.lower():
            for col_name in stats:
                if col_name in issue:
                    return {"operation": "fix_case", "column": col_name, "params": {"case": "title"}}

        if "date" in issue.lower() or "format" in issue.lower():
            for col_name in stats:
                if col_name in issue:
                    return {"operation": "fix_type", "column": col_name, "params": {"target_type": "datetime"}}

    # Last resort
    first_col = list(stats.keys())[0] if stats else ""
    return {"operation": "trim_whitespace", "column": first_col, "params": {}}


# ---------------------------------------------------------------------------
# Run one task episode
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, task_name: str) -> None:
    """Run a single task episode with the LLM agent."""
    config = TASK_CONFIGS[task_name]

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    # Create environment
    env = DataCleaningEnvironment(task_name=task_name)

    rewards: List[float] = []
    steps_taken = 0
    success = False

    try:
        # Reset
        obs = env.reset()
        obs_data = obs.model_dump()
        last_reward = 0.0

        for step in range(1, config.max_steps + 1):
            if obs.done:
                break

            # Get action from LLM
            action_dict = get_model_action(client, obs_data, step, last_reward)

            # Create action string for logging
            op = action_dict.get("operation", "unknown")
            col = action_dict.get("column", "")
            action_str = f"{op}('{col}')" if col else f"{op}()"

            # Step the environment
            try:
                action = DataCleaningAction(**action_dict)
                obs = env.step(action)
            except Exception as e:
                # If action construction fails, log and try fallback
                obs = env.step(DataCleaningAction(
                    operation=CleaningOperation.TRIM_WHITESPACE,
                    column=list(config.columns.keys())[0],
                ))
                action_str = f"fallback_trim('{list(config.columns.keys())[0]}')"

            obs_data = obs.model_dump()

            reward = obs.reward if obs.reward is not None else 0.0
            done = obs.done
            error = obs_data.get("metadata", {}).get("error", None)

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        # Grade the episode
        final_score = obs_data.get("data_quality_score", 0.0)
        issues_remaining = len(obs_data.get("issues_detected", []))
        result = grade_task(
            task_name=task_name,
            final_quality_score=final_score,
            steps_taken=steps_taken,
            issues_remaining=issues_remaining,
        )
        success = result.passed

    except Exception as e:
        print(f"[DEBUG] Task {task_name} error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all tasks sequentially."""
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_name in ALL_TASKS:
        run_task(client, task_name)
        print("", flush=True)  # Blank line between tasks


if __name__ == "__main__":
    main()
