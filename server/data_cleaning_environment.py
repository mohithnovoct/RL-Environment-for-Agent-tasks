# Copyright (c) 2026. All rights reserved.
# Data Cleaning Environment - Core Environment Logic

"""
DataCleaningEnvironment — OpenEnv server-side implementation.

Implements reset(), step(), and state for tabular data cleaning tasks.
The agent receives observations about data quality and issues, then
submits cleaning actions to improve the dataset toward a clean target.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
import pandas as pd

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State

import sys
import os

# Add parent dir so we can import models and data_generator
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import CleaningOperation, DataCleaningAction, DataCleaningObservation
from data_generator import generate_task_data, TASK_CONFIGS, TaskConfig


# ── Quality scoring ──────────────────────────────────────────────────────────

def _compute_null_score(df: pd.DataFrame, clean_df: pd.DataFrame) -> float:
    """Score based on fraction of non-null values that should be non-null."""
    expected_non_null = clean_df.notna().sum().sum()
    if expected_non_null == 0:
        return 1.0
    # Count cells that are non-null in our df where they should be
    # Use positional indexing to handle different-length DataFrames
    current_non_null = 0
    min_rows = min(len(df), len(clean_df))
    for col in clean_df.columns:
        if col in df.columns:
            clean_mask = clean_df[col].iloc[:min_rows].notna()
            current_non_null += df[col].iloc[:min_rows][clean_mask.values].notna().sum()
    # Count any extra rows beyond clean_df as fully non-null (they're duplicates)
    if len(df) > len(clean_df):
        extra_rows = len(df) - len(clean_df)
        for col in clean_df.columns:
            if col in df.columns:
                current_non_null += df[col].iloc[min_rows:].notna().sum()
    return min(current_non_null / expected_non_null, 1.0)


def _compute_type_score(df: pd.DataFrame, config: TaskConfig) -> float:
    """Score based on fraction of columns with correct dtype."""
    if not config.columns:
        return 1.0

    correct = 0
    total = 0
    for col, expected in config.columns.items():
        if col not in df.columns:
            continue
        total += 1
        col_dtype = str(df[col].dtype)
        if expected == "int" and col_dtype in ("int64", "int32", "Int64"):
            correct += 1
        elif expected == "float" and col_dtype in ("float64", "float32", "Float64"):
            correct += 1
        elif expected == "str" and col_dtype in ("object", "string"):
            correct += 1
        elif expected == "datetime" and "datetime" in col_dtype:
            correct += 1

    return correct / total if total > 0 else 1.0


def _compute_duplicate_score(df: pd.DataFrame, clean_df: pd.DataFrame) -> float:
    """Score: 1.0 if no excess duplicates vs the clean reference."""
    expected_rows = len(clean_df)
    if expected_rows == 0:
        return 1.0
    excess = max(0, len(df) - expected_rows)
    actual_dups = df.duplicated().sum()
    penalty = min(actual_dups, excess)
    return max(0.0, 1.0 - penalty / expected_rows)


def _compute_format_score(df: pd.DataFrame, clean_df: pd.DataFrame, config: TaskConfig) -> float:
    """Score based on how well values match the clean reference format."""
    if len(clean_df) == 0:
        return 1.0

    total_checks = 0
    correct = 0

    for col in clean_df.columns:
        if col not in df.columns:
            continue
        expected_type = config.columns.get(col, "str")

        # Compare against clean values for format consistency
        min_len = min(len(df), len(clean_df))
        for idx in range(min_len):
            clean_val = clean_df.iloc[idx][col]
            if pd.isna(clean_val):
                continue

            total_checks += 1
            if idx >= len(df):
                continue

            current_val = df.iloc[idx][col]
            if pd.isna(current_val):
                continue

            # Check formatting based on type
            if expected_type == "str":
                # Check for whitespace issues
                clean_str = str(clean_val).strip()
                current_str = str(current_val).strip()
                if current_str == current_val and clean_str.lower() == current_str.lower():
                    correct += 1
                elif current_str == current_val:
                    correct += 0.5  # partial credit if trimmed but case wrong
            elif expected_type == "datetime":
                # Check if it's a proper datetime or matches YYYY-MM-DD
                try:
                    pd.to_datetime(current_val)
                    correct += 1
                except (ValueError, TypeError):
                    pass
            elif expected_type in ("int", "float"):
                try:
                    float(current_val)
                    correct += 1
                except (ValueError, TypeError):
                    pass
            else:
                correct += 1

    return correct / total_checks if total_checks > 0 else 1.0


def compute_quality_score(df: pd.DataFrame, clean_df: pd.DataFrame, config: TaskConfig) -> float:
    """Compute the overall data quality score [0.0, 1.0].

    Weighted average of 4 components:
    - Null completeness (25%)
    - Type correctness (25%)
    - Duplicate absence (25%)
    - Format consistency (25%)
    """
    null_score = _compute_null_score(df, clean_df)
    type_score = _compute_type_score(df, config)
    dup_score = _compute_duplicate_score(df, clean_df)
    format_score = _compute_format_score(df, clean_df, config)

    total = 0.25 * null_score + 0.25 * type_score + 0.25 * dup_score + 0.25 * format_score
    return round(min(max(total, 0.0), 1.0), 4)


# ── Issue detection ──────────────────────────────────────────────────────────

def detect_issues(df: pd.DataFrame, config: TaskConfig) -> List[str]:
    """Detect remaining data quality issues for the observation."""
    issues: List[str] = []

    for col in df.columns:
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            issues.append(f"Column '{col}' has {null_count} null values")

    # Whitespace
    for col in df.select_dtypes(include=["object"]).columns:
        ws_count = 0
        for val in df[col].dropna():
            s = str(val)
            if s != s.strip():
                ws_count += 1
        if ws_count > 0:
            issues.append(f"Column '{col}' has {ws_count} values with extra whitespace")

    # Type issues
    for col, expected in config.columns.items():
        if col not in df.columns:
            continue
        if expected in ("int", "float"):
            non_numeric = 0
            for val in df[col].dropna():
                try:
                    float(val)
                except (ValueError, TypeError):
                    non_numeric += 1
            if non_numeric > 0:
                issues.append(f"Column '{col}' has {non_numeric} non-numeric values (expected {expected})")

    # Duplicates
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        issues.append(f"Dataset has {dup_count} duplicate rows")

    # Date format
    for col, expected in config.columns.items():
        if expected == "datetime" and col in df.columns:
            if str(df[col].dtype) not in ("datetime64[ns]", "datetime64[us]"):
                # Check if mixed formats exist
                formats_seen = set()
                for val in df[col].dropna().head(20):
                    s = str(val)
                    if re.match(r"\d{4}-\d{2}-\d{2}", s):
                        formats_seen.add("YYYY-MM-DD")
                    elif re.match(r"\d{2}/\d{2}/\d{4}", s):
                        formats_seen.add("MM/DD/YYYY")
                    elif re.match(r"[A-Z][a-z]{2} \d", s):
                        formats_seen.add("Mon DD, YYYY")
                    elif re.match(r"\d{2}\.\d{2}\.\d{4}", s):
                        formats_seen.add("DD.MM.YYYY")
                if len(formats_seen) > 1:
                    issues.append(f"Column '{col}' has mixed date formats: {formats_seen}")
                elif formats_seen and "YYYY-MM-DD" not in formats_seen:
                    issues.append(f"Column '{col}' dates are not in standard YYYY-MM-DD format")

    # Case inconsistency
    for col in df.select_dtypes(include=["object"]).columns:
        if col in config.columns and config.columns[col] == "str":
            vals = df[col].dropna().unique()
            if len(vals) > 1:
                # Check if there are case variants of the same value
                lower_map: Dict[str, set] = {}
                for v in vals:
                    s = str(v).strip()
                    key = s.lower()
                    if key not in lower_map:
                        lower_map[key] = set()
                    lower_map[key].add(s)
                inconsistent = {k: v for k, v in lower_map.items() if len(v) > 1}
                if inconsistent:
                    examples = list(inconsistent.values())[:2]
                    issues.append(f"Column '{col}' has inconsistent casing: {examples}")

    return issues


# ── Column stats for observation ─────────────────────────────────────────────

def get_column_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute per-column stats for the observation."""
    stats: Dict[str, Any] = {}
    for col in df.columns:
        col_data = df[col]
        sample_vals = col_data.dropna().head(5).tolist()
        stats[col] = {
            "dtype": str(col_data.dtype),
            "null_count": int(col_data.isna().sum()),
            "total_count": len(col_data),
            "unique_count": int(col_data.nunique()),
            "sample_values": [str(v) for v in sample_vals],
        }
    return stats


# ── Cleaning operation implementations ───────────────────────────────────────

def apply_cleaning_action(df: pd.DataFrame, action: DataCleaningAction,
                          config: TaskConfig) -> pd.DataFrame:
    """Apply a cleaning action to the DataFrame. Returns modified copy."""
    df = df.copy()
    col = action.column
    params = action.params
    op = action.operation

    if op == CleaningOperation.FILL_NULL:
        if col and col in df.columns:
            strategy = params.get("strategy", None)
            fill_value = params.get("value", None)
            if strategy == "mean":
                try:
                    numeric = pd.to_numeric(df[col], errors="coerce")
                    df[col] = df[col].fillna(numeric.mean())
                except Exception:
                    pass
            elif strategy == "median":
                try:
                    numeric = pd.to_numeric(df[col], errors="coerce")
                    df[col] = df[col].fillna(numeric.median())
                except Exception:
                    pass
            elif strategy == "mode":
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val.iloc[0])
            elif fill_value is not None:
                df[col] = df[col].fillna(fill_value)
            else:
                # Default: forward fill then backfill
                df[col] = df[col].ffill().bfill()

    elif op == CleaningOperation.FIX_TYPE:
        if col and col in df.columns:
            target_type = params.get("target_type", "str")
            if target_type == "int":
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].fillna(0).astype(int)
            elif target_type == "float":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif target_type == "str":
                df[col] = df[col].astype(str)
            elif target_type == "datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)

    elif op == CleaningOperation.REMOVE_DUPLICATES:
        df = df.drop_duplicates().reset_index(drop=True)

    elif op == CleaningOperation.STANDARDIZE_FORMAT:
        if col and col in df.columns:
            fmt = params.get("format", None)
            if fmt and config.columns.get(col) == "datetime":
                # Try to parse and reformat dates
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
                    df[col] = df[col].dt.strftime(fmt)
                except Exception:
                    pass
            decimal_places = params.get("decimal_places", None)
            if decimal_places is not None:
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce").round(decimal_places)
                except Exception:
                    pass

    elif op == CleaningOperation.TRIM_WHITESPACE:
        if col and col in df.columns:
            df[col] = df[col].apply(lambda x: str(x).strip() if pd.notna(x) else x)

    elif op == CleaningOperation.FIX_CASE:
        if col and col in df.columns:
            case = params.get("case", "title")
            def apply_case(x):
                if pd.isna(x):
                    return x
                s = str(x)
                if case == "lower":
                    return s.lower()
                elif case == "upper":
                    return s.upper()
                elif case == "title":
                    return s.title()
                return s
            df[col] = df[col].apply(apply_case)

    elif op == CleaningOperation.REMOVE_OUTLIERS:
        if col and col in df.columns:
            method = params.get("method", "iqr")
            threshold = params.get("threshold", 1.5)
            try:
                numeric = pd.to_numeric(df[col], errors="coerce")
                if method == "iqr":
                    q1 = numeric.quantile(0.25)
                    q3 = numeric.quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - threshold * iqr
                    upper = q3 + threshold * iqr
                    mask = (numeric >= lower) & (numeric <= upper) | numeric.isna()
                    df = df[mask].reset_index(drop=True)
                elif method == "zscore":
                    mean = numeric.mean()
                    std = numeric.std()
                    if std > 0:
                        z = (numeric - mean).abs() / std
                        mask = (z <= threshold) | numeric.isna()
                        df = df[mask].reset_index(drop=True)
            except Exception:
                pass

    elif op == CleaningOperation.DROP_COLUMN:
        if col and col in df.columns:
            df = df.drop(columns=[col])

    return df


# ── Environment class ────────────────────────────────────────────────────────

class DataCleaningEnvironment(Environment):
    """OpenEnv environment for tabular data cleaning.

    The agent receives messy data and applies cleaning operations to
    improve data quality toward a clean target state.
    """

    def __init__(self, task_name: str = "basic_cleanup"):
        super().__init__()
        self._task_name = task_name
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._dirty_df: Optional[pd.DataFrame] = None
        self._clean_df: Optional[pd.DataFrame] = None
        self._config: Optional[TaskConfig] = None
        self._prev_score: float = 0.0
        self._current_df: Optional[pd.DataFrame] = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """Reset the environment with a fresh messy dataset.

        Args:
            seed: Optional seed override (default uses task's built-in seed).
            episode_id: Optional episode identifier.
            **kwargs: Additional reset options. Can include 'task_name'.

        Returns:
            Initial DataCleaningObservation with data preview and stats.
        """
        # Allow task_name override in kwargs
        task_name = kwargs.get("task_name", self._task_name)
        if task_name in TASK_CONFIGS:
            self._task_name = task_name

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        # Generate fresh data
        self._dirty_df, self._clean_df, self._config = generate_task_data(self._task_name)
        self._current_df = self._dirty_df.copy()

        # Initial quality score
        self._prev_score = compute_quality_score(self._current_df, self._clean_df, self._config)

        # Build observation
        issues = detect_issues(self._current_df, self._config)
        stats = get_column_stats(self._current_df)
        preview = self._current_df.head(5).to_dict(orient="records")

        return DataCleaningObservation(
            done=False,
            reward=0.0,
            data_preview=preview,
            column_stats=stats,
            issues_detected=issues,
            data_quality_score=self._prev_score,
            task_name=self._task_name,
            step_number=0,
            max_steps=self._config.max_steps,
            metadata={
                "task_description": self._config.description,
                "difficulty": self._config.difficulty,
                "num_rows": len(self._current_df),
                "num_columns": len(self._current_df.columns),
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Apply a cleaning action and return the updated observation.

        Args:
            action: DataCleaningAction to execute.
            timeout_s: Optional timeout (unused).

        Returns:
            DataCleaningObservation with updated data state.
        """
        self._state.step_count += 1

        if self._current_df is None or self._clean_df is None or self._config is None:
            return DataCleaningObservation(
                done=True,
                reward=0.0,
                issues_detected=["Environment not initialized. Call reset() first."],
                metadata={"error": "Call reset() before step()"},
            )

        # Parse action
        if isinstance(action, DataCleaningAction):
            cleaning_action = action
        elif isinstance(action, dict):
            try:
                cleaning_action = DataCleaningAction(**action)
            except Exception as e:
                return DataCleaningObservation(
                    done=False,
                    reward=-0.1,  # Small penalty for invalid action
                    data_preview=self._current_df.head(5).to_dict(orient="records"),
                    column_stats=get_column_stats(self._current_df),
                    issues_detected=detect_issues(self._current_df, self._config) + [f"Invalid action: {e}"],
                    data_quality_score=self._prev_score,
                    task_name=self._task_name,
                    step_number=self._state.step_count,
                    max_steps=self._config.max_steps,
                    metadata={"error": str(e)},
                )
        else:
            # Try to extract from Action base
            try:
                action_dict = action.model_dump() if hasattr(action, "model_dump") else dict(action)
                cleaning_action = DataCleaningAction(**action_dict)
            except Exception as e:
                return DataCleaningObservation(
                    done=False,
                    reward=-0.1,
                    data_preview=self._current_df.head(5).to_dict(orient="records"),
                    column_stats=get_column_stats(self._current_df),
                    issues_detected=[f"Could not parse action: {e}"],
                    data_quality_score=self._prev_score,
                    task_name=self._task_name,
                    step_number=self._state.step_count,
                    max_steps=self._config.max_steps,
                    metadata={"error": str(e)},
                )

        # Apply the action
        try:
            self._current_df = apply_cleaning_action(
                self._current_df, cleaning_action, self._config
            )
        except Exception as e:
            return DataCleaningObservation(
                done=False,
                reward=-0.05,
                data_preview=self._current_df.head(5).to_dict(orient="records"),
                column_stats=get_column_stats(self._current_df),
                issues_detected=detect_issues(self._current_df, self._config) + [f"Action error: {e}"],
                data_quality_score=self._prev_score,
                task_name=self._task_name,
                step_number=self._state.step_count,
                max_steps=self._config.max_steps,
                metadata={"error": str(e)},
            )

        # Compute new score and reward
        new_score = compute_quality_score(self._current_df, self._clean_df, self._config)
        reward = new_score - self._prev_score  # Shaped reward: improvement delta
        self._prev_score = new_score

        # Check termination
        done = (
            new_score >= 0.95
            or self._state.step_count >= self._config.max_steps
        )

        # Build observation
        issues = detect_issues(self._current_df, self._config)
        stats = get_column_stats(self._current_df)
        preview = self._current_df.head(5).to_dict(orient="records")

        return DataCleaningObservation(
            done=done,
            reward=round(reward, 4),
            data_preview=preview,
            column_stats=stats,
            issues_detected=issues,
            data_quality_score=new_score,
            task_name=self._task_name,
            step_number=self._state.step_count,
            max_steps=self._config.max_steps,
            metadata={
                "action_applied": cleaning_action.operation.value,
                "target_column": cleaning_action.column,
                "num_rows": len(self._current_df),
            },
        )

    @property
    def state(self) -> State:
        """Get current environment state."""
        return self._state
