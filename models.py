# Copyright (c) 2026. All rights reserved.
# Data Cleaning Environment - Pydantic Models

"""
Typed Action and Observation models for the Data Cleaning Environment.

Action space:
    The agent picks a cleaning operation, a target column, and operation-specific
    parameters (e.g., fill value, target dtype, date format).

Observation space:
    The agent sees a preview of the current data, per-column statistics,
    a list of detected quality issues, and the current quality score.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation


# ── Cleaning operation enum ──────────────────────────────────────────────────

class CleaningOperation(str, Enum):
    """Supported data-cleaning operations."""

    FILL_NULL = "fill_null"
    """Fill null/missing values in a column. Params: {value: <fill_value>} or {strategy: 'mean'|'median'|'mode'}"""

    FIX_TYPE = "fix_type"
    """Convert column to correct dtype.  Params: {target_type: 'int'|'float'|'str'|'datetime'}"""

    REMOVE_DUPLICATES = "remove_duplicates"
    """Remove duplicate rows. Params: {} (no column needed; operates on whole table)"""

    STANDARDIZE_FORMAT = "standardize_format"
    """Standardize date/number formats. Params: {format: '%Y-%m-%d'} or {decimal_places: 2}"""

    TRIM_WHITESPACE = "trim_whitespace"
    """Strip leading/trailing whitespace from string column. Params: {}"""

    FIX_CASE = "fix_case"
    """Standardize text casing. Params: {case: 'lower'|'upper'|'title'}"""

    REMOVE_OUTLIERS = "remove_outliers"
    """Remove statistical outliers from numeric column. Params: {method: 'iqr'|'zscore', threshold: <float>}"""

    DROP_COLUMN = "drop_column"
    """Drop a column entirely. Params: {}"""


# ── Action model ─────────────────────────────────────────────────────────────

class DataCleaningAction(Action):
    """An action the agent takes to clean a dataset.

    Attributes:
        operation:  Which cleaning operation to apply.
        column:     Target column name (ignored for REMOVE_DUPLICATES).
        params:     Operation-specific parameters.
    """

    operation: CleaningOperation = Field(
        description="The cleaning operation to perform."
    )
    column: str = Field(
        default="",
        description="Target column name. Leave empty for row-level ops like REMOVE_DUPLICATES.",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Operation-specific parameters (e.g. fill value, target type).",
    )


# ── Column statistics (nested in Observation) ────────────────────────────────

class ColumnStats(Dict[str, Any]):
    """Convenience alias – just a plain dict serialized as JSON."""


# ── Observation model ────────────────────────────────────────────────────────

class DataCleaningObservation(Observation):
    """What the agent sees after each step.

    Inherits ``done``, ``reward``, ``metadata`` from the OpenEnv
    ``Observation`` base class.
    """

    data_preview: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="First N rows of the current dataset (list of row dicts).",
    )
    column_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Per-column statistics: dtype, null_count, unique_count, sample_values."
        ),
    )
    issues_detected: List[str] = Field(
        default_factory=list,
        description="Human-readable list of remaining data-quality issues.",
    )
    data_quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall data quality score in [0, 1].",
    )
    task_name: str = Field(
        default="",
        description="Name of the active task.",
    )
    step_number: int = Field(
        default=0,
        ge=0,
        description="Current step number in the episode.",
    )
    max_steps: int = Field(
        default=20,
        ge=1,
        description="Maximum steps allowed in this episode.",
    )
