# Copyright (c) 2026. All rights reserved.

"""
Data Cleaning Environment — An OpenEnv environment for tabular data cleaning.

This environment simulates real-world data wrangling tasks where an AI agent
must clean messy datasets through operations like filling nulls, fixing types,
removing duplicates, and standardizing formats.

Tasks:
    - basic_cleanup (easy): Fill nulls, trim whitespace
    - type_and_format (medium): Fix types, standardize dates and casing
    - full_pipeline (hard): All issue types combined

Example:
    >>> from data_cleaning_env import DataCleaningEnv, DataCleaningAction
    >>>
    >>> with DataCleaningEnv(base_url="http://localhost:8000").sync() as env:
    ...     result = env.reset()
    ...     result = env.step(DataCleaningAction(
    ...         operation="fill_null", column="age", params={"strategy": "mean"}
    ...     ))
"""

from models import (
    CleaningOperation,
    DataCleaningAction,
    DataCleaningObservation,
)
from client import DataCleaningEnv

__all__ = [
    "DataCleaningEnv",
    "DataCleaningAction",
    "DataCleaningObservation",
    "CleaningOperation",
]
