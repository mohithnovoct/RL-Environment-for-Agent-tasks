# Copyright (c) 2026. All rights reserved.
# Data Cleaning Environment - Seeded Data Generator

"""
Procedural generation of messy datasets for the Data Cleaning Environment.

Each task has a fixed seed so grading is deterministic and reproducible.
The generator creates a "clean" ground-truth DataFrame and then injects
specific issue types to produce the "dirty" version the agent must fix.
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


# ── Task definitions ─────────────────────────────────────────────────────────

@dataclass
class TaskConfig:
    """Configuration for a single data-cleaning task."""

    name: str
    difficulty: str  # easy | medium | hard
    seed: int
    max_steps: int
    num_rows: int
    columns: Dict[str, str]  # col_name -> expected dtype
    issue_types: List[str]
    description: str


TASK_CONFIGS: Dict[str, TaskConfig] = {
    # ── Easy: just nulls + whitespace on 3 columns ──────────────────────
    "basic_cleanup": TaskConfig(
        name="basic_cleanup",
        difficulty="easy",
        seed=42,
        max_steps=10,
        num_rows=50,
        columns={
            "name": "str",
            "age": "int",
            "email": "str",
        },
        issue_types=["null", "whitespace"],
        description=(
            "A small customer table with missing values and extra whitespace. "
            "Fill nulls and trim strings to clean it up."
        ),
    ),

    # ── Medium: type errors, date formats, casing on 5 columns ──────────
    "type_and_format": TaskConfig(
        name="type_and_format",
        difficulty="medium",
        seed=123,
        max_steps=15,
        num_rows=80,
        columns={
            "product_name": "str",
            "price": "float",
            "quantity": "int",
            "purchase_date": "datetime",
            "category": "str",
        },
        issue_types=["null", "type_error", "date_format", "case"],
        description=(
            "An e-commerce orders table with type mismatches, inconsistent "
            "date formats, and mixed-case category names."
        ),
    ),

    # ── Hard: all issue types across 7 columns ──────────────────────────
    "full_pipeline": TaskConfig(
        name="full_pipeline",
        difficulty="hard",
        seed=456,
        max_steps=20,
        num_rows=120,
        columns={
            "employee_name": "str",
            "department": "str",
            "salary": "float",
            "hire_date": "datetime",
            "phone": "str",
            "performance_score": "float",
            "status": "str",
        },
        issue_types=[
            "null", "whitespace", "type_error",
            "date_format", "case", "duplicates", "outliers",
        ],
        description=(
            "A full HR dataset with every kind of data-quality problem: "
            "missing values, wrong types, duplicates, outliers, inconsistent "
            "formats, and messy text. Requires a multi-step cleaning strategy."
        ),
    ),
}


# ── Clean data generators ────────────────────────────────────────────────────

def _generate_clean_basic(rng: random.Random, n: int) -> pd.DataFrame:
    """Generate clean customer data for the basic_cleanup task."""
    first_names = [
        "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank",
        "Grace", "Henry", "Ivy", "Jack", "Karen", "Leo",
        "Mia", "Nathan", "Olivia", "Peter", "Quinn", "Rachel",
        "Sam", "Tina", "Uma", "Victor", "Wendy", "Xavier",
        "Yara", "Zach",
    ]
    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
        "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez",
    ]
    domains = ["gmail.com", "yahoo.com", "outlook.com", "company.com"]

    rows = []
    for _ in range(n):
        first = rng.choice(first_names)
        last = rng.choice(last_names)
        name = f"{first} {last}"
        age = rng.randint(18, 75)
        domain = rng.choice(domains)
        email = f"{first.lower()}.{last.lower()}@{domain}"
        rows.append({"name": name, "age": age, "email": email})

    return pd.DataFrame(rows)


def _generate_clean_orders(rng: random.Random, n: int) -> pd.DataFrame:
    """Generate clean e-commerce order data for the type_and_format task."""
    products = [
        "Wireless Mouse", "USB Keyboard", "Monitor Stand", "Laptop Sleeve",
        "Webcam HD", "Desk Lamp", "Cable Organizer", "Mouse Pad XL",
        "Phone Charger", "Headphones Pro", "Screen Cleaner", "USB Hub",
    ]
    categories = ["Electronics", "Accessories", "Furniture", "Office Supplies"]

    rows = []
    for _ in range(n):
        product = rng.choice(products)
        price = round(rng.uniform(5.0, 299.99), 2)
        qty = rng.randint(1, 20)
        year = rng.randint(2022, 2025)
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        date = f"{year}-{month:02d}-{day:02d}"
        cat = rng.choice(categories)
        rows.append({
            "product_name": product,
            "price": price,
            "quantity": qty,
            "purchase_date": date,
            "category": cat,
        })

    return pd.DataFrame(rows)


def _generate_clean_hr(rng: random.Random, n: int) -> pd.DataFrame:
    """Generate clean HR data for the full_pipeline task."""
    first_names = [
        "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank",
        "Grace", "Henry", "Ivy", "Jack", "Karen", "Leo",
        "Mia", "Nathan", "Olivia", "Peter", "Quinn", "Rachel",
        "Sam", "Tina",
    ]
    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
        "Miller", "Davis", "Rodriguez", "Martinez",
    ]
    departments = ["Engineering", "Marketing", "Sales", "Finance", "HR", "Operations"]
    statuses = ["Active", "On Leave", "Terminated"]

    rows = []
    for _ in range(n):
        first = rng.choice(first_names)
        last = rng.choice(last_names)
        name = f"{first} {last}"
        dept = rng.choice(departments)
        salary = round(rng.uniform(35000, 150000), 2)
        year = rng.randint(2015, 2025)
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        hire_date = f"{year}-{month:02d}-{day:02d}"
        area_code = rng.randint(200, 999)
        phone = f"({area_code}) {rng.randint(200, 999)}-{rng.randint(1000, 9999)}"
        score = round(rng.uniform(1.0, 5.0), 1)
        status = rng.choice(statuses)
        rows.append({
            "employee_name": name,
            "department": dept,
            "salary": salary,
            "hire_date": hire_date,
            "phone": phone,
            "performance_score": score,
            "status": status,
        })

    return pd.DataFrame(rows)


# ── Issue injectors ──────────────────────────────────────────────────────────

def _inject_nulls(df: pd.DataFrame, rng: random.Random,
                  columns: List[str], frac: float = 0.15) -> pd.DataFrame:
    """Randomly set ~frac of values to NaN in selected columns."""
    df = df.copy()
    for col in columns:
        mask = [rng.random() < frac for _ in range(len(df))]
        df.loc[mask, col] = None
    return df


def _inject_whitespace(df: pd.DataFrame, rng: random.Random,
                       columns: List[str], frac: float = 0.20) -> pd.DataFrame:
    """Add leading/trailing whitespace to string columns."""
    df = df.copy()
    for col in columns:
        if df[col].dtype == object:
            for idx in df.index:
                if rng.random() < frac and pd.notna(df.at[idx, col]):
                    pad_left = " " * rng.randint(1, 4)
                    pad_right = " " * rng.randint(1, 4)
                    df.at[idx, col] = f"{pad_left}{df.at[idx, col]}{pad_right}"
    return df


def _inject_type_errors(df: pd.DataFrame, rng: random.Random,
                        columns: Dict[str, str], frac: float = 0.10) -> pd.DataFrame:
    """Convert some numeric values to bad string representations."""
    df = df.copy()
    word_map = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
                5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
                10: "ten", 15: "fifteen", 20: "twenty", 25: "twenty-five",
                30: "thirty", 50: "fifty", 100: "hundred"}

    for col, expected_type in columns.items():
        if expected_type in ("int", "float") and col in df.columns:
            # Convert column to object dtype first so we can insert strings
            df[col] = df[col].astype(object)
            for idx in df.index:
                if rng.random() < frac and pd.notna(df.at[idx, col]):
                    val = df.at[idx, col]
                    # Convert to string representation
                    if isinstance(val, (int, float)):
                        if int(val) in word_map and rng.random() < 0.5:
                            df.at[idx, col] = word_map[int(val)]
                        else:
                            df.at[idx, col] = f"${val}" if rng.random() < 0.5 else str(val) + " units"
    return df


def _inject_date_format_issues(df: pd.DataFrame, rng: random.Random,
                               columns: List[str]) -> pd.DataFrame:
    """Mix date formats: YYYY-MM-DD, MM/DD/YYYY, Mon DD, YYYY."""
    df = df.copy()
    month_names = [
        "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    formats = ["mdy_slash", "month_name", "dmy_dot"]

    for col in columns:
        for idx in df.index:
            if pd.notna(df.at[idx, col]):
                val = str(df.at[idx, col])
                try:
                    parts = val.split("-")
                    if len(parts) == 3:
                        y, m, d = parts
                        fmt = rng.choice(formats)
                        if fmt == "mdy_slash":
                            df.at[idx, col] = f"{m}/{d}/{y}"
                        elif fmt == "month_name":
                            mi = int(m)
                            if 1 <= mi <= 12:
                                df.at[idx, col] = f"{month_names[mi]} {int(d)}, {y}"
                        elif fmt == "dmy_dot":
                            df.at[idx, col] = f"{d}.{m}.{y}"
                except (ValueError, IndexError):
                    pass
    return df


def _inject_case_issues(df: pd.DataFrame, rng: random.Random,
                        columns: List[str]) -> pd.DataFrame:
    """Mix casing: lower, upper, title, random."""
    df = df.copy()
    for col in columns:
        if df[col].dtype == object:
            for idx in df.index:
                if pd.notna(df.at[idx, col]):
                    val = str(df.at[idx, col])
                    choice = rng.choice(["lower", "upper", "original"])
                    if choice == "lower":
                        df.at[idx, col] = val.lower()
                    elif choice == "upper":
                        df.at[idx, col] = val.upper()
    return df


def _inject_duplicates(df: pd.DataFrame, rng: random.Random,
                       count: int = 8) -> pd.DataFrame:
    """Duplicate some rows."""
    if len(df) == 0:
        return df
    dup_indices = [rng.randint(0, len(df) - 1) for _ in range(count)]
    dups = df.iloc[dup_indices].copy()
    df = pd.concat([df, dups], ignore_index=True)
    return df


def _inject_outliers(df: pd.DataFrame, rng: random.Random,
                     columns: List[str], count: int = 5) -> pd.DataFrame:
    """Inject extreme outlier values in numeric columns."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            try:
                numeric_vals = pd.to_numeric(df[col], errors="coerce")
                mean_val = numeric_vals.mean()
                std_val = numeric_vals.std()
                if pd.notna(mean_val) and pd.notna(std_val) and std_val > 0:
                    for _ in range(count):
                        idx = rng.randint(0, len(df) - 1)
                        multiplier = rng.choice([5, 8, 10, -3, -5])
                        outlier = mean_val + multiplier * std_val
                        df.at[idx, col] = round(outlier, 2)
            except (TypeError, ValueError):
                pass
    return df


# ── Main generator ───────────────────────────────────────────────────────────

def generate_task_data(task_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, TaskConfig]:
    """Generate dirty and clean DataFrames for a given task.

    Returns:
        (dirty_df, clean_df, task_config)

    Raises:
        ValueError: If task_name is not recognized.
    """
    if task_name not in TASK_CONFIGS:
        raise ValueError(
            f"Unknown task: {task_name!r}. "
            f"Available tasks: {list(TASK_CONFIGS.keys())}"
        )

    config = TASK_CONFIGS[task_name]
    rng = random.Random(config.seed)
    np.random.seed(config.seed)
    n = config.num_rows

    # Generate clean data
    if task_name == "basic_cleanup":
        clean_df = _generate_clean_basic(rng, n)
    elif task_name == "type_and_format":
        clean_df = _generate_clean_orders(rng, n)
    elif task_name == "full_pipeline":
        clean_df = _generate_clean_hr(rng, n)
    else:
        raise ValueError(f"No generator for task: {task_name}")

    # Create dirty copy
    dirty_df = clean_df.copy()

    # Inject issues based on task config
    str_cols = [c for c, t in config.columns.items() if t == "str" and c in dirty_df.columns]
    num_cols = [c for c, t in config.columns.items() if t in ("int", "float") and c in dirty_df.columns]
    date_cols = [c for c, t in config.columns.items() if t == "datetime" and c in dirty_df.columns]
    all_cols = list(config.columns.keys())

    if "null" in config.issue_types:
        null_targets = all_cols[:3] if len(all_cols) > 3 else all_cols
        dirty_df = _inject_nulls(dirty_df, rng, null_targets)

    if "whitespace" in config.issue_types:
        dirty_df = _inject_whitespace(dirty_df, rng, str_cols)

    if "type_error" in config.issue_types:
        dirty_df = _inject_type_errors(dirty_df, rng, config.columns)

    if "date_format" in config.issue_types:
        dirty_df = _inject_date_format_issues(dirty_df, rng, date_cols)

    if "case" in config.issue_types:
        case_targets = str_cols[-2:] if len(str_cols) > 2 else str_cols
        dirty_df = _inject_case_issues(dirty_df, rng, case_targets)

    if "duplicates" in config.issue_types:
        dup_count = max(3, n // 15)
        dirty_df = _inject_duplicates(dirty_df, rng, count=dup_count)

    if "outliers" in config.issue_types:
        dirty_df = _inject_outliers(dirty_df, rng, num_cols, count=max(2, n // 20))

    return dirty_df, clean_df, config


def list_tasks() -> List[str]:
    """Return sorted list of available task names."""
    return sorted(TASK_CONFIGS.keys())


def get_task_config(task_name: str) -> TaskConfig:
    """Get configuration for a specific task."""
    if task_name not in TASK_CONFIGS:
        raise ValueError(f"Unknown task: {task_name!r}")
    return TASK_CONFIGS[task_name]
