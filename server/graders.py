# Copyright (c) 2026. All rights reserved.
# Data Cleaning Environment - Task Graders

"""
Three deterministic task graders for the Data Cleaning Environment.

Each grader:
    - Uses a fixed seed for reproducibility
    - Returns a normalized score in [0.0, 1.0]
    - Has clear success/failure criteria
    - Produces deterministic results across runs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from data_generator import TASK_CONFIGS, generate_task_data


@dataclass
class GraderResult:
    """Result from running a task grader."""

    task_name: str
    difficulty: str
    score: float           # Normalized [0.0, 1.0]
    passed: bool           # score >= threshold
    details: Dict[str, float]  # Component scores
    issues_remaining: int
    steps_taken: int
    max_steps: int


# ── Grader thresholds ────────────────────────────────────────────────────────

PASS_THRESHOLDS = {
    "basic_cleanup": 0.80,       # Easy: 80% quality
    "type_and_format": 0.70,     # Medium: 70% quality
    "full_pipeline": 0.60,       # Hard: 60% quality
}


def grade_task(
    task_name: str,
    final_quality_score: float,
    steps_taken: int,
    issues_remaining: int,
    component_scores: Dict[str, float] | None = None,
) -> GraderResult:
    """Grade a completed task episode.

    Args:
        task_name: Name of the task.
        final_quality_score: The quality score at episode end [0.0, 1.0].
        steps_taken: Number of steps the agent took.
        issues_remaining: Number of unresolved issues.
        component_scores: Optional breakdown of quality components.

    Returns:
        GraderResult with normalized score and pass/fail.
    """
    config = TASK_CONFIGS[task_name]
    threshold = PASS_THRESHOLDS.get(task_name, 0.70)

    # Normalize score to [0.0, 1.0]
    score = min(max(final_quality_score, 0.0), 1.0)

    # Efficiency bonus: if agent finishes early, small bonus
    if steps_taken < config.max_steps and score >= threshold:
        efficiency = 1.0 - (steps_taken / config.max_steps)
        score = min(score + efficiency * 0.05, 1.0)  # Up to 5% bonus

    return GraderResult(
        task_name=task_name,
        difficulty=config.difficulty,
        score=round(score, 4),
        passed=score >= threshold,
        details=component_scores or {},
        issues_remaining=issues_remaining,
        steps_taken=steps_taken,
        max_steps=config.max_steps,
    )


def list_graded_tasks() -> List[str]:
    """Return list of all tasks that have graders."""
    return list(PASS_THRESHOLDS.keys())


def get_task_threshold(task_name: str) -> float:
    """Get the pass threshold for a task."""
    return PASS_THRESHOLDS.get(task_name, 0.70)
