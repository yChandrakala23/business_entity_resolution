"""Lightweight experiment tracking.

Owned by Person 4. Deliberately NOT a full MLOps system -- a single
append-only CSV, per the spec ("does NOT need to become a complex
MLOps system"). Purpose: catch leaderboard-overfitting, bad threshold
changes, and regressions between pipeline versions at a glance.
"""
from __future__ import annotations

import csv
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

FIELDNAMES = [
    "experiment_id",
    "timestamp",
    "blocking_version",
    "feature_version",
    "model_version",
    "threshold",
    "rel_margin",
    "macro_f05",
    "singleton_acc",
    "non_singleton_f05",
    "leaderboard_score",
    "notes",
]


def log_experiment(
    log_path: Path,
    blocking_version: str,
    feature_version: str,
    model_version: str,
    threshold: float,
    rel_margin: float,
    macro_f05: float,
    singleton_acc: Optional[float] = None,
    non_singleton_f05: Optional[float] = None,
    leaderboard_score: Optional[float] = None,
    notes: str = "",
) -> str:
    """Append one row to the experiment log. Returns the generated experiment_id.

    `*_version` fields are free-form strings you choose (e.g. a git
    short-hash, "v1", "tfidf+phonetic") -- whatever lets you tell two
    runs apart later. Creates the log file with a header if it
    doesn't exist yet.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    experiment_id = uuid.uuid4().hex[:8]
    row = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "blocking_version": blocking_version,
        "feature_version": feature_version,
        "model_version": model_version,
        "threshold": round(threshold, 4),
        "rel_margin": round(rel_margin, 4),
        "macro_f05": round(macro_f05, 4),
        "singleton_acc": round(singleton_acc, 4) if singleton_acc is not None else "",
        "non_singleton_f05": round(non_singleton_f05, 4) if non_singleton_f05 is not None else "",
        "leaderboard_score": leaderboard_score if leaderboard_score is not None else "",
        "notes": notes,
    }

    file_exists = log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    logger.info(
        "Logged experiment %s: macro_f05=%.4f threshold=%.3f rel_margin=%.3f -> %s",
        experiment_id, macro_f05, threshold, rel_margin, log_path,
    )
    return experiment_id


def load_experiments(log_path: Path):
    """Return all logged rows as a list of dicts (empty list if the log
    doesn't exist yet). Useful for a quick before/after comparison."""
    log_path = Path(log_path)
    if not log_path.exists():
        return []
    with open(log_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def best_experiment(log_path: Path, metric: str = "macro_f05"):
    """Return the row with the highest value for `metric` (None if empty
    or the metric column is blank in every row)."""
    rows = load_experiments(log_path)
    scored = [(float(r[metric]), r) for r in rows if r.get(metric)]
    if not scored:
        return None
    return max(scored, key=lambda pair: pair[0])[1]