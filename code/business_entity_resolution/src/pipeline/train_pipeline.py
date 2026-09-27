"""Training entrypoint for the Business Entity Resolution matcher.

Owned by Person 4. Separate from run_pipeline.py (inference) because
Person 3's EntityMatcher is stateful: it must be fit once on labeled
train data (source1/2/3 + train_ground_truth.tsv) before it can be
used for test-set inference. This module produces the saved model
artifact that run_pipeline.py loads.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pandas as pd

from src.matching.matcher import EntityMatcher
from src.pipeline.config import PipelineConfig
from src.pipeline.experiment_tracker import log_experiment
from src.pipeline.run_pipeline import (
    build_features,
    generate_candidates,
    load_sources,
    _adapt_candidates,
    _check_columns,
)

logger = logging.getLogger(__name__)


def _git_short_hash() -> str:
    """Current commit hash (8 chars), so every logged experiment is
    traceable to the exact code that produced it. Falls back to
    "unversioned" outside a git checkout rather than failing."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unversioned"


def run_training(config: PipelineConfig) -> dict:
    """Fit EntityMatcher on the train split and persist it.

    Returns the dict produced by EntityMatcher.fit_and_validate
    (macro_f05, singleton_acc, non_singleton_f05, abs_threshold,
    rel_margin), useful for experiment_tracker.py.
    """
    logger.info("Loading train data from %s", config.train_dir)
    train_data = load_sources(config.train_dir, "train")
    source1, source2, source3 = (
        train_data["source1"],
        train_data["source2"],
        train_data["source3"],
    )
    logger.info("Source 1 train records: %d", len(source1))

    ground_truth_path = config.train_dir / "train_ground_truth.tsv"
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Training ground truth not found: {ground_truth_path}")
    ground_truth = pd.read_csv(ground_truth_path, sep="\t", dtype=str, keep_default_na=False)
    _check_columns(
        ground_truth,
        {"source1_entity_id", "matched_entity_ids"},
        "train_ground_truth.tsv",
    )

    logger.info("Generating candidate pairs (Person 1 blocking) on train split")
    candidates = generate_candidates(source1, source2, source3)
    candidates = _adapt_candidates(candidates)
    logger.info("Train candidates generated: %d", len(candidates))

    logger.info("Building features (Person 2) on train split")
    features = build_features(candidates, source1, source2, source3)
    _check_columns(
        features, {"source1_entity_id", "candidate_entity_id"}, "Feature output"
    )

    all_train_s1_ids = source1["entity_id"].tolist()

    logger.info("Fitting EntityMatcher (Person 3)")
    matcher = EntityMatcher()
    results = matcher.fit_and_validate(features, ground_truth, all_train_s1_ids)

    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    matcher.save(str(config.model_path))
    logger.info("Saved trained matcher to %s", config.model_path)
    logger.info(
        "Local OOF Macro F0.5: %.4f (threshold=%.3f, rel_margin=%.3f)",
        results["macro_f05"],
        results["abs_threshold"],
        results["rel_margin"],
    )

    commit = _git_short_hash()
    experiment_id = log_experiment(
        log_path=config.output_dir / "experiments" / "experiments.csv",
        blocking_version=commit,
        feature_version=commit,
        model_version=commit,
        threshold=results["abs_threshold"],
        rel_margin=results["rel_margin"],
        macro_f05=results["macro_f05"],
        singleton_acc=results.get("singleton_acc"),
        non_singleton_f05=results.get("non_singleton_f05"),
        notes=f"n_train_s1={len(all_train_s1_ids)}",
    )
    logger.info("Logged as experiment %s", experiment_id)

    return results