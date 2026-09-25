"""End-to-end pipeline orchestration.

Owned by Person 4. This module coordinates the other teammates'
modules through the agreed contracts:

    generate_candidates(source1, source2, source3) -> candidates      # Person 1
    build_features(candidates, source1, source2, source3) -> features # Person 2
    predict_matches(features) -> predictions                          # Person 3

It intentionally does NOT implement blocking, feature engineering, or
modeling logic itself — those functions raise NotImplementedError
until a teammate's real implementation is wired in (see
"INTEGRATION POINTS" below).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Set

import pandas as pd

from src.pipeline.config import PipelineConfig
from src.pipeline.output_writer import write_candidate_pairs, write_matching_results
from src.pipeline.validator import validate_submission_files

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading (Person 4 owns this integration point but should defer to
# Person 1's loader once it exists — see the docstring below).
# ---------------------------------------------------------------------------

def load_sources(data_dir: Path, split: str) -> Dict[str, pd.DataFrame]:
    """Load the three source TSVs for a given split ("train" or "test").

    NOTE: If Person 1 builds a dedicated `load_train_data` /
    `load_test_data` function with extra logic (e.g. dtype coercion),
    swap it in here rather than duplicating loading logic elsewhere.
    """
    data_dir = Path(data_dir)
    return {
        "source1": pd.read_csv(data_dir / f"{split}_source1.tsv", sep="\t"),
        "source2": pd.read_csv(data_dir / f"{split}_source2.tsv", sep="\t"),
        "source3": pd.read_csv(data_dir / f"{split}_source3.tsv", sep="\t"),
    }


# ---------------------------------------------------------------------------
# INTEGRATION POINTS — Persons 1-3 plug in here.
#
# Each function currently raises NotImplementedError. Replace the body
# (or monkeypatch / import a teammate's module at the top of this file)
# once their real implementation lands. Do NOT invent a fake
# implementation in the meantime — a silent dummy predictor would
# produce a submission that looks valid but is meaningless.
# ---------------------------------------------------------------------------

def generate_candidates(
    source1: pd.DataFrame, source2: pd.DataFrame, source3: pd.DataFrame
) -> pd.DataFrame:
    """PERSON 1 CONTRACT — blocking / candidate generation.

    Expected return: long-form dataframe with columns
    ['source1_entity_id', 'candidate_entity_id'].
    """
    raise NotImplementedError(
        "Blocking implementation from Person 1 has not been integrated yet."
    )


def build_features(
    candidates: pd.DataFrame,
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
) -> pd.DataFrame:
    """PERSON 2 CONTRACT — feature engineering.

    Expected return: `candidates` with engineered similarity columns
    appended (e.g. name_jaro, name_levenshtein, address_similarity),
    preserving 'source1_entity_id' and 'candidate_entity_id'.
    """
    raise NotImplementedError(
        "Feature engineering implementation from Person 2 has not been integrated yet."
    )


def predict_matches(features: pd.DataFrame) -> pd.DataFrame:
    """PERSON 3 CONTRACT — matching model inference.

    Expected return: dataframe with columns
    ['source1_entity_id', 'candidate_entity_id', 'match_probability'].
    """
    raise NotImplementedError(
        "Matching model implementation from Person 3 has not been integrated yet."
    )


# ---------------------------------------------------------------------------
# Integration boundary validation + adapters
# ---------------------------------------------------------------------------

def _check_columns(df: pd.DataFrame, required: Set[str], context: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{context} is missing required columns: {sorted(missing)}")


def _adapt_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Rename common alternate column names to the pipeline's contract.

    Prefer this kind of lightweight adapter over rewriting a
    teammate's module when the shapes are compatible but the names
    differ.
    """
    rename_map = {
        "s1_id": "source1_entity_id",
        "s1_entity_id": "source1_entity_id",
        "candidate_id": "candidate_entity_id",
        "s2_s3_id": "candidate_entity_id",
    }
    cols_to_rename = {k: v for k, v in rename_map.items() if k in candidates.columns}
    if cols_to_rename:
        logger.info("Adapting blocking output columns: %s", cols_to_rename)
        candidates = candidates.rename(columns=cols_to_rename)
    _check_columns(candidates, {"source1_entity_id", "candidate_entity_id"}, "Blocking output")
    return candidates


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_pipeline(config: PipelineConfig) -> dict:
    """Run the full pipeline against the test split and write both
    required output files.

    Returns a small dict of summary stats, useful for logging and for
    experiment_tracker.py.
    """
    logger.info("Loading test data from %s", config.test_dir)
    test_data = load_sources(config.test_dir, "test")
    source1, source2, source3 = test_data["source1"], test_data["source2"], test_data["source3"]
    logger.info("Source 1 test records: %d", len(source1))
    logger.info("Source 2 test records: %d", len(source2))
    logger.info("Source 3 test records: %d", len(source3))

    logger.info("Generating candidate pairs (Person 1 blocking)")
    candidates = generate_candidates(source1, source2, source3)
    candidates = _adapt_candidates(candidates)
    logger.info("Candidates generated: %d", len(candidates))

    logger.info("Writing candidate_pairs.tsv")
    write_candidate_pairs(source1, candidates, config.candidate_pairs_path)

    logger.info("Building features (Person 2)")
    features = build_features(candidates, source1, source2, source3)
    _check_columns(features, {"source1_entity_id", "candidate_entity_id"}, "Feature output")

    logger.info("Running model inference (Person 3)")
    predictions = predict_matches(features)
    _check_columns(
        predictions,
        {"source1_entity_id", "candidate_entity_id", "match_probability"},
        "Model predictions",
    )

    logger.info("Applying threshold: %.3f", config.threshold)
    predicted_matches = predictions.loc[
        predictions["match_probability"] >= config.threshold,
        ["source1_entity_id", "candidate_entity_id"],
    ]
    logger.info("Matches predicted: %d", len(predicted_matches))

    logger.info("Writing matching_results.tsv")
    write_matching_results(source1, predicted_matches, config.matching_results_path)

    if not config.skip_validation:
        logger.info("Running internal validation")
        validate_submission_files(
            config.matching_results_path,
            config.candidate_pairs_path,
            config.test_dir,
        )
        logger.info("Internal validation passed")
    else:
        logger.info("Skipping internal validation (--skip-validation)")

    return {
        "n_source1": len(source1),
        "n_candidates": len(candidates),
        "n_matches": len(predicted_matches),
        "threshold": config.threshold,
    }