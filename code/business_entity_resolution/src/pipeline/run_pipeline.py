"""End-to-end inference orchestration.

Owned by Person 4. This module coordinates the other teammates'
modules through the agreed contracts:

    generate_candidates(source1, source2, source3) -> candidates      # Person 1
    build_features(candidates, source1, source2, source3) -> features # Person 2
    EntityMatcher (trained, loaded from disk)                         # Person 3

It intentionally does NOT implement blocking or feature engineering
logic itself -- those two functions raise NotImplementedError until a
teammate's real implementation is wired in.

Person 3's EntityMatcher is stateful and must be trained first --
see train_pipeline.run_training. This module only loads an already
-trained model for test-set inference.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Set

import pandas as pd

from src.matching.matcher import EntityMatcher
from src.pipeline.config import PipelineConfig
from src.pipeline.output_writer import (
    expand_grouped_ids,
    write_candidate_pairs,
    write_matching_results,
)
from src.pipeline.validator import validate_submission_files

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
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
# INTEGRATION POINTS -- Persons 1-2 plug in here.
#
# Each function raises NotImplementedError until a teammate's real
# implementation lands. Do NOT invent a fake implementation in the
# meantime -- a silent dummy would produce a submission that looks
# valid but is meaningless.
# ---------------------------------------------------------------------------

def _import_blocking_module():
    """Person 1's build_candidates.py is written as a standalone CLI script
    (`sys.path.insert(0, ".")` + `from normalize import ...`), which only
    resolves when run from inside src/blocking/ directly. To import it as a
    package without touching her file, we put src/blocking/ on sys.path
    ourselves before importing -- her internal `from normalize import ...`
    then resolves against that path.
    """
    import sys as _sys

    blocking_dir = Path(__file__).resolve().parents[1] / "blocking"
    if str(blocking_dir) not in _sys.path:
        _sys.path.insert(0, str(blocking_dir))
    from src.blocking import build_candidates  # noqa: F401

    return build_candidates


def generate_candidates(
    source1: pd.DataFrame, source2: pd.DataFrame, source3: pd.DataFrame
) -> pd.DataFrame:
    """PERSON 1 CONTRACT -- blocking / candidate generation.

    Wraps Chandrakala's src/blocking/build_candidates.py (five unioned
    blocking signals: name_prefix, name_char_prefix, name_token,
    addr_pincode, addr_token, addr_concat -- see src/blocking/README.md).
    Her `run()` entrypoint reads/writes files and does its own final
    TSV grouping; we instead call her building blocks directly and stop
    right before her grouping step, so write_candidate_pairs (the single
    place TSV serialization happens) does the grouping/writing instead.

    Returns long-form dataframe with columns
    ['source1_entity_id', 'candidate_entity_id', 'signal', 'score'].
    The extra 'signal'/'score' columns are harmless passengers for the
    writer (which only reads the two ID columns) and free bonus signal
    for Person 2/3 -- 'score' is numeric, so EntityMatcher's auto
    feature-selection will pick it up as a feature without any code
    change on their end.
    """
    bc = _import_blocking_module()

    s1 = bc.add_normalized_columns(source1.copy(), "S1")
    s2 = bc.add_normalized_columns(source2.copy(), "S2")
    s3 = bc.add_normalized_columns(source3.copy(), "S3")

    cand2 = bc.generate_candidates_for_source(s1, s2, "S2")
    cand3 = bc.generate_candidates_for_source(s1, s3, "S3")
    all_cand = pd.concat([cand2, cand3], ignore_index=True)

    cand_src_combined = pd.concat([s2, s3], ignore_index=True)
    scored = bc.score_and_cap(all_cand, s1, cand_src_combined)

    return scored[["source1_entity_id", "candidate_entity_id", "signal", "score"]]


def build_features(
    candidates: pd.DataFrame,
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
) -> pd.DataFrame:
    """PERSON 2 CONTRACT -- feature engineering.

    Expected return: `candidates` with engineered numeric similarity
    columns appended (e.g. name_jaccard, addr_tfidf), preserving
    'source1_entity_id' and 'candidate_entity_id'. Raw text columns
    (business_name, business_address, country, ...) may also be
    included -- EntityMatcher auto-selects numeric feature columns
    and ignores known non-feature columns, so no fixed feature-name
    list is required here.
    """
    raise NotImplementedError(
        "Feature engineering implementation from Person 2 has not been integrated yet."
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
# Orchestration (inference)
# ---------------------------------------------------------------------------

def run_pipeline(config: PipelineConfig) -> dict:
    """Run inference against the test split and write both required
    output files. Requires a model already trained and saved by
    train_pipeline.run_training at config.model_path.

    Returns a small dict of summary stats, useful for logging and for
    experiment_tracker.py.
    """
    if not config.model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {config.model_path}. "
            "Run training first: train_pipeline.run_training(config) "
            "(or `python main.py --mode train ...`)."
        )

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

    logger.info("Loading trained matcher from %s", config.model_path)
    matcher = EntityMatcher()
    matcher.load(str(config.model_path))

    if config.threshold is not None:
        logger.info(
            "Overriding tuned threshold %.3f -> %.3f",
            matcher.best_threshold, config.threshold,
        )
        matcher.best_threshold = config.threshold
    if config.rel_margin is not None:
        logger.info(
            "Overriding tuned rel_margin %.3f -> %.3f",
            matcher.best_rel_margin, config.rel_margin,
        )
        matcher.best_rel_margin = config.rel_margin

    all_s1_ids = source1["entity_id"].tolist()

    logger.info(
        "Running model inference (Person 3) with threshold=%.3f, rel_margin=%.3f",
        matcher.best_threshold, matcher.best_rel_margin,
    )
    grouped_predictions = matcher.predict_test(features, all_s1_ids)
    # grouped_predictions columns: [source1_entity_id, matched_entity_ids]
    # -- already thresholded, already one row per S1 entity.

    predicted_matches = expand_grouped_ids(
        grouped_predictions,
        ids_col="matched_entity_ids",
        value_col="candidate_entity_id",
    )
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
        "threshold": matcher.best_threshold,
        "rel_margin": matcher.best_rel_margin,
    }