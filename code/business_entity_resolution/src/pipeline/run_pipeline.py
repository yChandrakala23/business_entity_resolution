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

    dtype=str + keep_default_na=False is deliberate, not a default: a
    missing business_address (real data has these -- README notes
    "missing components") otherwise parses as NaN (a float), which
    crashes normalize_address's raw.strip() downstream. Every entity_id/
    business_name/business_address/country field should stay a literal
    string regardless of content.

    NOTE: If Person 1 builds a dedicated `load_train_data` /
    `load_test_data` function with extra logic, swap it in here rather
    than duplicating loading logic elsewhere -- keep the same dtype
    safety this function has.
    """
    data_dir = Path(data_dir)
    read = lambda name: pd.read_csv(data_dir / name, sep="\t", dtype=str, keep_default_na=False)
    return {
        "source1": read(f"{split}_source1.tsv"),
        "source2": read(f"{split}_source2.tsv"),
        "source3": read(f"{split}_source3.tsv"),
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

    Partitioned by country before calling her functions, same as her
    own standalone CLI's --country flag already recommends ("process
    only this country ... bounds memory/runtime losslessly" -- matches
    never cross countries, so this changes nothing about correctness,
    only memory footprint per call). Consistent with how build_features
    is chunked below.

    Returns long-form dataframe with columns
    ['source1_entity_id', 'candidate_entity_id', 'signal', 'score'].
    The extra 'signal'/'score' columns are harmless passengers for the
    writer (which only reads the two ID columns) and free bonus signal
    for Person 2/3 -- 'score' is numeric, so EntityMatcher's auto
    feature-selection will pick it up as a feature without any code
    change on their end.
    """
    bc = _import_blocking_module()

    chunks = []
    for country in sorted(source1["country"].unique()):
        sub_s1 = source1[source1["country"] == country]
        sub_s2 = source2[source2["country"] == country]
        sub_s3 = source3[source3["country"] == country]

        logger.info(
            "generate_candidates: country=%s, %d S1 / %d S2 / %d S3 rows",
            country, len(sub_s1), len(sub_s2), len(sub_s3),
        )

        s1 = bc.add_normalized_columns(sub_s1.copy(), f"S1[{country}]")
        s2 = bc.add_normalized_columns(sub_s2.copy(), f"S2[{country}]")
        s3 = bc.add_normalized_columns(sub_s3.copy(), f"S3[{country}]")

        cand2 = bc.generate_candidates_for_source(s1, s2, f"S2[{country}]")
        cand3 = bc.generate_candidates_for_source(s1, s3, f"S3[{country}]")
        all_cand = pd.concat([cand2, cand3], ignore_index=True)

        cand_src_combined = pd.concat([s2, s3], ignore_index=True)
        scored = bc.score_and_cap(all_cand, s1, cand_src_combined)
        chunks.append(scored[["source1_entity_id", "candidate_entity_id", "signal", "score"]])

    return pd.concat(chunks, ignore_index=True)


def build_features(
    candidates: pd.DataFrame,
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
) -> pd.DataFrame:
    """PERSON 2 CONTRACT -- feature engineering.

    Thin wrapper around Person 2's real src/features/build_features.py
    (Mansi). Her implementation is fully vectorized per-call (sparse
    CountVectorizer/TfidfVectorizer for Jaccard/TF-IDF, rapidfuzz for
    Levenshtein/Jaro-Winkler, jellyfish for phonetic match -- see her
    module docstring for the full vectorization rationale).

    Why this wrapper exists rather than calling her function directly:
    benchmarked on real sampled data, her single-call cost is
    ~0.04-0.06 ms/pair and NOT superlinear -- but extrapolated to the
    real full scale (~2.2M S1 x up to 100 candidates =~ 205M pairs)
    that's still >2 hours in one unchunked call, and one CountVectorizer/
    TfidfVectorizer fit_transform over the full corpus at once risks
    memory pressure alongside everything else the pipeline holds in
    memory concurrently.

    This wraps her function with the same per-country partitioning
    Person 1's blocking stage already uses, calling her build_features
    once per country and concatenating results. Since her vectorizers
    are refit fresh on every call already (no shared vocabulary across
    calls), this changes nothing about correctness -- TF-IDF/Jaccard
    are computed as row-aligned pairwise similarity, never compared
    across rows, so a per-country vocabulary is equivalent in kind to
    a global one, just bounded in size. Country is a required column
    on every real record, so this never drops or misroutes a pair.

    BUGFIX (verified against real src/blocking/normalize.py, which
    didn't exist yet when this feature module was written): her
    _unpack_addr() looked for a "canonical" key that Person 1's real
    normalize_address() calls "normalized" instead -- and her
    post-merge token cleanup checked `isinstance(v, list)`, but real
    tokens come back as `frozenset`, so every row's tokens were being
    silently wiped to `[]` (this actually crashed outright: "empty
    vocabulary" from CountVectorizer). Both are fixed directly in
    src/features/build_features.py -- confirmed against real sampled
    data, feature separation now matches her own README-person2.md
    validation table.
    """
    from src.features.build_features import build_features as _person2_build_features

    country_lookup = source1.drop_duplicates("entity_id").set_index("entity_id")["country"]
    candidate_country = candidates["source1_entity_id"].map(country_lookup)

    chunks = []
    for country, sub_idx in candidate_country.groupby(candidate_country).groups.items():
        sub_candidates = candidates.loc[sub_idx]
        s1_ids_needed = set(sub_candidates["source1_entity_id"])
        cand_ids_needed = set(sub_candidates["candidate_entity_id"])

        sub_s1 = source1[source1["entity_id"].isin(s1_ids_needed)]
        sub_s2 = source2[source2["entity_id"].isin(cand_ids_needed)]
        sub_s3 = source3[source3["entity_id"].isin(cand_ids_needed)]

        logger.info(
            "build_features: country=%s, %d pairs, %d S1 / %d S2 / %d S3 rows",
            country, len(sub_candidates), len(sub_s1), len(sub_s2), len(sub_s3),
        )
        chunks.append(_person2_build_features(sub_candidates, sub_s1, sub_s2, sub_s3))

    return pd.concat(chunks, ignore_index=True)


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