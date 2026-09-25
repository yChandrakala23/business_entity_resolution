"""Output writers for candidate_pairs.tsv and matching_results.tsv.

Owned by Person 4. This is the single source of truth for how the
final submission TSV files are serialized. Every Source 1 entity
must appear exactly once, singleton rows must have a literally empty
second field (never "NaN"/"None"/"[]"), and candidate/matched IDs
must be deduplicated and sorted for deterministic output.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_CANDIDATE_COLUMNS = {"source1_entity_id", "candidate_entity_id"}
REQUIRED_MATCH_COLUMNS = {"source1_entity_id", "candidate_entity_id"}


def _group_ids(
    df: pd.DataFrame,
    id_col: str = "source1_entity_id",
    value_col: str = "candidate_entity_id",
) -> Dict[str, List[str]]:
    """Group value_col by id_col into sorted, deduplicated lists."""
    grouped: Dict[str, List[str]] = {}
    for s1_id, sub in df.groupby(id_col)[value_col]:
        grouped[s1_id] = sorted(set(sub.tolist()))
    return grouped


def _validate_id_prefixes(df: pd.DataFrame, value_col: str, context: str) -> None:
    bad = df[~df[value_col].astype(str).str.startswith(("S2-", "S3-"))]
    if not bad.empty:
        raise ValueError(
            f"{context}: found IDs that are not S2-/S3- prefixed "
            f"(no S1 IDs are allowed in output lists): "
            f"{bad[value_col].unique().tolist()[:10]}"
        )


def _write_grouped_tsv(
    source1: pd.DataFrame,
    pairs: pd.DataFrame,
    output_path: Path,
    output_col_name: str,
    context: str,
) -> pd.DataFrame:
    if "entity_id" not in source1.columns:
        raise ValueError("source1 dataframe must contain an 'entity_id' column")

    pairs = pairs.dropna(subset=["source1_entity_id", "candidate_entity_id"])
    _validate_id_prefixes(pairs, "candidate_entity_id", context)

    grouped = _group_ids(pairs)

    rows = []
    for s1_id in source1["entity_id"]:
        ids = grouped.get(s1_id, [])
        rows.append({
            "source1_entity_id": s1_id,
            output_col_name: ",".join(ids),  # empty string for singletons
        })

    out_df = pd.DataFrame(rows, columns=["source1_entity_id", output_col_name])
    out_df = out_df.drop_duplicates(subset=["source1_entity_id"])

    n_expected = source1["entity_id"].nunique()
    if len(out_df) != n_expected:
        raise ValueError(
            f"{context}: expected {n_expected} unique Source 1 rows, got {len(out_df)}. "
            "This usually means the source1 dataframe itself has duplicate entity_ids."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, sep="\t", index=False)
    logger.info("Wrote %s: %d rows -> %s", output_path.name, len(out_df), output_path)
    return out_df


def write_candidate_pairs(
    source1: pd.DataFrame,
    candidates: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Write candidate_pairs.tsv.

    Parameters
    ----------
    source1:
        Full Source 1 dataframe (must contain an ``entity_id`` column).
        Used so every S1 entity is represented even with zero candidates.
    candidates:
        Long-form dataframe with columns ``source1_entity_id`` and
        ``candidate_entity_id`` — the FINAL candidate set actually
        sent into the matching model (post any filtering). Do not
        pass raw, unfiltered blocking output here if further
        candidate filters were applied downstream.
    output_path:
        Destination .tsv path.

    Returns
    -------
    The two-column dataframe that was written
    (``source1_entity_id``, ``candidate_entity_ids``) — handy for tests.
    """
    missing = REQUIRED_CANDIDATE_COLUMNS - set(candidates.columns)
    if missing:
        raise ValueError(
            f"Blocking output is missing required columns: {sorted(missing)}"
        )
    return _write_grouped_tsv(
        source1, candidates, output_path, "candidate_entity_ids", "candidate_pairs.tsv"
    )


def write_matching_results(
    source1: pd.DataFrame,
    predicted_matches: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Write matching_results.tsv.

    Parameters
    ----------
    source1:
        Full Source 1 dataframe (must contain an ``entity_id`` column).
    predicted_matches:
        Long-form dataframe with columns ``source1_entity_id`` and
        ``candidate_entity_id`` for pairs that PASSED the match
        threshold. Thresholding is expected to have already happened
        before this function is called — an optional
        ``match_probability`` column, if present, is ignored.
    output_path:
        Destination .tsv path.
    """
    missing = REQUIRED_MATCH_COLUMNS - set(predicted_matches.columns)
    if missing:
        raise ValueError(
            f"Predicted matches are missing required columns: {sorted(missing)}"
        )
    return _write_grouped_tsv(
        source1, predicted_matches, output_path, "matched_entity_ids", "matching_results.tsv"
    )