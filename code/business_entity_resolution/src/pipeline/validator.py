"""Internal validation checks for Business Entity Resolution outputs.

Owned by Person 4. Runs cheap, focused checks *before* invoking the
official utils/validate_submission.py script, so obvious mistakes are
caught early instead of burning a leaderboard submission attempt.
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class ValidationError(ValueError):
    """Raised when a submission file fails an internal consistency check."""


def _read_tsv(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise ValidationError(f"Expected file not found: {path}")
    # keep_default_na=False so a genuinely empty second field reads as ""
    # rather than being parsed as NaN.
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def _split_ids(cell: str) -> List[str]:
    cell = (cell or "").strip()
    if not cell:
        return []
    return [c.strip() for c in cell.split(",") if c.strip()]


def _load_test_ids(test_dir: Path) -> Tuple[Set[str], Set[str], Set[str]]:
    """Return (s1_ids, s2_ids, s3_ids) present in the test data."""
    test_dir = Path(test_dir)
    s1 = pd.read_csv(test_dir / "test_source1.tsv", sep="\t", dtype=str)
    s2 = pd.read_csv(test_dir / "test_source2.tsv", sep="\t", dtype=str)
    s3 = pd.read_csv(test_dir / "test_source3.tsv", sep="\t", dtype=str)
    return set(s1["entity_id"]), set(s2["entity_id"]), set(s3["entity_id"])


def _validate_grouped_file(
    df: pd.DataFrame,
    id_col: str,
    ids_col: str,
    s1_ids: Set[str],
    valid_s2_s3: Set[str],
    context: str,
) -> None:
    required_cols = {id_col, ids_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValidationError(f"{context} missing required columns: {sorted(missing)}")

    if df[id_col].duplicated().any():
        dupes = df.loc[df[id_col].duplicated(), id_col].tolist()
        raise ValidationError(f"{context}: duplicate {id_col} rows: {dupes[:10]}")

    seen = set(df[id_col])
    missing_s1 = s1_ids - seen
    if missing_s1:
        raise ValidationError(
            f"{context}: missing {len(missing_s1)} Source 1 test IDs, "
            f"e.g. {sorted(missing_s1)[:5]}"
        )
    extra_s1 = seen - s1_ids
    if extra_s1:
        raise ValidationError(
            f"{context}: has Source 1 IDs not present in the test set: "
            f"{sorted(extra_s1)[:5]}"
        )

    for _, row in df.iterrows():
        ids = _split_ids(row[ids_col])

        if len(ids) != len(set(ids)):
            raise ValidationError(f"{context}: duplicate IDs for {row[id_col]}: {ids}")

        bad_prefix = [i for i in ids if not i.startswith(("S2-", "S3-"))]
        if bad_prefix:
            raise ValidationError(
                f"{context}: non-S2-/S3- IDs for {row[id_col]}: {bad_prefix} "
                "(no S1 IDs are allowed in output lists)"
            )

        unknown = [i for i in ids if i not in valid_s2_s3]
        if unknown:
            raise ValidationError(
                f"{context}: IDs not present in test data for {row[id_col]}: {unknown}"
            )


def validate_candidate_pairs(candidate_pairs_path: Path, test_dir: Path) -> None:
    df = _read_tsv(candidate_pairs_path)
    s1_ids, s2_ids, s3_ids = _load_test_ids(test_dir)
    _validate_grouped_file(
        df, "source1_entity_id", "candidate_entity_ids",
        s1_ids, s2_ids | s3_ids, "candidate_pairs.tsv",
    )
    logger.info("candidate_pairs.tsv passed internal validation (%d rows).", len(df))


def validate_matching_results(matching_results_path: Path, test_dir: Path) -> None:
    df = _read_tsv(matching_results_path)
    s1_ids, s2_ids, s3_ids = _load_test_ids(test_dir)
    _validate_grouped_file(
        df, "source1_entity_id", "matched_entity_ids",
        s1_ids, s2_ids | s3_ids, "matching_results.tsv",
    )
    logger.info("matching_results.tsv passed internal validation (%d rows).", len(df))


def validate_match_subset(matching_results_path: Path, candidate_pairs_path: Path) -> None:
    """Ensure every matched ID is present in that S1 entity's candidate list."""
    matches = _read_tsv(matching_results_path)
    candidates = _read_tsv(candidate_pairs_path)

    cand_map = {
        row["source1_entity_id"]: set(_split_ids(row["candidate_entity_ids"]))
        for _, row in candidates.iterrows()
    }

    violations = {}
    for _, row in matches.iterrows():
        s1_id = row["source1_entity_id"]
        matched_ids = set(_split_ids(row["matched_entity_ids"]))
        not_in_candidates = matched_ids - cand_map.get(s1_id, set())
        if not_in_candidates:
            violations[s1_id] = sorted(not_in_candidates)

    if violations:
        sample = dict(list(violations.items())[:5])
        raise ValidationError(
            f"{len(violations)} Source 1 entities have matches not present in their "
            f"candidate set (matching_results.tsv must be a subset of "
            f"candidate_pairs.tsv). Examples: {sample}"
        )

    logger.info("matching_results.tsv is a valid subset of candidate_pairs.tsv.")


def validate_submission_files(
    matching_results_path: Path,
    candidate_pairs_path: Path,
    test_dir: Path,
) -> None:
    """Run all internal checks in one call. Raises ValidationError on failure."""
    validate_candidate_pairs(candidate_pairs_path, test_dir)
    validate_matching_results(matching_results_path, test_dir)
    validate_match_subset(matching_results_path, candidate_pairs_path)
    logger.info("All internal validation checks passed.")


def run_official_validator(
    validator_script: Path,
    matching_results_path: Path,
    candidate_pairs_path: Path,
    test_dir: Path,
) -> None:
    """Invoke the official utils/validate_submission.py script.

    Raises ValidationError if the script is missing or exits non-zero.
    """
    validator_script = Path(validator_script)
    if not validator_script.exists():
        raise ValidationError(
            f"Official validator not found at {validator_script}."
        )

    cmd = [
        sys.executable,
        str(validator_script),
        "--matching", str(matching_results_path),
        "--candidate", str(candidate_pairs_path),
        "--test-dir", str(test_dir),
    ]
    logger.info("Running official validator: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise ValidationError(
            f"Official validate_submission.py failed "
            f"(exit code {result.returncode}):\n{result.stdout}\n{result.stderr}"
        )

    logger.info("Official validator passed.\n%s", result.stdout)