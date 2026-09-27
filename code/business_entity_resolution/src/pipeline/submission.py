"""Final submission package assembly.

Owned by Person 4. Builds <team_name>_submission.zip with exactly the
structure the challenge requires:

    <team_name>_submission.zip
    |-- output/
    |   |-- matching_results.tsv
    |   `-- candidate_pairs.tsv
    |-- code/
    |   `-- business_entity_resolution/
    |       |-- src/
    |       |-- README.md
    |       `-- requirements.txt
    `-- Documentation_template.md

Run standalone from code/business_entity_resolution/:

    python -m src.pipeline.submission --team-name my_team \
        --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv \
        --code-dir . \
        --documentation ../../Documentation_template.md \
        --test-dir dataset/test
"""
from __future__ import annotations

import argparse
import logging
import zipfile
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

# Never worth including even if present under src/ -- dev artifacts,
# not code. Trained model files (.pkl) are deliberately excluded too:
# the spec only asks for output/ + code/ + the doc, and a reviewer
# regenerates results by running the pipeline, not by receiving a
# multi-hundred-MB pickle.
_EXCLUDE_DIR_NAMES = {
    "__pycache__", ".git", ".venv", "venv", "catboost_info",
    ".pytest_cache", ".ipynb_checkpoints", "models",
}
_EXCLUDE_SUFFIXES = {".pyc", ".pkl"}


class SubmissionError(ValueError):
    """Raised when a required submission file/folder is missing, or
    validation fails and run_validation=True."""


def _check_exists(path: Path, what: str) -> None:
    if not path.exists():
        raise SubmissionError(f"Missing required {what}: {path}")


def _iter_source_files(src_dir: Path) -> Iterable[Path]:
    for path in src_dir.rglob("*"):
        if path.is_dir():
            continue
        if any(part in _EXCLUDE_DIR_NAMES for part in path.parts):
            continue
        if path.suffix in _EXCLUDE_SUFFIXES:
            continue
        yield path


def build_submission_zip(
    team_name: str,
    matching_results_path: Path,
    candidate_pairs_path: Path,
    code_dir: Path,
    documentation_path: Path,
    output_zip_dir: Path = Path("."),
    run_validation: bool = True,
    test_dir: Optional[Path] = None,
    official_validator_script: Optional[Path] = None,
) -> Path:
    """Assemble <team_name>_submission.zip. Returns the zip's path.

    `code_dir` must point AT code/business_entity_resolution/ itself
    (its src/, README.md, requirements.txt land under
    code/business_entity_resolution/ inside the zip).

    Raises SubmissionError before writing anything if a required file
    is missing, or -- when run_validation=True -- if validation fails.
    Nothing is packaged from a failed run; fix the issue and re-run.
    """
    matching_results_path = Path(matching_results_path)
    candidate_pairs_path = Path(candidate_pairs_path)
    code_dir = Path(code_dir)
    documentation_path = Path(documentation_path)
    output_zip_dir = Path(output_zip_dir)

    _check_exists(matching_results_path, "matching_results.tsv")
    _check_exists(candidate_pairs_path, "candidate_pairs.tsv")
    _check_exists(code_dir / "src", "src/ directory")
    _check_exists(code_dir / "README.md", "README.md")
    _check_exists(code_dir / "requirements.txt", "requirements.txt")
    _check_exists(documentation_path, "Documentation_template.md")

    if run_validation:
        if test_dir is None:
            raise SubmissionError("run_validation=True requires test_dir to be given")
        from src.pipeline.validator import validate_submission_files

        logger.info("Running internal validation before packaging...")
        validate_submission_files(matching_results_path, candidate_pairs_path, test_dir)

        if official_validator_script is not None:
            from src.pipeline.validator import run_official_validator

            run_official_validator(
                official_validator_script, matching_results_path,
                candidate_pairs_path, test_dir, check_ids=True,
            )
        logger.info("Validation passed -- packaging.")

    output_zip_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_zip_dir / f"{team_name}_submission.zip"

    src_files = list(_iter_source_files(code_dir / "src"))
    if not src_files:
        raise SubmissionError(f"No source files found under {code_dir / 'src'}")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(matching_results_path, "output/matching_results.tsv")
        zf.write(candidate_pairs_path, "output/candidate_pairs.tsv")

        for f in src_files:
            arcname = "code/business_entity_resolution/src/" + str(
                f.relative_to(code_dir / "src")
            ).replace("\\", "/")
            zf.write(f, arcname)

        zf.write(code_dir / "README.md", "code/business_entity_resolution/README.md")
        zf.write(code_dir / "requirements.txt", "code/business_entity_resolution/requirements.txt")
        zf.write(documentation_path, "Documentation_template.md")

    logger.info(
        "Wrote %s (%d source files, %.1f MB)",
        zip_path, len(src_files), zip_path.stat().st_size / 1e6,
    )
    return zip_path


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Assemble the final submission zip")
    p.add_argument("--team-name", required=True)
    p.add_argument("--matching", type=Path, default=Path("output/matching_results.tsv"))
    p.add_argument("--candidate", type=Path, default=Path("output/candidate_pairs.tsv"))
    p.add_argument("--code-dir", type=Path, default=Path("."))
    p.add_argument("--documentation", type=Path, default=Path("../../Documentation_template.md"))
    p.add_argument("--output-zip-dir", type=Path, default=Path("."))
    p.add_argument("--test-dir", type=Path, default=Path("dataset/test"))
    p.add_argument("--skip-validation", action="store_true")
    p.add_argument("--official-validator-script", type=Path, default=Path("utils/validate_submission.py"))
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = _build_arg_parser().parse_args()

    zip_path = build_submission_zip(
        team_name=args.team_name,
        matching_results_path=args.matching,
        candidate_pairs_path=args.candidate,
        code_dir=args.code_dir,
        documentation_path=args.documentation,
        output_zip_dir=args.output_zip_dir,
        run_validation=not args.skip_validation,
        test_dir=args.test_dir,
        official_validator_script=args.official_validator_script,
    )
    logging.getLogger(__name__).info("Submission ready: %s", zip_path)


if __name__ == "__main__":
    main()