"""Pipeline configuration for the Business Entity Resolution project.

Owned by Person 4 (Pipeline / Infrastructure / Integration /
Documentation Lead). This is the single source of truth for
paths and the match threshold — no other module should hard-code
these values.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PipelineConfig:
    train_dir: Path
    test_dir: Path
    output_dir: Path
    threshold: float = 0.85
    model_path: Optional[Path] = None
    skip_validation: bool = False
    official_validator_script: Optional[Path] = None

    def __post_init__(self) -> None:
        self.train_dir = Path(self.train_dir)
        self.test_dir = Path(self.test_dir)
        self.output_dir = Path(self.output_dir)
        if self.model_path is not None:
            self.model_path = Path(self.model_path)
        if self.official_validator_script is not None:
            self.official_validator_script = Path(self.official_validator_script)

        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(
                f"threshold must be between 0 and 1, got {self.threshold}"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def matching_results_path(self) -> Path:
        return self.output_dir / "matching_results.tsv"

    @property
    def candidate_pairs_path(self) -> Path:
        return self.output_dir / "candidate_pairs.tsv"