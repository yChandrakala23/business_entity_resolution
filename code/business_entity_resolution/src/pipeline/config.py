"""Pipeline configuration for the Business Entity Resolution project.

Owned by Person 4. This is the single source of truth for paths and
run-time knobs.

NOTE on threshold/rel_margin: Person 3's EntityMatcher tunes its own
match-probability threshold and relative margin during training
(via a 2D grid search maximizing local Macro F0.5) and stores them
as `best_threshold` / `best_rel_margin` on the saved model. These
config fields are therefore OPTIONAL OVERRIDES for experimentation
only -- leave them as None to trust the value the model already
tuned; set them to force a specific cutoff at inference time without
retraining.
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
    threshold: Optional[float] = None
    rel_margin: Optional[float] = None
    model_path: Optional[Path] = None
    skip_validation: bool = False
    official_validator_script: Optional[Path] = None

    def __post_init__(self) -> None:
        self.train_dir = Path(self.train_dir)
        self.test_dir = Path(self.test_dir)
        self.output_dir = Path(self.output_dir)
        if self.official_validator_script is not None:
            self.official_validator_script = Path(self.official_validator_script)

        if self.threshold is not None and not (0.0 <= self.threshold <= 1.0):
            raise ValueError(
                f"threshold must be between 0 and 1, got {self.threshold}"
            )
        if self.rel_margin is not None and self.rel_margin < 0.0:
            raise ValueError(f"rel_margin must be >= 0, got {self.rel_margin}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.model_path is None:
            self.model_path = self.output_dir / "models" / "ensemble_matcher.pkl"
        else:
            self.model_path = Path(self.model_path)

    @property
    def matching_results_path(self) -> Path:
        return self.output_dir / "matching_results.tsv"

    @property
    def candidate_pairs_path(self) -> Path:
        return self.output_dir / "candidate_pairs.tsv"