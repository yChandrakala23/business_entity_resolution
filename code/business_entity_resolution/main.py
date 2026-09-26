"""CLI entry point for the Business Entity Resolution pipeline.

Train (fits and saves Person 3's EntityMatcher):
    python main.py --mode train \
        --train-dir dataset/train \
        --output-dir output

Predict (loads the saved model, runs test-set inference):
    python main.py --mode predict \
        --test-dir dataset/test \
        --output-dir output

--threshold / --rel-margin are optional overrides. Leave them unset
to use the threshold and margin the model already tuned during
training (recommended); set them only to experiment with a manual
cutoff without retraining.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.pipeline.config import PipelineConfig
from src.pipeline.run_pipeline import run_pipeline
from src.pipeline.train_pipeline import run_training


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Business Entity Resolution pipeline")
    parser.add_argument("--mode", choices=["train", "predict"], default="predict")
    parser.add_argument("--train-dir", type=Path, default=Path("dataset/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("dataset/test"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override the model's tuned match-probability threshold (0-1).",
    )
    parser.add_argument(
        "--rel-margin", type=float, default=None,
        help="Override the model's tuned relative margin from the top candidate.",
    )
    parser.add_argument(
        "--model-path", type=Path, default=None,
        help="Where to save (train mode) / load (predict mode) the matcher. "
             "Defaults to <output-dir>/models/ensemble_matcher.pkl",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip internal validation after writing outputs (predict mode only)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(levelname)s] %(message)s",
    )
    log = logging.getLogger(__name__)

    config = PipelineConfig(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
        rel_margin=args.rel_margin,
        model_path=args.model_path,
        skip_validation=args.skip_validation,
    )

    if args.mode == "train":
        summary = run_training(config)
    else:
        summary = run_pipeline(config)

    log.info("Pipeline finished (%s mode): %s", args.mode, summary)


if __name__ == "__main__":
    main()