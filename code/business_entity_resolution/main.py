"""CLI entry point for the Business Entity Resolution pipeline.

Usage:
    python main.py \
        --train-dir dataset/train \
        --test-dir dataset/test \
        --output-dir output \
        --threshold 0.85
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.pipeline.config import PipelineConfig
from src.pipeline.run_pipeline import run_pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Business Entity Resolution pipeline")
    parser.add_argument("--train-dir", type=Path, default=Path("dataset/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("dataset/test"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip internal validation after writing outputs",
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

    config = PipelineConfig(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
        model_path=args.model_path,
        skip_validation=args.skip_validation,
    )

    summary = run_pipeline(config)
    logging.getLogger(__name__).info("Pipeline finished: %s", summary)


if __name__ == "__main__":
    main()