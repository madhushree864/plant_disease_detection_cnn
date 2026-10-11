"""Opt-in held-out evaluation command: ``python -m src.evaluation.cli``."""

from __future__ import annotations

import argparse

from .orchestrator import evaluate_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("manifest")
    parser.add_argument("model", help="path to a saved Keras model")
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", default="outputs/evaluation")
    args = parser.parse_args()
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("TensorFlow is required for evaluation") from exc
    result = evaluate_model(tf.keras.models.load_model(args.model), args.archive, args.manifest,
                            split=args.split, batch_size=args.batch_size, output_dir=args.output_dir)
    print({key: result[key] for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1")})


if __name__ == "__main__":
    main()
