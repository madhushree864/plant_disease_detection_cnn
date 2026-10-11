"""Opt-in training command: ``python -m src.training.cli``."""

from __future__ import annotations

import argparse

from src.models import build_custom_cnn, build_efficientnetb0
from .config import TrainingConfig, configure_cpu
from .orchestrator import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("manifest")
    parser.add_argument("--model", choices=("custom_cnn", "efficientnetb0"), default="custom_cnn")
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shuffle-buffer-size", type=int, default=512)
    parser.add_argument("--output-dir", default="outputs/training")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    configure_cpu(args.threads)
    factory = build_custom_cnn if args.model == "custom_cnn" else build_efficientnetb0
    model = factory(num_classes=args.classes, weights=None) if args.model == "efficientnetb0" else factory(num_classes=args.classes)
    train_model(model, args.archive, args.manifest, config=TrainingConfig(
        epochs=args.epochs, batch_size=args.batch_size, output_dir=args.output_dir,
        shuffle_buffer_size=args.shuffle_buffer_size,
    ))


if __name__ == "__main__":
    main()
