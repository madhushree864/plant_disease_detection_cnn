"""Load a few batches from the verified file-backed dataset without training."""

from __future__ import annotations

import argparse

from src.data.extraction import extract_archive
from src.preprocessing import make_file_tf_dataset
from .orchestrator import _split, load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("manifest")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--shuffle-buffer-size", type=int, default=512)
    args = parser.parse_args()
    if args.batches < 1:
        parser.error("--batches must be positive")
    rows = _split(load_manifest(args.manifest), args.split)
    all_rows = load_manifest(args.manifest)
    extracted = extract_archive(args.archive, "data/extracted", manifest_rows=all_rows)
    dataset = make_file_tf_dataset(
        str(extracted),
        rows,
        training=args.split == "train",
        batch_size=args.batch_size,
        shuffle=args.split == "train",
        shuffle_buffer_size=args.shuffle_buffer_size,
    )
    for batch_number, (images, labels) in enumerate(dataset.take(args.batches), start=1):
        print(
            f"batch {batch_number}: images={tuple(images.shape)}, "
            f"labels={labels.numpy().tolist()}"
        )


if __name__ == "__main__":
    main()
