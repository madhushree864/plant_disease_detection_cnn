"""Command-line generation of a small CSV split manifest (images stay in the ZIP)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .archive import iter_image_entries, validate_images
from .split import stratified_split


def write_manifest(archive_path: str, output_path: str, seed: int = 42) -> dict[str, int]:
    entries = tuple(iter_image_entries(archive_path))
    validation = validate_images(archive_path, entries)
    if validation.unreadable:
        raise ValueError(f"{len(validation.unreadable)} unreadable image entries; refusing to create a manifest")
    records = stratified_split(entries, seed=seed)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("archive_path", "label", "class_index", "split"))
        writer.writeheader()
        writer.writerows(record.__dict__ for record in records)
    return {"images": len(records), "labels": len({record.label for record in records}), "readable": validation.readable}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", help="path to the unaugmented ZIP archive")
    parser.add_argument("--output", default="outputs/data_split_manifest.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(write_manifest(args.archive, args.output, args.seed))


if __name__ == "__main__":
    main()
