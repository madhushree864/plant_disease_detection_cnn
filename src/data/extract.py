"""Extract and verify the unaugmented archive: ``python -m src.data.extract``."""

from __future__ import annotations

import argparse

from .extraction import extract_archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", help="path to the unaugmented ZIP archive")
    parser.add_argument("--output", default="data/extracted")
    parser.add_argument("--manifest", help="manifest used to verify paths and counts")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    rows = None
    if args.manifest:
        from src.training.orchestrator import load_manifest
        rows = load_manifest(args.manifest)
    print(extract_archive(args.archive, args.output, manifest_rows=rows, force=args.force))


if __name__ == "__main__":
    main()
