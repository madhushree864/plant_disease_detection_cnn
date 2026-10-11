"""Safe, reusable extraction cache for the unaugmented dataset ZIP."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping

EXPECTED_IMAGES = 55_448
EXPECTED_CLASSES = 39
MARKER = ".extraction_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe ZIP member path: {name}")
    return path


def _manifest_counts(rows: Iterable[Mapping[str, object]]) -> tuple[int, int]:
    rows = tuple(rows)
    return len(rows), len({str(row["label"]) for row in rows})


def verify_extracted_cache(
    archive_path: str | Path,
    cache_dir: str | Path,
    *,
    manifest_rows: Iterable[Mapping[str, object]] | None = None,
    expected_images: int = EXPECTED_IMAGES,
    expected_classes: int = EXPECTED_CLASSES,
) -> bool:
    """Return true only when the cache marker, source, files, and counts agree."""
    source, cache = Path(archive_path), Path(cache_dir)
    marker_path = cache / MARKER
    if not source.is_file() or not marker_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker["source_sha256"] != _sha256(source):
            return False
        members = marker["members"]
        if marker["image_count"] != expected_images or marker["class_count"] != expected_classes:
            return False
        if manifest_rows is not None:
            count, classes = _manifest_counts(manifest_rows)
            if count != marker["image_count"] or classes != marker["class_count"]:
                return False
            expected_members = {str(row["archive_path"]) for row in manifest_rows}
            if expected_members != set(members):
                return False
        for name, details in members.items():
            target = cache.joinpath(*_safe_path(name).parts)
            if not isinstance(details, dict):
                return False
            if (not target.is_file() or target.stat().st_size != details["size"]
                    or _sha256(target) != details["sha256"]):
                return False
        return True
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def extract_archive(
    archive_path: str | Path,
    output_dir: str | Path = "data/extracted",
    *,
    manifest_rows: Iterable[Mapping[str, object]] | None = None,
    expected_images: int = EXPECTED_IMAGES,
    expected_classes: int = EXPECTED_CLASSES,
    force: bool = False,
) -> Path:
    """Extract image members once and return a verified cache directory.

    The ZIP is opened read-only. A sibling staging directory is atomically
    renamed into place, so an interrupted extraction cannot look complete.
    """
    source, destination = Path(archive_path), Path(output_dir)
    rows = tuple(manifest_rows) if manifest_rows is not None else None
    if not force and verify_extracted_cache(
        source, destination, manifest_rows=rows,
        expected_images=expected_images, expected_classes=expected_classes,
    ):
        return destination
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    members: dict[str, dict[str, object]] = {}
    classes: set[str] = set()
    try:
        with zipfile.ZipFile(source) as archive:
            for info in sorted(archive.infolist(), key=lambda item: item.filename):
                if info.is_dir():
                    continue
                relative = _safe_path(info.filename)
                if relative.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}:
                    continue
                target = staging.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as input_stream, target.open("wb") as output_stream:
                    shutil.copyfileobj(input_stream, output_stream)
                members[info.filename] = {
                    "size": info.file_size,
                    "sha256": _sha256(target),
                }
                if len(relative.parts) >= 2:
                    classes.add(relative.parts[-2])
        if len(members) != expected_images or len(classes) != expected_classes:
            raise ValueError(
                f"extracted dataset counts are {len(members)} images/{len(classes)} classes; "
                f"expected {expected_images}/{expected_classes}"
            )
        if rows is not None and {str(row["archive_path"]) for row in rows} != set(members):
            raise ValueError("manifest archive paths do not match extracted image members")
        marker = {
            "source_sha256": _sha256(source),
            "image_count": len(members),
            "class_count": len(classes),
            "members": members,
        }
        (staging / MARKER).write_text(json.dumps(marker, indent=2, sort_keys=True), encoding="utf-8")
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staging, destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    if not verify_extracted_cache(
        source, destination, manifest_rows=rows,
        expected_images=expected_images, expected_classes=expected_classes,
    ):
        raise RuntimeError("extraction cache failed post-write verification")
    return destination
