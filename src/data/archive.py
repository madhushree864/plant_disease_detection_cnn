"""Read-only inspection of the unaugmented dataset ZIP archive.

The functions in this module read archive members in place.  They never extract,
rename, or otherwise modify the source archive.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable, Iterator, Sequence
import zipfile

try:
    from PIL import Image
except ImportError:  # pragma: no cover - reported clearly when validation is called
    Image = None

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"})


@dataclass(frozen=True)
class ImageEntry:
    """An image member and its directory-derived class label."""

    archive_path: str
    label: str
    size: int


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating image members without extracting them."""

    total: int
    readable: int
    unreadable: tuple[str, ...]


def _label_for_member(name: str) -> str | None:
    path = PurePosixPath(name)
    if len(path.parts) < 3 or path.name.startswith("."):
        return None
    suffix = path.suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        return None
    # The final directory is the label; the leading directory is the archive root.
    return path.parts[-2]


def iter_image_entries(archive_path: str) -> Iterator[ImageEntry]:
    """Yield image entries in a stable archive-name order."""

    with zipfile.ZipFile(archive_path) as archive:
        members = sorted((info for info in archive.infolist() if not info.is_dir()), key=lambda i: i.filename)
        for info in members:
            label = _label_for_member(info.filename)
            if label is not None:
                yield ImageEntry(info.filename, label, info.file_size)


def discover_labels(entries: Iterable[ImageEntry]) -> tuple[str, ...]:
    """Return labels in canonical order, suitable for a class-to-index mapping."""

    return tuple(sorted({entry.label for entry in entries}))


def validate_images(archive_path: str, entries: Sequence[ImageEntry] | None = None) -> ValidationResult:
    """Decode-check every selected member while keeping the archive untouched."""

    if Image is None:
        raise RuntimeError("Pillow is required for image readability validation")
    selected = tuple(entries) if entries is not None else tuple(iter_image_entries(archive_path))
    unreadable: list[str] = []
    with zipfile.ZipFile(archive_path) as archive:
        for entry in selected:
            try:
                with archive.open(entry.archive_path) as stream:
                    with Image.open(stream) as image:
                        image.verify()
            except Exception:  # Pillow raises several format-specific exceptions.
                unreadable.append(entry.archive_path)
    return ValidationResult(len(selected), len(selected) - len(unreadable), tuple(unreadable))
