"""Deterministic class-stratified split and manifest records."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
from typing import Iterable, Mapping

from .archive import ImageEntry


@dataclass(frozen=True)
class SplitRecord:
    archive_path: str
    label: str
    class_index: int
    split: str


def class_mapping(entries: Iterable[ImageEntry]) -> dict[str, int]:
    """Map every discovered label to an index using one stable ordering."""

    return {label: index for index, label in enumerate(sorted({entry.label for entry in entries}))}


def stratified_split(
    entries: Iterable[ImageEntry],
    seed: int = 42,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[SplitRecord, ...]:
    """Create a deterministic 70/15/15 split within each class.

    The remainder is assigned to test, so every entry occurs exactly once.
    """

    entries = tuple(sorted(entries, key=lambda item: item.archive_path))
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave test data")
    mapping = class_mapping(entries)
    grouped: dict[str, list[ImageEntry]] = defaultdict(list)
    for entry in entries:
        grouped[entry.label].append(entry)

    records: list[SplitRecord] = []
    for label in sorted(grouped):
        members = grouped[label]
        random.Random(seed + mapping[label]).shuffle(members)
        # Round boundaries rather than flooring both portions, avoiding a
        # systematic under-allocation of validation examples.
        train_count = int(len(members) * train_fraction + 0.5)
        validation_boundary = int(len(members) * (train_fraction + validation_fraction) + 0.5)
        validation_count = validation_boundary - train_count
        for position, entry in enumerate(members):
            split = "train" if position < train_count else "validation" if position < train_count + validation_count else "test"
            records.append(SplitRecord(entry.archive_path, label, mapping[label], split))
    return tuple(sorted(records, key=lambda record: record.archive_path))
