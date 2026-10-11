import io
import zipfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from src.data.archive import ImageEntry, discover_labels, iter_image_entries, validate_images
from src.data.split import class_mapping, stratified_split


def make_archive(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for label in ("Class_B", "Class_A", "Class_C"):
            for number in range(10):
                image = Image.new("RGB", (4, 4), (number, 0, 0))
                payload = io.BytesIO()
                image.save(payload, format="JPEG")
                archive.writestr(f"dataset/{label}/image_{number}.jpg", payload.getvalue())


class DataModuleTests(unittest.TestCase):
    def test_class_discovery_and_readability(self):
        with TemporaryDirectory() as directory:
            archive = Path(directory) / "dataset.zip"
            make_archive(archive)
            entries = tuple(iter_image_entries(str(archive)))
            self.assertEqual(discover_labels(entries), ("Class_A", "Class_B", "Class_C"))
            result = validate_images(str(archive), entries)
            self.assertEqual((result.total, result.readable, result.unreadable), (30, 30, ()))

    def test_split_is_reproducible_and_has_no_leakage(self):
        with TemporaryDirectory() as directory:
            archive = Path(directory) / "dataset.zip"
            make_archive(archive)
            entries = tuple(iter_image_entries(str(archive)))
            first = stratified_split(entries, seed=42)
            second = stratified_split(entries, seed=42)
            self.assertEqual(first, second)
            self.assertEqual(len({record.archive_path for record in first}), len(first))
            for label in discover_labels(entries):
                counts = {split: sum(record.label == label and record.split == split for record in first)
                          for split in ("train", "validation", "test")}
                self.assertEqual(counts, {"train": 7, "validation": 2, "test": 1})

    def test_class_mapping_is_consistent_with_split(self):
        entries = tuple(
            ImageEntry(f"{label}/{number}", label, 1)
            for label in ("z", "a") for number in range(2)
        )
        mapping = class_mapping(entries)
        records = stratified_split(entries, seed=42)
        self.assertEqual(mapping, {"a": 0, "z": 1})
        self.assertTrue(all(record.class_index == mapping[record.label] for record in records))


if __name__ == "__main__":
    unittest.main()
