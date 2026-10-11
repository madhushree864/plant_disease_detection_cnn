import io
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from src.data.extraction import extract_archive, verify_extracted_cache
from src.preprocessing import make_file_tf_dataset


def _archive(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for label, value in (("Class_A", 20), ("Class_B", 80)):
            image = Image.new("RGB", (8, 8), (value, 0, 0))
            payload = io.BytesIO()
            image.save(payload, format="JPEG")
            archive.writestr(f"dataset/{label}/image.jpg", payload.getvalue())


class ExtractionAndFileLoaderTests(unittest.TestCase):
    def test_extraction_cache_reuse_and_source_verification(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "dataset.zip"
            cache = root / "extracted"
            _archive(archive)
            rows = tuple(
                {"archive_path": f"dataset/{label}/image.jpg", "label": label,
                 "class_index": str(index), "split": "validation"}
                for index, label in enumerate(("Class_A", "Class_B"))
            )
            extracted = extract_archive(archive, cache, manifest_rows=rows,
                                        expected_images=2, expected_classes=2)
            self.assertEqual(extracted, cache)
            self.assertTrue(verify_extracted_cache(
                archive, cache, manifest_rows=rows, expected_images=2, expected_classes=2
            ))
            self.assertEqual(extract_archive(
                archive, cache, manifest_rows=rows, expected_images=2, expected_classes=2
            ), cache)
            archive.write_bytes(archive.read_bytes() + b"changed")
            self.assertFalse(verify_extracted_cache(
                archive, cache, manifest_rows=rows, expected_images=2, expected_classes=2
            ))

    def test_file_loader_preserves_labels_and_shape(self):
        try:
            import tensorflow  # noqa: F401
        except ImportError:
            self.skipTest("TensorFlow is optional")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "dataset.zip"
            cache = root / "extracted"
            _archive(archive)
            rows = tuple(
                {"archive_path": f"dataset/{label}/image.jpg", "label": label,
                 "class_index": str(index), "split": "validation"}
                for index, label in enumerate(("Class_A", "Class_B"))
            )
            extract_archive(archive, cache, manifest_rows=rows,
                            expected_images=2, expected_classes=2)
            images, labels = next(iter(make_file_tf_dataset(
                str(cache), rows, batch_size=2, shuffle=False
            )))
            self.assertEqual(tuple(images.shape), (2, 224, 224, 3))
            self.assertEqual(labels.numpy().tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
