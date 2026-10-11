import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from src.profiling import benchmark_file_pipeline


class _FakeModel:
    def compile(self, **_kwargs):
        pass

    def train_on_batch(self, _images, _labels):
        return 0.0


class ProfilingTests(unittest.TestCase):
    def test_benchmark_rejects_unbounded_batch_counts(self):
        with self.assertRaises(ValueError):
            benchmark_file_pipeline(".", (), batches=19)

    def test_benchmark_returns_all_stage_measurements(self):
        try:
            import tensorflow  # noqa: F401
        except ImportError:
            self.skipTest("TensorFlow is optional")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index in range(320):
                payload = io.BytesIO()
                Image.new("RGB", (8, 8), (index % 255, 0, 0)).save(payload, format="JPEG")
                relative = f"image_{index}.jpg"
                (root / relative).write_bytes(payload.getvalue())
                rows.append({"archive_path": relative, "class_index": "0", "label": "A",
                             "split": "train"})
            with patch("src.profiling.configure_cpu"), patch(
                "src.profiling.build_custom_cnn", return_value=_FakeModel()
            ):
                result = benchmark_file_pipeline(root, rows, batches=20, batch_size=16)
            self.assertEqual(result["samples"], 320)
            self.assertEqual(result["image_size"], [224, 224])
            self.assertIn(result["dominant_stage"], {
                "preprocessing", "dataset_delivery", "train_on_batch"
            })
            self.assertGreaterEqual(result["dataset_delivery_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
