import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from src.preprocessing import make_tf_dataset, preprocess_image


class PreprocessingTests(unittest.TestCase):
    def test_output_shape_and_normalization(self):
        image = Image.new("L", (8, 4), 128)
        result = preprocess_image(image)
        self.assertEqual(result.shape, (224, 224, 3))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.allclose(result, 128 / 255))

    def test_evaluation_is_deterministic_and_training_can_augment(self):
        image = Image.new("RGB", (4, 2))
        image.putpixel((0, 0), (255, 0, 0))
        evaluation_a = preprocess_image(image, training=False, seed=1)
        evaluation_b = preprocess_image(image, training=False, seed=2)
        self.assertTrue(np.array_equal(evaluation_a, evaluation_b))
        # A seed that produces a flip differs from the corresponding unflipped image.
        train = preprocess_image(image, training=True, seed=2)
        self.assertFalse(np.array_equal(train, evaluation_a))

    @unittest.skipUnless(__import__("importlib").util.find_spec("tensorflow"), "TensorFlow not installed")
    def test_streaming_dataset_shape_labels_and_bounded_shuffle(self):
        import tensorflow as tf

        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "images.zip"
            rows = []
            with zipfile.ZipFile(archive_path, "w") as archive:
                for index, label in enumerate((3, 7, 3)):
                    member = f"class_{label}/image_{index}.png"
                    image = Image.new("RGB", (5, 3), (index * 40, 20, 10))
                    stream = io.BytesIO()
                    image.save(stream, format="PNG")
                    archive.writestr(member, stream.getvalue())
                    rows.append({"archive_path": member, "class_index": str(label), "label": str(label), "split": "train"})
            dataset = make_tf_dataset(
                str(archive_path), rows, training=True, batch_size=2,
                shuffle=True, shuffle_buffer_size=2,
            )
            images, labels = next(iter(dataset))
            self.assertEqual(tuple(images.shape[1:]), (224, 224, 3))
            self.assertEqual(labels.dtype, tf.int32)
            self.assertEqual(tuple(labels.shape), (2,))
            self.assertTrue(set(labels.numpy().tolist()).issubset({3, 7}))
            with self.assertRaises(ValueError):
                make_tf_dataset(str(archive_path), rows, shuffle_buffer_size=0)

    @unittest.skipUnless(__import__("importlib").util.find_spec("tensorflow"), "TensorFlow not installed")
    def test_evaluation_dataset_is_deterministic_and_preserves_split(self):
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "images.zip"
            rows = []
            with zipfile.ZipFile(archive_path, "w") as archive:
                for index in range(2):
                    member = f"validation/image_{index}.png"
                    image = Image.new("RGB", (5, 3), (index * 80, 0, 0))
                    stream = io.BytesIO()
                    image.save(stream, format="PNG")
                    archive.writestr(member, stream.getvalue())
                    rows.append({"archive_path": member, "class_index": str(index), "label": str(index), "split": "validation"})
            kwargs = dict(training=False, batch_size=2, shuffle=False)
            first = next(iter(make_tf_dataset(str(archive_path), rows, **kwargs)))
            second = next(iter(make_tf_dataset(str(archive_path), rows, **kwargs)))
            self.assertTrue(np.array_equal(first[0].numpy(), second[0].numpy()))
            self.assertEqual(first[1].numpy().tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
