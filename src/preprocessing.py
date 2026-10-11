"""CPU-friendly decoding and preprocessing for 224x224 model inputs."""

from __future__ import annotations

import io
from typing import Iterable, Mapping
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageOps

IMAGE_SIZE = (224, 224)


def decode_image_bytes(data: bytes) -> Image.Image:
    """Decode bytes as RGB, closing the underlying input safely."""
    with Image.open(io.BytesIO(data)) as image:
        return image.convert("RGB")


def preprocess_image(
    image: Image.Image | np.ndarray | bytes,
    *,
    training: bool = False,
    seed: int | None = None,
) -> np.ndarray:
    """Resize, optionally augment, and normalize one image to float32 [0, 1].

    Augmentation is limited to a seeded/random horizontal flip during training.
    Evaluation preprocessing is deterministic and never mutates the input.
    """
    if isinstance(image, bytes):
        image = decode_image_bytes(image)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    image = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    if training:
        rng = np.random.default_rng(seed)
        if rng.random() < 0.5:
            image = ImageOps.mirror(image)
    return np.asarray(image, dtype=np.float32) / 255.0


def preprocess_archive_member(
    archive_path: str,
    member_path: str,
    *,
    training: bool = False,
    seed: int | None = None,
) -> np.ndarray:
    """Read exactly one member from a ZIP and return a model-ready array."""
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(member_path) as stream:
            return preprocess_image(stream.read(), training=training, seed=seed)


def make_tf_dataset(
    archive_path: str,
    records: Iterable[Mapping[str, object]],
    *,
    training: bool = False,
    batch_size: int = 32,
    shuffle: bool | None = None,
    shuffle_buffer_size: int = 512,
    seed: int = 42,
):
    """Build a lazy, bounded-memory ``tf.data.Dataset``.

    The manifest rows remain metadata only. Each generator invocation opens the
    archive once (rather than once per image), and TensorFlow keeps at most
    ``shuffle_buffer_size`` decoded examples in its training shuffle buffer.
    """
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("TensorFlow is required for make_tf_dataset") from exc
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if shuffle_buffer_size < 1:
        raise ValueError("shuffle_buffer_size must be positive")
    rows = tuple(records)
    if shuffle is None:
        shuffle = training

    def generator():
        with zipfile.ZipFile(archive_path) as archive:
            for row in rows:
                with archive.open(str(row["archive_path"])) as stream:
                    yield preprocess_image(stream.read(), training=training), int(row["class_index"])

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )
    if shuffle:
        dataset = dataset.shuffle(
            min(shuffle_buffer_size, max(1, len(rows))),
            seed=seed,
            reshuffle_each_iteration=training,
        )
    dataset = dataset.batch(batch_size)
    options = tf.data.Options()
    options.experimental_deterministic = not training
    return dataset.with_options(options).prefetch(tf.data.AUTOTUNE)


def make_file_tf_dataset(
    extracted_dir: str,
    records: Iterable[Mapping[str, object]],
    *,
    training: bool = False,
    batch_size: int = 32,
    shuffle: bool | None = None,
    shuffle_buffer_size: int = 512,
    seed: int = 42,
):
    """Build the same dataset contract from an extracted class-preserving tree."""
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("TensorFlow is required for make_file_tf_dataset") from exc
    if batch_size < 1 or shuffle_buffer_size < 1:
        raise ValueError("batch_size and shuffle_buffer_size must be positive")
    rows = tuple(records)
    if shuffle is None:
        shuffle = training
    root = Path(extracted_dir)

    paths = []
    labels = []
    for row in rows:
        relative = PurePosixPath(str(row["archive_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe extracted member path: {relative}")
        paths.append(str(root.joinpath(*relative.parts)))
        labels.append(int(row["class_index"]))

    # TensorFlow performs file I/O, JPEG decode, and resize in parallel.  This
    # avoids running Python/Pillow in the input pipeline while retaining the
    # manifest's exact ordering and labels.
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def decode_and_resize(path, label):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE, method="bilinear")
        image = tf.cast(image, tf.float32) / 255.0
        if training:
            image = tf.image.random_flip_left_right(image, seed=seed)
        return image, tf.cast(label, tf.int32)

    dataset = dataset.map(decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE,
                          deterministic=not training)
    if shuffle:
        dataset = dataset.shuffle(min(shuffle_buffer_size, max(1, len(rows))), seed=seed,
                                  reshuffle_each_iteration=training)
    options = tf.data.Options()
    options.experimental_deterministic = not training
    return dataset.batch(batch_size).with_options(options).prefetch(tf.data.AUTOTUNE)
