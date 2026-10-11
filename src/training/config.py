"""Small, CPU-safe defaults for opt-in training runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 16
    shuffle_buffer_size: int = 512
    learning_rate: float = 1e-3
    output_dir: str = "outputs/training"
    checkpoint_name: str = "best_model.keras"
    seed: int = 42
    extracted_dir: str = "data/extracted"

    @property
    def checkpoint_path(self) -> str:
        return str(Path(self.output_dir) / self.checkpoint_name)


def configure_cpu(threads: int = 1) -> None:
    """Select a conservative TensorFlow CPU configuration (without importing it eagerly)."""
    if threads < 1:
        raise ValueError("threads must be positive")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("TensorFlow is required for training") from exc
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(threads)
