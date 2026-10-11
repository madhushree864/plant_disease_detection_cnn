"""Reusable, opt-in model training orchestration."""

from .config import TrainingConfig, configure_cpu
from .orchestrator import load_manifest, train_model

__all__ = ["TrainingConfig", "configure_cpu", "load_manifest", "train_model"]
