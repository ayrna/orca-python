"""Utility definitions for testing."""

from pathlib import Path

# Base path to the project root (assuming this file is at orca_python/testing/)
TEST_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Paths to test datasets and predictions
TEST_DATASETS_DIR = TEST_PROJECT_ROOT / "orca_python" / "testing" / "datasets"
TEST_PREDICTIONS_DIR = TEST_PROJECT_ROOT / "orca_python" / "testing" / "predictions"

# Constant random state for reproducibility
TEST_RANDOM_STATE = 0
