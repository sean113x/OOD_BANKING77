"""Project entry point."""

from __future__ import annotations

import sys
from pathlib import Path


CLASS_WISE_DIR = Path(__file__).parent / "embedding_class-wise methods"
sys.path.insert(0, str(CLASS_WISE_DIR))

from class_wise_experiment import run_class_wise_experiment


if __name__ == "__main__":
    run_class_wise_experiment()
