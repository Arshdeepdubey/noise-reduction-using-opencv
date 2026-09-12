"""Ensures the project root is importable as `noise_reduction` when
running `pytest` from anywhere inside the repository."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
