# tests/conftest.py
import os
import re
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
