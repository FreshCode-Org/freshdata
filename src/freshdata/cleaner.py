"""The cleaning pipeline and the reusable :class:`Cleaner` front-end."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Mapping

import pandas as pd

from ._util import memory_bytes
from .adapters.polars import is_polars_frame, to_pandas
from .config import CleanConfig, merge_options
from .engine import auto_missing, auto_outliers
from .engine.cache import build_engine_cache
from .report import CleanReport
from .steps.columns import normalize_column_names
from .steps.dtypes import fix_dtypes
from .steps.duplicates import drop_duplicate_rows
from .steps.memory import optimize_memory
from .steps.missing import impute_missing
from .steps.outliers import handle_outliers
from .steps.prune import drop_constant_columns, drop_empty_columns, drop_empty_rows
from .steps.strings import clean_strings

ProgressCallback = Callable[[