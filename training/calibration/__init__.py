"""Offline calibration training: features -> isotonic fit -> JSON tables.

The runtime never trains anything; it only loads the exported ``calib-v1``
JSON through the Phase-3 model registry. When the table is missing the
runtime stays safe and reports ``calibration_version="uncalibrated"``.
Deterministic protected-column checks are never probabilistic and are not
touched by calibration at all.
"""
