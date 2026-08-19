"""Path configuration for the figure scripts.

Inputs are read from DATA_DIR, figures written to OUT_DIR. Both default to
./data and ./outputs next to these scripts, and can be overridden with
environment variables:

    DATA_DIR=/path/to/data OUT_DIR=/path/to/figures python fig1_S2_S3_correlations.py

Place the required input files (see README) directly in DATA_DIR.
"""
import os

_here = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(_here, "data"))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(_here, "outputs"))
os.makedirs(OUT_DIR, exist_ok=True)
