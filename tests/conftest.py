"""Shared pytest configuration: make ``src/`` importable and expose data paths."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
DATA_DIR = REPO_ROOT / "data"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA_DIR


@pytest.fixture(scope="session")
def dem_path(data_dir: Path) -> Path:
    return data_dir / "DEM_LaPalma.tif"


@pytest.fixture(scope="session")
def gpkg_path(data_dir: Path) -> Path:
    return data_dir / "Emission_Pt.gpkg"
