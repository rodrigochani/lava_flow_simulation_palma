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


DEM_FILES = {"copernicus": "DEM_LaPalma_COP30.tif", "legacy": "DEM_LaPalma.tif"}


@pytest.fixture(scope="session", params=sorted(DEM_FILES))
def dem_source(request) -> str:
    """Run DEM-dependent tests on both the Copernicus (default) and the legacy DEM."""
    return request.param


@pytest.fixture(scope="session")
def dem_path(data_dir: Path, dem_source: str) -> Path:
    return data_dir / DEM_FILES[dem_source]


@pytest.fixture(scope="session")
def gpkg_path(data_dir: Path) -> Path:
    return data_dir / "Emission_Pt.gpkg"
