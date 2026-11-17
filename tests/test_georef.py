"""Georeferencing tests: DEM reprojection, cell size and emission point location."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile
from shapely.geometry import Point

import lava_flow_simulation_palma as lf

EMISSION_LAYER = "emision_pto"


def _emission_xy(gpkg_path, crs):
    gdf = gpd.read_file(gpkg_path, layer=EMISSION_LAYER).to_crs(crs)
    geom = gdf.geometry.iloc[0]
    return float(geom.x), float(geom.y)


def _sample_dem(dem: "lf.DEM", x: float, y: float) -> float:
    """Sample a DEM object with rasterio.sample by writing it to an in-memory GeoTIFF."""
    nrows, ncols = dem.elevation.shape
    with MemoryFile() as mem:
        with mem.open(
            driver="GTiff",
            height=nrows,
            width=ncols,
            count=1,
            dtype="float64",
            crs=dem.crs,
            transform=dem.transform,
            nodata=np.nan,
        ) as ds:
            ds.write(dem.elevation, 1)
        with mem.open() as ds:
            return float(next(ds.sample([(x, y)]))[0])


@pytest.fixture(scope="module")
def native_dem(dem_path):
    return lf.load_dem(dem_path, target_crs=None)


@pytest.fixture(scope="module")
def utm_dem(dem_path):
    return lf.load_dem(dem_path)  # default target CRS: UTM 28N


def test_native_grid_vent_matches_rasterio_index(native_dem, dem_path, gpkg_path):
    """Without reprojection, the vent must be exactly src.index(x, y) = (942, 593)."""
    vent = lf.load_emission_point(gpkg_path, native_dem, layer=EMISSION_LAYER)
    with rasterio.open(dem_path) as src:
        x, y = _emission_xy(gpkg_path, src.crs)
        expected_rc = src.index(x, y)
        sampled = float(next(src.sample([(x, y)]))[0])
    assert (vent.row, vent.col) == tuple(expected_rc) == (942, 593)
    assert native_dem.elevation[vent.row, vent.col] == pytest.approx(sampled)
    assert sampled == pytest.approx(1261.0)


def test_default_dem_is_utm28n(utm_dem):
    assert utm_dem.crs.to_epsg() == 32628


def test_cell_size_is_read_from_transform(utm_dem):
    dx, dy = abs(utm_dem.transform.a), abs(utm_dem.transform.e)
    assert dx == pytest.approx(dy)
    assert utm_dem.cell_size_m == pytest.approx(dx)
    # Native pixels are ~31 x 35 m in Web Mercator, i.e. ~27 x 31 m on the ground at 28.6 N.
    assert 20.0 < utm_dem.cell_size_m < 40.0


def test_reprojected_vent_is_inside_valid_and_matches_sample(utm_dem, gpkg_path):
    vent = lf.load_emission_point(gpkg_path, utm_dem, layer=EMISSION_LAYER)
    nrows, ncols = utm_dem.elevation.shape
    assert 0 <= vent.row < nrows and 0 <= vent.col < ncols
    z = utm_dem.elevation[vent.row, vent.col]
    assert np.isfinite(z) and z > 0.0

    x, y = _emission_xy(gpkg_path, utm_dem.crs)
    assert (vent.x, vent.y) == pytest.approx((x, y))
    # Elevation at the vent cell equals what rasterio.sample returns at the vent point.
    assert z == pytest.approx(_sample_dem(utm_dem, x, y))
    # ... and stays close to the native-grid value (bilinear resampling on a steep cone).
    assert abs(z - 1261.0) < 50.0


def test_vent_outside_dem_raises(utm_dem, tmp_path):
    gpkg = tmp_path / "far_away.gpkg"
    gpd.GeoDataFrame(geometry=[Point(-3.70, 40.42)], crs="EPSG:4326").to_file(
        gpkg, layer=EMISSION_LAYER
    )  # Madrid
    with pytest.raises(ValueError, match="outside"):
        lf.load_emission_point(gpkg, utm_dem, layer=EMISSION_LAYER)


def test_vent_on_sea_raises(utm_dem, tmp_path):
    gpkg = tmp_path / "sea.gpkg"
    gpd.GeoDataFrame(geometry=[Point(-17.99, 28.62)], crs="EPSG:4326").to_file(
        gpkg, layer=EMISSION_LAYER
    )  # open sea west of La Palma, inside the DEM extent
    with pytest.raises(ValueError, match="nodata|sea"):
        lf.load_emission_point(gpkg, utm_dem, layer=EMISSION_LAYER)


def test_cell_size_override_warns(dem_path):
    with pytest.warns(UserWarning, match="cell size"):
        dem = lf.load_dem(dem_path, cell_size_override_m=10.0)
    assert dem.cell_size_m == 10.0
