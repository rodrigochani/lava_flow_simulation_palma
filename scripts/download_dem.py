#!/usr/bin/env python3
"""Download Copernicus GLO-30 tiles, mosaic them and crop them to La Palma.

Source: Copernicus DEM GLO-30 Public, as Cloud Optimized GeoTIFFs in the AWS Open Data
bucket ``s3://copernicus-dem-30m`` (https://registry.opendata.aws/copernicus-dem/).

Steps:
1. Download tile N28 W018 over HTTPS. If that fails, fall back to
   ``aws s3 cp --no-sign-request``.
2. If La Palma's land touches the tile's western edge, download tile N28 W019 as well
   and mosaic the two. (La Palma reaches ~18.007 W, so in practice both are needed.)
3. Crop to the island's land extent plus a margin, and check that no land touches the
   edge of the crop, i.e. that the island is complete.
4. Write ``data/DEM_LaPalma_COP30.tif``: float32, EPSG:4326, DEFLATE, with provenance
   tags.

Usage:
    python scripts/download_dem.py [--out data/DEM_LaPalma_COP30.tif] [--cache-dir DIR]
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.windows import Window, from_bounds

BUCKET = "copernicus-dem-30m"
HTTP_ROOT = f"https://{BUCKET}.s3.amazonaws.com"
MAIN_TILE = "Copernicus_DSM_COG_10_N28_00_W018_00_DEM"
WEST_TILE = "Copernicus_DSM_COG_10_N28_00_W019_00_DEM"

# Generous search box around La Palma (lon/lat, EPSG:4326). It excludes the other islands.
SEARCH_BOUNDS = (-18.20, 28.35, -17.60, 28.95)  # west, south, east, north
MARGIN_DEG = 0.02  # ~2 km of sea around the island in the final crop

ATTRIBUTION = (
    "produced using Copernicus WorldDEM-30 (c) DLR e.V. 2010-2014 and (c) Airbus Defence and "
    "Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; "
    "all rights reserved"
)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "DEM_LaPalma_COP30.tif"


def download_tile(name: str, cache_dir: Path) -> Path:
    """Return a local path to ``<name>.tif``, downloading it if needed."""
    dest = cache_dir / f"{name}.tif"
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Using cached {dest}")
        return dest
    url = f"{HTTP_ROOT}/{name}/{name}.tif"
    tmp = dest.with_suffix(".part")
    try:
        print(f"Downloading {url}")
        with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f)
        tmp.replace(dest)
        return dest
    except Exception as exc:  # HTTP failure: try the AWS CLI without credentials
        tmp.unlink(missing_ok=True)
        print(f"HTTP download failed ({exc}); trying the AWS CLI.", file=sys.stderr)
    if shutil.which("aws") is None:
        raise RuntimeError(f"Could not download {name}: HTTP failed and the 'aws' CLI is not installed.")
    s3_uri = f"s3://{BUCKET}/{name}/{name}.tif"
    subprocess.run(["aws", "s3", "cp", "--no-sign-request", s3_uri, str(dest)], check=True)
    return dest


def land_touches_west_edge(tile: Path) -> bool:
    """True if land (> 0 m) lies on the tile's first column within the La Palma latitudes."""
    with rasterio.open(tile) as src:
        _, south, _, north = SEARCH_BOUNDS
        row_top, _ = src.index(src.bounds.left + 1e-9, north)
        row_bot, _ = src.index(src.bounds.left + 1e-9, south)
        col = src.read(1, window=Window(0, row_top, 1, row_bot - row_top + 1))
    return bool(np.any(col > 0))


def mosaic_and_crop(tiles: list[Path]) -> tuple[np.ndarray, "rasterio.Affine", dict]:
    srcs = [rasterio.open(t) for t in tiles]
    try:
        profile = srcs[0].profile.copy()
        # Ocean areas have no tiles and are 0 m in the product, so fill with 0.
        data, transform = merge(srcs, bounds=SEARCH_BOUNDS, nodata=0.0)
    finally:
        for s in srcs:
            s.close()
    data = data[0].astype(np.float32)

    land = data > 0
    if not land.any():
        raise RuntimeError("No land found in the search box; check the tiles.")
    rows, cols = np.nonzero(land)
    # Land must not touch the search box itself, or the island would be cut.
    if rows.min() == 0 or cols.min() == 0 or rows.max() == data.shape[0] - 1 or cols.max() == data.shape[1] - 1:
        raise RuntimeError("Land touches the search box edge: La Palma may be incomplete.")
    res = abs(transform.a)
    pad = int(round(MARGIN_DEG / res))
    r0, r1 = max(rows.min() - pad, 0), min(rows.max() + pad, data.shape[0] - 1)
    c0, c1 = max(cols.min() - pad, 0), min(cols.max() + pad, data.shape[1] - 1)
    crop = data[r0 : r1 + 1, c0 : c1 + 1]
    crop_transform = transform * rasterio.Affine.translation(c0, r0)

    border = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]])
    if np.any(border > 0):
        raise RuntimeError("Land touches the crop border: La Palma is incomplete.")
    return crop, crop_transform, profile


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="Directory where raw tiles are kept (default: a temporary directory).")
    args = p.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        cache = args.cache_dir or Path(tmp)
        cache.mkdir(parents=True, exist_ok=True)
        tiles = [download_tile(MAIN_TILE, cache)]
        if land_touches_west_edge(tiles[0]):
            print("La Palma is cut at 18 W in the main tile; adding the western tile and mosaicking.")
            tiles.append(download_tile(WEST_TILE, cache))
        else:
            print("La Palma lies entirely in the main tile.")
        crop, transform, profile = mosaic_and_crop(tiles)

    profile.update(
        driver="GTiff", dtype="float32", count=1, height=crop.shape[0], width=crop.shape[1],
        transform=transform, crs="EPSG:4326", nodata=None, compress="deflate", predictor=3,
        tiled=True, blockxsize=256, blockysize=256,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(args.out, "w", **profile) as dst:
        dst.write(crop, 1)
        dst.update_tags(
            PRODUCT="Copernicus DEM GLO-30 Public (DSM)",
            SOURCE=f"s3://{BUCKET}/ (https://registry.opendata.aws/copernicus-dem/)",
            TILES=",".join(t.stem for t in tiles),
            PROCESSING="mosaicked, cropped to La Palma + 0.02 deg margin; ocean = 0 m",
            ACCESS_DATE=dt.date.today().isoformat(),
            ATTRIBUTION=ATTRIBUTION,
        )
    size_mb = args.out.stat().st_size / 1e6
    print(f"Wrote {args.out} ({crop.shape[0]} x {crop.shape[1]} px, {size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
