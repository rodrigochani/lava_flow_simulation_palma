#!/usr/bin/env python3
"""La Palma lava flow simulation (English-only outputs).

All generated artifacts are written to ./output (or --output-dir).

Expected repo layout:
  data/      input DEMs and optional emission point
  output/    generated frames, GIFs, plots
  src/       code (this file)
  notebooks/ interactive notebook
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Headless-safe by default (prevents ICE/X11 errors). Use --show for interactive windows.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    import imageio  # type: ignore

try:
    import rasterio
except Exception:  # pragma: no cover
    rasterio = None  # type: ignore

try:
    import geopandas as gpd
except Exception:  # pragma: no cover
    gpd = None  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def format_volume(v_m3: float) -> str:
    return f"{v_m3:,.1f} m³"


def gaussian_dem(n: int = 100, height: float = 100.0, sigma: float = 15.0) -> np.ndarray:
    x = np.arange(n, dtype=float)
    y = np.arange(n, dtype=float)
    X, Y = np.meshgrid(x, y, indexing="xy")
    cx, cy = n / 2.0, n / 2.0
    dem = height * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sigma**2))
    dem[dem < 0] = 0
    return dem.astype(np.float32)


def dem_stats(dem: np.ndarray) -> Dict[str, float]:
    return {
        "min": float(np.nanmin(dem)),
        "max": float(np.nanmax(dem)),
        "mean": float(np.nanmean(dem)),
        "std": float(np.nanstd(dem)),
    }


# --------------------------------------------------------------------------- georeferencing
DEFAULT_TARGET_CRS = "EPSG:32628"  # WGS 84 / UTM zone 28N (metric on the ground at La Palma)
DEFAULT_EMISSION_LAYER = "emision_pto"
# Mercator variants: map units are metres, but ground distances are scaled by 1/cos(lat).
_NON_METRIC_PROJECTED_EPSG = {3857, 3395, 900913, 3785, 102100}


@dataclass
class DEM:
    """Elevation grid (float64, NaN = nodata) plus its georeferencing."""

    elevation: np.ndarray
    transform: Optional["Affine"] = None
    crs: Optional["CRS"] = None
    cell_size_m: Optional[float] = None

    @property
    def shape(self) -> Tuple[int, int]:
        return self.elevation.shape  # type: ignore[return-value]

    @property
    def nodata_mask(self) -> np.ndarray:
        return ~np.isfinite(self.elevation)


@dataclass
class EmissionPoint:
    row: int
    col: int
    x: float
    y: float


def _require_rasterio() -> None:
    if rasterio is None:
        raise RuntimeError("rasterio is required to load GeoTIFF DEMs. Install with: pip install rasterio")


def ground_cell_size(transform: "Affine", crs: Optional["CRS"]) -> Optional[float]:
    """Return the ground cell size in metres, or None if the grid cannot provide one.

    A cell size is only trusted when the CRS is projected, not a Mercator variant (whose
    "metres" are inflated by 1/cos(latitude)), and the pixels are square.
    """
    dx, dy = abs(transform.a), abs(transform.e)
    if crs is None or not crs.is_projected:
        return None
    if crs.to_epsg() in _NON_METRIC_PROJECTED_EPSG:
        return None
    if not np.isclose(dx, dy, rtol=1e-6, atol=0.0):
        return None
    return float(dx)


def load_dem(
    path: Path,
    target_crs: Optional[str] = DEFAULT_TARGET_CRS,
    resolution_m: Optional[float] = None,
    cell_size_override_m: Optional[float] = None,
) -> DEM:
    """Load a GeoTIFF DEM, optionally reprojecting it (bilinear) to ``target_crs``.

    Nodata pixels become NaN. The cell size is read from the (reprojected) transform;
    ``cell_size_override_m`` replaces it and emits a warning, because areas and volumes
    are then inconsistent with the georeferenced grid.
    """
    _require_rasterio()
    from rasterio.crs import CRS
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float64)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        src_crs, src_transform = src.crs, src.transform
        dst_crs = CRS.from_user_input(target_crs) if target_crs is not None else None

        if dst_crs is None or dst_crs == src_crs:
            elevation, transform, crs = data, src_transform, src_crs
        else:
            res = None if resolution_m is None else (resolution_m, resolution_m)
            transform, width, height = calculate_default_transform(
                src_crs, dst_crs, src.width, src.height, *src.bounds, resolution=res
            )
            elevation = np.full((height, width), np.nan, dtype=np.float64)
            reproject(
                source=data,
                destination=elevation,
                src_transform=src_transform,
                src_crs=src_crs,
                src_nodata=np.nan,
                dst_transform=transform,
                dst_crs=dst_crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
            crs = dst_crs

    cell = ground_cell_size(transform, crs)
    if cell_size_override_m is not None:
        warnings.warn(
            f"Overriding the georeferenced cell size ({cell if cell is not None else 'undefined'} m) "
            f"with {cell_size_override_m} m; areas and volumes will not match the DEM grid.",
            UserWarning,
            stacklevel=2,
        )
        cell = float(cell_size_override_m)
    elif cell is None:
        warnings.warn(
            f"DEM in {crs} has no reliable ground cell size (geographic/Mercator CRS or "
            "non-square pixels); reproject it to a metric CRS before simulating.",
            UserWarning,
            stacklevel=2,
        )
    return DEM(elevation=elevation, transform=transform, crs=crs, cell_size_m=cell)


def load_emission_point(
    gpkg_path: Path,
    dem: DEM,
    layer: str = DEFAULT_EMISSION_LAYER,
    sea_level_m: float = 0.0,
) -> EmissionPoint:
    """Read the vent from a GeoPackage layer, reproject it to the DEM CRS and validate it.

    Raises ValueError if the point falls outside the DEM, on nodata, or on the sea.
    """
    if gpd is None:
        raise RuntimeError("geopandas is required to read the emission point. Install with: pip install geopandas")
    from rasterio.transform import rowcol

    gdf = gpd.read_file(gpkg_path, layer=layer)
    if len(gdf) < 1:
        raise ValueError(f"No features found in layer '{layer}' of {gpkg_path}")
    if len(gdf) > 1:
        warnings.warn(f"Layer '{layer}' has {len(gdf)} features; using the first one.", UserWarning, stacklevel=2)
    if gdf.crs is None:
        raise ValueError(f"Layer '{layer}' of {gpkg_path} has no CRS; cannot place the vent on the DEM.")
    geom = gdf.to_crs(dem.crs).geometry.iloc[0]
    x, y = float(geom.x), float(geom.y)

    row, col = rowcol(dem.transform, x, y)
    row, col = int(row), int(col)
    nrows, ncols = dem.shape
    if not (0 <= row < nrows and 0 <= col < ncols):
        raise ValueError(
            f"Emission point ({x:.1f}, {y:.1f}) in {dem.crs} falls outside the DEM "
            f"(pixel {row}, {col}; grid is {nrows} x {ncols})."
        )
    z = dem.elevation[row, col]
    if not np.isfinite(z):
        raise ValueError(f"Emission point ({x:.1f}, {y:.1f}) falls on a nodata pixel ({row}, {col}).")
    if z <= sea_level_m:
        raise ValueError(
            f"Emission point ({x:.1f}, {y:.1f}) falls on the sea: elevation {z:.1f} m <= sea level "
            f"({sea_level_m} m) at pixel ({row}, {col})."
        )
    return EmissionPoint(row=row, col=col, x=x, y=y)


def summit_rc(elevation: np.ndarray) -> Tuple[int, int]:
    """(row, col) of the highest valid cell; used as the vent for synthetic DEMs."""
    r, c = np.unravel_index(np.nanargmax(elevation), elevation.shape)
    return int(r), int(c)


def choose_emission_point(dem: np.ndarray, emission_rc: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
    if emission_rc is None:
        return summit_rc(dem)
    r, c = emission_rc
    if not (0 <= r < dem.shape[0] and 0 <= c < dem.shape[1]):
        raise ValueError(f"Emission point {emission_rc} is outside the grid {dem.shape}.")
    return int(r), int(c)


@dataclass
class SimulationParams:
    cell_size_m: float = 10.0
    emission_rate_m3_per_min: float = 1000.0
    pulse_minutes: int = 60
    pulses: int = 3

    initial_temp_c: float = 1150.0
    ambient_temp_c: float = 25.0
    cooling_constant_per_min: float = 0.03
    solidification_temp_c: float = 900.0

    viscosity_ref_pa_s: float = 1e5
    arrhenius_beta: float = 0.02
    flow_coef: float = 0.45

    lava_epsilon_m: float = 1e-6
    snapshot_every_min: int = 1
    max_minutes_per_pulse: int = 240


def viscosity_from_temp(temp_c: np.ndarray, params: SimulationParams) -> np.ndarray:
    return params.viscosity_ref_pa_s * np.exp(params.arrhenius_beta * (params.initial_temp_c - temp_c))


def newton_cooling(temp_c: np.ndarray, params: SimulationParams, dt_min: float = 1.0) -> np.ndarray:
    Ta = params.ambient_temp_c
    k = params.cooling_constant_per_min
    return Ta + (temp_c - Ta) * np.exp(-k * dt_min)


def moore_neighbors(r: int, c: int, nrows: int, ncols: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < nrows and 0 <= cc < ncols:
                out.append((rr, cc))
    return out


def compute_volumes(lava_m: np.ndarray, solid_m: np.ndarray, cell_area_m2: float) -> Tuple[float, float]:
    lava_v = float(np.nansum(np.clip(lava_m, 0.0, None)) * cell_area_m2)
    solid_v = float(np.nansum(np.clip(solid_m, 0.0, None)) * cell_area_m2)
    return lava_v, solid_v


def simulate_one_minute(
    dem: np.ndarray,
    lava_m: np.ndarray,
    solid_m: np.ndarray,
    temp_c: np.ndarray,
    params: SimulationParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nrows, ncols = dem.shape
    total_height = dem + solid_m + lava_m

    temp_c = newton_cooling(temp_c, params, dt_min=1.0)
    mu = viscosity_from_temp(temp_c, params)
    mu_norm = mu / np.nanmax(mu)
    mobility = 1.0 / (1.0 + mu_norm)

    new_lava = lava_m.copy()

    for r in range(nrows):
        for c in range(ncols):
            if lava_m[r, c] <= params.lava_epsilon_m:
                continue
            h_here = total_height[r, c]
            nbrs = moore_neighbors(r, c, nrows, ncols)
            downhill = []
            for rr, cc in nbrs:
                d = h_here - total_height[rr, cc]
                if d > 0:
                    downhill.append((rr, cc, d))
            if not downhill:
                continue

            flow_amount = params.flow_coef * float(mobility[r, c]) * float(lava_m[r, c])
            flow_amount = float(np.clip(flow_amount, 0.0, float(lava_m[r, c])))

            dsum = sum(d for _, _, d in downhill)
            if dsum <= 0:
                continue

            new_lava[r, c] -= flow_amount
            for rr, cc, d in downhill:
                new_lava[rr, cc] += flow_amount * (d / dsum)

    new_lava = np.clip(new_lava, 0.0, None)

    solidify = (temp_c <= params.solidification_temp_c) & (new_lava > params.lava_epsilon_m)
    solid_m = solid_m + np.where(solidify, new_lava, 0.0)
    new_lava = np.where(solidify, 0.0, new_lava)

    temp_c = np.where(new_lava > params.lava_epsilon_m, temp_c, params.ambient_temp_c)
    return new_lava, solid_m, temp_c


def render_frame(
    dem: np.ndarray,
    lava_m: np.ndarray,
    emission_rc: Tuple[int, int],
    minute: int,
    active_lava_volume_m3: float,
    out_png: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=100)

    dem_masked = np.ma.masked_where(dem <= 0, dem)
    ax.imshow(dem_masked, cmap="terrain", origin="upper")

    lava_masked = np.ma.masked_where(lava_m <= 1e-6, lava_m)
    if np.ma.count(lava_masked) > 0:
        im = ax.imshow(lava_masked, cmap="inferno", origin="upper", alpha=0.85)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Active lava thickness (m)")

    er, ec = emission_rc
    ax.scatter([ec], [er], s=40, marker="o", label="Emission point")
    ax.legend(loc="upper right", fontsize=8)

    ax.set_title(f"Time: {minute} min\nActive lava volume: {active_lava_volume_m3:,.1f} m³", fontsize=10)
    ax.set_xlabel("Column (px)")
    ax.set_ylabel("Row (px)")
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def build_gif(frame_paths: List[Path], out_gif: Path, fps: int = 8) -> None:
    imgs = [imageio.imread(p) for p in frame_paths]
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_gif, imgs, fps=fps)


def run_simulation(
    dem: np.ndarray,
    params: SimulationParams,
    output_dir: Path,
    emission_rc: Optional[Tuple[int, int]] = None,
    make_gif: bool = True,
) -> Dict[str, object]:
    output_dir = ensure_dir(output_dir)

    dem = dem.astype(np.float32)
    emission_rc = choose_emission_point(dem, emission_rc)

    cell_area = params.cell_size_m * params.cell_size_m
    lava_m = np.zeros_like(dem, dtype=np.float32)
    solid_m = np.zeros_like(dem, dtype=np.float32)
    temp_c = np.full_like(dem, params.ambient_temp_c, dtype=np.float32)

    for pulse in range(1, params.pulses + 1):
        print(f"\n=== Pulse {pulse} ===")
        frames_dir = ensure_dir(output_dir / f"pulse_{pulse}" / "frames")
        frame_paths: List[Path] = []

        pulse_emitted_m3 = 0.0

        # Emission phase
        for minute in range(1, params.pulse_minutes + 1):
            dV = params.emission_rate_m3_per_min
            dh = dV / cell_area
            er, ec = emission_rc
            lava_m[er, ec] += dh
            temp_c[er, ec] = params.initial_temp_c
            pulse_emitted_m3 += dV

            lava_m, solid_m, temp_c = simulate_one_minute(dem, lava_m, solid_m, temp_c, params)

            lava_v, solid_v = compute_volumes(lava_m, solid_m, cell_area)
            total_v = lava_v + solid_v
            diff = pulse_emitted_m3 - total_v

            if minute % 10 == 0:
                print(
                    f"Pulse {pulse} - minute {minute:3d} | "
                    f"emitted={format_volume(pulse_emitted_m3)} | "
                    f"active={format_volume(lava_v)} | "
                    f"solid={format_volume(solid_v)} | "
                    f"total={format_volume(total_v)} | "
                    f"diff(emitted-total)={diff:,.1f} m³"
                )

            if params.snapshot_every_min > 0 and (minute % params.snapshot_every_min == 0):
                out_png = frames_dir / f"frame_{minute:04d}.png"
                render_frame(dem, lava_m, emission_rc, minute, lava_v, out_png)
                frame_paths.append(out_png)

        # Relaxation phase (no emission)
        for extra in range(1, max(1, params.max_minutes_per_pulse - params.pulse_minutes) + 1):
            minute = params.pulse_minutes + extra
            lava_m, solid_m, temp_c = simulate_one_minute(dem, lava_m, solid_m, temp_c, params)

            lava_v, solid_v = compute_volumes(lava_m, solid_m, cell_area)
            total_v = lava_v + solid_v
            diff = pulse_emitted_m3 - total_v

            if minute % 10 == 0:
                print(
                    f"Pulse {pulse} - minute {minute:3d} | "
                    f"emitted={format_volume(pulse_emitted_m3)} | "
                    f"active={format_volume(lava_v)} | "
                    f"solid={format_volume(solid_v)} | "
                    f"total={format_volume(total_v)} | "
                    f"diff(emitted-total)={diff:,.1f} m³"
                )

            if params.snapshot_every_min > 0 and (minute % params.snapshot_every_min == 0):
                out_png = frames_dir / f"frame_{minute:04d}.png"
                render_frame(dem, lava_m, emission_rc, minute, lava_v, out_png)
                frame_paths.append(out_png)

            if lava_v <= 1e-6:
                print(f"Pulse {pulse} stopped at minute {minute}: no active lava remains.")
                break

        gif_path = output_dir / f"pulse_{pulse}_animation.gif"
        if make_gif and frame_paths:
            build_gif(frame_paths, gif_path, fps=8)
            print(f'Pulse {pulse} animation saved as "{gif_path.name}"')

        # Add solidified lava to DEM for the next pulse, then reset layers
        dem = dem + solid_m
        lava_m[:] = 0.0
        solid_m[:] = 0.0
        temp_c[:] = params.ambient_temp_c

        stats = dem_stats(dem)
        print(f"--- DEM statistics after pulse {pulse} ---")
        print(f"Min: {stats['min']:.2f} m")
        print(f"Max: {stats['max']:.2f} m")
        print(f"Mean: {stats['mean']:.2f} m")
        print(f"Std dev: {stats['std']:.2f} m")

    return {"output_dir": str(output_dir), "emission_rc": emission_rc}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="La Palma lava flow simulation (English-only outputs).")
    p.add_argument("--mode", choices=["gaussian", "real"], default="real")
    p.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    p.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))

    # Your renamed files in data/
    p.add_argument("--dem-path", type=str, default="DEM_LaPalma.tif")
    p.add_argument("--emission-gpkg", type=str, default="Emission_Pt.gpkg")

    p.add_argument("--pulses", type=int, default=3)
    p.add_argument("--pulse-minutes", type=int, default=60)
    p.add_argument("--rate", type=float, default=1000.0)
    p.add_argument("--emission-layer", type=str, default=DEFAULT_EMISSION_LAYER)
    p.add_argument("--target-crs", type=str, default=DEFAULT_TARGET_CRS,
                   help="CRS the real DEM is reprojected to (bilinear). Default: UTM 28N.")
    p.add_argument("--resolution", type=float, default=None,
                   help="Target resolution (m) of the reprojected DEM. Default: chosen by GDAL.")
    p.add_argument("--cell-size", type=float, default=None,
                   help="Real mode: override the cell size read from the DEM (emits a warning). "
                        "Gaussian mode: cell size of the synthetic grid (default 10 m).")
    p.add_argument("--snapshot-every", type=int, default=1)
    p.add_argument("--max-minutes-per-pulse", type=int, default=240)
    p.add_argument("--no-gif", action="store_true")

    # Gaussian options
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--height", type=float, default=100.0)
    p.add_argument("--sigma", type=float, default=15.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = ensure_dir(Path(args.output_dir))

    if args.mode == "gaussian":
        elevation = gaussian_dem(n=args.n, height=args.height, sigma=args.sigma)
        cell_size = float(args.cell_size) if args.cell_size is not None else 10.0
        dem = DEM(elevation=elevation.astype(np.float64), cell_size_m=cell_size)
        emission_rc = summit_rc(dem.elevation)
    else:
        dem_path = Path(args.dem_path)
        if not dem_path.is_absolute():
            dem_path = data_dir / dem_path
        gpkg = Path(args.emission_gpkg)
        if not gpkg.is_absolute():
            gpkg = data_dir / gpkg
        dem = load_dem(dem_path, target_crs=args.target_crs, resolution_m=args.resolution,
                       cell_size_override_m=args.cell_size)
        if dem.cell_size_m is None:
            raise SystemExit("The DEM has no reliable ground cell size; use a metric --target-crs.")
        vent = load_emission_point(gpkg, dem, layer=args.emission_layer)
        emission_rc = (vent.row, vent.col)
        print(f"DEM: {dem.shape[0]} x {dem.shape[1]} cells, {dem.cell_size_m:.2f} m, CRS {dem.crs}")
        print(f"Vent: pixel ({vent.row}, {vent.col}), x={vent.x:.1f}, y={vent.y:.1f}, "
              f"z={dem.elevation[vent.row, vent.col]:.1f} m")

    params = SimulationParams(
        cell_size_m=float(dem.cell_size_m),
        emission_rate_m3_per_min=float(args.rate),
        pulse_minutes=int(args.pulse_minutes),
        pulses=int(args.pulses),
        snapshot_every_min=int(args.snapshot_every),
        max_minutes_per_pulse=int(args.max_minutes_per_pulse),
    )

    run_simulation(
        dem=dem.elevation,
        params=params,
        output_dir=out_dir,
        emission_rc=emission_rc,
        make_gif=(not args.no_gif),
    )

    print("\nDone.")
    print(f"Outputs written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
