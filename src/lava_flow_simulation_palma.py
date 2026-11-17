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


@dataclass
class SimulationParams:
    """Model parameters. Values are illustrative, NOT calibrated against observations."""

    emission_rate_m3_per_min: float = 1000.0
    pulse_minutes: int = 60
    pulses: int = 3
    max_minutes_per_pulse: int = 240

    initial_temp_c: float = 1150.0
    ambient_temp_c: float = 25.0
    cooling_constant_per_min: float = 0.03
    solidification_temp_c: float = 900.0

    viscosity_ref_pa_s: float = 1e5
    arrhenius_beta: float = 0.02
    flow_coef: float = 0.45

    sea_level_m: float = 0.0
    lava_epsilon_m: float = 1e-6
    snapshot_every_min: int = 1


def viscosity_from_temp(temp_c: np.ndarray, params: SimulationParams) -> np.ndarray:
    return params.viscosity_ref_pa_s * np.exp(params.arrhenius_beta * (params.initial_temp_c - temp_c))


def mobility(temp_c: np.ndarray, params: SimulationParams) -> np.ndarray:
    """Dimensionless, local mobility ``mu(T0) / mu(T)`` in (0, 1].

    Criterion: lava erupted at T0 has mobility 1 and mobility decreases with the
    Arrhenius-type viscosity increase as the cell cools. The value depends only on the
    cell's own temperature, not on the rest of the grid (the previous normalisation by
    the global viscosity maximum made the vent's mobility depend on the coldest cell on
    the map). Values are clipped to [0, 1] for temperatures above T0.
    """
    ratio = viscosity_from_temp(params.initial_temp_c, params) / viscosity_from_temp(temp_c, params)
    return np.clip(ratio, 0.0, 1.0)


def newton_cooling(temp_c: np.ndarray, params: SimulationParams, dt_min: float = 1.0) -> np.ndarray:
    Ta = params.ambient_temp_c
    k = params.cooling_constant_per_min
    return Ta + (temp_c - Ta) * np.exp(-k * dt_min)


# Cell kinds on the padded grid.
LAND, SEA, BOUNDARY = 0, 1, 2
# Moore neighbourhood: (dr, dc, distance in cell units).
_NEIGHBOURS = [
    (dr, dc, float(np.hypot(dr, dc))) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)
]


class LavaFlowModel:
    """Cellular-automaton lava flow on a regular grid with a 1-minute time step.

    State arrays are stored with a one-cell ghost ring around the grid. Ghost cells and
    sea/nodata cells are sinks: lava that reaches them leaves the domain and is
    accounted for as ``lost_boundary_m3`` / ``lost_sea_m3`` in the mass balance.
    """

    def __init__(
        self,
        elevation: np.ndarray,
        cell_size_m: float,
        vent_rc: Tuple[int, int],
        params: Optional[SimulationParams] = None,
    ) -> None:
        self.params = params if params is not None else SimulationParams()
        if not (cell_size_m and cell_size_m > 0):
            raise ValueError(f"cell_size_m must be positive, got {cell_size_m!r}")
        elev = np.asarray(elevation, dtype=np.float64)
        self.shape: Tuple[int, int] = elev.shape  # type: ignore[assignment]
        self.cell_size_m = float(cell_size_m)
        self.cell_area_m2 = self.cell_size_m**2

        sea_level = self.params.sea_level_m
        sea = ~np.isfinite(elev) | (elev <= sea_level)
        r, c = int(vent_rc[0]), int(vent_rc[1])
        if not (0 <= r < self.shape[0] and 0 <= c < self.shape[1]):
            raise ValueError(f"Vent {vent_rc} is outside the grid {self.shape}.")
        if sea[r, c]:
            raise ValueError(f"Vent {vent_rc} is on a sea/nodata cell.")
        self.vent_rc = (r, c)
        self._vent_p = (r + 1, c + 1)

        ground = np.where(np.isfinite(elev), elev, sea_level)
        self._ground = np.pad(ground, 1, mode="edge")  # ghost ring: zero-gradient ground
        self._kind = np.full(self._ground.shape, BOUNDARY, dtype=np.int8)
        self._kind[1:-1, 1:-1] = np.where(sea, SEA, LAND)

        self._h = np.zeros(self._ground.shape)  # active lava thickness (m)
        self._solid = np.zeros(self._ground.shape)  # solidified lava thickness (m)
        self._heat = np.zeros(self._ground.shape)  # heat content h*T (m * degC)

        self.minute = 0
        self.emitted_m3 = 0.0
        self.lost_sea_m3 = 0.0
        self.lost_boundary_m3 = 0.0

    # ------------------------------------------------------------------ views
    @property
    def lava(self) -> np.ndarray:
        return self._h[1:-1, 1:-1]

    @property
    def solid(self) -> np.ndarray:
        return self._solid[1:-1, 1:-1]

    @property
    def temperature(self) -> np.ndarray:
        """Lava temperature (degC); cells without lava are at ambient temperature."""
        return self._temperature()[1:-1, 1:-1]

    def _temperature(self) -> np.ndarray:
        h = self._h
        out = np.full(h.shape, self.params.ambient_temp_c)
        np.divide(self._heat, h, out=out, where=h > 0)
        return out

    @property
    def sink_mask(self) -> np.ndarray:
        return self._kind[1:-1, 1:-1] == SEA

    def footprint(self, eps: Optional[float] = None) -> np.ndarray:
        eps = self.params.lava_epsilon_m if eps is None else eps
        return (self.lava > eps) | (self.solid > eps)

    def set_state(self, lava: np.ndarray, temperature: np.ndarray, solid: Optional[np.ndarray] = None) -> None:
        """Overwrite the state (used by tests). Lava already present is not counted as emitted."""
        self._h[1:-1, 1:-1] = lava
        self._heat[1:-1, 1:-1] = np.where(lava > 0, lava * temperature, 0.0)
        if solid is not None:
            self._solid[1:-1, 1:-1] = solid
        self.emitted_m3 += float(np.sum(lava) + (0.0 if solid is None else np.sum(solid))) * self.cell_area_m2

    # ----------------------------------------------------------- bookkeeping
    def mass_balance(self) -> Dict[str, float]:
        A = self.cell_area_m2
        active = float(np.sum(self._h)) * A
        solidified = float(np.sum(self._solid)) * A
        accounted = active + solidified + self.lost_sea_m3 + self.lost_boundary_m3
        rel = abs(self.emitted_m3 - accounted) / self.emitted_m3 if self.emitted_m3 > 0 else 0.0
        return {
            "minute": float(self.minute),
            "emitted_m3": self.emitted_m3,
            "active_m3": active,
            "solidified_m3": solidified,
            "lost_sea_m3": self.lost_sea_m3,
            "lost_boundary_m3": self.lost_boundary_m3,
            "relative_error": rel,
        }

    # -------------------------------------------------------------- dynamics
    def step(self, emission_rate_m3_per_min: float, dt_min: float = 1.0) -> None:
        """Advance one time step: emit, cool, flow (advecting heat), drain sinks, solidify."""
        p = self.params
        A = self.cell_area_m2
        h, heat = self._h, self._heat

        # 1. Emission at the vent, at the eruption temperature.
        if emission_rate_m3_per_min > 0:
            vr, vc = self._vent_p
            dh = emission_rate_m3_per_min * dt_min / A
            h[vr, vc] += dh
            heat[vr, vc] += dh * p.initial_temp_c
            self.emitted_m3 += emission_rate_m3_per_min * dt_min

        # 2. Newtonian cooling of cells holding lava (applied to T, stored as h*T).
        wet = h > 0
        heat[wet] = h[wet] * newton_cooling(heat[wet] / h[wet], p, dt_min)
        temp = self._temperature()

        # 3. Mobility.
        mob = mobility(temp, p)

        # 4. Flow towards lower neighbours (synchronous update).
        #    - outflow = flow_coef * mobility * min(h, max_drop / 2): moving more than half
        #      of the largest surface drop would invert the gradient with the lowest
        #      neighbour (overshoot / checkerboard oscillations on gentle slopes);
        #    - it is split among lower neighbours in proportion to the surface slope
        #      (drop / distance; diagonals are sqrt(2) cells away);
        #    - each volume q moved from A to B carries heat q*T_A, so B ends at the
        #      volume-weighted mean temperature of its previous and incoming lava.
        z = self._ground + self._solid + h
        new_h = h.copy()
        new_heat = heat.copy()
        for r, c in np.argwhere(h > p.lava_epsilon_m):
            downhill = []
            for dr, dc, dist in _NEIGHBOURS:
                d = z[r, c] - z[r + dr, c + dc]
                if d > 0:
                    downhill.append((r + dr, c + dc, d, d / dist))
            if not downhill:
                continue
            max_drop = max(d for _, _, d, _ in downhill)
            out = p.flow_coef * float(mob[r, c]) * min(h[r, c], 0.5 * max_drop)
            t_src = temp[r, c]
            ssum = sum(sl for _, _, _, sl in downhill)
            new_h[r, c] -= out
            new_heat[r, c] -= out * t_src
            for rr, cc, _, sl in downhill:
                q = out * (sl / ssum)
                new_h[rr, cc] += q
                new_heat[rr, cc] += q * t_src
        h[:] = new_h
        heat[:] = new_heat

        # 5. Sinks: lava reaching the sea/nodata or the grid boundary leaves the domain.
        for kind, attr in ((SEA, "lost_sea_m3"), (BOUNDARY, "lost_boundary_m3")):
            mask = self._kind == kind
            setattr(self, attr, getattr(self, attr) + float(np.sum(h[mask])) * A)
            h[mask] = 0.0
            heat[mask] = 0.0

        # 6. Solidification of cells that cooled below the threshold.
        temp = self._temperature()
        solidify = (h > 0) & (temp <= p.solidification_temp_c)
        self._solid[solidify] += h[solidify]
        h[solidify] = 0.0
        heat[solidify] = 0.0
        heat[h <= 0] = 0.0
        self.minute += 1


def render_frame(model: LavaFlowModel, base: np.ndarray, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=100)
    ax.imshow(np.ma.masked_where(~np.isfinite(base) | (base <= 0), base), cmap="terrain", origin="upper")
    lava_masked = np.ma.masked_where(model.lava <= model.params.lava_epsilon_m, model.lava)
    if np.ma.count(lava_masked) > 0:
        im = ax.imshow(lava_masked, cmap="inferno", origin="upper", alpha=0.85)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Active lava thickness (m)")
    er, ec = model.vent_rc
    ax.scatter([ec], [er], s=40, marker="o", label="Emission point")
    ax.legend(loc="upper right", fontsize=8)
    active = model.mass_balance()["active_m3"]
    ax.set_title(f"Time: {model.minute} min\nActive lava volume: {active:,.1f} m³", fontsize=10)
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


def _log(record: Dict[str, float], pulse: int) -> None:
    print(
        f"Pulse {pulse} - minute {int(record['minute']):4d} | "
        f"emitted={format_volume(record['emitted_m3'])} | "
        f"active={format_volume(record['active_m3'])} | "
        f"solid={format_volume(record['solidified_m3'])} | "
        f"lost(sea)={format_volume(record['lost_sea_m3'])} | "
        f"lost(boundary)={format_volume(record['lost_boundary_m3'])} | "
        f"rel.err={record['relative_error']:.1e}"
    )


def run_simulation(model: LavaFlowModel, output_dir: Path, make_gif: bool = True) -> List[Dict[str, float]]:
    """Run all pulses. Lava still active at the end of a pulse carries over to the next one.

    Returns one mass-balance record per minute.
    """
    p = model.params
    output_dir = ensure_dir(output_dir)
    base = model._ground[1:-1, 1:-1].copy()
    records: List[Dict[str, float]] = []

    for pulse in range(1, p.pulses + 1):
        print(f"\n=== Pulse {pulse} ===")
        frames_dir = output_dir / f"pulse_{pulse}" / "frames"
        frame_paths: List[Path] = []
        n_minutes = max(p.max_minutes_per_pulse, p.pulse_minutes)
        for minute in range(1, n_minutes + 1):
            erupting = minute <= p.pulse_minutes
            model.step(p.emission_rate_m3_per_min if erupting else 0.0)
            rec = model.mass_balance()
            rec["pulse"] = float(pulse)
            rec["footprint_cells"] = float(model.footprint().sum())
            records.append(rec)
            if minute % 10 == 0:
                _log(rec, pulse)
            if make_gif and p.snapshot_every_min > 0 and minute % p.snapshot_every_min == 0:
                out_png = frames_dir / f"frame_{minute:04d}.png"
                render_frame(model, base, out_png)
                frame_paths.append(out_png)
            if not erupting and rec["active_m3"] <= 0.0:
                print(f"Pulse {pulse} stopped at minute {minute}: no active lava remains.")
                break

        if make_gif and frame_paths:
            gif_path = output_dir / f"pulse_{pulse}_animation.gif"
            build_gif(frame_paths, gif_path, fps=8)
            print(f'Pulse {pulse} animation saved as "{gif_path.name}"')

        stats = dem_stats(np.where(model.sink_mask, np.nan, base + model.solid))
        print(f"--- Topography (land cells, incl. deposits) after pulse {pulse} ---")
        print(f"Min: {stats['min']:.2f} m | Max: {stats['max']:.2f} m | Mean: {stats['mean']:.2f} m | Std: {stats['std']:.2f} m")

    return records


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
        emission_rate_m3_per_min=float(args.rate),
        pulse_minutes=int(args.pulse_minutes),
        pulses=int(args.pulses),
        snapshot_every_min=int(args.snapshot_every),
        max_minutes_per_pulse=int(args.max_minutes_per_pulse),
    )
    model = LavaFlowModel(dem.elevation, dem.cell_size_m, emission_rc, params)
    run_simulation(model, out_dir, make_gif=(not args.no_gif))

    print("\nDone.")
    print(f"Outputs written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
