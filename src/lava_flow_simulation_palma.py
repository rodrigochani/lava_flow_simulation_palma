#!/usr/bin/env python3
"""Cellular-automaton lava flow simulator inspired by the 2021 La Palma eruption.

Model (one step = 1 minute, Moore neighbourhood): lava is emitted at the vent at T0,
cools by Newtonian cooling, flows towards lower neighbours with a mobility
mu(T0)/mu(T) from an Arrhenius-type viscosity, carries its heat with it, and solidifies
when its temperature drops below a threshold. Sea/nodata cells and the grid edge are
sinks. Parameters are illustrative and uncalibrated; see README.md ("Limitations").

Outputs are written to output/<mode>/ (or --output-dir).
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import asdict, dataclass
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
    cell_size_override_m: Optional[float] = None

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
    return DEM(elevation=elevation, transform=transform, crs=crs, cell_size_m=cell,
               cell_size_override_m=cell_size_override_m)


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
    # Bulk Newtonian cooling rate. 0.003/min means lava erupted at 1150 C reaches the
    # 900 C solidification threshold ~84 min after leaving the vent (if not remixed with
    # hotter lava). Illustrative value, not derived from field data. The former default
    # (0.03/min, ~8 min) combined with mu(T0)/mu(T) mobility froze flows near the vent.
    cooling_constant_per_min: float = 0.003
    solidification_temp_c: float = 900.0

    viscosity_ref_pa_s: float = 1e5
    arrhenius_beta: float = 0.02
    flow_coef: float = 0.45

    sea_level_m: float = 0.0
    lava_epsilon_m: float = 1e-6


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
        self._search: Optional[Tuple[int, int, int, int]] = None  # see _window()
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
        """Overwrite the state (used by tests). The volumes set here are added to
        ``emitted_m3`` so that the mass balance stays closed."""
        self._h[1:-1, 1:-1] = lava
        self._heat[1:-1, 1:-1] = np.where(lava > 0, lava * temperature, 0.0)
        if solid is not None:
            self._solid[1:-1, 1:-1] = solid
        self._search = None  # rescan the full grid on the next step
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
    def _window(self) -> Tuple[slice, slice]:
        """Padded-grid window holding every cell with lava (and the vent) plus a 1-cell margin.

        Lava moves at most one cell per step, so the next search only needs the previous
        window grown by one cell; the full grid is never scanned after the first step.
        """
        nr, nc = self._h.shape
        if self._search is None:
            r0, r1, c0, c1 = 0, nr - 1, 0, nc - 1
        else:
            r0, r1, c0, c1 = self._search
        sub = self._h[r0 : r1 + 1, c0 : c1 + 1] > 0
        rows = np.flatnonzero(sub.any(axis=1))
        cols = np.flatnonzero(sub.any(axis=0))
        vr, vc = self._vent_p
        lo_r = min(vr, r0 + rows[0]) if rows.size else vr
        hi_r = max(vr, r0 + rows[-1]) if rows.size else vr
        lo_c = min(vc, c0 + cols[0]) if cols.size else vc
        hi_c = max(vc, c0 + cols[-1]) if cols.size else vc
        # Lava never rests on ghost cells, so the 1-cell margin stays inside the padded grid.
        lo_r, hi_r, lo_c, hi_c = lo_r - 1, hi_r + 1, lo_c - 1, hi_c + 1
        self._search = (max(lo_r - 1, 0), min(hi_r + 1, nr - 1), max(lo_c - 1, 0), min(hi_c + 1, nc - 1))
        return slice(lo_r, hi_r + 1), slice(lo_c, hi_c + 1)

    def step(self, emission_rate_m3_per_min: float, dt_min: float = 1.0) -> None:
        """Advance one time step: emit, cool, flow (advecting heat), drain sinks, solidify.

        All operations are vectorised over the window around the lava (see ``_window``).
        """
        p = self.params
        A = self.cell_area_m2

        # 1. Emission at the vent, at the eruption temperature.
        if emission_rate_m3_per_min > 0:
            vr, vc = self._vent_p
            dh = emission_rate_m3_per_min * dt_min / A
            self._h[vr, vc] += dh
            self._heat[vr, vc] += dh * p.initial_temp_c
            self.emitted_m3 += emission_rate_m3_per_min * dt_min

        win = self._window()
        h = self._h[win]  # views: in-place updates write to the model state
        heat = self._heat[win]
        kind = self._kind[win]

        # 2. Newtonian cooling of cells holding lava (applied to T, stored as h*T).
        wet = h > 0
        temp = np.full(h.shape, p.ambient_temp_c)
        temp[wet] = newton_cooling(heat[wet] / h[wet], p, dt_min)
        heat[wet] = h[wet] * temp[wet]

        # 3. Mobility (local, see ``mobility``).
        # 4. Flow towards lower neighbours (synchronous update).
        #    - outflow = flow_coef * mobility * min(h, max_drop / 2): moving more than half
        #      of the largest surface drop would invert the gradient with the lowest
        #      neighbour (overshoot / checkerboard oscillations on gentle slopes);
        #    - it is split among lower neighbours in proportion to the surface slope
        #      (drop / distance; diagonals are sqrt(2) cells away);
        #    - each volume q moved from A to B carries heat q*T_A, so B ends at the
        #      volume-weighted mean temperature of its previous and incoming lava.
        #    Sources are the window's inner cells; the outer ring only receives.
        z = self._ground[win] + self._solid[win] + h
        zi, hi, ti = z[1:-1, 1:-1], h[1:-1, 1:-1], temp[1:-1, 1:-1]
        source = hi > p.lava_epsilon_m
        nr, nc = z.shape
        drops = np.empty((len(_NEIGHBOURS),) + zi.shape)
        slopes = np.empty_like(drops)
        for k, (dr, dc, dist) in enumerate(_NEIGHBOURS):
            d = zi - z[1 + dr : nr - 1 + dr, 1 + dc : nc - 1 + dc]
            d = np.where(source & (d > 0), d, 0.0)
            drops[k] = d
            slopes[k] = d / dist
        ssum = slopes.sum(axis=0)
        moving = ssum > 0
        out = np.where(
            moving, p.flow_coef * mobility(ti, p) * np.minimum(hi, 0.5 * drops.max(axis=0)), 0.0
        )
        share = out / np.where(moving, ssum, 1.0)
        new_h = h.copy()
        new_heat = heat.copy()
        new_h[1:-1, 1:-1] -= out
        new_heat[1:-1, 1:-1] -= out * ti
        for k, (dr, dc, _) in enumerate(_NEIGHBOURS):
            q = share * slopes[k]
            new_h[1 + dr : nr - 1 + dr, 1 + dc : nc - 1 + dc] += q
            new_heat[1 + dr : nr - 1 + dr, 1 + dc : nc - 1 + dc] += q * ti
        h[:] = new_h
        heat[:] = new_heat

        # 5. Sinks: lava reaching the sea/nodata or the grid boundary leaves the domain.
        for sink_kind, attr in ((SEA, "lost_sea_m3"), (BOUNDARY, "lost_boundary_m3")):
            mask = kind == sink_kind
            if mask.any():
                setattr(self, attr, getattr(self, attr) + float(np.sum(h[mask])) * A)
                h[mask] = 0.0
                heat[mask] = 0.0

        # 6. Solidification of cells that cooled below the threshold.
        wet = h > 0
        temp = np.full(h.shape, p.ambient_temp_c)
        temp[wet] = heat[wet] / h[wet]
        solidify = wet & (temp <= p.solidification_temp_c)
        self._solid[win][solidify] += h[solidify]
        h[solidify] = 0.0
        heat[~(h > 0)] = 0.0
        self.minute += 1


# ------------------------------------------------------------------------------- outputs
# Chart ink and categorical colours (fixed order, never cycled).
_INK, _INK_2, _GRID, _SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
_SERIES = {"active_m3": "#2a78d6", "solidified_m3": "#eb6834", "lost_sea_m3": "#1baf7a",
           "lost_boundary_m3": "#eda100"}
_SERIES_LABELS = {"active_m3": "Active (molten)", "solidified_m3": "Solidified",
                  "lost_sea_m3": "Lost to sea/nodata", "lost_boundary_m3": "Lost through grid edge"}
_SEA_COLOR = "#cde2fb"


@dataclass
class Snapshot:
    minute: int
    row0: int
    col0: int
    thickness: np.ndarray  # lava + solid (m) inside the window


def _style_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_INK_2)
    ax.tick_params(colors=_INK_2, labelsize=8)
    ax.xaxis.label.set_color(_INK_2)
    ax.yaxis.label.set_color(_INK_2)


def _grid_transform(dem: DEM) -> "Affine":
    """Georeferenced transform, or a local north-up one (origin at the lower-left corner)."""
    from affine import Affine

    if dem.transform is not None and dem.cell_size_m is not None and dem.cell_size_override_m is None:
        return dem.transform
    cell = float(dem.cell_size_m)
    return Affine(cell, 0.0, 0.0, 0.0, -cell, dem.shape[0] * cell)


def footprint_metrics(model: LavaFlowModel) -> Dict[str, float]:
    fp = model.footprint()
    rows, cols = np.nonzero(fp)
    thick = model.lava + model.solid
    if rows.size == 0:
        return {"footprint_cells": 0.0, "footprint_area_km2": 0.0, "runout_m": 0.0, "max_thickness_m": 0.0}
    vr, vc = model.vent_rc
    runout = float(np.max(np.hypot(rows - vr, cols - vc))) * model.cell_size_m
    return {
        "footprint_cells": float(rows.size),
        "footprint_area_km2": rows.size * model.cell_area_m2 / 1e6,
        "runout_m": runout,  # straight-line distance vent -> farthest footprint cell centre
        "max_thickness_m": float(thick.max()),
    }


def _crop_box(model: LavaFlowModel, margin_frac: float = 0.35, min_margin: int = 15) -> Tuple[int, int, int, int]:
    fp = model.footprint()
    rows, cols = np.nonzero(fp)
    vr, vc = model.vent_rc
    r0, r1 = min(rows.min(initial=vr), vr), max(rows.max(initial=vr), vr)
    c0, c1 = min(cols.min(initial=vc), vc), max(cols.max(initial=vc), vc)
    m = max(min_margin, int(margin_frac * max(r1 - r0, c1 - c0)))
    nr, nc = model.shape
    return max(r0 - m, 0), min(r1 + m, nr - 1), max(c0 - m, 0), min(c1 + m, nc - 1)


def _plot_map(ax, model: LavaFlowModel, dem: DEM, box, thickness: np.ndarray, label: str, vmax: float):
    from matplotlib.colors import LightSource, ListedColormap

    r0, r1, c0, c1 = box
    tr = _grid_transform(dem)
    x0, y0 = tr * (c0, r0)
    x1, y1 = tr * (c1 + 1, r1 + 1)
    extent = (x0 / 1e3, x1 / 1e3, y1 / 1e3, y0 / 1e3)
    ground = np.where(np.isfinite(dem.elevation), dem.elevation, model.params.sea_level_m)
    win = (slice(r0, r1 + 1), slice(c0, c1 + 1))
    shade = LightSource(azdeg=315, altdeg=45).hillshade(
        ground[win], vert_exag=1.0, dx=model.cell_size_m, dy=model.cell_size_m
    )
    ax.imshow(shade, cmap="gray", vmin=0, vmax=1, extent=extent, interpolation="nearest")
    sea = model.sink_mask[win]
    if sea.any():
        ax.imshow(np.ma.masked_where(~sea, sea), cmap=ListedColormap([_SEA_COLOR]), extent=extent,
                  interpolation="nearest")
    thick = np.ma.masked_where(thickness[win] <= model.params.lava_epsilon_m, thickness[win])
    reds = plt.get_cmap("Reds")
    cmap = ListedColormap(reds(np.linspace(0.3, 1.0, 256)))
    im = ax.imshow(thick, cmap=cmap, vmin=0.0, vmax=vmax, extent=extent, interpolation="nearest", alpha=0.9)
    vx, vy = tr * (model.vent_rc[1] + 0.5, model.vent_rc[0] + 0.5)
    ax.scatter([vx / 1e3], [vy / 1e3], s=60, marker="^", facecolor=_INK, edgecolor=_SURFACE, linewidth=1.5,
               zorder=5, label="Vent")
    ax.legend(loc="upper right", fontsize=8, frameon=True, facecolor=_SURFACE, edgecolor=_GRID)
    georef = dem.transform is not None and dem.crs is not None and dem.cell_size_override_m is None
    ax.set_xlabel(f"Easting (km, {dem.crs})" if georef else "x (km)")
    ax.set_ylabel(f"Northing (km, {dem.crs})" if georef else "y (km)")
    ax.set_title(label, fontsize=10, color=_INK, loc="left")
    _style_axes(ax)
    return im


def plot_final_thickness(model: LavaFlowModel, dem: DEM, out_png: Path, title: str) -> None:
    box = _crop_box(model)
    thickness = model.lava + model.solid
    met = footprint_metrics(model)
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=150, facecolor=_SURFACE)
    vmax = max(float(np.percentile(thickness[thickness > model.params.lava_epsilon_m], 99)), 1e-3) \
        if met["footprint_cells"] else 1.0
    label = (f"{title}\nFinal lava thickness after {model.minute} min "
             f"(area {met['footprint_area_km2']:.3f} km², run-out {met['runout_m']:,.0f} m)")
    im = _plot_map(ax, model, dem, box, thickness, label, vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, extend="max")
    cbar.set_label("Lava thickness, molten + solidified (m)", color=_INK_2, fontsize=8)
    cbar.ax.tick_params(colors=_INK_2, labelsize=8)
    cbar.outline.set_visible(False)
    fig.text(0.01, 0.01, "Illustrative, uncalibrated parameters. Hillshade: pre-eruption DEM. "
             "Light blue: sea/nodata (sink).", fontsize=7, color=_INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, facecolor=_SURFACE)
    plt.close(fig)


def plot_mass_balance(records: List[Dict[str, float]], out_png: Path, title: str) -> None:
    t = np.array([r["minute"] for r in records])
    fig, (ax, ax_err) = plt.subplots(2, 1, figsize=(7.5, 6.0), dpi=150, sharex=True, facecolor=_SURFACE,
                                     gridspec_kw={"height_ratios": [3, 1.2]})
    scale = 1e6 if max(r["emitted_m3"] for r in records) >= 1e6 else 1e3
    unit = "10⁶ m³" if scale == 1e6 else "10³ m³"
    emitted = np.array([r["emitted_m3"] for r in records]) / scale
    ax.plot(t, emitted, color=_INK, lw=2, ls="--", label="Emitted")
    end_labels = [(emitted[-1], "Emitted")]
    for key, color in _SERIES.items():
        y = np.array([r[key] for r in records]) / scale
        ax.plot(t, y, color=color, lw=2, label=_SERIES_LABELS[key])
        if y[-1] > 0.02 * max(emitted[-1], 1e-12):
            end_labels.append((y[-1], _SERIES_LABELS[key]))
    # Direct labels at the line ends, nudged apart so they never overlap.
    gap = 0.06 * max(emitted.max(), 1e-12)
    placed: List[float] = []
    for y_end, text in sorted(end_labels, key=lambda e: e[0]):
        y_lab = max(y_end, placed[-1] + gap) if placed else y_end
        placed.append(y_lab)
        ax.annotate(text, (t[-1], y_end), xytext=(t[-1] + 0.01 * (t[-1] - t[0]), y_lab), textcoords="data",
                    fontsize=8, color=_INK_2, va="center", annotation_clip=False)
    ax.set_xlim(t[0] - 0.02 * (t[-1] - t[0]), t[-1] + 0.02 * (t[-1] - t[0]))
    ax.set_ylabel(f"Volume ({unit})")
    ax.set_title(f"{title}\nMass balance: emitted = active + solidified + lost", fontsize=10,
                 color=_INK, loc="left")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(True, color=_GRID, lw=0.8)
    ax.set_axisbelow(True)
    _style_axes(ax)

    err = np.array([max(r["relative_error"], 1e-18) for r in records])
    ax_err.semilogy(t, err, color=_INK_2, lw=1.5)
    ax_err.axhline(1e-6, color=_INK_2, lw=1, ls=":")
    ax_err.annotate("1e-6 tolerance", (t[0], 1e-6), xytext=(2, 3), textcoords="offset points",
                    fontsize=7, color=_INK_2)
    ax_err.set_ylim(1e-18, 1e-4)
    ax_err.set_ylabel("Relative error")
    ax_err.set_xlabel("Time since start of eruption (min)")
    ax_err.grid(True, color=_GRID, lw=0.8)
    _style_axes(ax_err)
    fig.align_ylabels()
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, facecolor=_SURFACE)
    plt.close(fig)


def write_mass_balance_csv(records: List[Dict[str, float]], out_csv: Path) -> None:
    cols = ["pulse", "minute", "emitted_m3", "active_m3", "solidified_m3", "lost_sea_m3",
            "lost_boundary_m3", "relative_error", "footprint_cells"]
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in records:
            f.write(",".join(f"{r[c]:.0f}" if c in ("pulse", "minute", "footprint_cells") else f"{r[c]:.6e}"
                             for c in cols) + "\n")


def render_animation(model: LavaFlowModel, dem: DEM, snapshots: List[Snapshot], out_gif: Path,
                     title: str, fps: int = 6) -> None:
    """Render stored snapshots on the final crop box and assemble them into a GIF."""
    if not snapshots:
        return
    box = _crop_box(model)
    vmax = max(float(np.max([s.thickness.max() for s in snapshots])), 1e-3)
    frames = []
    full = np.zeros(model.shape)
    for snap in snapshots:
        full[:] = 0.0
        h, w = snap.thickness.shape
        full[snap.row0 : snap.row0 + h, snap.col0 : snap.col0 + w] = snap.thickness
        fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=90, facecolor=_SURFACE)
        im = _plot_map(ax, model, dem, box, full, f"{title} | t = {snap.minute} min", vmax)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Lava thickness (m)", color=_INK_2, fontsize=8)
        cbar.outline.set_visible(False)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_gif, frames, duration=1.0 / fps)


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


def run_simulation(model: LavaFlowModel, snapshot_every_min: int = 0,
                   verbose: bool = True) -> Tuple[List[Dict[str, float]], List[Snapshot]]:
    """Run all pulses. Each pulse erupts for ``pulse_minutes`` and then relaxes without
    emission up to ``max_minutes_per_pulse`` (or until no molten lava remains). Lava still
    molten at the end of a pulse carries over to the next one.

    Returns one mass-balance record per minute and the stored thickness snapshots.
    """
    p = model.params
    records: List[Dict[str, float]] = []
    snapshots: List[Snapshot] = []
    for pulse in range(1, p.pulses + 1):
        if verbose:
            print(f"\n=== Pulse {pulse} ===")
        n_minutes = max(p.max_minutes_per_pulse, p.pulse_minutes)
        for minute in range(1, n_minutes + 1):
            erupting = minute <= p.pulse_minutes
            model.step(p.emission_rate_m3_per_min if erupting else 0.0)
            rec = model.mass_balance()
            rec["pulse"] = float(pulse)
            rec["footprint_cells"] = float(model.footprint().sum())
            records.append(rec)
            if verbose and minute % 10 == 0:
                _log(rec, pulse)
            if snapshot_every_min > 0 and model.minute % snapshot_every_min == 0:
                rows, cols = np.nonzero(model.footprint())
                if rows.size:
                    win = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
                    snapshots.append(Snapshot(model.minute, int(rows.min()), int(cols.min()),
                                              (model.lava + model.solid)[win].copy()))
            if not erupting and rec["active_m3"] <= 0.0:
                if verbose:
                    print(f"Pulse {pulse} stopped at minute {minute}: no molten lava remains.")
                break
    return records, snapshots


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    d = SimulationParams()
    p = argparse.ArgumentParser(description="La Palma lava flow simulation (cellular automaton).")
    p.add_argument("--mode", choices=["gaussian", "real"], default="real")
    p.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    p.add_argument("--output-dir", type=str, default=None,
                   help="Default: output/<mode>/")

    g = p.add_argument_group("real DEM")
    g.add_argument("--dem-path", type=str, default="DEM_LaPalma.tif")
    g.add_argument("--emission-gpkg", type=str, default="Emission_Pt.gpkg")
    g.add_argument("--emission-layer", type=str, default=DEFAULT_EMISSION_LAYER)
    g.add_argument("--target-crs", type=str, default=DEFAULT_TARGET_CRS,
                   help="CRS the real DEM is reprojected to (bilinear). Default: UTM 28N.")
    g.add_argument("--resolution", type=float, default=None,
                   help="Target resolution (m) of the reprojected DEM. Default: chosen by GDAL.")
    g.add_argument("--cell-size", type=float, default=None,
                   help="Real mode: override the cell size read from the DEM (emits a warning). "
                        "Gaussian mode: cell size of the synthetic grid (default 10 m).")

    g = p.add_argument_group("synthetic Gaussian DEM")
    g.add_argument("--n", type=int, default=100)
    g.add_argument("--height", type=float, default=100.0)
    g.add_argument("--sigma", type=float, default=15.0)

    g = p.add_argument_group("eruption and physics (illustrative defaults)")
    g.add_argument("--pulses", type=int, default=d.pulses)
    g.add_argument("--pulse-minutes", type=int, default=d.pulse_minutes)
    g.add_argument("--max-minutes-per-pulse", type=int, default=d.max_minutes_per_pulse)
    g.add_argument("--rate", type=float, default=d.emission_rate_m3_per_min, help="Effusion rate (m3/min).")
    g.add_argument("--initial-temp", type=float, default=d.initial_temp_c)
    g.add_argument("--solidification-temp", type=float, default=d.solidification_temp_c)
    g.add_argument("--cooling", type=float, default=d.cooling_constant_per_min,
                   help="Newtonian cooling constant k (1/min).")
    g.add_argument("--beta", type=float, default=d.arrhenius_beta, help="Arrhenius-type viscosity slope (1/C).")
    g.add_argument("--flow-coef", type=float, default=d.flow_coef)

    g = p.add_argument_group("outputs")
    g.add_argument("--snapshot-every", type=int, default=10, help="Minutes between GIF frames (0 = none).")
    g.add_argument("--no-gif", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    import json

    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    out_dir = ensure_dir(Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR / args.mode)

    if args.mode == "gaussian":
        elevation = gaussian_dem(n=args.n, height=args.height, sigma=args.sigma)
        cell_size = float(args.cell_size) if args.cell_size is not None else 10.0
        dem = DEM(elevation=elevation.astype(np.float64), cell_size_m=cell_size)
        emission_rc = summit_rc(dem.elevation)
        title = f"Synthetic Gaussian DEM ({args.n} x {args.n} cells, {cell_size:g} m)"
        vent_info: Dict[str, object] = {"row": emission_rc[0], "col": emission_rc[1]}
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
        z_vent = float(dem.elevation[vent.row, vent.col])
        print(f"DEM: {dem.shape[0]} x {dem.shape[1]} cells, {dem.cell_size_m:.2f} m, CRS {dem.crs}")
        print(f"Vent: pixel ({vent.row}, {vent.col}), x={vent.x:.1f}, y={vent.y:.1f}, z={z_vent:.1f} m")
        title = f"La Palma DEM ({dem.crs}, {dem.cell_size_m:.1f} m cells)"
        vent_info = {"row": vent.row, "col": vent.col, "x": vent.x, "y": vent.y, "z_m": z_vent}

    params = SimulationParams(
        emission_rate_m3_per_min=float(args.rate),
        pulse_minutes=int(args.pulse_minutes),
        pulses=int(args.pulses),
        max_minutes_per_pulse=int(args.max_minutes_per_pulse),
        initial_temp_c=float(args.initial_temp),
        solidification_temp_c=float(args.solidification_temp),
        cooling_constant_per_min=float(args.cooling),
        arrhenius_beta=float(args.beta),
        flow_coef=float(args.flow_coef),
    )
    model = LavaFlowModel(dem.elevation, dem.cell_size_m, emission_rc, params)
    records, snapshots = run_simulation(model, snapshot_every_min=0 if args.no_gif else args.snapshot_every)

    final_png = out_dir / "final_thickness.png"
    mb_png = out_dir / "mass_balance.png"
    mb_csv = out_dir / "mass_balance.csv"
    plot_final_thickness(model, dem, final_png, title)
    plot_mass_balance(records, mb_png, title)
    write_mass_balance_csv(records, mb_csv)
    outputs = [final_png, mb_png, mb_csv]
    if snapshots:
        gif = out_dir / "animation.gif"
        render_animation(model, dem, snapshots, gif, title)
        outputs.append(gif)

    summary = {
        "mode": args.mode,
        "command_line_args": vars(args),
        "params": asdict(params),
        "grid": {"shape": list(dem.shape), "cell_size_m": dem.cell_size_m,
                 "crs": str(dem.crs) if dem.crs else None},
        "vent": vent_info,
        "minutes_simulated": model.minute,
        "final_mass_balance": model.mass_balance(),
        "max_relative_error": max(r["relative_error"] for r in records),
        "footprint": footprint_metrics(model),
        "note": "Illustrative, uncalibrated parameters; not validated against observations.",
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    outputs.append(summary_path)

    print("\nDone. Outputs:")
    for o in outputs:
        print(f"  {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
