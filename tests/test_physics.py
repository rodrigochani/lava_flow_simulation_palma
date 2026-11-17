"""Physics tests: propagation, mass conservation, symmetry, heat advection, mobility."""

from __future__ import annotations

import numpy as np
import pytest

import lava_flow_simulation_palma as lf

RATE = 1000.0  # m3/min
CELL = 10.0  # m


def _gaussian_model(n: int = 60, **param_overrides) -> "lf.LavaFlowModel":
    elev = lf.gaussian_dem(n=n, height=100.0, sigma=15.0)
    params = lf.SimulationParams(**param_overrides)
    return lf.LavaFlowModel(elev, CELL, lf.summit_rc(elev), params)


def _front_distance(model: "lf.LavaFlowModel") -> float:
    """Largest distance (in cells) between the vent and any footprint cell."""
    rows, cols = np.nonzero(model.footprint())
    vr, vc = model.vent_rc
    return float(np.max(np.hypot(rows - vr, cols - vc)))


# --------------------------------------------------------------------------- propagation
def test_gaussian_footprint_grows_beyond_3x3_and_front_advances():
    model = _gaussian_model(n=60)
    footprint, front = {}, {}
    for minute in range(1, 21):
        model.step(RATE)
        if minute in (5, 10, 20):
            footprint[minute] = int(model.footprint().sum())
            front[minute] = _front_distance(model)
    assert footprint[20] > 9, f"footprint after 20 min is only {footprint[20]} cells"
    assert footprint[5] < footprint[10] < footprint[20]
    assert front[5] < front[10] < front[20]


# ------------------------------------------------------------------------ mass balance
def _assert_balance(model: "lf.LavaFlowModel") -> None:
    mb = model.mass_balance()
    accounted = mb["active_m3"] + mb["solidified_m3"] + mb["lost_sea_m3"] + mb["lost_boundary_m3"]
    assert mb["emitted_m3"] > 0
    rel = abs(mb["emitted_m3"] - accounted) / mb["emitted_m3"]
    assert rel < 1e-6
    assert mb["relative_error"] == pytest.approx(rel, abs=1e-12)


def test_mass_conservation_gaussian_every_step():
    model = _gaussian_model(n=60)
    for minute in range(1, 121):
        model.step(RATE if minute <= 60 else 0.0)
        _assert_balance(model)


def test_mass_conservation_with_sea_and_boundary_sinks():
    """Inclined plane draining into a sea strip; lava must leave through sea and edges."""
    nrows, ncols = 40, 30
    rows = np.arange(nrows, dtype=float)[:, None]
    elev = np.repeat(200.0 - 5.0 * rows, ncols, axis=1)  # slopes towards the last rows
    elev[-5:, :] = 0.0  # sea strip (elevation <= sea level)
    elev[-5:, :5] = np.nan  # nodata patch, also a sink
    params = lf.SimulationParams(cooling_constant_per_min=0.0)  # never solidifies
    model = lf.LavaFlowModel(elev, CELL, (2, 2), params)  # vent close to the left edge
    for minute in range(1, 151):
        model.step(RATE if minute <= 60 else 0.0)
        _assert_balance(model)
    mb = model.mass_balance()
    assert mb["lost_sea_m3"] > 0.0
    assert mb["lost_boundary_m3"] > 0.0
    # Sink cells never hold lava or deposits.
    sink = model.sink_mask
    assert np.all(model.lava[sink] == 0.0) and np.all(model.solid[sink] == 0.0)


# ---------------------------------------------------------------------------- symmetry
def test_flat_dem_spreads_symmetrically():
    n = 41
    elev = np.full((n, n), 100.0)
    model = lf.LavaFlowModel(elev, CELL, (n // 2, n // 2), lf.SimulationParams())
    for minute in range(1, 41):
        model.step(RATE if minute <= 30 else 0.0)
    thick = model.lava + model.solid
    assert model.footprint().sum() > 9
    for transformed in (thick[::-1, :], thick[:, ::-1], thick.T, thick[::-1, ::-1].T):
        np.testing.assert_allclose(transformed, thick, rtol=1e-9, atol=1e-12)


# ------------------------------------------------------------------------ heat advection
def test_without_cooling_all_lava_stays_at_eruption_temperature():
    """With no cooling, advection must carry the vent temperature everywhere."""
    model = _gaussian_model(n=40, cooling_constant_per_min=0.0)
    for _ in range(30):
        model.step(RATE)
    wet = model.lava > 0
    assert wet.sum() > 9
    np.testing.assert_allclose(model.temperature[wet], model.params.initial_temp_c, rtol=1e-12)
    np.testing.assert_array_equal(model.temperature[~wet], model.params.ambient_temp_c)
    assert model.mass_balance()["solidified_m3"] == 0.0


def test_mixing_is_volume_weighted():
    """Moving q from a hot cell into a cell holding cooler lava gives the weighted mean."""
    params = lf.SimulationParams(cooling_constant_per_min=0.0)
    # Two open cells, A=(1,1) and B=(1,2) one metre lower, fenced in by high walls so
    # that A can only drain into B and B cannot drain anywhere.
    elev = np.full((3, 4), 1000.0)
    elev[1, 1], elev[1, 2] = 100.0, 99.0
    lava = np.zeros_like(elev)
    lava[1, 1], lava[1, 2] = 2.0, 0.5
    temp = np.full_like(elev, params.ambient_temp_c)
    temp[1, 1], temp[1, 2] = 1100.0, 1000.0
    model = lf.LavaFlowModel(elev, CELL, (1, 1), params)
    model.set_state(lava=lava, temperature=temp)
    model.step(0.0)
    q = 2.0 - model.lava[1, 1]
    assert q > 0
    assert model.lava[1, 2] == pytest.approx(0.5 + q, rel=1e-12)
    expected = (0.5 * 1000.0 + q * 1100.0) / (0.5 + q)
    assert model.temperature[1, 2] == pytest.approx(expected, rel=1e-12)
    assert model.temperature[1, 1] == pytest.approx(1100.0, rel=1e-12)


# ---------------------------------------------------------------------------- mobility
def test_mobility_is_local_and_bounded():
    params = lf.SimulationParams()
    temps = np.array([params.initial_temp_c, 1100.0, 1000.0, params.solidification_temp_c])
    mob = lf.mobility(temps, params)
    assert mob[0] == pytest.approx(1.0)
    assert np.all((mob > 0) & (mob <= 1.0))
    assert np.all(np.diff(mob) < 0)
    # Local: the value for one cell does not depend on the other cells in the array.
    assert lf.mobility(temps[:2], params)[1] == pytest.approx(mob[1])
    expected = lf.viscosity_from_temp(params.initial_temp_c, params) / lf.viscosity_from_temp(1100.0, params)
    assert mob[1] == pytest.approx(expected)
