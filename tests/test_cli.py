"""Smoke test: the command-line entry point runs end to end and writes its outputs."""

from __future__ import annotations

import json

import lava_flow_simulation_palma as lf


def test_gaussian_cli_writes_outputs(tmp_path):
    rc = lf.main([
        "--mode", "gaussian", "--n", "30", "--pulses", "2", "--pulse-minutes", "5",
        "--max-minutes-per-pulse", "15", "--snapshot-every", "5", "--output-dir", str(tmp_path),
    ])
    assert rc == 0
    for name in ("final_thickness.png", "mass_balance.png", "mass_balance.csv", "animation.gif", "summary.json"):
        assert (tmp_path / name).stat().st_size > 0, name
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["max_relative_error"] < 1e-6
    assert summary["final_mass_balance"]["emitted_m3"] == 2 * 5 * summary["params"]["emission_rate_m3_per_min"]
    lines = (tmp_path / "mass_balance.csv").read_text().splitlines()
    assert lines[0].startswith("pulse,minute,emitted_m3")
    assert len(lines) - 1 == summary["minutes_simulated"]
