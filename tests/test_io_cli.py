import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sc_microgrid import (read_instance, synthetic_fixed_charge, synthetic_supply_chain, synthetic_vpp,
                          synthetic_warehouses, write_instance)
from sc_microgrid.cli import main

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("make", [synthetic_fixed_charge, lambda: synthetic_warehouses(hours=24),
                                  lambda: synthetic_supply_chain(periods=2, hours=6), synthetic_vpp])
def test_round_trip(tmp_path, make):
    inst = make()
    back = read_instance(write_instance(inst, tmp_path / "case"))
    assert type(back) is type(inst)
    for name, value in vars(inst).items():
        other = getattr(back, name)
        if isinstance(value, dict) and value and isinstance(next(iter(value.values())), np.ndarray):
            assert all(np.allclose(value[k], other[k]) for k in value)
        elif isinstance(value, np.ndarray):
            assert np.allclose(value, other)
        elif name in ("technologies", "storage", "truck", "meta"):
            continue
        else:
            assert value == other, name


def test_site_names_kept_as_text(tmp_path):
    inst = synthetic_warehouses(hours=24)
    names = {"W-North": "001", "W-Central": "NA", "W-South": "3"}
    ren = lambda d: {(names.get(k[0], k[0]),) + tuple(k[1:]) if isinstance(k, tuple) else names.get(k, k): v  # noqa: E731
                     for k, v in d.items()}
    for f in ("fixed_cost_usd", "capacity_items", "site_load_mw", "ship_cost_usd_per_item_km", "distance_km",
              "max_capacity_mw", "cf", "buy_price_usd_per_mwh", "sell_price_usd_per_mwh", "operating_share"):
        setattr(inst, f, ren(getattr(inst, f)))
    inst.sites = [names[n] for n in inst.sites]
    back = read_instance(write_instance(inst, tmp_path / "names"))
    assert back.sites == ["001", "NA", "3"]
    assert np.allclose(back.cf["001", "pv"], inst.cf["001", "pv"])


def test_read_errors(tmp_path):
    d = write_instance(synthetic_supply_chain(periods=2, hours=6), tmp_path / "c")
    data = json.loads((d / "instance.json").read_text())
    data["fixed_cost"] = 1
    (d / "instance.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unknown keys"):
        read_instance(d)
    d = write_instance(synthetic_supply_chain(periods=2, hours=6), tmp_path / "d")
    hourly = pd.read_csv(d / "hourly.csv").iloc[1:]
    hourly.to_csv(d / "hourly.csv", index=False)
    with pytest.raises(ValueError, match="complete grid"):
        read_instance(d)
    d = write_instance(synthetic_warehouses(hours=24), tmp_path / "e")
    hourly = pd.read_csv(d / "hourly.csv").rename(columns={"cf_pv": "cf_solar"})
    hourly.to_csv(d / "hourly.csv", index=False)
    with pytest.raises(ValueError, match="missing columns"):
        read_instance(d)
    with pytest.raises(ValueError, match="not found"):
        read_instance(tmp_path / "nothing")


def test_cli(tmp_path):
    for model in ("fixed-charge", "wmla", "fmla", "vpp"):
        case, out = tmp_path / model, tmp_path / f"out_{model}"
        assert main(["synthetic", model, str(case), "--hours", "24", "--periods", "2"]) == 0
        assert main(["run", str(case), "--out-dir", str(out)]) == 0
        summary = json.loads((out / "summary.json").read_text())
        assert summary["solver"]["status"] == "convergenceCriteriaSatisfied"
    assert main(["run", str(tmp_path / "wmla"), "--enumerate", "--out-dir", str(tmp_path / "enum")]) == 0
    assert main(["risk", str(tmp_path / "fmla"), "--scale", "low=0.8", "--scale", "high=1.2",
                 "--prob", "low=0.4", "--prob", "high=0.6", "--phi", "0.5", "--out-dir", str(tmp_path / "risk")]) == 0
    assert len(pd.read_csv(tmp_path / "risk" / "risk.csv")) == 1
    assert main(["run", str(tmp_path / "missing")]) == 2
    assert main(["run", str(tmp_path / "vpp"), "--allow-spill", "--time-limit", "60", "--out-dir", str(tmp_path / "spill")]) == 0


def test_minimal_example_runs():
    out = subprocess.run([sys.executable, str(ROOT / "examples" / "minimal.py")], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert "open factories" in out.stdout
