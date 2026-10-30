"""Command line: write synthetic instances, solve instance directories, evaluate scenario risk.

    sc-microgrid synthetic fmla my_case/                 # also: fixed-charge, wmla, vpp
    sc-microgrid run my_case/ --out-dir outputs/case     # model chosen by instance.json
    sc-microgrid risk my_case/ --scale low=0.8 --scale medium=1 --scale high=1.2 \\
        --prob low=0.15 --prob medium=0.6 --prob high=0.25 --phi 0.8 0.9 --out-dir outputs/risk
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo

from .fmla import build_fmla, fmla_summary
from .instances import FixedChargeInstance, SupplyChainInstance, VPPInstance, WarehouseInstance
from .io import read_instance, write_instance
from .location import build_fixed_charge, fixed_charge_summary
from .results import keyed_table, write_summary
from .risk import risk_table, scenario_costs
from .solve import SolveError, model_size, solve
from .synthetic import synthetic_fixed_charge, synthetic_supply_chain, synthetic_vpp, synthetic_warehouses
from .vpp import day_ahead, intraday, rolling_horizon, schedule_table
from .wmla import build_wmla, solve_enumerated, wmla_summary


def _pairs(items, name):
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--{name} expects NAME=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        out[k] = float(v)
    return out


def cmd_synthetic(a) -> int:
    makers = {"fixed-charge": lambda: synthetic_fixed_charge(seed=a.seed),
              "wmla": lambda: synthetic_warehouses(hours=a.hours or 168, seed=a.seed),
              "fmla": lambda: synthetic_supply_chain(periods=a.periods or 4, hours=a.hours or 24, seed=a.seed),
              "vpp": lambda: synthetic_vpp(hours=a.hours or 24, seed=a.seed)}
    path = write_instance(makers[a.model](), a.directory)
    print(f"wrote synthetic {a.model} instance to {path}")
    return 0


def cmd_run(a) -> int:
    inst = read_instance(a.directory)
    out = Path(a.out_dir)
    opts = dict(solver=a.solver, mip_gap=a.mip_gap, time_limit=a.time_limit)
    if isinstance(inst, FixedChargeInstance):
        m = build_fixed_charge(inst)
        res = solve(m, **opts)
        summary = fixed_charge_summary(m, inst)
    elif isinstance(inst, WarehouseInstance):
        if a.enumerate:
            m, res, _, table = solve_enumerated(inst, **opts)
            pd.DataFrame(table).to_csv(_mk(out) / "open_sets.csv", index=False)
        else:
            m = build_wmla(inst, a.formulation or "big_m")
            if a.formulation == "bilinear":
                if a.solver != "gurobi":
                    raise ValueError("wmla --formulation bilinear needs --solver gurobi, or use --enumerate")
                opts["options"] = {"NonConvex": 2}
            res = solve(m, **opts)
        summary = wmla_summary(m, inst)
        keyed_table(summary["capacity_mw"], ["site", "technology"]).to_csv(_mk(out) / "capacity_mw.csv", index=False)
    elif isinstance(inst, SupplyChainInstance):
        if a.formulation == "bilinear":
            raise ValueError("the fmla model is implemented in its big-value form only (omit --formulation)")
        m = build_fmla(inst, integer_flows=not a.relax_integers, no_microgrid=a.no_microgrid)
        res = solve(m, **opts)
        summary = fmla_summary(m, inst)
        keyed_table(summary["capacity_mw"], ["site", "technology"]).to_csv(_mk(out) / "capacity_mw.csv", index=False)
        keyed_table(summary["production_items"], ["product", "period", "factory"]).to_csv(out / "production_items.csv", index=False)
    elif isinstance(inst, VPPInstance):
        da = day_ahead(inst, allow_spill=a.allow_spill)
        res = solve(da, solver=a.solver, time_limit=a.time_limit)
        sales = [pyo.value(da.sale[t]) for t in range(1, inst.hours + 1)]
        schedule_table(da, inst).to_csv(_mk(out) / "day_ahead.csv", index=False)
        idm = intraday(inst, sales, allow_spill=a.allow_spill)
        res_id = solve(idm, solver=a.solver, time_limit=a.time_limit)
        schedule_table(idm, inst).to_csv(out / "intraday.csv", index=False)
        rh = rolling_horizon(inst, sales, a.prediction_horizon, a.control_horizon, solver=a.solver,
                             allow_spill=a.allow_spill, time_limit=a.time_limit)
        rh.to_csv(out / "rolling_horizon.csv", index=False)
        summary = {"day_ahead_cost_usd": res.objective, "intraday_cost_usd": res_id.objective,
                   "rolling_horizon_cost_usd": float(rh["cost_usd"].sum()),
                   "prediction_horizon_h": a.prediction_horizon, "control_horizon_h": a.control_horizon}
    else:  # pragma: no cover
        raise SystemExit(f"unsupported instance {type(inst).__name__}")
    summary["solver"] = {"status": res.status, "objective": res.objective, "bound": res.bound,
                         "runtime_s": res.runtime_s, "solver": res.solver}
    if not isinstance(inst, VPPInstance):
        summary["model_size"] = model_size(m)
    path = write_summary(summary, _mk(out))
    print(f"{res.status}: objective {res.objective:,.2f}; results in {path.parent}")
    return 0


def cmd_risk(a) -> int:
    inst = read_instance(a.directory)
    if not isinstance(inst, SupplyChainInstance):
        raise SystemExit("risk needs an fmla instance")
    scale, prob = _pairs(a.scale, "scale"), _pairs(a.prob, "prob")
    costs = scenario_costs(inst, scale, solver=a.solver, mip_gap=a.mip_gap, time_limit=a.time_limit)
    cost = {k: v["cost_usd"] for k, v in costs.items()}
    rows = risk_table(cost, prob, a.phi, solver=a.solver, time_limit=a.time_limit)
    out = _mk(Path(a.out_dir))
    pd.DataFrame([{"scenario": k, "cf_scale": scale[k], "probability": prob[k], "cost_usd": cost[k]} for k in cost]) \
        .to_csv(out / "scenario_costs.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "risk.csv", index=False)
    write_summary({k: v["summary"] for k, v in costs.items()}, out, "scenario_summaries.json")
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


def _mk(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sc-microgrid", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synthetic", help="write a synthetic instance directory")
    s.add_argument("model", choices=["fixed-charge", "wmla", "fmla", "vpp"])
    s.add_argument("directory")
    s.add_argument("--hours", type=int)
    s.add_argument("--periods", type=int)
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(func=cmd_synthetic)

    def solver_args(p):
        p.add_argument("--solver", choices=["highs", "gurobi"], default="highs")
        p.add_argument("--mip-gap", type=float, default=1e-6)
        p.add_argument("--time-limit", type=float)
        p.add_argument("--out-dir", default="outputs")

    r = sub.add_parser("run", help="solve an instance directory")
    r.add_argument("directory")
    r.add_argument("--formulation", choices=["big_m", "bilinear"], help="wmla only: big_m (default) or bilinear (Gurobi)")
    r.add_argument("--enumerate", action="store_true", help="wmla: bilinear model solved over every open-site set")
    r.add_argument("--no-microgrid", action="store_true", help="fmla: grid supply only")
    r.add_argument("--relax-integers", action="store_true", help="fmla: continuous product flows")
    r.add_argument("--prediction-horizon", type=int, default=3, help="vpp rolling horizon (hours)")
    r.add_argument("--control-horizon", type=int, default=1, help="vpp rolling horizon (hours)")
    r.add_argument("--allow-spill", action="store_true", help="vpp: spill extension")
    solver_args(r)
    r.set_defaults(func=cmd_run)

    k = sub.add_parser("risk", help="scenario costs and VaR/CVaR/minimax for an fmla instance")
    k.add_argument("directory")
    k.add_argument("--scale", action="append", required=True, help="NAME=FACTOR (capacity factors times FACTOR)")
    k.add_argument("--prob", action="append", required=True, help="NAME=PROBABILITY")
    k.add_argument("--phi", type=float, nargs="+", default=[0.8, 0.9, 0.95])
    solver_args(k)
    k.set_defaults(func=cmd_risk)

    a = ap.parse_args(argv)
    try:
        return a.func(a)
    except (ValueError, SolveError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
