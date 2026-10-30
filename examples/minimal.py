"""Solve Parts A, B, C and E once on synthetic data with HiGHS (a few seconds).

Model 5.1 and the risk criteria (Part D) are in risk_scenarios.py.

The synthetic regions are invented; the numbers say nothing about real places or firms.
"""

import pyomo.environ as pyo

from sc_microgrid import (build_fixed_charge, build_fmla, build_wmla, day_ahead, fixed_charge_summary, fmla_summary,
                          intraday, rolling_horizon, solve, synthetic_fixed_charge, synthetic_supply_chain,
                          synthetic_vpp, synthetic_warehouses, wmla_summary)

# Part A: classical fixed-charge location (thesis Model 4.1)
fc = synthetic_fixed_charge()
m = build_fixed_charge(fc)
solve(m)
s = fixed_charge_summary(m, fc)
print(f"[fixed charge] cost {s['total_cost_usd']:,.0f} $; open sites {s['open_sites']}")

# Part B: warehouses with on-site microgrids (thesis Model 4.4), one week weighted to a year
wh = synthetic_warehouses(hours=168)
m = build_wmla(wh, "big_m")
solve(m)
s = wmla_summary(m, wh)
mw = {f"{n}/{g}": round(v, 2) for (n, g), v in s["capacity_mw"].items() if v > 1e-6}
print(f"[warehouses] cost {s['total_cost_usd']:,.0f} $/yr; open {s['open_sites']}; capacity MW {mw}")

# Part C: factories, warehouses and microgrids (AMM 2023), four one-day periods weighted to a year
sc = synthetic_supply_chain(periods=4, hours=24)
m = build_fmla(sc)
solve(m)
s = fmla_summary(m, sc)
mw = {f"{x}/{g}": round(v, 2) for (x, g), v in s["capacity_mw"].items() if v > 1e-6}
print(f"[supply chain] cost {s['total_cost_usd']:,.0f} $/yr; open factories {s['open_factories']}, "
      f"warehouses {s['open_warehouses']}; capacity MW {mw}")
print("               cost components:", {k: f"{v:,.0f}" for k, v in s["components_usd"].items()})

# Part E: virtual power plant, day-ahead plan then rolling-horizon intraday operation (thesis Ch. 3)
vpp = synthetic_vpp()
da = day_ahead(vpp)
r_da = solve(da)
sales = [pyo.value(da.sale[t]) for t in range(1, vpp.hours + 1)]
r_id = solve(intraday(vpp, sales))
rh = rolling_horizon(vpp, sales, prediction_horizon=3, control_horizon=1)
print(f"[VPP] day-ahead cost {r_da.objective:,.0f} $; intraday {r_id.objective:,.0f} $; "
      f"rolling horizon (PH 3 h, CH 1 h) {rh.cost_usd.sum():,.0f} $; imbalance bought {rh.imbalance_purchase_mw.sum():.1f} MWh")
