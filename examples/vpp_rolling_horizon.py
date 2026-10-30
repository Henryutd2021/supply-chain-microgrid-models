"""Day-ahead planning and rolling-horizon intraday operation of the synthetic VPP (thesis Ch. 3).

Compares prediction horizons of 1 to 24 hours with a control horizon of 1 hour. Beyond the
control horizon each window sees the forecasts; inside it, the realized values.
"""

import pyomo.environ as pyo
import pandas as pd

from sc_microgrid import day_ahead, intraday, rolling_horizon, solve, synthetic_vpp

vpp = synthetic_vpp()
da = day_ahead(vpp)
r_da = solve(da)
sales = [pyo.value(da.sale[t]) for t in range(1, vpp.hours + 1)]
r_id = solve(intraday(vpp, sales))
print(f"day-ahead plan: cost {r_da.objective:,.0f} $, sales {sum(sales):,.1f} MWh")
print(f"intraday, whole day known: cost {r_id.objective:,.0f} $")
rows = []
for ph in (1, 2, 3, 6, 12, 24):
    rh = rolling_horizon(vpp, sales, prediction_horizon=ph, control_horizon=1)
    rows.append({"prediction_horizon_h": ph, "cost_usd": rh.cost_usd.sum(),
                 "imbalance_bought_mwh": rh.imbalance_purchase_mw.sum(), "chp_mwh": rh.chp_mw.sum()})
print(pd.DataFrame(rows).round(1).to_string(index=False))
