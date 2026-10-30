"""Capacity-factor scenarios and risk criteria (thesis Chapter 5) on the synthetic supply chain.

1. Expected-value design: the FMLA model with scenario-weighted capacity factors (Model 5.1).
2. Scenario costs C_w: the deterministic model solved under each scenario.
3. VaR, CVaR and minimax of the scenario costs (Models 5.2-5.4) for several confidence levels.
"""

from dataclasses import replace

import pandas as pd

from sc_microgrid import (build_fmla, expected_cf, fmla_summary, risk_table, scale_cf, scenario_costs, solve,
                          synthetic_cf_scenarios, synthetic_supply_chain, with_cf)

inst = synthetic_supply_chain(periods=4, hours=24)
inst = replace(inst, factory_load_mw={k: 0.0 for k in inst.factories})   # Model 5.1: no factory base load (Eq. 5.17)
scale, prob = synthetic_cf_scenarios()

ev = with_cf(inst, expected_cf([scale_cf(inst, scale[w]).cf for w in scale], [prob[w] for w in scale]))
m = build_fmla(ev)
solve(m)
s = fmla_summary(m, ev)
print(f"expected-value design: cost {s['total_cost_usd']:,.0f} $/yr; factories {s['open_factories']}; "
      f"warehouses {s['open_warehouses']}")

costs = scenario_costs(inst, scale)
table = pd.DataFrame([{"scenario": w, "cf_scale": scale[w], "probability": prob[w], "cost_usd_per_yr": costs[w]["cost_usd"],
                       "factories": " ".join(costs[w]["summary"]["open_factories"])} for w in scale])
print(table.to_string(index=False))
cost = {w: costs[w]["cost_usd"] for w in scale}
print(pd.DataFrame(risk_table(cost, prob, [0.5, 0.8, 0.9, 0.95])).to_string(index=False))
