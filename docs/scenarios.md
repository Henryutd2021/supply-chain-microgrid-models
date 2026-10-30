# Scenarios and sensitivity cases

The publications study each model under families of input changes. This document lists the families and shows how to express them with the framework. Values are the user's choice; [`examples/sensitivity.py`](../examples/sensitivity.py) and [`examples/risk_scenarios.py`](../examples/risk_scenarios.py) run a set of them on the synthetic supply chain.

## How a case is built

A case is an instance with some inputs changed, solved again. Instances are plain dataclasses, so a case is a few lines:

```python
from dataclasses import replace
from sc_microgrid import synthetic_supply_chain, build_fmla, solve, fmla_summary

base = synthetic_supply_chain()                       # or read_instance("my_case/")
case = replace(base, buy_price_usd_per_mwh={x: p * 0.75 for x, p in base.buy_price_usd_per_mwh.items()})
m = build_fmla(case)
solve(m)
print(fmla_summary(m, case)["components_usd"])
```

Nested objects (technologies, storage) are changed the same way, for example `replace(base.technologies["pv"], capacity_cost_usd_per_mw=...)`.

## Families of the three-tier model (AMM 2023 Section 5; thesis Section 4.5)

| Family | What changes | Inputs |
|---|---|---|
| production planning | holding and backorder costs, energy use per item, production cost, product demand (each scaled up and down) | `holding_cost_usd_per_item_period`, `backorder_cost_usd_per_item_period`, `energy_mwh_per_item`, `production_cost_usd_per_item`, `demand_mean_items`, `demand_std_items` |
| microgrid sizing | electricity sale and purchase prices, time-of-use purchase price, capacity cost of each technology and of storage, capacity factors | `sell_price_usd_per_mwh`, `buy_price_usd_per_mwh`, `technologies[g].capacity_cost_usd_per_mw`, `storage.capacity_cost_usd_per_mwh`, `cf` (or `scale_cf`) |
| grid supply only | no generation or storage anywhere | `build_fmla(inst, no_microgrid=True)` |
| product variety | more products, each with its own demand, energy use, resource use, weight and costs | `products` and every product-indexed input |
| network size | more candidate factories, warehouses or stores | site lists and every site-indexed input |

Comparing a case with the grid-only case gives the value of the on-site microgrids under the case's assumptions.

## Families of the two-tier model (thesis Sections 4.1–4.2; SEGRE 2023)

| Family | What changes | Inputs |
|---|---|---|
| classical location versus microgrid location | the fixed-charge model of Part A against the WMLA model on the same network | `build_fixed_charge`, `build_wmla` |
| purchase price | electricity purchase price lowered in steps | `buy_price_usd_per_mwh` |
| technology cost | wind turbine or PV capacity cost lowered | `technologies[g].capacity_cost_usd_per_mw` |
| open-warehouse sets | the cost of every feasible set of open warehouses | `solve_enumerated(inst)` returns the table |

## Capacity-factor scenarios and risk criteria (thesis Chapter 5)

1. For the thesis setting, set the factory base load to zero (`replace(inst, factory_load_mw={k: 0.0 for k in inst.factories})`); Model 5.1 has no base-load term in (5.17).
2. Define the scenarios: a medium scenario (the representative capacity factors) and lower and higher scenarios, each with a probability. Scaling the medium series is one simple way (`scale_cf`); historical years are another (`with_cf`).
3. Solve the expected-value model (Model 5.1): `with_cf(inst, expected_cf(scenarios, probabilities))`.
4. Solve the deterministic model under each scenario: `scenario_costs(inst, scale)` gives $C_\omega$ and each scenario's design.
5. Evaluate the scenario costs with VaR, CVaR and minimax over a range of confidence levels $\phi$: `risk_table(cost, prob, phis)`. VaR and CVaR do not decrease as $\phi$ rises; minimax does not depend on it.

From the command line:

```bash
sc-microgrid risk my_case/ --scale low=0.8 --scale medium=1 --scale high=1.2 \
    --prob low=0.15 --prob medium=0.6 --prob high=0.25 --phi 0.5 0.8 0.9 --out-dir outputs/risk
```

## VPP (thesis Chapter 3)

| Family | What changes | Inputs |
|---|---|---|
| forecast error | delivery-day loads and renewable output against the forecasts | `load_realized_mw`, `heat_load_realized_mw`, `renewable_realized_mw` |
| rolling-horizon settings | prediction horizon and control horizon | `rolling_horizon(inst, sales, prediction_horizon, control_horizon)` |
| information in the window | forecasts or delivery-day values beyond the control horizon | `forecast_beyond_control` |
| prices | day-ahead and imbalance prices | `day_ahead_price_usd_per_mwh`, `imbalance_price_usd_per_mwh` |
| equipment | CHP limits and heat-to-power ratio, battery and thermal storage limits | `chp_*`, `heat_to_power`, `battery_*`, `thermal_*` |

[`examples/vpp_rolling_horizon.py`](../examples/vpp_rolling_horizon.py) compares prediction horizons from 1 to 24 hours.
