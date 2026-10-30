"""Sensitivity cases of the kind studied in the articles, on the synthetic supply chain.

Each case changes one group of inputs and re-solves the FMLA model (AMM 2023 formulation).
Families follow docs/scenarios.md: production planning, microgrid sizing, energy prices.
"""

import copy

import numpy as np
import pandas as pd

from sc_microgrid import build_fmla, fmla_summary, solve, synthetic_supply_chain


def case(name, change=None, **build):
    inst = synthetic_supply_chain(periods=4, hours=24)
    if change:
        change(inst)
    m = build_fmla(inst, **build)
    solve(m)
    s = fmla_summary(m, inst)
    return {"case": name, "cost_usd_per_yr": s["total_cost_usd"],
            "factories": " ".join(s["open_factories"]), "warehouses": " ".join(s["open_warehouses"]),
            "wind_mw": sum(v for (x, g), v in s["capacity_mw"].items() if g == "wind"),
            "pv_mw": sum(v for (x, g), v in s["capacity_mw"].items() if g == "pv"),
            "storage_mwh": sum(s["storage_mwh"].values()),
            "bought_mwh": s["energy_bought_mwh"], "sold_mwh": s["energy_sold_mwh"]}


def scale(d, f):
    for k in d:
        d[k] = d[k] * f


rows = [
    case("benchmark"),
    case("grid supply only", no_microgrid=True),
    case("PV capacity cost -50%", lambda i: setattr(i.technologies["pv"], "capacity_cost_usd_per_mw",
                                                    i.technologies["pv"].capacity_cost_usd_per_mw * 0.5)),
    case("storage capacity cost -75%", lambda i: setattr(i.storage, "capacity_cost_usd_per_mwh",
                                                         i.storage.capacity_cost_usd_per_mwh * 0.25)),
    case("purchase price +25%", lambda i: scale(i.buy_price_usd_per_mwh, 1.25)),
    case("flat purchase price", lambda i: [i.buy_price_usd_per_mwh.__setitem__(x, np.full_like(v, v.mean()))
                                           for x, v in list(i.buy_price_usd_per_mwh.items())]),
    case("energy per item x2", lambda i: scale(i.energy_mwh_per_item, 2.0)),
    case("holding and backorder cost x2", lambda i: (scale(i.holding_cost_usd_per_item_period, 2.0),
                                                     scale(i.backorder_cost_usd_per_item_period, 2.0))),
    case("demand +20%", lambda i: scale(i.demand_mean_items, 1.2)),
]
pd.set_option("display.width", 160)
print(pd.DataFrame(rows).round(2).to_string(index=False))
