"""Facility and microgrid location-allocation (FMLA) in a three-layer supply chain.

Factories k make products i in periods j and ship them to warehouses n, which serve stores s.
Every open factory or warehouse can install wind (or other) generation g, PV and storage, and
buys from and sells to the grid every hour t of every period. docs/formulation.md, Part C.

The formulation is the big-value form of AMM 2023 Eqs. (1)-(35) (thesis Model 4.7; with expected
capacity factors, thesis Model 5.1). The objective and the constraints are linear in all decision
variables: a MILP, solved with HiGHS or Gurobi.
"""

from __future__ import annotations

from dataclasses import replace
from statistics import NormalDist
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pyomo.environ as pyo

from .instances import SupplyChainInstance

def z_value(service_level: float) -> float:
    """z_alpha of the deterministic equivalent (35) of the chance constraint (10)."""
    return NormalDist().inv_cdf(service_level)


def build_fmla(inst: SupplyChainInstance, *, integer_flows: bool = True, no_microgrid: bool = False) -> pyo.ConcreteModel:
    """Build the FMLA model (AMM 2023 Eqs. 1-35, big-value form).

    ``integer_flows=False`` relaxes production, shipments, inventory and backorders to continuous
    variables (an option for large instances; the articles use integers, Eqs. 7 and 16).
    ``no_microgrid=True`` fixes all generation and storage capacities at zero (grid supply only).
    """
    inst.validate()
    pmax_top = max(inst.max_capacity_mw.values())
    if inst.big_m < pmax_top:
        raise ValueError(f"big_m ({inst.big_m}) must be at least the largest max_capacity_mw ({pmax_top}); "
                         "see docs/formulation.md, conventions")
    K, N, S, I, R = inst.factories, inst.warehouses, inst.stores, inst.products, inst.resources
    G = list(inst.technologies)
    J, T = inst.periods, inst.hours
    P1, P0 = range(1, J + 1), range(0, J + 1)
    H1, H0 = range(1, T + 1), range(0, T + 1)
    tech, es, truck = inst.technologies, inst.storage, inst.truck
    W, M = inst.horizon_weight, inst.big_m
    dom = pyo.NonNegativeIntegers if integer_flows else pyo.NonNegativeReals

    m = pyo.ConcreteModel("fmla")
    # decision variables, AMM Table 2
    m.u = pyo.Var(K, domain=pyo.Binary)                       # factory open
    m.uh = pyo.Var(N, domain=pyo.Binary)                      # warehouse open
    m.v = pyo.Var(I, P1, K, N, domain=pyo.Binary)             # route factory -> warehouse used
    m.vh = pyo.Var(I, N, S, domain=pyo.Binary)                # store served by warehouse
    m.x = pyo.Var(I, P1, K, N, domain=dom)                    # production shipped k -> n
    m.xh = pyo.Var(I, P1, N, S, domain=dom)                   # shipment n -> s
    m.y = pyo.Var(I, P0, N, domain=dom)                       # inventory (period 0 = initial)
    m.z = pyo.Var(I, P1, N, domain=dom)                       # backorders
    m.Pk = pyo.Var(K, G, domain=pyo.NonNegativeReals)         # P^c_gk (MW)
    m.Pn = pyo.Var(N, G, domain=pyo.NonNegativeReals)         # P^c_gn (MW)
    m.Bck = pyo.Var(K, domain=pyo.NonNegativeReals)           # B^c_k (MWh)
    m.Bcn = pyo.Var(N, domain=pyo.NonNegativeReals)           # B^c_n (MWh)
    m.Bk = pyo.Var(P1, K, H0, domain=pyo.NonNegativeReals)    # B_jkt (MWh)
    m.Bn = pyo.Var(P1, N, H0, domain=pyo.NonNegativeReals)    # B_jnt (MWh)
    m.Esk = pyo.Var(P1, K, H1, domain=pyo.NonNegativeReals)   # E^se_jkt
    m.Ebk = pyo.Var(P1, K, H1, domain=pyo.NonNegativeReals)   # E^bu_jkt
    m.Esn = pyo.Var(P1, N, H1, domain=pyo.NonNegativeReals)   # E^se_jnt
    m.Ebn = pyo.Var(P1, N, H1, domain=pyo.NonNegativeReals)   # E^bu_jnt
    # convention: no backorder is opened in the last period (thesis Eq. 4.74; AMM Eq. 1 costs j < J)
    for i in I:
        for n in N:
            m.z[i, J, n].fix(0)
    if no_microgrid:
        for c in (m.Pk, m.Pn, m.Bck, m.Bcn):
            for idx in c:
                c[idx].fix(0)

    p, e = inst.production_cost_usd_per_item, inst.energy_mwh_per_item
    h, b = inst.holding_cost_usd_per_item_period, inst.backorder_cost_usd_per_item_period
    pi, pih = inst.ship_cost_fw_usd_per_item_km, inst.ship_cost_ws_usd_per_item_km
    d, dh = inst.distance_fw_km, inst.distance_ws_km
    mi, gam = inst.product_weight_kg, inst.warehouse_capacity_items
    cf, rb, rs = inst.cf, inst.buy_price_usd_per_mwh, inst.sell_price_usd_per_mwh

    def g_om(g):   # tau_g (b_g - c_g)
        return tech[g].operating_hours_per_step * (tech[g].om_cost_usd_per_mwh - tech[g].carbon_credit_usd_per_mwh)

    def g_cap(g):  # phi_g a_g
        return tech[g].capital_recovery_factor * tech[g].capacity_cost_usd_per_mw

    cf_sum = {(x, g): float(cf[x, g].sum()) for x in inst.sites for g in G}

    # ---- objective: AMM Eq. (1)
    obj = (
        sum(inst.factory_fixed_cost_usd[k] * m.u[k] for k in K)
        + W * sum(p[i, k] * m.x[i, j, k, n] + pi[i] * d[k, n] * m.x[i, j, k, n]
                  for i in I for j in P1 for k in K for n in N)
        + sum(inst.warehouse_fixed_cost_usd[n] * m.uh[n] for n in N)
        + W * sum(h[i] * m.y[i, j, n] for i in I for j in P1 for n in N)
        + W * sum(b[i] * m.z[i, j, n] for i in I for j in range(1, J) for n in N)
        + W * sum(pih[i] * dh[n, s] * m.xh[i, j, n, s] for i in I for j in P1 for n in N for s in S)
        + sum(g_cap(g) * m.Pk[k, g] + W * g_om(g) * cf_sum[k, g] * m.Pk[k, g] for g in G for k in K)
        + sum(es.annual_cost_usd_per_mwh * m.Bck[k] for k in K)
        + W * sum(rb[k][j - 1, t - 1] * m.Ebk[j, k, t] - rs[k][j - 1, t - 1] * m.Esk[j, k, t]
                  for k in K for j in P1 for t in H1)
        + sum(g_cap(g) * m.Pn[n, g] + W * g_om(g) * cf_sum[n, g] * m.Pn[n, g] for g in G for n in N)
        + sum(es.annual_cost_usd_per_mwh * m.Bcn[n] for n in N)
        + W * sum(rb[n][j - 1, t - 1] * m.Ebn[j, n, t] - rs[n][j - 1, t - 1] * m.Esn[j, n, t]
                  for n in N for j in P1 for t in H1)
    )
    m.cost = pyo.Objective(expr=obj, sense=pyo.minimize)

    # ---- facility location: AMM (2)-(5)
    m.eq2_single_source = pyo.Constraint(I, S, rule=lambda m, i, s: sum(m.vh[i, n, s] for n in N) == 1)
    m.eq3_inbound_capacity = pyo.Constraint(I, P1, N, rule=lambda m, i, j, n:
                                            sum(m.x[i, j, k, n] for k in K) <= gam[i, n] * m.uh[n])
    m.eq4_outbound_capacity = pyo.Constraint(I, P1, N, rule=lambda m, i, j, n:
                                             sum(m.xh[i, j, n, s] for s in S) <= gam[i, n] * m.uh[n])
    m.eq5_store_open = pyo.Constraint(I, N, S, rule=lambda m, i, n, s: m.vh[i, n, s] <= m.uh[n])

    # ---- production and inventory: AMM (8), (9), (35), (11)-(14)
    def inflow(m, i, j, n):
        return sum(m.x[i, j, k, n] for k in K)

    def balance(m, i, j, n):
        out = sum(m.xh[i, j, n, s] for s in S)
        if j == 1:
            return inflow(m, i, 1, n) + m.y[i, 0, n] - m.y[i, 1, n] + m.z[i, 1, n] == out
        return inflow(m, i, j, n) + m.y[i, j - 1, n] - m.y[i, j, n] - m.z[i, j - 1, n] + m.z[i, j, n] == out
    m.eq8_9_inventory_balance = pyo.Constraint(I, P1, N, rule=balance)

    z_a = z_value(inst.service_level)
    m.eq35_service_level = pyo.Constraint(I, P1, S, rule=lambda m, i, j, s:
                                          sum(m.xh[i, j, n, s] for n in N)
                                          >= inst.demand_mean_items[i, j, s] + z_a * inst.demand_std_items[i, j, s])
    m.eq11_resources = pyo.Constraint(P1, K, R, rule=lambda m, j, k, r:
                                      sum(inst.resource_use[i, k, r] * m.x[i, j, k, n] for i in I for n in N)
                                      <= inst.resource_available[j, k, r] * m.u[k])
    m.eq12_route_used = pyo.Constraint(I, P1, K, N, rule=lambda m, i, j, k, n: m.x[i, j, k, n] <= M * m.v[i, j, k, n])
    m.eq13_store_route = pyo.Constraint(I, P1, N, S, rule=lambda m, i, j, n, s: m.xh[i, j, n, s] <= M * m.vh[i, n, s])
    m.eq14_initial_inventory = pyo.Constraint(I, N, rule=lambda m, i, n: m.y[i, 0, n] == 0)

    # ---- hourly energy balances: AMM (17), (22); production spread over the period (thesis 4.81-4.82)
    q, mev = truck.energy_mwh_per_kg_km, truck.weight_kg
    nfw, nws = inst.trips_fw_per_hour, inst.trips_ws_per_hour

    def factory_load(m, j, k):   # MWh in each hour of period j (does not depend on t)
        return (inst.factory_load_mw[k] * m.u[k]
                + sum((e[i, k] + q * d[k, n] * mi[i]) * m.x[i, j, k, n] for i in I for n in N) / T
                + q * mev * sum(nfw[k, n] * d[k, n] * m.v[i, j, k, n] for i in I for n in N))

    def warehouse_load(m, j, n):
        return (inst.warehouse_load_mw[n] * m.uh[n]
                + q * mev * sum(nfw[k, n] * d[k, n] * m.v[i, j, k, n] for i in I for k in K)
                + q * sum(dh[n, s] * mi[i] * m.xh[i, j, n, s] for i in I for s in S) / T
                + 2 * q * mev * sum(nws[n, s] * dh[n, s] * m.vh[i, n, s] for i in I for s in S))

    def gen(x, var, j, t):
        return sum(tech[g].operating_hours_per_step * cf[x, g][j - 1, t - 1] * var[x, g] for g in G)

    Lk = {(j, k): factory_load(m, j, k) for j in P1 for k in K}
    Ln = {(j, n): warehouse_load(m, j, n) for j in P1 for n in N}
    m.eq17_factory_energy = pyo.Constraint(P1, K, H1, rule=lambda m, j, k, t:
                                           Lk[j, k] + m.Bk[j, k, t] - m.Bk[j, k, t - 1] + m.Esk[j, k, t] - m.Ebk[j, k, t]
                                           == gen(k, m.Pk, j, t))
    m.eq22_warehouse_energy = pyo.Constraint(P1, N, H1, rule=lambda m, j, n, t:
                                             Ln[j, n] + m.Bn[j, n, t] - m.Bn[j, n, t - 1] + m.Esn[j, n, t] - m.Ebn[j, n, t]
                                             == gen(n, m.Pn, j, t))

    # ---- microgrid bounds: AMM (18)-(30)
    pmax = inst.max_capacity_mw
    m.eq18_max_capacity_k = pyo.Constraint(K, G, rule=lambda m, k, g: m.Pk[k, g] <= pmax[k, g] * m.u[k])
    m.eq19_storage_open_k = pyo.Constraint(K, rule=lambda m, k: m.Bck[k] <= M * m.u[k])
    m.eq20_sale_open_k = pyo.Constraint(P1, K, H1, rule=lambda m, j, k, t: m.Esk[j, k, t] <= M * m.u[k])
    m.eq21_purchase_open_k = pyo.Constraint(P1, K, H1, rule=lambda m, j, k, t: m.Ebk[j, k, t] <= M * m.u[k])
    m.eq23_max_capacity_n = pyo.Constraint(N, G, rule=lambda m, n, g: m.Pn[n, g] <= pmax[n, g] * m.uh[n])
    m.eq24_storage_open_n = pyo.Constraint(N, rule=lambda m, n: m.Bcn[n] <= M * m.uh[n])
    m.eq25_sale_open_n = pyo.Constraint(P1, N, H1, rule=lambda m, j, n, t: m.Esn[j, n, t] <= M * m.uh[n])
    m.eq26_purchase_open_n = pyo.Constraint(P1, N, H1, rule=lambda m, j, n, t: m.Ebn[j, n, t] <= M * m.uh[n])
    # (27)-(28): with (19) and (24), B^c u = B^c at every feasible point, so the bound is written linearly
    m.eq27_storage_k = pyo.Constraint(P1, K, H1, rule=lambda m, j, k, t: m.Bk[j, k, t] <= m.Bck[k])
    m.eq28_storage_n = pyo.Constraint(P1, N, H1, rule=lambda m, j, n, t: m.Bn[j, n, t] <= m.Bcn[n])
    # (29)-(30): storage full at the start and at the end of the horizon; thesis (4.85)-(4.86): periods are consecutive
    m.eq29_start_k = pyo.Constraint(K, rule=lambda m, k: m.Bk[1, k, 0] == m.Bck[k])
    m.eq29_end_k = pyo.Constraint(K, rule=lambda m, k: m.Bk[J, k, T] == m.Bck[k])
    m.eq30_start_n = pyo.Constraint(N, rule=lambda m, n: m.Bn[1, n, 0] == m.Bcn[n])
    m.eq30_end_n = pyo.Constraint(N, rule=lambda m, n: m.Bn[J, n, T] == m.Bcn[n])
    m.eq4_85_carry_k = pyo.Constraint(range(2, J + 1), K, rule=lambda m, j, k: m.Bk[j, k, 0] == m.Bk[j - 1, k, T])
    m.eq4_86_carry_n = pyo.Constraint(range(2, J + 1), N, rule=lambda m, j, n: m.Bn[j, n, 0] == m.Bn[j - 1, n, T])
    return m


# --------------------------------------------------------------------- capacity-factor scenarios (Ch. 5)

def scale_cf(inst: SupplyChainInstance, factor: float) -> SupplyChainInstance:
    """A copy of ``inst`` with every capacity factor multiplied by ``factor`` and capped at 1."""
    if not (np.isfinite(factor) and factor >= 0):
        raise ValueError(f"cf scale factor must be finite and non-negative, got {factor}")
    return replace(inst, cf={k: np.minimum(v * factor, 1.0) for k, v in inst.cf.items()})


def with_cf(inst: SupplyChainInstance, cf: Mapping) -> SupplyChainInstance:
    """A copy of ``inst`` with the capacity factors replaced by ``cf`` (same keys and shapes)."""
    out = replace(inst, cf={k: np.asarray(v, dtype=float) for k, v in cf.items()})
    return out.validate()


def expected_cf(scenarios: Sequence[Mapping], probabilities: Sequence[float]) -> Dict:
    """sum_w beta_w lambda^w: the scenario-weighted capacity factors of thesis Model 5.1 (Eqs. 5.1-5.2)."""
    if len(scenarios) != len(probabilities) or not scenarios:
        raise ValueError("one probability per scenario is needed")
    probs = [float(x) for x in probabilities]
    if any((not np.isfinite(x)) or x < 0 for x in probs) or abs(sum(probs) - 1.0) > 1e-9:
        raise ValueError(f"scenario probabilities must be non-negative and sum to 1, got {probs}")
    keys = list(scenarios[0])
    if any(set(sc) != set(keys) for sc in scenarios):
        raise ValueError("all scenarios need the same capacity-factor keys")
    arrays = []
    for w, sc in enumerate(scenarios):   # check every scenario before averaging (averages can hide errors)
        arr = {k: np.asarray(sc[k], dtype=float) for k in keys}
        for k, a in arr.items():
            if a.shape != np.asarray(scenarios[0][k]).shape:
                raise ValueError(f"scenario {w}: cf[{k}] has shape {a.shape}, scenario 0 has {np.asarray(scenarios[0][k]).shape}")
            if not np.all(np.isfinite(a)) or np.any(a < 0) or np.any(a > 1):
                raise ValueError(f"scenario {w}: cf[{k}] must be finite and within [0, 1]")
        arrays.append(arr)
    return {k: sum(p * arr[k] for p, arr in zip(probs, arrays)) for k in keys}


# ----------------------------------------------------------------------------------------------- results

def fmla_summary(m: pyo.ConcreteModel, inst: SupplyChainInstance) -> Dict[str, object]:
    """Cost components, open sites, capacities and energy traded of a solved model.

    Components follow the terms of AMM Eq. (1) and add up to the objective.
    """
    val = pyo.value
    K, N, S, I = inst.factories, inst.warehouses, inst.stores, inst.products
    G, J, T, W = list(inst.technologies), inst.periods, inst.hours, inst.horizon_weight
    tech = inst.technologies
    P1, H1 = range(1, J + 1), range(1, T + 1)
    opened = {**{k: val(m.u[k]) > 0.5 for k in K}, **{n: val(m.uh[n]) > 0.5 for n in N}}

    def cap_mw(x, g):
        return val(m.Pk[x, g]) if x in K else val(m.Pn[x, g])

    def ess(x):
        return val(m.Bck[x]) if x in K else val(m.Bcn[x])

    def trade(x, kind):
        if x in K:
            var = m.Ebk if kind == "buy" else m.Esk
        else:
            var = m.Ebn if kind == "buy" else m.Esn
        return var, (inst.buy_price_usd_per_mwh if kind == "buy" else inst.sell_price_usd_per_mwh)[x]

    comp = {
        "factories_usd": sum(inst.factory_fixed_cost_usd[k] * val(m.u[k]) for k in K),
        "warehouses_usd": sum(inst.warehouse_fixed_cost_usd[n] * val(m.uh[n]) for n in N),
        "production_usd": W * sum(inst.production_cost_usd_per_item[i, k] * val(m.x[i, j, k, n])
                                  for i in I for j in P1 for k in K for n in N),
        "transport_usd": W * (sum(inst.ship_cost_fw_usd_per_item_km[i] * inst.distance_fw_km[k, n] * val(m.x[i, j, k, n])
                                  for i in I for j in P1 for k in K for n in N)
                              + sum(inst.ship_cost_ws_usd_per_item_km[i] * inst.distance_ws_km[n, s] * val(m.xh[i, j, n, s])
                                    for i in I for j in P1 for n in N for s in S)),
        "holding_usd": W * sum(inst.holding_cost_usd_per_item_period[i] * val(m.y[i, j, n]) for i in I for j in P1 for n in N),
        "backorder_usd": W * sum(inst.backorder_cost_usd_per_item_period[i] * val(m.z[i, j, n])
                                 for i in I for j in range(1, J) for n in N),
        "generation_capacity_usd": sum(tech[g].capital_recovery_factor * tech[g].capacity_cost_usd_per_mw * cap_mw(x, g)
                                       for x in inst.sites for g in G if opened[x]),
        "generation_om_net_of_credits_usd": W * sum(
            tech[g].operating_hours_per_step * (tech[g].om_cost_usd_per_mwh - tech[g].carbon_credit_usd_per_mwh)
            * float(inst.cf[x, g].sum()) * cap_mw(x, g) for x in inst.sites for g in G if opened[x]),
        "storage_usd": inst.storage.annual_cost_usd_per_mwh * sum(ess(x) for x in inst.sites if opened[x]),
        "electricity_bought_usd": 0.0,
        "electricity_sold_usd": 0.0,
    }
    bought = sold = 0.0
    for x in inst.sites:
        if not opened[x]:
            continue
        for kind in ("buy", "sell"):
            var, price = trade(x, kind)
            mwh = sum(val(var[j, x, t]) for j in P1 for t in H1)
            usd = sum(price[j - 1, t - 1] * val(var[j, x, t]) for j in P1 for t in H1)
            if kind == "buy":
                comp["electricity_bought_usd"] += W * usd
                bought += W * mwh
            else:
                comp["electricity_sold_usd"] -= W * usd
                sold += W * mwh
    return {
        "total_cost_usd": val(m.cost),
        "components_usd": comp,
        "open_factories": [k for k in K if opened[k]],
        "open_warehouses": [n for n in N if opened[n]],
        "store_assignment": {(i, s): n for i in I for s in S for n in N if val(m.vh[i, n, s]) > 0.5},
        # a closed site has no microgrid (its capacities are forced to zero by Eqs. 18-19 and 23-24)
        "capacity_mw": {(x, g): (cap_mw(x, g) if opened[x] else 0.0) for x in inst.sites for g in G},
        "storage_mwh": {x: (ess(x) if opened[x] else 0.0) for x in inst.sites},
        "energy_bought_mwh": bought,
        "energy_sold_mwh": sold,
        "production_items": {(i, j, k): sum(val(m.x[i, j, k, n]) for n in N) for i in I for j in P1 for k in K},
        "inventory_items": {(i, j, n): val(m.y[i, j, n]) for i in I for j in P1 for n in N},
        "backorder_items": {(i, j, n): val(m.z[i, j, n]) for i in I for j in P1 for n in N},
    }
