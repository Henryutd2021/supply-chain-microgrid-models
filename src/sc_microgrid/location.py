"""Classical capacitated fixed-charge location with single sourcing (thesis Model 4.1).

    min   sum_n F_n u_n + sum_s sum_n q_s c_ns d_ns v_ns                       (4.1)
    s.t.  sum_n v_ns = 1                                     for all s          (4.2)
          sum_s q_s v_ns <= gamma_n u_n                      for all n          (4.3)
          v_ns <= u_n                                        for all n, s       (4.4)
          u_n, v_ns binary                                                       (4.5)-(4.6)
"""

from __future__ import annotations

import pyomo.environ as pyo

from .instances import FixedChargeInstance


def build_fixed_charge(inst: FixedChargeInstance) -> pyo.ConcreteModel:
    inst.validate()
    N, S = inst.sites, inst.stores
    F, cap, q = inst.fixed_cost_usd, inst.capacity_items, inst.demand_items
    c, d = inst.ship_cost_usd_per_item_km, inst.distance_km

    m = pyo.ConcreteModel("fixed_charge")
    m.u = pyo.Var(N, domain=pyo.Binary)
    m.v = pyo.Var(N, S, domain=pyo.Binary)
    m.cost = pyo.Objective(expr=sum(F[n] * m.u[n] for n in N)
                           + sum(q[s] * c[n, s] * d[n, s] * m.v[n, s] for n in N for s in S))
    m.eq4_2_single_source = pyo.Constraint(S, rule=lambda m, s: sum(m.v[n, s] for n in N) == 1)
    m.eq4_3_capacity = pyo.Constraint(N, rule=lambda m, n: sum(q[s] * m.v[n, s] for s in S) <= cap[n] * m.u[n])
    m.eq4_4_open_site = pyo.Constraint(N, S, rule=lambda m, n, s: m.v[n, s] <= m.u[n])
    return m


def fixed_charge_summary(m: pyo.ConcreteModel, inst: FixedChargeInstance) -> dict:
    """Total cost, open sites and the store-to-site assignment of a solved model."""
    val = pyo.value
    return {
        "total_cost_usd": val(m.cost),
        "fixed_cost_usd": sum(inst.fixed_cost_usd[n] * val(m.u[n]) for n in inst.sites),
        "open_sites": [n for n in inst.sites if val(m.u[n]) > 0.5],
        "assignment": {s: n for n in inst.sites for s in inst.stores if val(m.v[n, s]) > 0.5},
    }
