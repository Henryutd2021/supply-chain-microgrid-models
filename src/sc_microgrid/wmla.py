"""Warehouse and microgrid location-allocation (WMLA) in a two-layer network.

Two formulations of the same decision problem (docs/formulation.md, Part B):

``"bilinear"``  thesis Model 4.2, Eqs. (4.7)-(4.19): the objective and the
                hourly energy balance multiply the microgrid and trade terms by the location
                binary u^_n. With the location binaries fixed, every product has a constant factor
                and the model is a MILP; :func:`solve_enumerated` solves the model exactly that
                way, one open-warehouse set at a time. Without fixed binaries the model has
                products of variables and needs a solver for non-convex quadratic problems (Gurobi).
``"big_m"``     thesis Model 4.4, Eqs. (4.48)-(4.63): the balance is linear and the microgrid
                variables of a closed site are forced to zero by big-M bounds. A MILP.

SEGRE 2023 presents the same two-tier model. Index sets: sites n (candidate warehouses), stores s,
technologies g, hours t = 1..T.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Dict, Iterable, Optional, Sequence

import pyomo.environ as pyo

from .instances import WarehouseInstance
from .solve import INFEASIBLE, SolveError, SolveResult, solve

FORMULATIONS = ("bilinear", "big_m")


def build_wmla(inst: WarehouseInstance, formulation: str = "big_m",
               open_sites: Optional[Iterable[str]] = None) -> pyo.ConcreteModel:
    """Build the WMLA model.

    ``open_sites`` fixes the location decision u^_n (1 for the listed sites, 0 otherwise). With the
    bilinear formulation this turns every product into a linear term; the microgrid variables of a
    closed site are then fixed at zero (docs/formulation.md, conventions).
    """
    if formulation not in FORMULATIONS:
        raise ValueError(f"formulation must be one of {FORMULATIONS}, got {formulation!r}")
    inst.validate()
    N, S, G = inst.sites, inst.stores, list(inst.technologies)
    T = inst.hours
    H1, H0 = range(1, T + 1), range(0, T + 1)
    tech, es, truck = inst.technologies, inst.storage, inst.truck
    W, M = inst.horizon_weight, inst.big_m
    fixed = None
    if open_sites is not None:
        chosen = set(open_sites)           # materialize once (open_sites may be an iterator)
        unknown = chosen - set(N)
        if unknown:
            raise ValueError(f"open_sites contains unknown sites {sorted(unknown)}")
        fixed = {n: (1 if n in chosen else 0) for n in N}

    m = pyo.ConcreteModel(f"wmla_{formulation}")
    m.bilinear = formulation == "bilinear" and fixed is None   # products of variables remain
    m.u = pyo.Var(N, domain=pyo.Binary)                       # u^_n
    m.v = pyo.Var(N, S, domain=pyo.Binary)                    # v^_ns
    m.P = pyo.Var(N, G, domain=pyo.NonNegativeReals)          # P^c_gn (MW)
    m.Bc = pyo.Var(N, domain=pyo.NonNegativeReals)            # B^c_n (MWh)
    m.B = pyo.Var(N, H0, domain=pyo.NonNegativeReals)         # B_nt (MWh)
    m.Ese = pyo.Var(N, H1, domain=pyo.NonNegativeReals)       # E^se_nt (MWh)
    m.Ebu = pyo.Var(N, H1, domain=pyo.NonNegativeReals)       # E^bu_nt (MWh)
    if fixed is not None:
        for n in N:
            m.u[n].fix(fixed[n])
            if fixed[n] == 0:   # a closed warehouse has no microgrid
                for g in G:
                    m.P[n, g].fix(0)
                m.Bc[n].fix(0)
                for t in H0:
                    m.B[n, t].fix(0)
                for t in H1:
                    m.Ese[n, t].fix(0)
                    m.Ebu[n, t].fix(0)

    def u_of(n):   # the location binary, or its fixed value
        return fixed[n] if fixed is not None else m.u[n]

    if formulation == "bilinear":   # thesis Eq. (4.7): microgrid and trade terms times u^_n
        def site(expr, n):
            return expr * u_of(n)
    else:                           # thesis Eq. (4.48)
        def site(expr, n):
            return expr

    cf_sum = {(n, g): float(inst.cf[n, g].sum()) for n in N for g in G}
    rb, rs = inst.buy_price_usd_per_mwh, inst.sell_price_usd_per_mwh
    m.cost = pyo.Objective(expr=(
        sum(inst.fixed_cost_usd[n] * m.u[n] for n in N)
        + W * sum(inst.demand_items[s] * inst.ship_cost_usd_per_item_km[n, s] * inst.distance_km[n, s] * m.v[n, s]
                  for n in N for s in S)
        + sum(site(sum(tech[g].capital_recovery_factor * tech[g].capacity_cost_usd_per_mw * m.P[n, g]
                       + W * tech[g].operating_hours_per_step
                       * (tech[g].om_cost_usd_per_mwh - tech[g].carbon_credit_usd_per_mwh) * cf_sum[n, g] * m.P[n, g]
                       for g in G)
                   + es.annual_cost_usd_per_mwh * m.Bc[n]
                   + W * sum(rb[n][t - 1] * m.Ebu[n, t] - rs[n][t - 1] * m.Ese[n, t] for t in H1), n)
              for n in N)))

    # location: thesis (4.8)-(4.10) = (4.49)-(4.51)
    m.eq4_8_single_source = pyo.Constraint(S, rule=lambda m, s: sum(m.v[n, s] for n in N) == 1)
    m.eq4_9_capacity = pyo.Constraint(N, rule=lambda m, n:
                                    sum(inst.demand_items[s] * m.v[n, s] for s in S) <= inst.capacity_items[n] * m.u[n])
    m.eq4_10_open_site = pyo.Constraint(N, S, rule=lambda m, n, s: m.v[n, s] <= m.u[n])

    # hourly energy balance: thesis (4.11) (bilinear) or (4.52) (big-M)
    def transport(n):   # MWh per hour: product weight and empty truck round trip, spread over T hours
        return (inst.product_weight_kg * truck.energy_mwh_per_kg_km
                * sum(inst.distance_km[n, s] * inst.demand_items[s] * m.v[n, s] for s in S) / T
                + 2 * truck.weight_kg * truck.energy_mwh_per_kg_km * sum(inst.distance_km[n, s] * m.v[n, s] for s in S) / T)

    tr = {n: transport(n) for n in N}
    share = inst.operating_share

    def gen(n, t):
        return sum(tech[g].operating_hours_per_step * inst.cf[n, g][t - 1] * m.P[n, g] for g in G)

    if formulation == "bilinear":
        def balance(m, n, t):
            u = u_of(n)
            return (share[n][t - 1] * inst.site_load_mw[n] * m.u[n] + tr[n]
                    + (m.B[n, t] - m.B[n, t - 1] + m.Ese[n, t] - m.Ebu[n, t]) * u == gen(n, t) * u)
        m.eq4_11_energy_balance = pyo.Constraint(N, H1, rule=balance)
    else:
        m.eq4_52_energy_balance = pyo.Constraint(N, H1, rule=lambda m, n, t:
                                                 share[n][t - 1] * inst.site_load_mw[n] * m.u[n] + tr[n]
                                                 + m.B[n, t] - m.B[n, t - 1] + m.Ese[n, t] - m.Ebu[n, t] == gen(n, t))
        m.eq4_53_generation_open = pyo.Constraint(N, G, rule=lambda m, n, g: m.P[n, g] <= M * m.u[n])
        m.eq4_54_storage_open = pyo.Constraint(N, rule=lambda m, n: m.Bc[n] <= M * m.u[n])
        m.eq4_55_sale_open = pyo.Constraint(N, H1, rule=lambda m, n, t: m.Ese[n, t] <= M * m.u[n])

    # microgrid limits: thesis (4.12)-(4.14) = (4.56)-(4.58)
    m.eq4_12_max_capacity = pyo.Constraint(N, G, rule=lambda m, n, g: m.P[n, g] <= inst.max_capacity_mw[n, g])
    m.eq4_13_storage_level = pyo.Constraint(N, H1, rule=lambda m, n, t: m.B[n, t] <= m.Bc[n])
    m.eq4_14_storage_start = pyo.Constraint(N, rule=lambda m, n: m.B[n, 0] == m.Bc[n])
    m.eq4_14_storage_end = pyo.Constraint(N, rule=lambda m, n: m.B[n, T] == m.Bc[n])
    return m


def feasible_open_sets(inst: WarehouseInstance) -> list:
    """Every non-empty set of sites whose total capacity covers total demand."""
    inst.validate()
    need = sum(inst.demand_items.values())
    out = []
    for r in range(1, len(inst.sites) + 1):
        for subset in combinations(inst.sites, r):
            if sum(inst.capacity_items[n] for n in subset) >= need:
                out.append(subset)
    return out


def solve_enumerated(inst: WarehouseInstance, solver: str = "highs", mip_gap: float = 1e-6,
                     open_sets: Optional[Sequence[Sequence[str]]] = None, **kwargs):
    """Solve the bilinear WMLA exactly by fixing each open-warehouse set in turn.

    Returns ``(model, result, open_set, table)`` for the cheapest set; ``table`` lists every set
    tried with its optimal cost, or ``nan`` and the status for a set proven infeasible. Any other
    failure (time limit, numerical trouble) raises :class:`SolveError`, because the enumeration
    would then be incomplete. For N candidate sites there are at most 2^N - 1 sets, so this
    suits small candidate sets.
    """
    if kwargs.get("require_optimal", True) is False:
        raise ValueError("solve_enumerated needs optimal subproblems; require_optimal=False is not allowed")
    kwargs.pop("require_optimal", None)
    sets = [tuple(s) for s in (open_sets if open_sets is not None else feasible_open_sets(inst))]
    if not sets:
        raise SolveError("no set of open sites has enough capacity for the total demand")
    best, table = None, []
    for subset in sets:
        m = build_wmla(inst, "bilinear", open_sites=subset)
        try:
            res = solve(m, solver=solver, mip_gap=mip_gap, **kwargs)
        except SolveError as exc:
            if exc.status not in INFEASIBLE:
                raise SolveError(f"enumeration incomplete: open set {subset}: {exc}", exc.status) from None
            table.append({"open_sites": subset, "total_cost_usd": float("nan"), "status": exc.status})
            continue
        if not math.isfinite(res.objective):   # defensive: solve() returns only optimal results here
            raise SolveError(f"enumeration incomplete: open set {subset}: no finite objective", res.status)
        table.append({"open_sites": subset, "total_cost_usd": res.objective, "status": res.status})
        if best is None or res.objective < best[1].objective:
            best = (m, res, subset)
    if best is None:
        raise SolveError("every open-warehouse set is infeasible", "provenInfeasible")
    return best[0], best[1], best[2], table


def wmla_summary(m: pyo.ConcreteModel, inst: WarehouseInstance) -> Dict[str, object]:
    """Cost components, open sites, assignment, capacities and energy traded of a solved model.

    Microgrid and trade quantities are reported for open sites only (a closed site has no
    microgrid); the components then add up to the objective of either formulation.
    """
    val = pyo.value
    N, S, G = inst.sites, inst.stores, list(inst.technologies)
    T, W = inst.hours, inst.horizon_weight
    tech = inst.technologies
    H1 = range(1, T + 1)
    op = [n for n in N if val(m.u[n]) > 0.5]
    comp = {
        "warehouses_usd": sum(inst.fixed_cost_usd[n] * val(m.u[n]) for n in N),
        "transport_usd": W * sum(inst.demand_items[s] * inst.ship_cost_usd_per_item_km[n, s] * inst.distance_km[n, s]
                                 * val(m.v[n, s]) for n in N for s in S),
        "generation_capacity_usd": sum(tech[g].capital_recovery_factor * tech[g].capacity_cost_usd_per_mw * val(m.P[n, g])
                                       for n in op for g in G),
        "generation_om_net_of_credits_usd": W * sum(
            tech[g].operating_hours_per_step * (tech[g].om_cost_usd_per_mwh - tech[g].carbon_credit_usd_per_mwh)
            * float(inst.cf[n, g].sum()) * val(m.P[n, g]) for n in op for g in G),
        "storage_usd": inst.storage.annual_cost_usd_per_mwh * sum(val(m.Bc[n]) for n in op),
        "electricity_bought_usd": W * sum(inst.buy_price_usd_per_mwh[n][t - 1] * val(m.Ebu[n, t]) for n in op for t in H1),
        "electricity_sold_usd": -W * sum(inst.sell_price_usd_per_mwh[n][t - 1] * val(m.Ese[n, t]) for n in op for t in H1),
    }
    return {"total_cost_usd": val(m.cost), "components_usd": comp, "open_sites": op,
            "assignment": {s: n for n in N for s in S if val(m.v[n, s]) > 0.5},
            "capacity_mw": {(n, g): (val(m.P[n, g]) if n in op else 0.0) for n in N for g in G},
            "storage_mwh": {n: (val(m.Bc[n]) if n in op else 0.0) for n in N},
            "energy_bought_mwh": W * sum(val(m.Ebu[n, t]) for n in op for t in H1),
            "energy_sold_mwh": W * sum(val(m.Ese[n, t]) for n in op for t in H1)}
