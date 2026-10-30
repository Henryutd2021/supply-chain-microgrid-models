from itertools import product

import numpy as np
import pyomo.environ as pyo
import pytest

from conftest import needs_gurobi
from sc_microgrid import (build_fixed_charge, build_wmla, feasible_open_sets, fixed_charge_summary, solve,
                          solve_enumerated, synthetic_fixed_charge, synthetic_warehouses, wmla_summary)


def brute_force_fixed_charge(inst):
    best = np.inf
    for assign in product(inst.sites, repeat=len(inst.stores)):
        load = {n: 0.0 for n in inst.sites}
        cost = 0.0
        for s, n in zip(inst.stores, assign):
            load[n] += inst.demand_items[s]
            cost += inst.demand_items[s] * inst.ship_cost_usd_per_item_km[n, s] * inst.distance_km[n, s]
        if any(load[n] > inst.capacity_items[n] for n in inst.sites):
            continue
        cost += sum(inst.fixed_cost_usd[n] for n in inst.sites if load[n] > 0)
        best = min(best, cost)
    return best


def test_fixed_charge_matches_brute_force():
    inst = synthetic_fixed_charge()
    m = build_fixed_charge(inst)
    res = solve(m, mip_gap=0)
    assert res.objective == pytest.approx(brute_force_fixed_charge(inst), rel=1e-9)
    s = fixed_charge_summary(m, inst)
    assert set(s["assignment"]) == set(inst.stores)
    assert set(s["assignment"].values()) <= set(s["open_sites"])


@pytest.fixture(scope="module")
def wmla_case():
    inst = synthetic_warehouses(hours=24)
    mb = build_wmla(inst, "big_m")
    rb = solve(mb)
    me, re_, best, table = solve_enumerated(inst)
    return inst, mb, rb, me, re_, best, table


def balance_residual(m, inst):
    """Largest residual of the hourly energy balance at open sites, recomputed from the solution."""
    val = pyo.value
    T, G = inst.hours, list(inst.technologies)
    worst = 0.0
    for n in inst.sites:
        if val(m.u[n]) < 0.5:
            continue
        tr = (inst.product_weight_kg * inst.truck.energy_mwh_per_kg_km
              * sum(inst.distance_km[n, s] * inst.demand_items[s] * val(m.v[n, s]) for s in inst.stores) / T
              + 2 * inst.truck.weight_kg * inst.truck.energy_mwh_per_kg_km
              * sum(inst.distance_km[n, s] * val(m.v[n, s]) for s in inst.stores) / T)
        for t in range(1, T + 1):
            lhs = (inst.operating_share[n][t - 1] * inst.site_load_mw[n] + tr + val(m.B[n, t]) - val(m.B[n, t - 1])
                   + val(m.Ese[n, t]) - val(m.Ebu[n, t]))
            rhs = sum(inst.technologies[g].operating_hours_per_step * inst.cf[n, g][t - 1] * val(m.P[n, g]) for g in G)
            worst = max(worst, abs(lhs - rhs))
    return worst


def test_wmla_big_m_equals_enumeration(wmla_case):
    inst, mb, rb, me, re_, best, table = wmla_case
    assert rb.objective == pytest.approx(re_.objective, rel=1e-6)
    assert set(wmla_summary(mb, inst)["open_sites"]) == set(best)
    assert len(table) == len(feasible_open_sets(inst))
    assert min(r["total_cost_usd"] for r in table) == pytest.approx(re_.objective)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_wmla_forms_agree_on_other_synthetic_cases(seed):
    inst = synthetic_warehouses(hours=24, seed=seed)
    rb = solve(build_wmla(inst, "big_m"))
    _, re_, best, _ = solve_enumerated(inst)
    assert rb.objective == pytest.approx(re_.objective, rel=1e-6)


def test_wmla_solution_properties(wmla_case):
    inst, mb, rb, me, re_, best, table = wmla_case
    for m in (mb, me):
        s = wmla_summary(m, inst)
        assert sum(s["components_usd"].values()) == pytest.approx(pyo.value(m.cost), rel=1e-9)
        assert balance_residual(m, inst) < 1e-6
        for n in inst.sites:
            assert pyo.value(m.B[n, 0]) == pytest.approx(pyo.value(m.Bc[n]), abs=1e-6)
            assert pyo.value(m.B[n, inst.hours]) == pytest.approx(pyo.value(m.Bc[n]), abs=1e-6)
            if n not in s["open_sites"]:
                assert all(abs(pyo.value(m.P[n, g])) < 1e-6 for g in inst.technologies)
        load = {n: sum(inst.demand_items[st] for st, w in s["assignment"].items() if w == n) for n in inst.sites}
        assert all(load[n] <= inst.capacity_items[n] + 1e-6 for n in inst.sites)


def test_wmla_rejects_unknown_site():
    with pytest.raises(ValueError, match="unknown sites"):
        build_wmla(synthetic_warehouses(hours=24), "bilinear", open_sites=["nowhere"])
    with pytest.raises(ValueError, match="unknown sites"):
        build_wmla(synthetic_warehouses(hours=24), "bilinear", open_sites=iter(["W-North", "nowhere"]))


def test_open_sites_iterator(wmla_case):
    inst = wmla_case[0]
    m = build_wmla(inst, "bilinear", open_sites=iter(["W-Central", "W-South"]))
    assert [n for n in inst.sites if pyo.value(m.u[n]) > 0.5] == ["W-Central", "W-South"]


def test_enumeration_stops_on_incomplete_subproblem(monkeypatch):
    import sc_microgrid.wmla as wmla
    from sc_microgrid.solve import SolveError

    def fake_solve(m, **kw):
        raise SolveError("solver highs returned maxTimeLimit", "maxTimeLimit")
    monkeypatch.setattr(wmla, "solve", fake_solve)
    with pytest.raises(SolveError, match="enumeration incomplete"):
        wmla.solve_enumerated(synthetic_warehouses(hours=24))
    with pytest.raises(ValueError, match="require_optimal"):
        wmla.solve_enumerated(synthetic_warehouses(hours=24), require_optimal=False)


@needs_gurobi
def test_wmla_bilinear_direct_with_gurobi(wmla_case):
    inst, mb, rb, me, re_, best, table = wmla_case
    m = build_wmla(inst, "bilinear")
    res = solve(m, solver="gurobi", options={"NonConvex": 2})
    assert res.objective == pytest.approx(re_.objective, rel=1e-6)
