import numpy as np
import pyomo.environ as pyo
import pytest

from sc_microgrid import (build_fmla, expected_cf, fmla_summary, scale_cf, solve, synthetic_cf_scenarios,
                          synthetic_supply_chain, with_cf, z_value)


@pytest.fixture(scope="module")
def solved():
    inst = synthetic_supply_chain(periods=3, hours=12)
    m = build_fmla(inst)
    res = solve(m)
    return inst, m, res


def residuals(m, inst):
    """Largest residuals of the factory and warehouse energy balances (AMM Eqs. 17 and 22)."""
    val = pyo.value
    K, N, S, I, G = inst.factories, inst.warehouses, inst.stores, inst.products, list(inst.technologies)
    J, T = inst.periods, inst.hours
    q, mev = inst.truck.energy_mwh_per_kg_km, inst.truck.weight_kg
    worst = 0.0
    for j in range(1, J + 1):
        for k in K:
            load = (inst.factory_load_mw[k] * val(m.u[k])
                    + sum((inst.energy_mwh_per_item[i, k] + q * inst.distance_fw_km[k, n] * inst.product_weight_kg[i])
                          * val(m.x[i, j, k, n]) for i in I for n in N) / T
                    + q * mev * sum(inst.trips_fw_per_hour[k, n] * inst.distance_fw_km[k, n] * val(m.v[i, j, k, n])
                                    for i in I for n in N))
            for t in range(1, T + 1):
                gen = sum(inst.technologies[g].operating_hours_per_step * inst.cf[k, g][j - 1, t - 1] * val(m.Pk[k, g])
                          for g in G)
                lhs = load + val(m.Bk[j, k, t]) - val(m.Bk[j, k, t - 1]) + val(m.Esk[j, k, t]) - val(m.Ebk[j, k, t])
                worst = max(worst, abs(lhs - gen))
        for n in N:
            load = (inst.warehouse_load_mw[n] * val(m.uh[n])
                    + q * mev * sum(inst.trips_fw_per_hour[k, n] * inst.distance_fw_km[k, n] * val(m.v[i, j, k, n])
                                    for i in I for k in K)
                    + q * sum(inst.distance_ws_km[n, s] * inst.product_weight_kg[i] * val(m.xh[i, j, n, s])
                              for i in I for s in S) / T
                    + 2 * q * mev * sum(inst.trips_ws_per_hour[n, s] * inst.distance_ws_km[n, s] * val(m.vh[i, n, s])
                                        for i in I for s in S))
            for t in range(1, T + 1):
                gen = sum(inst.technologies[g].operating_hours_per_step * inst.cf[n, g][j - 1, t - 1] * val(m.Pn[n, g])
                          for g in G)
                lhs = load + val(m.Bn[j, n, t]) - val(m.Bn[j, n, t - 1]) + val(m.Esn[j, n, t]) - val(m.Ebn[j, n, t])
                worst = max(worst, abs(lhs - gen))
    return worst


def test_fmla_solution(solved):
    inst, m, res = solved
    val = pyo.value
    s = fmla_summary(m, inst)
    assert sum(s["components_usd"].values()) == pytest.approx(res.objective, rel=1e-9)
    assert residuals(m, inst) < 1e-6
    z = z_value(inst.service_level)
    for (i, j, st), mu in inst.demand_mean_items.items():
        served = sum(val(m.xh[i, j, n, st]) for n in inst.warehouses)
        assert served >= mu + z * inst.demand_std_items[i, j, st] - 1e-6
    for v in m.x.values():
        assert abs(val(v) - round(val(v))) < 1e-6
    for i in inst.products:
        for n in inst.warehouses:
            assert m.z[i, inst.periods, n].fixed and val(m.z[i, inst.periods, n]) == 0
    for x in inst.sites:
        Bc = val(m.Bck[x]) if x in inst.factories else val(m.Bcn[x])
        B = m.Bk if x in inst.factories else m.Bn
        assert val(B[1, x, 0]) == pytest.approx(Bc, abs=1e-6)
        assert val(B[inst.periods, x, inst.hours]) == pytest.approx(Bc, abs=1e-6)
    for k in inst.factories:
        if k not in s["open_factories"]:
            assert abs(sum(s["production_items"][i, j, k] for i in inst.products for j in range(1, inst.periods + 1))) < 1e-6


def test_balances_with_partial_operating_hours():
    inst = synthetic_supply_chain(periods=2, hours=12)
    inst.technologies["pv"].operating_hours_per_step = 0.5
    m = build_fmla(inst)
    solve(m)
    assert residuals(m, inst) < 1e-6


def test_options_are_keyword_only():
    inst = synthetic_supply_chain(periods=2, hours=6)
    with pytest.raises(TypeError):
        build_fmla(inst, "big_m")
    with pytest.raises(TypeError):
        build_fmla(inst, False)


def test_no_microgrid_costs_at_least_as_much(solved):
    inst, m, res = solved
    m0 = build_fmla(inst, no_microgrid=True)
    r0 = solve(m0)
    assert r0.objective >= res.objective - 1e-6 * abs(res.objective)
    assert all(v == 0 for v in fmla_summary(m0, inst)["capacity_mw"].values())


def test_relaxed_flows_bound_the_integer_model(solved):
    inst, m, res = solved
    r = solve(build_fmla(inst, integer_flows=False))
    assert r.objective <= res.objective * (1 + 1e-6)


def test_cf_scenario_helpers():
    inst = synthetic_supply_chain(periods=2, hours=6)
    hi = scale_cf(inst, 10.0)
    assert all(np.all(v <= 1.0) for v in hi.cf.values())
    scale, prob = synthetic_cf_scenarios()
    scen = [scale_cf(inst, scale[k]).cf for k in scale]
    exp = expected_cf(scen, [prob[k] for k in scale])
    key = ("F-East", "pv")
    assert np.allclose(exp[key], sum(prob[k] * scale_cf(inst, scale[k]).cf[key] for k in scale))
    assert with_cf(inst, exp).cf[key].shape == (2, 6)
    with pytest.raises(ValueError, match="sum to 1"):
        expected_cf(scen, [0.5, 0.5, 0.5])
    bad = [dict(scen[0]), dict(scen[1])]
    bad[0][key] = scen[0][key] - 0.5                       # an average could hide this
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        expected_cf(bad, [0.5, 0.5])
    bad = [dict(scen[0]), dict(scen[1])]
    bad[1][key] = scen[1][key][0]                          # would broadcast silently
    with pytest.raises(ValueError, match="shape"):
        expected_cf(bad, [0.5, 0.5])


def test_big_m_below_capacity_bound_is_refused():
    inst = synthetic_supply_chain(periods=2, hours=6)
    inst.big_m = 1.0
    with pytest.raises(ValueError, match="big_m"):
        build_fmla(inst)
