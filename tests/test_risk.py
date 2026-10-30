import pyomo.environ as pyo
import pytest

from sc_microgrid import (cvar_model, cvar_value, expected_value, minimax_model, minimax_value, risk_table,
                          scenario_costs, solve, synthetic_cf_scenarios, synthetic_supply_chain, var_model, var_value)

COST = {"a": 100.0, "b": 140.0, "c": 115.0, "d": 180.0}
PROB = {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}


@pytest.mark.parametrize("phi", [0.05, 0.5, 0.65, 0.8, 0.9, 0.95])
def test_models_match_closed_forms(phi):
    mv, mc, mm = var_model(COST, PROB, phi), cvar_model(COST, PROB, phi), minimax_model(COST)
    solve(mv, mip_gap=0)
    solve(mc)
    solve(mm)
    assert pyo.value(mv.VaR) == pytest.approx(var_value(COST, PROB, phi))
    assert pyo.value(mc.obj) == pytest.approx(cvar_value(COST, PROB, phi))
    assert pyo.value(mm.M) == pytest.approx(minimax_value(COST)) == 180.0
    assert var_value(COST, PROB, phi) <= cvar_value(COST, PROB, phi) + 1e-9 <= 180.0 + 1e-9


def test_known_values():
    assert var_value(COST, PROB, 0.5) == 115.0           # P(C > 115) = 0.4 <= 0.5
    assert var_value(COST, PROB, 0.9) == 140.0           # P(C > 140) = 0.1 <= 0.1
    assert cvar_value(COST, PROB, 0.9) == pytest.approx(180.0)
    assert expected_value(COST, PROB) == pytest.approx(0.4 * 100 + 0.3 * 140 + 0.2 * 115 + 0.1 * 180)
    rows = risk_table(COST, PROB, [0.5, 0.9])
    assert [r["VaR"] for r in rows] == pytest.approx([115.0, 140.0])


def test_invalid_inputs():
    with pytest.raises(ValueError, match="strictly"):
        var_value(COST, PROB, 1.0)
    with pytest.raises(ValueError, match="sum to 1"):
        cvar_value(COST, {k: 0.5 for k in COST}, 0.9)
    with pytest.raises(ValueError, match="same scenarios"):
        expected_value(COST, {"a": 1.0})
    with pytest.raises(ValueError, match="cost range"):
        var_model(COST, PROB, 0.5, big_m=1.0)


def test_scenario_costs_order():
    inst = synthetic_supply_chain(periods=2, hours=12)
    scale, prob = synthetic_cf_scenarios()
    out = scenario_costs(inst, scale)
    cost = {k: v["cost_usd"] for k, v in out.items()}
    assert cost["low"] >= cost["medium"] - 1e-6 * cost["medium"] >= cost["high"] - 2e-6 * cost["medium"]
    assert minimax_value(cost) == cost["low"]
