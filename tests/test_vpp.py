import numpy as np
import pyomo.environ as pyo
import pytest
from dataclasses import replace

from sc_microgrid import SolveError, day_ahead, intraday, rolling_horizon, schedule_table, solve, synthetic_vpp


@pytest.fixture(scope="module")
def plan():
    v = synthetic_vpp()
    da = day_ahead(v)
    res = solve(da)
    sales = np.array([pyo.value(da.sale[t]) for t in range(1, v.hours + 1)])
    return v, da, res, sales


def test_day_ahead_balances(plan):
    v, da, res, sales = plan
    tab = schedule_table(da, v)
    power = tab.chp_mw + v.renewable_mw + tab.battery_discharge_mw - tab.battery_charge_mw - tab.day_ahead_sale_mw
    heat = v.heat_to_power * tab.chp_mw + tab.thermal_discharge_mw - tab.thermal_charge_mw
    assert np.allclose(power, v.load_mw, atol=1e-6)
    assert np.allclose(heat, v.heat_load_mw, atol=1e-6)
    assert tab.battery_mwh.between(v.battery_min_mwh - 1e-6, v.battery_max_mwh + 1e-6).all()
    assert tab.chp_mw.between(v.chp_min_mw - 1e-6, v.chp_max_mw + 1e-6).all()


def test_intraday_keeps_sales(plan):
    v, da, res, sales = plan
    m = intraday(v, sales)
    solve(m)
    tab = schedule_table(m, v)
    power = (tab.chp_mw + v.renewable_realized_mw + tab.battery_discharge_mw + tab.imbalance_purchase_mw
             - tab.battery_charge_mw - sales)
    assert np.allclose(power, v.load_realized_mw, atol=1e-6)
    assert (tab.imbalance_purchase_mw > 1e-6).any()      # the realized day has shortfalls


def test_rolling_horizon(plan):
    v, da, res, sales = plan
    full = intraday(v, sales)
    r_full = solve(full)
    rh = rolling_horizon(v, sales, prediction_horizon=v.hours, control_horizon=1, forecast_beyond_control=False)
    assert list(rh.hour) == list(range(1, v.hours + 1))
    assert rh.cost_usd.sum() == pytest.approx(r_full.objective, rel=1e-6)   # same information, same optimum
    rh3 = rolling_horizon(v, sales, prediction_horizon=3, control_horizon=2)
    assert len(rh3) == v.hours
    level = np.concatenate([[v.battery_initial_mwh], rh3.battery_mwh.to_numpy()])
    assert np.allclose(np.diff(level), rh3.battery_charge_mw - rh3.battery_discharge_mw, atol=1e-6)
    assert rh3.cost_usd.sum() >= r_full.objective - 1e-6 * abs(r_full.objective)
    with pytest.raises(ValueError):
        rolling_horizon(v, sales, prediction_horizon=2, control_horizon=3)


def test_spill_extension(plan):
    v, da, res, sales = plan
    surplus = replace(v, renewable_realized_mw=v.renewable_mw * 3, load_realized_mw=None, heat_load_realized_mw=None)
    with pytest.raises(SolveError):
        solve(intraday(surplus, sales))
    m = intraday(surplus, sales, allow_spill=True)
    solve(m)
    assert schedule_table(m, surplus).spill_mw.sum() > 0
    negative = replace(v, imbalance_price_usd_per_mwh=v.imbalance_price_usd_per_mwh - 500)
    with pytest.raises(ValueError, match="non-negative imbalance prices"):
        intraday(negative, sales, allow_spill=True)


def test_initial_storage_within_capacity(plan):
    v = plan[0]
    with pytest.raises(ValueError, match="battery_initial_mwh exceeds"):
        replace(v, battery_initial_mwh=v.battery_max_mwh + 1).validate()
    ok = replace(v, battery_initial_mwh=20.0, thermal_initial_mwh=30.0).validate()
    m = day_ahead(ok)
    solve(m)
    assert pyo.value(m.soc[0]) == pytest.approx(20.0) and pyo.value(m.sth[0]) == pytest.approx(30.0)
