"""Virtual power plant scheduling: day-ahead, intraday and rolling horizon (thesis Chapter 3).

A VPP with a CHP unit, renewables, a battery and a thermal store first plans its day-ahead sales
E^m_t (Model DA, Eqs. 3.1-3.10). On the delivery day the sales are fixed and shortfalls are
bought as imbalance energy E^b_t (Model ID, Eqs. 3.11-3.20). The rolling-horizon algorithm
(Section 3.1.2, Fig. 3.1) solves Model ID over a prediction horizon, implements the control
horizon, updates the storage levels and moves on. All three are linear programs.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from .instances import VPPInstance
from .solve import solve


def _common(m: pyo.ConcreteModel, inst: VPPInstance, T: int) -> None:
    """Variables and constraints shared by Models DA and ID: Eqs. (3.3)-(3.10) / (3.13)-(3.20)."""
    H1, H0 = range(1, T + 1), range(0, T + 1)
    m.chp = pyo.Var(H1, bounds=(inst.chp_min_mw, inst.chp_max_mw))          # E^chp_t, (3.8)
    m.e_ch = pyo.Var(H1, domain=pyo.NonNegativeReals)                       # E^c_t
    m.e_dis = pyo.Var(H1, domain=pyo.NonNegativeReals)                      # E^d_t
    m.th_ch = pyo.Var(H1, domain=pyo.NonNegativeReals)                      # P^c_t
    m.th_dis = pyo.Var(H1, domain=pyo.NonNegativeReals)                     # P^d_t
    m.soc = pyo.Var(H0)                                                     # S^e_t
    m.sth = pyo.Var(H0)                                                     # S^th_t
    m.eq3_4_battery = pyo.Constraint(H1, rule=lambda m, t: m.soc[t] == m.soc[t - 1] + m.e_ch[t] - m.e_dis[t])
    m.eq3_5_thermal = pyo.Constraint(H1, rule=lambda m, t: m.sth[t] == m.sth[t - 1] + m.th_ch[t] - m.th_dis[t])
    m.eq3_6_thermal_limits = pyo.Constraint(H1, rule=lambda m, t: pyo.inequality(inst.thermal_min_mwh, m.sth[t], inst.thermal_max_mwh))
    m.eq3_7_battery_limits = pyo.Constraint(H1, rule=lambda m, t: pyo.inequality(inst.battery_min_mwh, m.soc[t], inst.battery_max_mwh))
    m.eq3_10_battery_initial = pyo.Constraint(expr=m.soc[0] == inst.battery_initial_mwh)
    m.eq3_10_thermal_initial = pyo.Constraint(expr=m.sth[0] == inst.thermal_initial_mwh)


def day_ahead(inst: VPPInstance, allow_spill: bool = False) -> pyo.ConcreteModel:
    """Model DA, Eqs. (3.1)-(3.10), on the forecast series.

    ``allow_spill=True`` adds a non-negative heat-dump term to the heat balance (an extension, off
    by default; surplus power can already be sold day-ahead).
    """
    inst.validate()
    T = inst.hours
    H1 = range(1, T + 1)
    m = pyo.ConcreteModel("vpp_day_ahead")
    _common(m, inst, T)
    m.sale = pyo.Var(H1, domain=pyo.NonNegativeReals)                       # E^m_t
    if allow_spill:
        m.heat_dump = pyo.Var(H1, domain=pyo.NonNegativeReals)
    m.cost = pyo.Objective(expr=sum(inst.fuel_cost_usd_per_mwh * m.chp[t]
                                    + inst.storage_cost_usd_per_mwh * (m.e_ch[t] + m.e_dis[t])
                                    - inst.day_ahead_price_usd_per_mwh[t - 1] * m.sale[t] for t in H1))
    m.eq3_2_power_balance = pyo.Constraint(H1, rule=lambda m, t:
                                           m.chp[t] + inst.renewable_mw[t - 1] + m.e_dis[t] - m.e_ch[t] - m.sale[t]
                                           == inst.load_mw[t - 1])
    m.eq3_3_heat_balance = pyo.Constraint(H1, rule=lambda m, t:
                                          inst.heat_to_power * m.chp[t] + m.th_dis[t] - m.th_ch[t]
                                          - (m.heat_dump[t] if allow_spill else 0) == inst.heat_load_mw[t - 1])
    return m


def intraday(inst: VPPInstance, sales_mw: Sequence[float], realized: bool = True,
             allow_spill: bool = False) -> pyo.ConcreteModel:
    """Model ID, Eqs. (3.11)-(3.20), with the day-ahead sales E^m_t fixed at ``sales_mw``.

    ``realized=True`` uses the delivery-day series (``*_realized_mw``), otherwise the forecasts.
    ``allow_spill=True`` adds non-negative spill terms to the power and heat balances (an extension,
    off by default; see docs/formulation.md).
    """
    inst.validate()
    T = inst.hours
    H1 = range(1, T + 1)
    sales = np.asarray(sales_mw, dtype=float)
    if sales.shape != (T,) or not np.all(np.isfinite(sales)) or np.any(sales < 0):
        raise ValueError(f"sales_mw must hold {T} finite, non-negative values")
    if allow_spill and np.any(inst.imbalance_price_usd_per_mwh < 0):
        raise ValueError("allow_spill needs non-negative imbalance prices (buying to spill would be unbounded)")
    load = inst.load_realized_mw if realized else inst.load_mw
    heat = inst.heat_load_realized_mw if realized else inst.heat_load_mw
    ren = inst.renewable_realized_mw if realized else inst.renewable_mw
    m = pyo.ConcreteModel("vpp_intraday")
    _common(m, inst, T)
    m.buy = pyo.Var(H1, domain=pyo.NonNegativeReals)                        # E^b_t
    if allow_spill:
        m.spill = pyo.Var(H1, domain=pyo.NonNegativeReals)
        m.heat_dump = pyo.Var(H1, domain=pyo.NonNegativeReals)
    m.cost = pyo.Objective(expr=sum(inst.fuel_cost_usd_per_mwh * m.chp[t]
                                    + inst.storage_cost_usd_per_mwh * (m.e_ch[t] + m.e_dis[t])
                                    + inst.imbalance_price_usd_per_mwh[t - 1] * m.buy[t]
                                    - inst.day_ahead_price_usd_per_mwh[t - 1] * sales[t - 1] for t in H1))
    m.eq3_12_power_balance = pyo.Constraint(H1, rule=lambda m, t:
                                            m.chp[t] + ren[t - 1] + m.e_dis[t] + m.buy[t] - m.e_ch[t] - sales[t - 1]
                                            - (m.spill[t] if allow_spill else 0) == load[t - 1])
    m.eq3_13_heat_balance = pyo.Constraint(H1, rule=lambda m, t:
                                           inst.heat_to_power * m.chp[t] + m.th_dis[t] - m.th_ch[t]
                                           - (m.heat_dump[t] if allow_spill else 0) == heat[t - 1])
    return m


def schedule_table(m: pyo.ConcreteModel, inst: VPPInstance, first_hour: int = 1) -> pd.DataFrame:
    """Hourly schedule of a solved Model DA or ID."""
    val = pyo.value
    T = len(m.chp)
    rows = []
    for t in range(1, T + 1):
        r = {"hour": first_hour + t - 1, "chp_mw": val(m.chp[t]), "battery_charge_mw": val(m.e_ch[t]),
             "battery_discharge_mw": val(m.e_dis[t]), "battery_mwh": val(m.soc[t]),
             "thermal_charge_mw": val(m.th_ch[t]), "thermal_discharge_mw": val(m.th_dis[t]), "thermal_mwh": val(m.sth[t])}
        if hasattr(m, "sale"):
            r["day_ahead_sale_mw"] = val(m.sale[t])
        if hasattr(m, "buy"):
            r["imbalance_purchase_mw"] = val(m.buy[t])
        if hasattr(m, "spill"):
            r["spill_mw"] = val(m.spill[t])
        if hasattr(m, "heat_dump"):
            r["heat_dump_mw"] = val(m.heat_dump[t])
        rows.append(r)
    return pd.DataFrame(rows)


def rolling_horizon(inst: VPPInstance, sales_mw: Sequence[float], prediction_horizon: int = 3,
                    control_horizon: int = 1, forecast_beyond_control: bool = True, solver: str = "highs",
                    allow_spill: bool = False, time_limit: Optional[float] = None) -> pd.DataFrame:
    """Rolling-horizon solution of Model ID (thesis Section 3.1.2).

    At hour h, Model ID is solved over hours h .. h + PH - 1 (truncated at the end of the
    scheduling horizon) from the current storage levels. The first CH hours are implemented and
    recorded, the storage levels are carried over, and h moves forward by CH. Hours inside the
    control horizon use the realized series; with ``forecast_beyond_control`` the rest of the
    window uses the forecasts, otherwise the realized series. Returns one row per hour, with the
    hourly cost of the implemented decisions in ``cost_usd``.
    """
    inst.validate()
    T = inst.hours
    if not (1 <= control_horizon <= prediction_horizon):
        raise ValueError("need 1 <= control_horizon <= prediction_horizon")
    sales = np.asarray(sales_mw, dtype=float)
    if sales.shape != (T,):
        raise ValueError(f"sales_mw must hold {T} values")
    soc, sth = inst.battery_initial_mwh, inst.thermal_initial_mwh
    rows: List[Dict[str, float]] = []
    h = 0                                   # 0-based start hour of the window
    while h < T:
        end = min(h + prediction_horizon, T)
        n_ctrl = min(control_horizon, end - h)
        sl = slice(h, end)

        def mix(realized, forecast):
            out = realized[sl].copy()
            if forecast_beyond_control:
                out[n_ctrl:] = forecast[sl][n_ctrl:]
            return out
        window = replace(inst,
                         load_mw=mix(inst.load_realized_mw, inst.load_mw),
                         heat_load_mw=mix(inst.heat_load_realized_mw, inst.heat_load_mw),
                         renewable_mw=mix(inst.renewable_realized_mw, inst.renewable_mw),
                         day_ahead_price_usd_per_mwh=inst.day_ahead_price_usd_per_mwh[sl],
                         imbalance_price_usd_per_mwh=inst.imbalance_price_usd_per_mwh[sl],
                         battery_initial_mwh=soc, thermal_initial_mwh=sth,
                         load_realized_mw=None, heat_load_realized_mw=None, renewable_realized_mw=None)
        m = intraday(window, sales[sl], realized=False, allow_spill=allow_spill)
        solve(m, solver=solver, time_limit=time_limit)
        table = schedule_table(m, window, first_hour=h + 1)
        val = pyo.value
        for t in range(1, n_ctrl + 1):
            r = table.iloc[t - 1].to_dict()
            r["day_ahead_sale_mw"] = float(sales[h + t - 1])
            r["cost_usd"] = (inst.fuel_cost_usd_per_mwh * val(m.chp[t])
                             + inst.storage_cost_usd_per_mwh * (val(m.e_ch[t]) + val(m.e_dis[t]))
                             + window.imbalance_price_usd_per_mwh[t - 1] * val(m.buy[t])
                             - window.day_ahead_price_usd_per_mwh[t - 1] * sales[h + t - 1])
            rows.append(r)
        # carry the storage levels over, clipped to their limits (removes solver round-off)
        soc = min(max(val(m.soc[n_ctrl]), inst.battery_min_mwh, 0.0), inst.battery_max_mwh)
        sth = min(max(val(m.sth[n_ctrl]), inst.thermal_min_mwh, 0.0), inst.thermal_max_mwh)
        h += n_ctrl
    return pd.DataFrame(rows)
