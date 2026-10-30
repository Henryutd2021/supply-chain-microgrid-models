"""Scenario costs and risk criteria (thesis Chapter 5, Models 5.2-5.4).

The thesis evaluates the costs C_w of capacity-factor scenarios w, each the optimal cost of the
deterministic FMLA model under that scenario, with three criteria:

* VaR (Model 5.2, Eqs. 5.41-5.43): the smallest cost level exceeded with probability at most 1 - phi;
* CVaR (Model 5.3, Eqs. 5.45-5.47): zeta + E[(C - zeta)+] / (1 - phi), minimized over zeta;
* minimax (Model 5.4, Eqs. 5.48-5.49): the largest scenario cost.

Each criterion is available as the optimization model of the thesis (:func:`var_model`,
:func:`cvar_model`, :func:`minimax_model`) and in closed form (:func:`var_value`,
:func:`cvar_value`, :func:`minimax_value`); the tests check that both agree.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import pyomo.environ as pyo

from .fmla import build_fmla, fmla_summary, scale_cf
from .instances import SupplyChainInstance
from .solve import solve


def _check(cost: Mapping[str, float], prob: Mapping[str, float], phi: Optional[float] = None) -> List[str]:
    sc = list(cost)
    if not sc:
        raise ValueError("at least one scenario is needed")
    if set(sc) != set(prob):
        raise ValueError("costs and probabilities must have the same scenarios")
    for s in sc:
        if not math.isfinite(float(cost[s])):
            raise ValueError(f"cost of scenario {s} is not finite")
        if not (math.isfinite(float(prob[s])) and prob[s] >= 0):
            raise ValueError(f"probability of scenario {s} must be finite and non-negative")
    if abs(sum(prob.values()) - 1.0) > 1e-9:
        raise ValueError(f"scenario probabilities must sum to 1, got {sum(prob.values())}")
    if phi is not None and not 0.0 < phi < 1.0:
        raise ValueError(f"confidence level phi must lie strictly between 0 and 1, got {phi}")
    return sc


# ------------------------------------------------------------------------------ optimization models

def var_model(cost: Mapping[str, float], prob: Mapping[str, float], phi: float,
              big_m: Optional[float] = None) -> pyo.ConcreteModel:
    """Model 5.2: min VaR s.t. C_w - VaR <= M delta_w (5.42), sum_w beta_w delta_w <= 1 - phi (5.43).

    ``big_m`` defaults to the cost range plus one, which never cuts off a VaR between the
    smallest and the largest scenario cost.
    """
    sc = _check(cost, prob, phi)
    spread = max(cost.values()) - min(cost.values())
    M = spread + 1.0 if big_m is None else float(big_m)
    if not (math.isfinite(M) and M >= spread):
        raise ValueError(f"big_m must be finite and at least the cost range {spread}, got {big_m}")
    m = pyo.ConcreteModel("VaR")
    m.delta = pyo.Var(sc, domain=pyo.Binary)
    m.VaR = pyo.Var()
    m.obj = pyo.Objective(expr=m.VaR)
    m.eq5_42 = pyo.Constraint(sc, rule=lambda m, s: cost[s] - m.VaR <= M * m.delta[s])
    m.eq5_43 = pyo.Constraint(expr=sum(prob[s] * m.delta[s] for s in sc) <= 1 - phi)
    return m


def cvar_model(cost: Mapping[str, float], prob: Mapping[str, float], phi: float) -> pyo.ConcreteModel:
    """Model 5.3: min zeta + 1/(1-phi) sum_w beta_w eta_w (5.45) s.t. C_w - zeta <= eta_w (5.46), eta_w >= 0 (5.47)."""
    sc = _check(cost, prob, phi)
    m = pyo.ConcreteModel("CVaR")
    m.zeta = pyo.Var()
    m.eta = pyo.Var(sc, domain=pyo.NonNegativeReals)
    m.obj = pyo.Objective(expr=m.zeta + sum(prob[s] * m.eta[s] for s in sc) / (1 - phi))
    m.eq5_46 = pyo.Constraint(sc, rule=lambda m, s: cost[s] - m.zeta <= m.eta[s])
    return m


def minimax_model(cost: Mapping[str, float]) -> pyo.ConcreteModel:
    """Model 5.4: min M s.t. M >= C_w for every scenario (5.48)-(5.49)."""
    sc = list(cost)
    m = pyo.ConcreteModel("minimax")
    m.M = pyo.Var()
    m.obj = pyo.Objective(expr=m.M)
    m.eq5_49 = pyo.Constraint(sc, rule=lambda m, s: m.M >= cost[s])
    return m


# ------------------------------------------------------------------------------------- closed forms

def var_value(cost: Mapping[str, float], prob: Mapping[str, float], phi: float) -> float:
    """Smallest scenario cost V with P(C > V) <= 1 - phi."""
    _check(cost, prob, phi)
    for v in sorted(cost.values()):
        if sum(prob[s] for s, c in cost.items() if c > v) <= 1 - phi + 1e-12:
            return float(v)
    return float(max(cost.values()))


def cvar_value(cost: Mapping[str, float], prob: Mapping[str, float], phi: float) -> float:
    """min over zeta of zeta + E[(C - zeta)+]/(1 - phi); the minimum lies at a scenario cost."""
    _check(cost, prob, phi)
    return float(min(z + sum(prob[s] * max(c - z, 0.0) for s, c in cost.items()) / (1 - phi)
                     for z in cost.values()))


def minimax_value(cost: Mapping[str, float]) -> float:
    return float(max(cost.values()))


def expected_value(cost: Mapping[str, float], prob: Mapping[str, float]) -> float:
    _check(cost, prob)
    return float(sum(prob[s] * cost[s] for s in cost))


def risk_table(cost: Mapping[str, float], prob: Mapping[str, float], phis: Sequence[float],
               solver: str = "highs", time_limit: Optional[float] = None) -> List[Dict[str, float]]:
    """VaR, CVaR (from Models 5.2 and 5.3), minimax and expected cost for each confidence level."""
    rows = []
    for phi in phis:
        mv, mc = var_model(cost, prob, phi), cvar_model(cost, prob, phi)
        solve(mv, solver=solver, mip_gap=0.0, time_limit=time_limit)
        solve(mc, solver=solver, time_limit=time_limit)
        rows.append({"phi": float(phi), "VaR": pyo.value(mv.VaR), "CVaR": pyo.value(mc.obj),
                     "minimax": minimax_value(cost), "expected": expected_value(cost, prob)})
    return rows


# --------------------------------------------------------------------------------- scenario costs

def scenario_costs(inst: SupplyChainInstance, cf_scale: Mapping[str, float], solver: str = "highs",
                   mip_gap: float = 1e-6, build: Optional[Callable] = None, **solve_kwargs) -> Dict[str, dict]:
    """C_w for capacity-factor scenarios built by scaling every capacity factor (capped at 1).

    Solves the deterministic FMLA model once per scenario and returns, per scenario, the cost and
    the summary of the design. ``build`` defaults to :func:`build_fmla` with the big-M formulation.
    """
    build = build or build_fmla
    out = {}
    for name, factor in cf_scale.items():
        sc_inst = scale_cf(inst, factor)
        m = build(sc_inst)
        res = solve(m, solver=solver, mip_gap=mip_gap, **solve_kwargs)
        out[name] = {"cost_usd": res.objective, "summary": fmla_summary(m, sc_inst)}
    return out
