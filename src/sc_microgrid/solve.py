"""Solve a Pyomo model with HiGHS (default) or Gurobi."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Mapping, Optional

import pyomo.environ as pyo
from pyomo.contrib.solver.common.factory import SolverFactory
from pyomo.contrib.solver.common.results import SolutionStatus, TerminationCondition

SOLVERS = {"highs": "highs", "gurobi": "gurobi_persistent"}


class SolveError(RuntimeError):
    """The solver did not return an optimal solution (within the requested MIP gap).

    ``status`` is the Pyomo termination condition name, or None when the solve did not start.
    """

    def __init__(self, message: str, status: Optional[str] = None):
        super().__init__(message)
        self.status = status


INFEASIBLE = ("provenInfeasible", "locallyInfeasible")


@dataclass
class SolveResult:
    status: str               # Pyomo termination condition
    objective: float          # incumbent objective (nan if none)
    bound: float              # best bound reported by the solver (nan if none)
    runtime_s: float
    solver: str


def model_size(m: pyo.ConcreteModel) -> dict:
    """Numbers of variables (all and integer or binary) and active constraints."""
    n_var = n_int = 0
    for v in m.component_data_objects(pyo.Var, descend_into=True):
        n_var += 1
        n_int += 0 if v.is_continuous() else 1
    n_con = sum(1 for _ in m.component_data_objects(pyo.Constraint, active=True, descend_into=True))
    return {"variables": n_var, "integer_variables": n_int, "constraints": n_con}


def solve(m: pyo.ConcreteModel, solver: str = "highs", mip_gap: float = 1e-6, time_limit: Optional[float] = None,
          threads: Optional[int] = None, options: Optional[Mapping[str, object]] = None, tee: bool = False,
          require_optimal: bool = True) -> SolveResult:
    """Solve ``m`` in place and load the solution.

    ``mip_gap`` is the relative MIP gap at which the solver stops; mixed-integer objectives are
    optimal up to this gap. ``options`` are passed to the solver unchanged (HiGHS or Gurobi option
    names). Models with products of variables (the bilinear two-tier form) need ``solver="gurobi"``.
    Raises :class:`SolveError` unless the solver reports an optimal solution, or, with
    ``require_optimal=False``, returns the status and any feasible solution found.
    """
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {sorted(SOLVERS)}, got {solver!r}")
    if solver == "highs" and getattr(m, "bilinear", False):
        raise SolveError("this model has products of variables (bilinear formulation); use solver='gurobi'")
    opt = SolverFactory(SOLVERS[solver])
    if not opt.available():
        raise SolveError(f"solver {solver!r} is not available in this environment")
    kwargs = dict(tee=tee, load_solutions=False, raise_exception_on_nonoptimal_result=False, rel_gap=mip_gap)
    if time_limit is not None:
        kwargs["time_limit"] = time_limit
    if threads is not None:
        kwargs["threads"] = threads
    if options:
        kwargs["solver_options"] = dict(options)
    t0 = time.time()
    res = opt.solve(m, **kwargs)
    runtime = time.time() - t0
    feasible = res.solution_status in (SolutionStatus.feasible, SolutionStatus.optimal)
    if feasible:
        res.solution_loader.load_vars()
    optimal = res.termination_condition == TerminationCondition.convergenceCriteriaSatisfied
    if require_optimal and not optimal:
        raise SolveError(f"solver {solver} returned {res.termination_condition.name}", res.termination_condition.name)
    objective = float(res.incumbent_objective) if feasible and res.incumbent_objective is not None else float("nan")
    bound = float(res.objective_bound) if res.objective_bound is not None else float("nan")
    return SolveResult(status=res.termination_condition.name, objective=objective, bound=bound,
                       runtime_s=round(runtime, 3), solver=solver)
