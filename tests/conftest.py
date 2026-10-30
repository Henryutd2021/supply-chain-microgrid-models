import pytest

from sc_microgrid.solve import SolverFactory, SOLVERS


def gurobi_available() -> bool:
    try:
        return bool(SolverFactory(SOLVERS["gurobi"]).available())
    except Exception:
        return False


needs_gurobi = pytest.mark.skipif(not gurobi_available(), reason="Gurobi (gurobipy with a licence) is not available")
