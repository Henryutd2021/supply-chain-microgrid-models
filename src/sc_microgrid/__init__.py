"""sc_microgrid: location-allocation of facilities and on-site microgrids in supply chains with
transactive energy, and scheduling of a virtual power plant.

Implements the formulations of

* H. Li, T. Jin, C. Novoa, "Facility and microgrid location-allocation for integrated supply chain
  and transactive energy operations", Applied Mathematical Modelling 119 (2023) 119-136;
* H. Li, T. Jin, E. Z. Fainman, "Locating and allocating warehouse facilities and prosumer microgrid
  in transactive energy paradigm", 2023 International Conference on Smart Electrical Grid and
  Renewable Energy (SEGRE), IEEE, pp. 410-415;
* H. Li, "Planning Virtual Power Plants for Smart Factories and Industry 4.0 with Renewable and
  Thermal Cogeneration", M.S. thesis, Texas State University, 2021.

See docs/formulation.md. Typical use::

    from sc_microgrid import synthetic_supply_chain, build_fmla, solve, fmla_summary
    inst = synthetic_supply_chain()
    m = build_fmla(inst)               # AMM 2023 formulation, a MILP
    solve(m)                           # HiGHS by default
    print(fmla_summary(m, inst)["open_factories"])
"""

from .fmla import build_fmla, expected_cf, fmla_summary, scale_cf, with_cf, z_value
from .instances import (FixedChargeInstance, Storage, SupplyChainInstance, Technology, Truck, VPPInstance,
                        WarehouseInstance)
from .io import read_instance, write_instance
from .location import build_fixed_charge, fixed_charge_summary
from .risk import (cvar_model, cvar_value, expected_value, minimax_model, minimax_value, risk_table, scenario_costs,
                   var_model, var_value)
from .solve import SolveError, SolveResult, model_size, solve
from .synthetic import (capital_recovery_factor, synthetic_cf_scenarios, synthetic_fixed_charge, synthetic_supply_chain,
                        synthetic_vpp, synthetic_warehouses)
from .vpp import day_ahead, intraday, rolling_horizon, schedule_table
from .wmla import build_wmla, feasible_open_sets, solve_enumerated, wmla_summary

__version__ = "1.0.0"

__all__ = [
    "build_fmla", "expected_cf", "fmla_summary", "scale_cf", "with_cf", "z_value",
    "FixedChargeInstance", "Storage", "SupplyChainInstance", "Technology", "Truck", "VPPInstance", "WarehouseInstance",
    "read_instance", "write_instance", "build_fixed_charge", "fixed_charge_summary",
    "cvar_model", "cvar_value", "expected_value", "minimax_model", "minimax_value", "risk_table", "scenario_costs",
    "var_model", "var_value", "SolveError", "SolveResult", "model_size", "solve",
    "capital_recovery_factor", "synthetic_cf_scenarios", "synthetic_fixed_charge", "synthetic_supply_chain",
    "synthetic_vpp", "synthetic_warehouses", "day_ahead", "intraday", "rolling_horizon", "schedule_table",
    "build_wmla", "feasible_open_sets", "solve_enumerated", "wmla_summary", "__version__",
]
