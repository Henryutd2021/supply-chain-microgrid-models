# Supply chains with on-site microgrids: location-allocation and VPP scheduling framework

A documented, runnable implementation of the optimization models in:

> Honglin Li, Tongdan Jin, Clara Novoa. **Facility and microgrid location-allocation for integrated supply chain and transactive energy operations.** *Applied Mathematical Modelling* 119 (2023) 119–136. [doi:10.1016/j.apm.2023.02.014](https://doi.org/10.1016/j.apm.2023.02.014)
>
> Honglin Li, Tongdan Jin, Emily Zhu Fainman. **Locating and allocating warehouse facilities and prosumer microgrid in transactive energy paradigm.** *2023 International Conference on Smart Electrical Grid and Renewable Energy (SEGRE)*, IEEE, pp. 410–415. [doi:10.1109/SEGRE58867.2023.00071](https://doi.org/10.1109/SEGRE58867.2023.00071)
>
> Honglin Li. **Planning Virtual Power Plants for Smart Factories and Industry 4.0 with Renewable and Thermal Cogeneration.** M.S. thesis, Texas State University, 2021. [hdl:10877/17829](https://hdl.handle.net/10877/17829)

The models decide where to open factories and warehouses, how to route products through a supply chain, and how much wind, solar and storage capacity each open site should install, when every site can buy electricity from and sell it to the grid every hour (transactive energy). A second family schedules a virtual power plant (VPP) with a CHP unit, renewables, a battery and a thermal store in the day-ahead and intraday markets.

## Research questions

- **Supply chain with on-site microgrids.** Which candidate factories and warehouses should a multi-tier supply chain open, how should production, inventory and shipments be planned, and how much generation and storage should each open site install, so that the net annual cost of facilities, logistics and energy is minimal? On-site generation adds energy as a fourth flow to the material, cash and information flows of supply-chain design: a site with good wind or sun can be worth a longer transport distance.
- **Risk of weather-dependent designs.** How do the cost and the design change across capacity-factor scenarios, and how do VaR, CVaR and minimax rank the outcomes?
- **VPP market participation.** How should a VPP plan its day-ahead sales and then operate on the delivery day, when loads and renewable output differ from the forecasts, and how long should the prediction horizon of a rolling-horizon scheme be?

## The models

| Part | Model | Sources | Type |
|---|---|---|---|
| A | capacitated fixed-charge location with single sourcing | thesis Model 4.1 | binary program |
| B | warehouses with on-site microgrids (WMLA): location, store assignment, wind/PV/storage sizing, hourly storage and grid trade | thesis Models 4.2 and 4.4; the same two-tier model as SEGRE 2023 | MILP (big-value form); bilinear form solved exactly by enumerating open-warehouse sets, or with Gurobi |
| C | factories, warehouses and on-site microgrids (FMLA): the above for a three-tier network, with production planning, inventory and backorders, production resources, a chance-constrained service level and electric-truck energy | AMM 2023; thesis Model 4.7 | MILP (big-value form) |
| C | expected-value design over capacity-factor scenarios | thesis Model 5.1 | MILP |
| D | VaR, CVaR and minimax of the scenario costs | thesis Models 5.2–5.4 | MILP, LP |
| E | VPP day-ahead plan, intraday operation with fixed sales, rolling horizon | thesis Models 3.1–3.2, Section 3.1.2 | LPs |

| Aspect | In the publications | In this framework |
|---|---|---|
| Network | 3 candidate warehouses and 6 stores (thesis Section 4.1; SEGRE 2023); 4 candidate factories, 2 candidate warehouses and 4 stores, 2 or 8 products (AMM) | any sets |
| Time | 52 weekly production periods of 168 hours | any $`J`$ periods of $`T`$ hours; shorter horizons weighted to a year |
| Renewables | wind turbines and PV, hourly capacity factors per candidate city | any technologies with hourly capacity factors per site |
| Demand | normal, met at a service level through a chance constraint | same, deterministic equivalent with $`z_\alpha`$ |
| Energy | hourly balance per open site: facility load, production energy, electric-truck energy, storage, purchase and sale | same |

The complete mathematics, keyed to the publications' equation numbers, is in [docs/formulation.md](docs/formulation.md).

## What is in this repository

| Path | Content |
|---|---|
| [`docs/formulation.md`](docs/formulation.md) | sets, parameters, variables, objectives and every constraint of Parts A–E, with the publications' equation numbers; conventions; extensions |
| [`docs/data_pipeline.md`](docs/data_pipeline.md) | how each input is prepared (network, costs, demand statistics, capacity factors from weather records, prices), QA checks, the input file format |
| [`docs/scenarios.md`](docs/scenarios.md) | the sensitivity-case families of the publications, capacity-factor scenarios and risk criteria, VPP settings, and how to express them |
| [`docs/architecture.md`](docs/architecture.md) | modules, data flow, solvers, problem size, output tables, tests |
| [`src/sc_microgrid/`](src/sc_microgrid/) | the implementation (Pyomo; HiGHS by default, Gurobi optional), input validation, input format, preprocessing helpers, summaries, command line |
| [`examples/`](examples/) | `minimal.py` (Parts A, B, C and E once, a few seconds), `risk_scenarios.py` (Model 5.1 and the risk criteria of Part D), `sensitivity.py`, `vpp_rolling_horizon.py`, `prepare_inputs.py`; all on synthetic data |
| [`tests/`](tests/) | 52 tests: validation, input format, optimal solves checked against brute force, closed forms and recomputed balances, formulations against each other, command line |

## Quick start

```bash
git clone https://github.com/Henryutd2021/supply-chain-microgrid-models.git
cd supply-chain-microgrid-models
python3 -m venv .venv && source .venv/bin/activate    # needs Python 3.10+ (check: python3 --version); if older, use e.g. python3.12
pip install -e ".[test]"
python examples/minimal.py                             # Parts A, B, C, E on synthetic data, HiGHS
python examples/risk_scenarios.py                      # capacity-factor scenarios, VaR, CVaR, minimax
pytest
```

With conda instead: `conda env create -f environment.yml && conda activate sc-microgrid && pip install -e .`

The command line works on instance directories:

```bash
sc-microgrid synthetic fmla my_case/                   # write a synthetic instance (also: fixed-charge, wmla, vpp)
sc-microgrid run my_case/ --out-dir outputs/base       # solve; writes summary.json and tables
sc-microgrid run my_case/ --no-microgrid --out-dir outputs/grid_only
sc-microgrid risk my_case/ --scale low=0.8 --scale medium=1 --scale high=1.2 \
    --prob low=0.15 --prob medium=0.6 --prob high=0.25 --phi 0.5 0.8 0.9 --out-dir outputs/risk
```

In Python:

```python
from sc_microgrid import read_instance, build_fmla, solve, fmla_summary

inst = read_instance("my_case/")             # your own data, see docs/data_pipeline.md
m = build_fmla(inst)                         # AMM 2023 formulation
solve(m)                                     # or solve(m, solver="gurobi")
s = fmla_summary(m, inst)
print(s["open_factories"], s["open_warehouses"])
print({k: float(v) for k, v in s["components_usd"].items()})   # cost components, $/yr
```

**About the example data:** `sc_microgrid.synthetic` generates small invented networks, weather and prices with round, illustrative costs. They show how the models behave and how to read their output. Their numbers say nothing about any real place, company or market.

## Using your own data

1. Prepare the inputs as described in [docs/data_pipeline.md](docs/data_pipeline.md). `sc_microgrid.preprocess` covers generic steps: capacity factors from wind speed and irradiance, periods from an hourly year, demand statistics, time-of-use prices; `examples/prepare_inputs.py` shows them end to end.
2. Write the instance directory (`instance.json`, `hourly.csv`) or build the dataclass in Python and call `write_instance`. Inputs are validated when read: names carry units, keys must match, hourly grids must be complete, values must be finite and in range.
3. Solve the base case and the sensitivity cases of interest ([docs/scenarios.md](docs/scenarios.md)).

A full year of a small three-tier network (52 weeks × 168 h) has over 100,000 variables, a few thousand of them integer; see [docs/architecture.md](docs/architecture.md#problem-size) for sizes and solve times.

## What this repository does not contain

This repository presents the modelling framework. It does not contain the code, the input data or the results used for the publications. For their results, see the publications.

## Citation

If you use this framework, please cite the publication of the model you use (above). Citation metadata for the publications and this software are in [`CITATION.cff`](CITATION.cff).

## Licence

- Code: [MIT](LICENSE).
- Documentation (`README.md`, `docs/`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
