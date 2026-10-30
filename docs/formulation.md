# Mathematical formulation

This document states the optimization models as implemented by this framework. They come from three publications:

- **AMM 2023.** H. Li, T. Jin, C. Novoa, "Facility and microgrid location-allocation for integrated supply chain and transactive energy operations", *Applied Mathematical Modelling* 119 (2023) 119–136, [doi:10.1016/j.apm.2023.02.014](https://doi.org/10.1016/j.apm.2023.02.014), Section 3.
- **SEGRE 2023.** H. Li, T. Jin, E. Z. Fainman, "Locating and allocating warehouse facilities and prosumer microgrid in transactive energy paradigm", *2023 International Conference on Smart Electrical Grid and Renewable Energy (SEGRE)*, IEEE, pp. 410–415, [doi:10.1109/SEGRE58867.2023.00071](https://doi.org/10.1109/SEGRE58867.2023.00071). It presents the same two-tier model as thesis Section 4.2 (Part B).
- **Thesis.** H. Li, *Planning Virtual Power Plants for Smart Factories and Industry 4.0 with Renewable and Thermal Cogeneration*, M.S. thesis, Texas State University, 2021, [hdl:10877/17829](https://hdl.handle.net/10877/17829), Chapters 3–5.

Equation numbers follow AMM 2023 and the thesis, so each part can be matched to its source: plain numbers such as (17) are AMM 2023, numbers such as (4.52) or (3.12) are the thesis. Each constraint in the code is named after its equation (for example `eq17_factory_energy`, `eq4_52_energy_balance`, `eq3_12_power_balance`).

| Part | Model | Sources | Type | Code |
|---|---|---|---|---|
| [A](#part-a-classical-fixed-charge-location) | capacitated fixed-charge location | thesis Model 4.1 | binary program | [`location.py`](../src/sc_microgrid/location.py) |
| [B](#part-b-warehouses-with-on-site-microgrids-wmla) | warehouses with on-site microgrids (WMLA) | thesis Models 4.2 and 4.4; SEGRE 2023 | MILP; bilinear form solved by enumeration | [`wmla.py`](../src/sc_microgrid/wmla.py) |
| [C](#part-c-factories-warehouses-and-microgrids-fmla) | factories, warehouses and on-site microgrids (FMLA) | AMM 2023; thesis Models 4.7 and 5.1 | MILP | [`fmla.py`](../src/sc_microgrid/fmla.py) |
| [D](#part-d-risk-criteria-over-capacity-factor-scenarios) | risk criteria over capacity-factor scenarios | thesis Models 5.2–5.4 | MILP, LP | [`risk.py`](../src/sc_microgrid/risk.py) |
| [E](#part-e-virtual-power-plant-scheduling) | virtual power plant: day-ahead, intraday, rolling horizon | thesis Models 3.1–3.2, Section 3.1.2 | LPs | [`vpp.py`](../src/sc_microgrid/vpp.py) |

[Conventions](#conventions-of-this-implementation) lists the definitions that the framework makes explicit. [Extensions](#extensions-off-by-default-not-part-of-the-published-formulations) lists options that are not part of the published formulations; all are off by default.

---

## Part A: classical fixed-charge location

A set of candidate distribution centres (warehouses) $`n = 1..N`$ serves stores $`s = 1..S`$. Each store is served by exactly one open site (single sourcing).

| Symbol | Meaning | Unit | Code |
|---|---|---|---|
| $`F_n`$ | fixed cost of opening site $`n`$ | \$ | `fixed_cost_usd` |
| $`\gamma_n`$ | capacity of site $`n`$ | items | `capacity_items` |
| $`q_s`$ | demand of store $`s`$ | items | `demand_items` |
| $`c_{ns}`$ | shipping cost | \$/item/km | `ship_cost_usd_per_item_km` |
| $`d_{ns}`$ | distance | km | `distance_km` |
| $`u_n \in \{0,1\}`$ | site $`n`$ is open | – | `u` |
| $`v_{ns} \in \{0,1\}`$ | store $`s`$ is served by site $`n`$ | – | `v` |

```math
\min \; \sum_n F_n u_n + \sum_s \sum_n q_s c_{ns} d_{ns} v_{ns} \qquad (4.1)
```

subject to

```math
\sum_n v_{ns} = 1 \quad \forall s \qquad (4.2)
```

```math
\sum_s q_s v_{ns} \le \gamma_n u_n \quad \forall n \qquad (4.3)
```

```math
v_{ns} \le u_n \quad \forall n, s \qquad (4.4)
```

```math
u_n, v_{ns} \in \{0,1\} \qquad (4.5–4.6)
```

---

## Part B: warehouses with on-site microgrids (WMLA)

Part A is extended with an on-site microgrid at every open warehouse: renewable generation of technologies $`g`$ (wind turbines, PV, …), energy storage (ES) and two-way trade with the main grid in every hour $`t = 1..T`$. Electric trucks carry products to the stores; their energy is part of the warehouse load.

### Parameters

| Symbol | Meaning | Unit | Code (`WarehouseInstance`) |
|---|---|---|---|
| $`F_n`$, $`\gamma_n`$ | annualized fixed cost, capacity of warehouse $`n`$ | \$/yr, items | `fixed_cost_usd`, `capacity_items` |
| $`D_s`$ | demand of store $`s`$ over the horizon | items | `demand_items` |
| $`\pi_{ns}`$, $`d_{ns}`$ | shipping cost, distance | \$/item/km, km | `ship_cost_usd_per_item_km`, `distance_km` |
| $`L_n`$ | power demand of warehouse $`n`$ | MW | `site_load_mw` |
| $`\tau^{w}_{nt}`$ | operating share of warehouse $`n`$ in hour $`t`$ | – | `operating_share` (default 1) |
| $`m_p`$ | unit weight of the product | kg/item | `product_weight_kg` |
| $`m_{EV}`$, $`q_{EV}`$ | truck self-weight, truck energy intensity | kg, MWh/(kg·km) | `truck.weight_kg`, `truck.energy_mwh_per_kg_km` |
| $`a_g`$, $`\phi_g`$ | capacity cost and capital recovery factor of technology $`g`$ | \$/MW, 1/yr | `technologies[g]` |
| $`b_g`$, $`c_g`$ | O&M cost and carbon credit per MWh generated | \$/MWh | `technologies[g]` |
| $`\tau_g`$ | operating hours of $`g`$ per time step | h | `technologies[g].operating_hours_per_step` |
| $`d_{ES}`$, $`\phi_{ES}`$, $`b_{ES}`$ | ES capacity cost, capital recovery factor, maintenance cost | \$/MWh, 1/yr, <span>\$</span>/MWh | `storage` |
| $`P^{max}_{gn}`$ | largest installable capacity | MW | `max_capacity_mw` |
| $`\lambda_{gnt}`$ | capacity factor of $`g`$ at $`n`$ in hour $`t`$ | – | `cf[(n, g)]` |
| $`\rho^{bu}_{nt}`$, $`\rho^{se}_{nt}`$ | electricity purchase and sale prices | \$/MWh | `buy_price_usd_per_mwh`, `sell_price_usd_per_mwh` |
| $`M`$ | big value of thesis Model 4.4 | – | `big_m` |

### Decision variables

| Symbol | Meaning | Unit | Code |
|---|---|---|---|
| $`\hat u_n \in \{0,1\}`$ | warehouse $`n`$ is open | – | `u` |
| $`\hat v_{ns} \in \{0,1\}`$ | store $`s`$ is served by warehouse $`n`$ | – | `v` |
| $`P^c_{gn}`$ | installed capacity of $`g`$ at $`n`$ | MW | `P` |
| $`B^c_n`$ | ES capacity at $`n`$ | MWh | `Bc` |
| $`B_{nt}`$ | energy stored at the end of hour $`t`$ ($`t = 0`$ is the start) | MWh | `B` |
| $`E^{se}_{nt}`$, $`E^{bu}_{nt}`$ | energy sold to and bought from the grid in hour $`t`$ | MWh | `Ese`, `Ebu` |

All continuous variables are non-negative.

### Formulation `"big_m"` (thesis Model 4.4, Eqs. 4.48–4.63)

```math
\begin{aligned}
\min \;& \sum_n F_n \hat u_n + \sum_n \sum_s D_s \pi_{ns} d_{ns} \hat v_{ns} + \sum_g \sum_n \phi_g a_g P^c_{gn} + \sum_t \sum_g \sum_n \tau_g (b_g - c_g) \lambda_{gnt} P^c_{gn}
\\ &+ (\phi_{ES} d_{ES} + b_{ES}) \sum_n B^c_n + \sum_n \sum_t \left(\rho^{bu}_{nt} E^{bu}_{nt} - \rho^{se}_{nt} E^{se}_{nt}\right)
\end{aligned}
\qquad (4.48)
```

subject to the location constraints (4.49)–(4.51), which are (4.2)–(4.4) with $`\gamma_n`$ and $`D_s`$:

```math
\sum_n \hat v_{ns} = 1 \;\; \forall s, \qquad \sum_s D_s \hat v_{ns} \le \gamma_n \hat u_n \;\; \forall n, \qquad \hat v_{ns} \le \hat u_n \;\; \forall n,s
```

the hourly energy balance of an open warehouse

```math
\tau^{w}_{nt} L_n \hat u_n + \frac{1}{T} m_p q_{EV} \sum_s d_{ns} D_s \hat v_{ns} + \frac{1}{T} 2 m_{EV} q_{EV} \sum_s d_{ns} \hat v_{ns} + B_{nt} - B_{n,t-1} + E^{se}_{nt} - E^{bu}_{nt} = \sum_g \tau_g \lambda_{gnt} P^c_{gn} \quad \forall n, t
\qquad (4.52)
```

the big-value bounds, which tie the generation and storage capacities and the sales of a site to its opening decision

```math
P^c_{gn} \le M \hat u_n \;\; \forall g, n, \qquad B^c_n \le M \hat u_n \;\; \forall n \qquad (4.53–4.54)
```

```math
E^{se}_{nt} \le M \hat u_n \;\; \forall n, t \qquad (4.55)
```

and the microgrid limits

```math
0 \le P^c_{gn} \le P^{max}_{gn} \qquad (4.56)
```

```math
0 \le B_{nt} \le B^c_n \qquad (4.57)
```

```math
B_{n,0} = B_{n,T} = B^c_n \qquad (4.58)
```

with (4.59)–(4.63) the variable domains. In (4.52) the second term is the energy to move the products, the third the energy of the empty truck's round trip; both are spread evenly over the $`T`$ hours.

### Formulation `"bilinear"` (thesis Model 4.2, Eqs. 4.7–4.19)

The microgrid and trade terms are multiplied by the location binary instead of being bounded by $`M`$:

```math
\begin{aligned}
\min \;& \sum_n F_n \hat u_n + \sum_n \sum_s D_s \pi_{ns} d_{ns} \hat v_{ns} + \sum_g \sum_n \phi_g a_g P^c_{gn} \hat u_n + \sum_t \sum_g \sum_n \tau_g (b_g - c_g) \lambda_{gnt} P^c_{gn} \hat u_n
\\ &+ (\phi_{ES} d_{ES} + b_{ES}) \sum_n B^c_n \hat u_n + \sum_n \sum_t \left(\rho^{bu}_{nt} E^{bu}_{nt} - \rho^{se}_{nt} E^{se}_{nt}\right) \hat u_n
\end{aligned}
\qquad (4.7)
```

```math
\tau^{w}_{nt} L_n \hat u_n + \frac{1}{T} m_p q_{EV} \sum_s d_{ns} D_s \hat v_{ns} + \frac{1}{T} 2 m_{EV} q_{EV} \sum_s d_{ns} \hat v_{ns} + \left(B_{nt} - B_{n,t-1} + E^{se}_{nt} - E^{bu}_{nt}\right) \hat u_n = \sum_g \tau_g \lambda_{gnt} P^c_{gn} \hat u_n \quad \forall n, t
\qquad (4.11)
```

together with the location constraints (4.8)–(4.10), the microgrid limits (4.12)–(4.14) and the domains (4.15)–(4.19), which are the same as (4.49)–(4.51), (4.56)–(4.58) and (4.59)–(4.63).

**Solving the bilinear form.** For a fixed vector of location binaries, every product in (4.7) and (4.11) has a constant factor and the model is a MILP. [`solve_enumerated`](../src/sc_microgrid/wmla.py) solves it for every set of open warehouses whose capacity covers the total demand (at most $`2^N - 1`$ sets) and keeps the cheapest, which is the optimum of the bilinear model. Sets proven infeasible are skipped; any other solver failure stops the enumeration with an error, so a returned optimum always covers every set. With free binaries the model needs a solver for non-convex quadratic programs (`solve(..., solver="gurobi", options={"NonConvex": 2})`).

| Bilinear form (Model 4.2) | Big-value form (Model 4.4) | Content | Code |
|---|---|---|---|
| (4.7) | (4.48) | objective | `cost` |
| (4.8)–(4.10) | (4.49)–(4.51) | single sourcing, capacity, open site | `eq4_8_single_source`, `eq4_9_capacity`, `eq4_10_open_site` |
| (4.11) | (4.52) | hourly energy balance | `eq4_11_energy_balance` / `eq4_52_energy_balance` |
| – | (4.53)–(4.55) | big-value bounds | `eq4_53_generation_open`, `eq4_54_storage_open`, `eq4_55_sale_open` |
| (4.12)–(4.14) | (4.56)–(4.58) | capacity limit, storage limit, storage full at start and end | `eq4_12_max_capacity`, `eq4_13_storage_level`, `eq4_14_storage_start`, `eq4_14_storage_end` |
| (4.15)–(4.19) | (4.59)–(4.63) | domains | variable domains |

SEGRE 2023 presents the same two-tier model and also gives a linearization of the products of binary and continuous variables, which this framework does not use: it solves the bilinear form by enumeration, or uses the big-value form (4.48)–(4.63).

---

## Part C: factories, warehouses and microgrids (FMLA)

A three-tier network: candidate factories $`k`$ make products $`i`$ in production periods $`j = 1..J`$ (weeks in AMM 2023) and ship them to candidate warehouses $`n`$, which serve the stores $`s`$. Every open factory and warehouse can install renewable generation $`g`$ and ES and trades with the grid in every hour $`t = 1..T`$ of every period. Demand is random and normally distributed. Each store receives a product from one dedicated warehouse; a warehouse can be supplied by several factories.

### Parameters (AMM 2023 Table A1)

| Symbol | Meaning | Unit | Code (`SupplyChainInstance`) |
|---|---|---|---|
| $`F_k`$, $`\hat F_n`$ | annualized cost of opening factory $`k`$, warehouse $`n`$ | \$/yr | `factory_fixed_cost_usd`, `warehouse_fixed_cost_usd` |
| $`L_k`$, $`\hat L_n`$ | base load of an open factory, power demand of a warehouse | MW | `factory_load_mw`, `warehouse_load_mw` |
| $`p_{ijk}`$ | production cost (energy excluded) | \$/item | `production_cost_usd_per_item[(i, k)]` |
| $`e_{ik}`$ | energy to make one item | MWh/item | `energy_mwh_per_item[(i, k)]` |
| $`h_{ijn}`$, $`b_{ijn}`$ | holding and backorder cost | \$/item/period | `holding_cost_usd_per_item_period[i]`, `backorder_cost_usd_per_item_period[i]` |
| $`\pi_{ikn}`$, $`\hat\pi_{ins}`$ | shipping cost factory→warehouse, warehouse→store | \$/item/km | `ship_cost_fw_usd_per_item_km[i]`, `ship_cost_ws_usd_per_item_km[i]` |
| $`d_{kn}`$, $`\hat d_{ns}`$ | distances | km | `distance_fw_km`, `distance_ws_km` |
| $`m_i`$ | unit weight | kg/item | `product_weight_kg` |
| $`w_{ikr}`$ | amount of resource $`r`$ (labour hours, machine hours, …) per item | – | `resource_use[(i, k, r)]` |
| $`W_{jkr}`$ | availability of resource $`r`$ in period $`j`$ at factory $`k`$ | – | `resource_available[(j, k, r)]` |
| $`\gamma_{in}`$ | inventory capacity of warehouse $`n`$ for product $`i`$ | items | `warehouse_capacity_items[(i, n)]` |
| $`D_{ijs}`$ | demand, normal with mean $`E[D_{ijs}]`$ and variance $`Var(D_{ijs})`$ | items/period | `demand_mean_items`, `demand_std_items` ($`\sqrt{Var}`$) |
| $`\alpha`$ | probability of meeting the demand | – | `service_level` |
| $`n_{jknt}`$, $`\hat n_{jnst}`$ | truck trips per hour between the tiers | 1/h | `trips_fw_per_hour`, `trips_ws_per_hour` |
| $`m_{EV}`$, $`q_{EV}`$ | truck self-weight, truck energy intensity | kg, MWh/(kg·km) | `truck` |
| $`a_g`$, $`\phi_g`$, $`b_g`$, $`c_g`$, $`\tau_g`$ | as in Part B | | `technologies[g]` |
| $`d_{ES}`$, $`\phi_{ES}`$, $`b_{ES}`$ | as in Part B | | `storage` |
| $`P^{max}_{gk}`$, $`P^{max}_{gn}`$ | largest installable capacity | MW | `max_capacity_mw[(site, g)]` |
| $`\lambda_{gjkt}`$, $`\lambda_{gjnt}`$ | capacity factors | – | `cf[(site, g)]`, shape $`(J, T)`$ |
| $`\rho^{bu}`$, $`\rho^{se}`$ | purchase and sale prices per site, period and hour | \$/MWh | `buy_price_usd_per_mwh[site]`, `sell_price_usd_per_mwh[site]` |
| $`M`$ | big value | – | `big_m` |

### Decision variables (AMM 2023 Table 2)

| Symbol | Meaning | Code |
|---|---|---|
| $`u_k`$, $`\hat u_n \in \{0,1\}`$ | factory $`k`$, warehouse $`n`$ is open | `u`, `uh` |
| $`v_{ijkn} \in \{0,1\}`$ | product $`i`$ made by factory $`k`$ is shipped to warehouse $`n`$ in period $`j`$ | `v` |
| $`\hat v_{ins} \in \{0,1\}`$ | product $`i`$ of store $`s`$ is served by warehouse $`n`$ | `vh` |
| $`x_{ijkn} \in \mathbb{Z}_{\ge 0}`$ | amount of $`i`$ made by $`k`$ in period $`j`$ and shipped to $`n`$ | `x` |
| $`\hat x_{ijns} \in \mathbb{Z}_{\ge 0}`$ | amount of $`i`$ shipped from $`n`$ to $`s`$ in period $`j`$ | `xh` |
| $`y_{ijn}`$, $`z_{ijn} \in \mathbb{Z}_{\ge 0}`$ | inventory and backorders at the end of period $`j`$ | `y`, `z` |
| $`P^c_{gk}`$, $`P^c_{gn}`$ | installed capacity (MW) | `Pk`, `Pn` |
| $`B^c_k`$, $`B^c_n`$ | ES capacity (MWh) | `Bck`, `Bcn` |
| $`B_{jkt}`$, $`B_{jnt}`$ | energy stored at the end of hour $`t`$ of period $`j`$ (MWh) | `Bk`, `Bn` |
| $`E^{se}_{jkt}`$, $`E^{bu}_{jkt}`$, $`E^{se}_{jnt}`$, $`E^{bu}_{jnt}`$ | energy sold and bought (MWh) | `Esk`, `Ebk`, `Esn`, `Ebn` |

### Formulation (AMM 2023 Eqs. 1–35; thesis Model 4.7, FMLA3)

```math
\begin{aligned}
\min \;& \sum_k F_k u_k + \sum_{i,j,k,n} p_{ijk} x_{ijkn} + \sum_{i,j,k,n} \pi_{ikn} d_{kn} x_{ijkn}
+ \sum_n \hat F_n \hat u_n + \sum_{i,n} \sum_{j=1}^{J} h_{ijn} y_{ijn} + \sum_{i,n} \sum_{j=1}^{J-1} b_{ijn} z_{ijn} + \sum_{i,j,n,s} \hat\pi_{ins} \hat d_{ns} \hat x_{ijns}
\\ &+ \sum_{g,k} \phi_g a_g P^c_{gk} + \sum_{g,j,k,t} \tau_{g} (b_{g} - c_{g}) \lambda_{gjkt} P^c_{gk} + (\phi_{ES} d_{ES} + b_{ES}) \sum_k B^c_k + \sum_{j,k,t} \left(\rho^{bu}_{jkt} E^{bu}_{jkt} - \rho^{se}_{jkt} E^{se}_{jkt}\right)
\\ &+ \text{the same three microgrid terms for the warehouses } n
\end{aligned}
\qquad (1)
```

**Facility location**

```math
\sum_n \hat v_{ins} = 1 \quad \forall i, s \qquad (2)
```

```math
\sum_k x_{ijkn} \le \gamma_{in} \hat u_n \quad \forall i, j, n \qquad (3)
```

```math
\sum_s \hat x_{ijns} \le \gamma_{in} \hat u_n \quad \forall i, j, n \qquad (4)
```

```math
\hat v_{ins} \le \hat u_n \quad \forall i, n, s \qquad (5)
```

with (6)–(7) the domains of $`\hat u`$, $`\hat v`$, $`x`$, $`\hat x`$.

**Production and inventory**

```math
\sum_k x_{i1kn} + y_{i0n} - y_{i1n} + z_{i1n} = \sum_s \hat x_{i1ns} \quad \forall i, n \qquad (8)
```

```math
\sum_k x_{ijkn} + y_{i,j-1,n} - y_{ijn} - z_{i,j-1,n} + z_{ijn} = \sum_s \hat x_{ijns} \quad j = 2..J,\; \forall i, n \qquad (9)
```

```math
\Pr\Big(\sum_n \hat x_{ijns} \ge D_{ijs}\Big) \ge \alpha \quad \forall i, j, s \qquad (10)
```

```math
\sum_i \sum_n w_{ikr} x_{ijkn} \le W_{jkr} u_k \quad \forall j, k, r \qquad (11)
```

```math
x_{ijkn} \le M v_{ijkn} \quad \forall i, j, k, n \qquad (12)
```

```math
\hat x_{ijns} \le M \hat v_{ins} \quad \forall i, j, n, s \qquad (13)
```

```math
y_{i0n} = 0 \quad \forall i, n \qquad (14)
```

with (15)–(16) the domains. For normal demand the chance constraint (10) is replaced by its deterministic equivalent (AMM Section 3.3)

```math
\sum_n \hat x_{ijns} \ge E[D_{ijs}] + z_\alpha \sqrt{Var(D_{ijs})} \quad \forall i, j, s \qquad (35)
```

where $`z_\alpha`$ is the $`\alpha`$-quantile of the standard normal distribution (`z_value`).

**Energy balances.** At a factory, in every hour $`t`$ of every period $`j`$:

```math
L_k u_k + \frac{1}{T} \sum_i \sum_n (e_{ik} + q_{EV} d_{kn} m_i) x_{ijkn} + q_{EV} m_{EV} \sum_i \sum_n n_{jknt} d_{kn} v_{ijkn} + B_{jkt} - B_{jk,t-1} + E^{se}_{jkt} - E^{bu}_{jkt} = \sum_g \tau_{g} \lambda_{gjkt} P^c_{gk}
\qquad (17)
```

and at a warehouse:

```math
\hat L_n \hat u_n + q_{EV} m_{EV} \sum_i \sum_k n_{jknt} d_{kn} v_{ijkn} + \frac{q_{EV}}{T} \sum_i \sum_s \hat d_{ns} m_i \hat x_{ijns} + 2 q_{EV} m_{EV} \sum_i \sum_s \hat n_{jnst} \hat d_{ns} \hat v_{ins} + B_{jnt} - B_{jn,t-1} + E^{se}_{jnt} - E^{bu}_{jnt} = \sum_g \tau_{g} \lambda_{gjnt} P^c_{gn}
\qquad (22)
```

The production quantity of a period is spread evenly over its $`T`$ hours (thesis Eqs. 4.81–4.82: $`x^t_{ijkn} = x_{ijkn} / T`$).

**Microgrid bounds**

```math
0 \le P^c_{gk} \le P^{max}_{gk} u_k \qquad (18)
```

```math
B^c_k \le M u_k,\quad E^{se}_{jkt} \le M u_k,\quad E^{bu}_{jkt} \le M u_k \qquad (19–21)
```

```math
0 \le P^c_{gn} \le P^{max}_{gn} \hat u_n,\quad B^c_n \le M \hat u_n,\quad E^{se}_{jnt} \le M \hat u_n,\quad E^{bu}_{jnt} \le M \hat u_n \qquad (23–26)
```

```math
0 \le B_{jkt} \le B^c_k u_k, \qquad 0 \le B_{jnt} \le B^c_n \hat u_n \qquad (27–28)
```

```math
B_{1k0} = B_{JkT} = B^c_k, \qquad B_{1n0} = B_{JnT} = B^c_n \qquad (29–30)
```

with (31)–(34) the non-negativity of the energy and storage variables. Periods are consecutive, so storage carries over from one period to the next (thesis Eqs. 4.85–4.86):

```math
B_{jk0} = B_{j-1,k,T}, \qquad B_{jn0} = B_{j-1,n,T} \qquad j = 2..J
```

**Thesis Model 4.7 (FMLA3)** has the same structure (thesis Eqs. 4.111–4.121 with 4.65–4.68, 4.70–4.71, 4.76–4.78, 4.81–4.94, 4.105–4.109). Its factory balance (4.112) has no base-load term; set `factory_load_mw` to zero to obtain it. The thesis also states bilinear forms of this model (Section 4.3); the framework implements the big-value form.

### Capacity-factor scenarios (thesis Model 5.1, SFMLA)

Thesis Chapter 5 treats the wind and solar capacity factors as random, with scenarios $`\omega = 1..W`$ of probability $`\beta_\omega`$ ($`\sum_\omega \beta_\omega = 1`$, Eq. 5.2). Model 5.1 is the big-value model with the constraints of FMLA3 (Eqs. 5.3–5.40) and scenario-weighted capacity factors $`\sum_\omega \beta_\omega \lambda^\omega`$. This framework uses the weighted values everywhere the capacity factors appear: in the O&M term of the objective (5.1) and in both hourly balances, at factories (5.17) and at warehouses (5.22).

In code:

```python
from dataclasses import replace
inst5 = replace(inst, factory_load_mw={k: 0.0 for k in inst.factories})   # (5.17) has no factory base load
inst5 = with_cf(inst5, expected_cf([cf_1, cf_2, cf_3], [beta_1, beta_2, beta_3]))
m = build_fmla(inst5)                                                      # big-value formulation
```

The capacity bounds (5.18), (5.23) ($`P^c \le M u`$) and (5.35) ($`P^c \le P^{max}`$) are implemented as (18) and (23) ($`P^c \le P^{max} u`$); see the conventions.

---

## Part D: risk criteria over capacity-factor scenarios

The cost $`C_\omega`$ of scenario $`\omega`$ is the optimal cost of the deterministic model under that scenario's capacity factors (`scenario_costs`). Three criteria evaluate the scenario costs at a confidence level $`\phi \in (0, 1)`$:

**VaR (Model 5.2)**

```math
\min\; VaR \qquad (5.41)
```

```math
C_\omega - VaR \le M \delta_\omega \quad \forall \omega \qquad (5.42)
```

```math
\sum_\omega \beta_\omega \delta_\omega \le 1 - \phi, \qquad \delta_\omega \in \{0, 1\} \qquad (5.43)
```

**CVaR (Model 5.3)**

```math
\min\; \zeta + \frac{1}{1 - \phi} \sum_\omega \beta_\omega \eta_\omega \qquad (5.45)
```

```math
C_\omega - \zeta \le \eta_\omega \quad \forall \omega \qquad (5.46)
```

```math
\eta_\omega \ge 0 \qquad (5.47)
```

**Minimax (Model 5.4)**

```math
\min\; M \qquad (5.48)
```

```math
M \ge C_\omega \quad \forall \omega \qquad (5.49)
```

The thesis uses these criteria to evaluate the scenario results; they do not change the design of a scenario. The code solves each model (`var_model`, `cvar_model`, `minimax_model`) and also gives the closed forms (`var_value`: the smallest scenario cost $`V`$ with $`\Pr(C > V) \le 1 - \phi`$; `cvar_value`: $`\min_\zeta \zeta + E[(C - \zeta)^+]/(1 - \phi)`$; `minimax_value`: $`\max_\omega C_\omega`$). The tests check that both agree.

---

## Part E: virtual power plant scheduling

A virtual power plant (VPP) combines a CHP unit, renewable generation, a battery and a thermal store. It first sells its expected surplus in the day-ahead market; on the delivery day the sales are fixed and shortfalls are bought as imbalance energy.

| Symbol | Meaning | Unit | Code (`VPPInstance`) |
|---|---|---|---|
| $`E^p_t`$, $`P^{th}_t`$ | electrical and thermal load | MW | `load_mw`, `heat_load_mw` (forecast); `*_realized_mw` (delivery day) |
| $`E^r_t`$ | renewable output | MW | `renewable_mw`, `renewable_realized_mw` |
| $`\tau^{se}_t`$ | day-ahead price | \$/MWh | `day_ahead_price_usd_per_mwh` |
| $`c_p`$ | price of imbalance energy | \$/MWh | `imbalance_price_usd_per_mwh` |
| $`c_f`$, $`c_o`$ | CHP fuel cost, storage operating cost per MWh charged or discharged | \$/MWh | `fuel_cost_usd_per_mwh`, `storage_cost_usd_per_mwh` |
| $`p_t`$ | CHP heat-to-power ratio | – | `heat_to_power` |
| $`E^{chp}_{min}`$, $`E^{chp}_{max}`$ | CHP output limits | MW | `chp_min_mw`, `chp_max_mw` |
| $`S^e_{min}`$, $`S^e_{max}`$, $`S^{th}_{min}`$, $`S^{th}_{max}`$ | battery and thermal storage limits | MWh | `battery_*_mwh`, `thermal_*_mwh` |
| $`E^{chp}_t`$ | CHP electrical output | MW | `chp` |
| $`E^m_t`$ | energy sold day-ahead | MW | `sale` |
| $`E^b_t`$ | imbalance energy bought on the delivery day | MW | `buy` |
| $`E^c_t`$, $`E^d_t`$ | battery charge and discharge | MW | `e_ch`, `e_dis` |
| $`P^c_t`$, $`P^d_t`$ | thermal storage charge and discharge | MW | `th_ch`, `th_dis` |
| $`S^e_t`$, $`S^{th}_t`$ | battery and thermal storage levels | MWh | `soc`, `sth` |

**Model DA (Model 3.1)**

```math
\min \sum_{t=1}^{T} \left(c_f E^{chp}_t + c_o E^c_t + c_o E^d_t - \tau^{se}_t E^m_t\right) \qquad (3.1)
```

```math
E^{chp}_t + E^r_t + E^d_t - E^c_t - E^m_t = E^p_t \qquad (3.2)
```

```math
p_t E^{chp}_t + P^d_t - P^c_t = P^{th}_t \qquad (3.3)
```

```math
S^e_t = S^e_{t-1} + E^c_t - E^d_t \qquad (3.4)
```

```math
S^{th}_t = S^{th}_{t-1} + P^c_t - P^d_t \qquad (3.5)
```

```math
S^{th}_{min} \le S^{th}_t \le S^{th}_{max} \qquad (3.6)
```

```math
S^e_{min} \le S^e_t \le S^e_{max} \qquad (3.7)
```

```math
E^{chp}_{min} \le E^{chp}_t \le E^{chp}_{max} \qquad (3.8)
```

```math
E^m_t, E^c_t, E^d_t, P^d_t, P^c_t \ge 0 \qquad (3.9)
```

```math
S^{th}_0 = S^e_0 = 0 \qquad (3.10)
```

for all $`t = 1..T`$, on the forecast series.

**Model ID (Model 3.2).** With $`E^m_t`$ fixed at the day-ahead values and the delivery-day series:

```math
\min \sum_{t=1}^{T} \left(c_f E^{chp}_t + c_o E^c_t + c_o E^d_t + c_p E^b_t - \tau^{se}_t E^m_t\right) \qquad (3.11)
```

```math
E^{chp}_t + E^r_t + E^d_t + E^b_t - E^c_t - E^m_t = E^p_t \qquad (3.12)
```

and (3.13)–(3.20), which repeat (3.3)–(3.10).

**Rolling horizon (Section 3.1.2, Fig. 3.1).** The scheduling horizon (24 h) is covered by iterations. In each iteration Model ID is solved over a prediction horizon of $`PH`$ hours from the current storage levels; the decisions of the first $`CH`$ hours (the control horizon) are implemented, the storage levels are updated, and the window moves forward by $`CH`$ hours. The cost of the schedule is the sum over the implemented hours (`rolling_horizon`).

---

## Conventions of this implementation

**Units and horizon**

- Money in \$, power in MW, energy in MWh, one time step = one hour, product quantities in items per period. Names in the input files carry their units ([data_pipeline.md](data_pipeline.md)).
- Fixed costs, capacity costs and ES costs are annual; operating terms are summed over the modelled horizon. For a horizon shorter than a year (examples, tests, screening runs), `horizon_weight` $`W`$ multiplies every operating term (production, transport, holding, backorders, O&M net of carbon credits and grid trade) so that they stay annual next to the annual costs; for a full year $`W = 1`$. Truck and transport energy are spread over the modelled $`T`$ hours as in (17), (22) and (4.52).

**Index reductions in the input format**

- $`p_{ijk}`$ is given per product and factory, and $`h_{ijn}`$, $`b_{ijn}`$, $`\pi_{ikn}`$, $`\hat\pi_{ins}`$ per product; they are held constant over the other indices.
- $`b_{gk}`$, $`b_{gn}`$, $`c_{gk}`$, $`c_{gn}`$, $`\tau_{gk}`$, $`\tau_{gn}`$ are given per technology.
- The trip rates $`n_{jknt}`$, $`\hat n_{jnst}`$ are given per link (trips per hour) and held constant over periods and hours. For time-varying schedules, build the instance in Python and adapt `factory_load`/`warehouse_load` in `fmla.py`.
- $`p_t`$, $`c_f`$, $`c_o`$ of Part E are held constant over $`t`$; $`c_p`$ may vary by hour.

**Three-tier model**

- Service level: the deterministic equivalent (35) applies per product, period and store to the total shipped by all warehouses, $`\sum_n \hat x_{ijns}`$ (thesis Eqs. 4.78 and 5.16 are likewise indexed by product, period and store).
- Capacity bounds: thesis FMLA3 and Model 5.1 bound $`P^c`$ by $`M u`$ ((4.113), (4.118), (5.18), (5.23)) and by $`P^{max}`$ ((4.89), (5.35)). With $`M \ge P^{max}`$ these two bounds together equal AMM (18) and (23), $`P^c \le P^{max} u`$, which the code uses; `build_fmla` refuses a `big_m` below the largest $`P^{max}`$.
- Last period: no backorder is opened in the last period, $`z_{iJn} = 0`$ (thesis Eq. 4.74), consistent with the backorder cost of (1), which runs over $`j = 1..J-1`$.
- Storage limits (27)–(28): with (19) and (24), $`B^c_k u_k = B^c_k`$ and $`B^c_n \hat u_n = B^c_n`$ at every feasible point, so the big-value formulation writes the bounds as $`B_{jkt} \le B^c_k`$ and $`B_{jnt} \le B^c_n`$, which are linear.
- Storage carries over between consecutive periods (thesis Eqs. 4.85–4.86), and is full at the start and at the end of the horizon (29)–(30).
- Product quantities are integers (Eqs. 7 and 16). `integer_flows=False` relaxes them (extension).

**Big value $`M`$**

- $`M`$ must not cut off a feasible solution: it has to exceed every quantity it bounds (shipments in (12)–(13), ES capacity and hourly trade in (19)–(26) and (4.53)–(4.55)). Values far larger than needed make the LP relaxation weak and can let a solver accept a "closed" site that still trades a little energy, within its integrality tolerance. Choose the smallest safe value, for example the largest possible shipment or hourly energy of the instance.

**Two-tier models**

- In the big-value formulation, the framework bounds both the generation capacities ($`P^c_{gn} \le M \hat u_n`$, `eq4_53_generation_open`) and the storage capacity ($`B^c_n \le M \hat u_n`$, `eq4_54_storage_open`) of a site by its opening decision, thesis (4.53)–(4.54).

- When the location binaries are fixed (enumeration), the microgrid variables of a closed warehouse are fixed at zero: a closed warehouse has no microgrid. Their objective terms are multiplied by $`\hat u_n = 0`$ in (4.7) in any case.

**Risk criteria**

- $`C_\omega`$ is the optimal cost of the deterministic big-value model under scenario $`\omega`$; scenarios can be any capacity-factor arrays (`with_cf`), or scale factors of a base case capped at 1 (`scale_cf`, `scenario_costs`).
- In Model 5.2, $`M`$ defaults to the range of the scenario costs plus one, which never cuts off a VaR between the smallest and the largest cost; a smaller $`M`$ is refused.

**VPP**

- $`E^b_t \ge 0`$; the day-ahead sales are an input of Model ID.
- The initial storage levels of (3.10) are inputs (`battery_initial_mwh`, `thermal_initial_mwh`, default 0, at most the storage capacity); the rolling horizon passes the implemented levels from one window to the next.
- At the end of the scheduling horizon the prediction window is truncated. Inside the control horizon the window uses the delivery-day values; beyond it, the forecasts (`forecast_beyond_control=True`) or the delivery-day values.
- Equations (3.2), (3.3), (3.12) and (3.13) are balances without a spill term. With fixed day-ahead sales, a delivery-day surplus of power has to be stored in the battery and a surplus of heat in the thermal store; the spill extension below adds explicit outlets.

---

## Extensions (off by default; not part of the published formulations)

| Option | Effect |
|---|---|
| `build_fmla(..., integer_flows=False)` | production, shipments, inventory and backorders become continuous |
| `build_fmla(..., no_microgrid=True)` | all generation and ES capacities fixed at zero (grid supply only) |
| `intraday(..., allow_spill=True)`, `rolling_horizon(..., allow_spill=True)` | non-negative spill terms on the left-hand side of (3.12) and (3.13), with zero cost; needs non-negative imbalance prices |
| `day_ahead(..., allow_spill=True)` | a non-negative heat-dump term in (3.3) |
