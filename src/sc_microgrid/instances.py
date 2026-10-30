"""The data interface: one dataclass per model family, with validation.

* :class:`FixedChargeInstance`  classical capacitated fixed-charge location (thesis Model 4.1).
* :class:`WarehouseInstance`    warehouses, stores and on-site microgrids (thesis Models 4.2 and 4.4;
  the two-tier model of SEGRE 2023).
* :class:`SupplyChainInstance`  factories, warehouses, stores and on-site microgrids (AMM 2023
  FMLA; thesis Models 4.7 and 5.1).
* :class:`VPPInstance`          a virtual power plant with CHP, battery and thermal storage
  (thesis Models 3.1 and 3.2).

Units are part of the names (``_usd``, ``_mw``, ``_mwh``, ``_km``, ``_kg``, ``_items``). Hourly series
are numpy arrays: shape ``(T,)`` for the two-layer model and the VPP, ``(J, T)`` (period, hour of the
period) for the three-layer model. Hours are 1..T in the models; arrays are indexed from 0.
Every ``validate`` raises ``ValueError`` with the name of the offending input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------------------------- checks

def _finite(name: str, value: float, lower: Optional[float] = 0.0, upper: Optional[float] = None) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if lower is not None and v < lower:
        raise ValueError(f"{name} must be >= {lower}, got {v}")
    if upper is not None and v > upper:
        raise ValueError(f"{name} must be <= {upper}, got {v}")
    return v


def _names(kind: str, names) -> List[str]:
    out = [str(n) for n in names]
    if not out:
        raise ValueError(f"{kind}: at least one is needed")
    if len(set(out)) != len(out):
        raise ValueError(f"{kind}: names must be unique, got {out}")
    return out


def _table(name: str, table: Mapping, keys, lower: Optional[float] = 0.0, upper: Optional[float] = None) -> dict:
    """Check that ``table`` has exactly the keys ``keys`` and finite values in [lower, upper]."""
    keys = list(keys)
    missing = [k for k in keys if k not in table]
    if missing:
        raise ValueError(f"{name}: missing entries for {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    extra = [k for k in table if k not in set(keys)]
    if extra:
        raise ValueError(f"{name}: unknown entries {extra[:5]}{' ...' if len(extra) > 5 else ''}")
    return {k: _finite(f"{name}[{k}]", table[k], lower, upper) for k in keys}


def _series(name: str, a, shape: Tuple[int, ...], lower: Optional[float] = 0.0, upper: Optional[float] = None) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.shape != shape:
        raise ValueError(f"{name} has shape {a.shape}, expected {shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN or infinite values")
    if lower is not None and np.any(a < lower):
        raise ValueError(f"{name} contains values below {lower}")
    if upper is not None and np.any(a > upper):
        raise ValueError(f"{name} contains values above {upper}")
    return a


# ----------------------------------------------------------------------------- shared microgrid data

@dataclass
class Technology:
    """A renewable generation technology g (wind turbine, PV, ...)."""

    capacity_cost_usd_per_mw: float          # a_g
    capital_recovery_factor: float           # phi_g (1/yr)
    om_cost_usd_per_mwh: float               # b_g
    carbon_credit_usd_per_mwh: float = 0.0   # c_g
    operating_hours_per_step: float = 1.0    # tau_g: hours of operation per time step

    def validate(self, name: str) -> None:
        _finite(f"technology {name}: capacity_cost_usd_per_mw", self.capacity_cost_usd_per_mw)
        _finite(f"technology {name}: capital_recovery_factor", self.capital_recovery_factor, 0.0, 1.0)
        _finite(f"technology {name}: om_cost_usd_per_mwh", self.om_cost_usd_per_mwh)
        _finite(f"technology {name}: carbon_credit_usd_per_mwh", self.carbon_credit_usd_per_mwh)
        _finite(f"technology {name}: operating_hours_per_step", self.operating_hours_per_step, 0.0, 1.0)


@dataclass
class Storage:
    """Energy storage (ES) at a site."""

    capacity_cost_usd_per_mwh: float         # d_ES
    capital_recovery_factor: float           # phi_ES (1/yr)
    maintenance_cost_usd_per_mwh: float      # b_ES, per MWh of capacity and year

    def validate(self) -> None:
        _finite("storage: capacity_cost_usd_per_mwh", self.capacity_cost_usd_per_mwh)
        _finite("storage: capital_recovery_factor", self.capital_recovery_factor, 0.0, 1.0)
        _finite("storage: maintenance_cost_usd_per_mwh", self.maintenance_cost_usd_per_mwh)

    @property
    def annual_cost_usd_per_mwh(self) -> float:
        """(phi_ES d_ES + b_ES): annualized capacity cost plus maintenance, per MWh of capacity."""
        return self.capital_recovery_factor * self.capacity_cost_usd_per_mwh + self.maintenance_cost_usd_per_mwh


@dataclass
class Truck:
    """Electric trucks that move products between tiers."""

    weight_kg: float                         # m_EV, truck self-weight
    energy_mwh_per_kg_km: float              # q_EV, energy per kg and km

    def validate(self) -> None:
        _finite("truck: weight_kg", self.weight_kg)
        _finite("truck: energy_mwh_per_kg_km", self.energy_mwh_per_kg_km)


# ---------------------------------------------------------------------- classical fixed-charge model

@dataclass
class FixedChargeInstance:
    """Capacitated fixed-charge location with single sourcing (thesis Model 4.1)."""

    sites: List[str]                                     # n: candidate warehouse (DC) sites
    stores: List[str]                                    # s: demand nodes
    fixed_cost_usd: Dict[str, float]                     # F_n
    capacity_items: Dict[str, float]                     # gamma_n
    demand_items: Dict[str, float]                       # q_s
    ship_cost_usd_per_item_km: Dict[Tuple[str, str], float]   # c_ns
    distance_km: Dict[Tuple[str, str], float]            # d_ns

    def validate(self) -> "FixedChargeInstance":
        self.sites = _names("sites", self.sites)
        self.stores = _names("stores", self.stores)
        self.fixed_cost_usd = _table("fixed_cost_usd", self.fixed_cost_usd, self.sites)
        self.capacity_items = _table("capacity_items", self.capacity_items, self.sites)
        self.demand_items = _table("demand_items", self.demand_items, self.stores)
        pairs = [(n, s) for n in self.sites for s in self.stores]
        self.ship_cost_usd_per_item_km = _table("ship_cost_usd_per_item_km", self.ship_cost_usd_per_item_km, pairs)
        self.distance_km = _table("distance_km", self.distance_km, pairs)
        if sum(self.capacity_items.values()) < sum(self.demand_items.values()):
            raise ValueError("total site capacity is below total demand: no feasible assignment")
        return self


# ------------------------------------------------------------------------------ two-layer WMLA model

@dataclass
class WarehouseInstance:
    """Warehouses with on-site microgrids serving stores (thesis Models 4.2, 4.4; SEGRE 2023).

    ``cf[(n, g)]``, ``buy_price_usd_per_mwh[n]``, ``sell_price_usd_per_mwh[n]`` and ``operating_share[n]``
    have shape (T,). ``demand_items`` is the demand of the whole horizon; the transport energy is
    spread evenly over its T hours.
    """

    sites: List[str]
    stores: List[str]
    technologies: Dict[str, Technology]
    storage: Storage
    truck: Truck
    hours: int                                           # T
    fixed_cost_usd: Dict[str, float]                     # F_n (annualized)
    capacity_items: Dict[str, float]                     # gamma_n
    demand_items: Dict[str, float]                       # D_s
    ship_cost_usd_per_item_km: Dict[Tuple[str, str], float]   # pi_ns
    distance_km: Dict[Tuple[str, str], float]            # d_ns
    site_load_mw: Dict[str, float]                       # L_n
    product_weight_kg: float                             # m_p
    max_capacity_mw: Dict[Tuple[str, str], float]        # P^max_gn, key (site, technology)
    cf: Dict[Tuple[str, str], np.ndarray]                # lambda_gnt
    buy_price_usd_per_mwh: Dict[str, np.ndarray]         # rho^bu_nt
    sell_price_usd_per_mwh: Dict[str, np.ndarray]        # rho^se_nt
    operating_share: Optional[Dict[str, np.ndarray]] = None   # tau^w_nt in [0, 1]; default 1
    big_m: float = 1.0e4                                 # M of thesis Model 4.4
    horizon_weight: float = 1.0                          # see docs/formulation.md, conventions

    def validate(self) -> "WarehouseInstance":
        self.sites = _names("sites", self.sites)
        self.stores = _names("stores", self.stores)
        if not self.technologies:
            raise ValueError("technologies: at least one is needed")
        for g, tech in self.technologies.items():
            tech.validate(g)
        self.storage.validate()
        self.truck.validate()
        if int(self.hours) != self.hours or self.hours < 1:
            raise ValueError(f"hours must be a positive integer, got {self.hours}")
        self.hours = int(self.hours)
        T, G = self.hours, list(self.technologies)
        self.fixed_cost_usd = _table("fixed_cost_usd", self.fixed_cost_usd, self.sites)
        self.capacity_items = _table("capacity_items", self.capacity_items, self.sites)
        self.demand_items = _table("demand_items", self.demand_items, self.stores)
        pairs = [(n, s) for n in self.sites for s in self.stores]
        self.ship_cost_usd_per_item_km = _table("ship_cost_usd_per_item_km", self.ship_cost_usd_per_item_km, pairs)
        self.distance_km = _table("distance_km", self.distance_km, pairs)
        self.site_load_mw = _table("site_load_mw", self.site_load_mw, self.sites)
        self.product_weight_kg = _finite("product_weight_kg", self.product_weight_kg)
        site_tech = [(n, g) for n in self.sites for g in G]
        self.max_capacity_mw = _table("max_capacity_mw", self.max_capacity_mw, site_tech)
        _table("cf", {k: 0.0 for k in self.cf}, site_tech)          # same keys
        self.cf = {k: _series(f"cf[{k}]", self.cf[k], (T,), 0.0, 1.0) for k in site_tech}
        self.buy_price_usd_per_mwh = {n: _series(f"buy_price_usd_per_mwh[{n}]", self.buy_price_usd_per_mwh[n], (T,), None)
                                      for n in _keys("buy_price_usd_per_mwh", self.buy_price_usd_per_mwh, self.sites)}
        self.sell_price_usd_per_mwh = {n: _series(f"sell_price_usd_per_mwh[{n}]", self.sell_price_usd_per_mwh[n], (T,), None)
                                       for n in _keys("sell_price_usd_per_mwh", self.sell_price_usd_per_mwh, self.sites)}
        if self.operating_share is None:
            self.operating_share = {n: np.ones(T) for n in self.sites}
        self.operating_share = {n: _series(f"operating_share[{n}]", self.operating_share[n], (T,), 0.0, 1.0)
                                for n in _keys("operating_share", self.operating_share, self.sites)}
        self.big_m = _finite("big_m", self.big_m, 0.0)
        if self.big_m == 0:
            raise ValueError("big_m must be positive")
        self.horizon_weight = _finite("horizon_weight", self.horizon_weight, 0.0)
        if sum(self.capacity_items.values()) < sum(self.demand_items.values()):
            raise ValueError("total warehouse capacity is below total demand: no feasible assignment")
        return self


def _keys(name: str, table: Mapping, keys) -> List:
    _table(name, {k: 0.0 for k in table}, keys)
    return list(keys)


# ---------------------------------------------------------------------------- three-layer FMLA model

@dataclass
class SupplyChainInstance:
    """Factories, warehouses and stores with on-site microgrids (AMM 2023 FMLA; thesis Ch. 4-5).

    Hourly arrays ``cf[(site, g)]``, ``buy_price_usd_per_mwh[site]`` and ``sell_price_usd_per_mwh[site]``
    have shape (J, T); a site is a factory or a warehouse, so their names must differ. Product
    flows are per period (e.g. a week of T = 168 hours).
    """

    factories: List[str]                                 # k
    warehouses: List[str]                                # n
    stores: List[str]                                    # s
    products: List[str]                                  # i
    resources: List[str]                                 # r (e.g. labour hours, machine hours)
    technologies: Dict[str, Technology]                  # g
    storage: Storage
    truck: Truck
    periods: int                                         # J
    hours: int                                           # T, hours per period
    factory_fixed_cost_usd: Dict[str, float]             # F_k (annualized)
    warehouse_fixed_cost_usd: Dict[str, float]           # F^_n (annualized)
    factory_load_mw: Dict[str, float]                    # L_k, base load of an open factory
    warehouse_load_mw: Dict[str, float]                  # L^_n
    production_cost_usd_per_item: Dict[Tuple[str, str], float]   # p_ik, key (product, factory)
    energy_mwh_per_item: Dict[Tuple[str, str], float]            # e_ik, key (product, factory)
    holding_cost_usd_per_item_period: Dict[str, float]           # h_i
    backorder_cost_usd_per_item_period: Dict[str, float]         # b_i
    ship_cost_fw_usd_per_item_km: Dict[str, float]               # pi_i, factory -> warehouse
    ship_cost_ws_usd_per_item_km: Dict[str, float]               # pi^_i, warehouse -> store
    product_weight_kg: Dict[str, float]                          # m_i
    resource_use: Dict[Tuple[str, str, str], float]              # w_ikr, key (product, factory, resource)
    resource_available: Dict[Tuple[int, str, str], float]        # W_jkr, key (period, factory, resource)
    warehouse_capacity_items: Dict[Tuple[str, str], float]       # gamma_in, key (product, warehouse)
    demand_mean_items: Dict[Tuple[str, int, str], float]         # E[D_ijs], key (product, period, store)
    demand_std_items: Dict[Tuple[str, int, str], float]          # sqrt(Var(D_ijs))
    service_level: float                                         # alpha
    distance_fw_km: Dict[Tuple[str, str], float]                 # d_kn
    distance_ws_km: Dict[Tuple[str, str], float]                 # d^_ns
    trips_fw_per_hour: Dict[Tuple[str, str], float]              # n_jknt, held constant over j and t
    trips_ws_per_hour: Dict[Tuple[str, str], float]              # n^_jnst, held constant over j and t
    max_capacity_mw: Dict[Tuple[str, str], float]                # P^max, key (site, technology)
    cf: Dict[Tuple[str, str], np.ndarray]                        # lambda, key (site, technology), (J, T)
    buy_price_usd_per_mwh: Dict[str, np.ndarray]                 # rho^bu, (J, T)
    sell_price_usd_per_mwh: Dict[str, np.ndarray]                # rho^se, (J, T)
    big_m: float = 1.0e4                                         # M
    horizon_weight: float = 1.0
    meta: Dict[str, str] = field(default_factory=dict)

    @property
    def sites(self) -> List[str]:
        return list(self.factories) + list(self.warehouses)

    def validate(self) -> "SupplyChainInstance":
        self.factories = _names("factories", self.factories)
        self.warehouses = _names("warehouses", self.warehouses)
        self.stores = _names("stores", self.stores)
        self.products = _names("products", self.products)
        self.resources = _names("resources", self.resources)
        _names("factories and warehouses", self.sites)
        if not self.technologies:
            raise ValueError("technologies: at least one is needed")
        for g, tech in self.technologies.items():
            tech.validate(g)
        self.storage.validate()
        self.truck.validate()
        for name in ("periods", "hours"):
            v = getattr(self, name)
            if int(v) != v or v < 1:
                raise ValueError(f"{name} must be a positive integer, got {v}")
            setattr(self, name, int(v))
        J, T, G = self.periods, self.hours, list(self.technologies)
        K, N, S, I, R = self.factories, self.warehouses, self.stores, self.products, self.resources
        P = range(1, J + 1)
        self.factory_fixed_cost_usd = _table("factory_fixed_cost_usd", self.factory_fixed_cost_usd, K)
        self.warehouse_fixed_cost_usd = _table("warehouse_fixed_cost_usd", self.warehouse_fixed_cost_usd, N)
        self.factory_load_mw = _table("factory_load_mw", self.factory_load_mw, K)
        self.warehouse_load_mw = _table("warehouse_load_mw", self.warehouse_load_mw, N)
        ik = [(i, k) for i in I for k in K]
        self.production_cost_usd_per_item = _table("production_cost_usd_per_item", self.production_cost_usd_per_item, ik)
        self.energy_mwh_per_item = _table("energy_mwh_per_item", self.energy_mwh_per_item, ik)
        self.holding_cost_usd_per_item_period = _table("holding_cost_usd_per_item_period", self.holding_cost_usd_per_item_period, I)
        self.backorder_cost_usd_per_item_period = _table("backorder_cost_usd_per_item_period", self.backorder_cost_usd_per_item_period, I)
        self.ship_cost_fw_usd_per_item_km = _table("ship_cost_fw_usd_per_item_km", self.ship_cost_fw_usd_per_item_km, I)
        self.ship_cost_ws_usd_per_item_km = _table("ship_cost_ws_usd_per_item_km", self.ship_cost_ws_usd_per_item_km, I)
        self.product_weight_kg = _table("product_weight_kg", self.product_weight_kg, I)
        self.resource_use = _table("resource_use", self.resource_use, [(i, k, r) for i in I for k in K for r in R])
        self.resource_available = _table("resource_available", self.resource_available,
                                         [(j, k, r) for j in P for k in K for r in R])
        self.warehouse_capacity_items = _table("warehouse_capacity_items", self.warehouse_capacity_items,
                                               [(i, n) for i in I for n in N])
        ijs = [(i, j, s) for i in I for j in P for s in S]
        self.demand_mean_items = _table("demand_mean_items", self.demand_mean_items, ijs)
        self.demand_std_items = _table("demand_std_items", self.demand_std_items, ijs)
        self.service_level = _finite("service_level", self.service_level, 0.0, 1.0)
        if not 0.0 < self.service_level < 1.0:
            raise ValueError("service_level must lie strictly between 0 and 1")
        kn, ns = [(k, n) for k in K for n in N], [(n, s) for n in N for s in S]
        self.distance_fw_km = _table("distance_fw_km", self.distance_fw_km, kn)
        self.distance_ws_km = _table("distance_ws_km", self.distance_ws_km, ns)
        self.trips_fw_per_hour = _table("trips_fw_per_hour", self.trips_fw_per_hour, kn)
        self.trips_ws_per_hour = _table("trips_ws_per_hour", self.trips_ws_per_hour, ns)
        site_tech = [(x, g) for x in self.sites for g in G]
        self.max_capacity_mw = _table("max_capacity_mw", self.max_capacity_mw, site_tech)
        _table("cf", {k: 0.0 for k in self.cf}, site_tech)
        self.cf = {k: _series(f"cf[{k}]", self.cf[k], (J, T), 0.0, 1.0) for k in site_tech}
        self.buy_price_usd_per_mwh = {x: _series(f"buy_price_usd_per_mwh[{x}]", self.buy_price_usd_per_mwh[x], (J, T), None)
                                      for x in _keys("buy_price_usd_per_mwh", self.buy_price_usd_per_mwh, self.sites)}
        self.sell_price_usd_per_mwh = {x: _series(f"sell_price_usd_per_mwh[{x}]", self.sell_price_usd_per_mwh[x], (J, T), None)
                                       for x in _keys("sell_price_usd_per_mwh", self.sell_price_usd_per_mwh, self.sites)}
        self.big_m = _finite("big_m", self.big_m, 0.0)
        if self.big_m == 0:
            raise ValueError("big_m must be positive")
        self.horizon_weight = _finite("horizon_weight", self.horizon_weight, 0.0)
        return self


# ----------------------------------------------------------------------------------------- VPP model

@dataclass
class VPPInstance:
    """A virtual power plant: CHP, renewables, battery and thermal storage (thesis Ch. 3).

    Series of shape (T,): electrical load E^p_t, thermal load P^th_t, renewable output E^r_t,
    day-ahead price, imbalance price. Optional ``*_realized`` series are the values seen on the
    delivery day (intraday model); they default to the forecast series.
    """

    load_mw: np.ndarray                      # E^p_t
    heat_load_mw: np.ndarray                 # P^th_t
    renewable_mw: np.ndarray                 # E^r_t
    day_ahead_price_usd_per_mwh: np.ndarray  # tau^se_t
    imbalance_price_usd_per_mwh: np.ndarray  # c_p (per hour)
    fuel_cost_usd_per_mwh: float             # c_f
    storage_cost_usd_per_mwh: float          # c_o, per MWh charged or discharged
    heat_to_power: float                     # p_t, CHP heat-to-power ratio (held constant)
    chp_min_mw: float
    chp_max_mw: float
    battery_min_mwh: float
    battery_max_mwh: float
    thermal_min_mwh: float
    thermal_max_mwh: float
    battery_initial_mwh: float = 0.0         # Eq. (3.10)
    thermal_initial_mwh: float = 0.0
    load_realized_mw: Optional[np.ndarray] = None
    heat_load_realized_mw: Optional[np.ndarray] = None
    renewable_realized_mw: Optional[np.ndarray] = None

    @property
    def hours(self) -> int:
        return int(np.asarray(self.load_mw).shape[0])

    def validate(self) -> "VPPInstance":
        T = self.hours
        if T < 1:
            raise ValueError("load_mw must have at least one hour")
        for name in ("load_mw", "heat_load_mw", "renewable_mw"):
            setattr(self, name, _series(name, getattr(self, name), (T,)))
        for name in ("day_ahead_price_usd_per_mwh", "imbalance_price_usd_per_mwh"):
            setattr(self, name, _series(name, getattr(self, name), (T,), None))
        for name, base in (("load_realized_mw", "load_mw"), ("heat_load_realized_mw", "heat_load_mw"),
                           ("renewable_realized_mw", "renewable_mw")):
            v = getattr(self, name)
            setattr(self, name, getattr(self, base).copy() if v is None else _series(name, v, (T,)))
        for name in ("fuel_cost_usd_per_mwh", "storage_cost_usd_per_mwh", "heat_to_power", "chp_min_mw", "chp_max_mw",
                     "battery_min_mwh", "battery_max_mwh", "thermal_min_mwh", "thermal_max_mwh",
                     "battery_initial_mwh", "thermal_initial_mwh"):
            setattr(self, name, _finite(name, getattr(self, name)))
        for lo, hi in (("chp_min_mw", "chp_max_mw"), ("battery_min_mwh", "battery_max_mwh"),
                       ("thermal_min_mwh", "thermal_max_mwh")):
            if getattr(self, lo) > getattr(self, hi):
                raise ValueError(f"{lo} exceeds {hi}")
        if self.battery_initial_mwh > self.battery_max_mwh:
            raise ValueError("battery_initial_mwh exceeds battery_max_mwh")
        if self.thermal_initial_mwh > self.thermal_max_mwh:
            raise ValueError("thermal_initial_mwh exceeds thermal_max_mwh")
        return self
