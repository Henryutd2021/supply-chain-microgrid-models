"""Synthetic example instances.

Small invented regions with round, illustrative costs. They show how the models behave and how
to read their output; their numbers say nothing about any real location, company or market.
Every generator is deterministic for a given ``seed``.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .instances import (FixedChargeInstance, Storage, SupplyChainInstance, Technology, Truck, VPPInstance,
                        WarehouseInstance)

HOURS_PER_YEAR = 8760


def capital_recovery_factor(rate: float, years: int) -> float:
    """Annuity factor r (1+r)^n / ((1+r)^n - 1)."""
    if rate == 0:
        return 1.0 / years
    f = (1 + rate) ** years
    return rate * f / (f - 1)


def illustrative_technologies() -> Dict[str, Technology]:
    return {
        "wind": Technology(capacity_cost_usd_per_mw=1.3e6, capital_recovery_factor=capital_recovery_factor(0.06, 25),
                           om_cost_usd_per_mwh=9.0, carbon_credit_usd_per_mwh=2.0),
        "pv": Technology(capacity_cost_usd_per_mw=0.9e6, capital_recovery_factor=capital_recovery_factor(0.06, 25),
                         om_cost_usd_per_mwh=3.0, carbon_credit_usd_per_mwh=6.0),
    }


def illustrative_storage() -> Storage:
    return Storage(capacity_cost_usd_per_mwh=0.3e6, capital_recovery_factor=capital_recovery_factor(0.06, 10),
                   maintenance_cost_usd_per_mwh=2.0)


def illustrative_truck() -> Truck:
    return Truck(weight_kg=6000.0, energy_mwh_per_kg_km=1.0e-7)


def synthetic_cf(rng: np.random.Generator, hours: int, kind: str, quality: float = 1.0) -> np.ndarray:
    """An hourly capacity-factor series starting at midnight.

    ``pv``: a daylight bell (hours 6-18) scaled by a daily clearness drawn in [0.4, 1].
    ``wind``: a mean-reverting random walk around 0.3. ``quality`` scales the series (capped at 1).
    """
    t = np.arange(hours)
    if kind == "pv":
        hod = t % 24
        bell = np.clip(np.sin(np.pi * (hod - 6) / 12), 0, None)
        clear = rng.uniform(0.4, 1.0, size=hours // 24 + 1)[t // 24]
        cf = 0.85 * bell * clear
    elif kind == "wind":
        cf = np.empty(hours)
        x = 0.3
        for h in range(hours):
            x = 0.3 + 0.85 * (x - 0.3) + rng.normal(0, 0.08)
            cf[h] = x
    else:
        raise ValueError(f"unknown capacity-factor kind {kind!r}")
    return np.clip(cf * quality, 0.0, 1.0)


def tou_price(hours: int, peak: float, offpeak: float, peak_hours: Tuple[int, int] = (8, 20)) -> np.ndarray:
    """A daily time-of-use profile: ``peak`` in [peak_hours[0], peak_hours[1]) of each day."""
    hod = np.arange(hours) % 24
    return np.where((hod >= peak_hours[0]) & (hod < peak_hours[1]), peak, offpeak).astype(float)


# ---------------------------------------------------------------------------------- fixed charge

def synthetic_fixed_charge(seed: int = 3) -> FixedChargeInstance:
    rng = np.random.default_rng(seed)
    sites, stores = ["DC-A", "DC-B", "DC-C"], [f"S{i}" for i in range(1, 7)]
    return FixedChargeInstance(
        sites=sites, stores=stores,
        fixed_cost_usd={"DC-A": 5000.0, "DC-B": 8000.0, "DC-C": 11000.0},
        capacity_items={"DC-A": 400.0, "DC-B": 450.0, "DC-C": 500.0},
        demand_items={s: float(rng.integers(60, 141)) for s in stores},
        ship_cost_usd_per_item_km={(n, s): float(rng.choice([0.5, 1.0, 1.5])) for n in sites for s in stores},
        distance_km={(n, s): float(rng.integers(20, 121)) for n in sites for s in stores},
    ).validate()


# -------------------------------------------------------------------------------------------- WMLA

def synthetic_warehouses(hours: int = 168, seed: int = 5) -> WarehouseInstance:
    """Three candidate warehouses with different wind and sun, five stores, one modelled week
    (``hours``) weighted to a year."""
    rng = np.random.default_rng(seed)
    sites, stores = ["W-North", "W-Central", "W-South"], [f"S{i}" for i in range(1, 6)]
    techs = illustrative_technologies()
    quality = {"W-North": {"wind": 1.3, "pv": 0.8}, "W-Central": {"wind": 1.0, "pv": 1.0},
               "W-South": {"wind": 0.7, "pv": 1.25}}
    year_share = hours / HOURS_PER_YEAR
    demand_year = {s: float(rng.integers(40, 81)) * 1000 for s in stores}
    return WarehouseInstance(
        sites=sites, stores=stores, technologies=techs, storage=illustrative_storage(), truck=illustrative_truck(),
        hours=hours,
        fixed_cost_usd={"W-North": 1.6e6, "W-Central": 2.0e6, "W-South": 1.5e6},
        capacity_items={n: 180000.0 * year_share for n in sites},
        demand_items={s: v * year_share for s, v in demand_year.items()},
        ship_cost_usd_per_item_km={(n, s): 0.04 for n in sites for s in stores},
        distance_km={(n, s): float(rng.integers(50, 401)) for n in sites for s in stores},
        site_load_mw={n: 2.0 for n in sites},
        product_weight_kg=10.0,
        max_capacity_mw={(n, g): 10.0 for n in sites for g in techs},
        cf={(n, g): synthetic_cf(rng, hours, g, quality[n][g]) for n in sites for g in techs},
        buy_price_usd_per_mwh={n: tou_price(hours, 85.0, 45.0) for n in sites},
        sell_price_usd_per_mwh={n: np.full(hours, 40.0) for n in sites},
        big_m=1.0e3,
        horizon_weight=HOURS_PER_YEAR / hours,
    ).validate()


# -------------------------------------------------------------------------------------------- FMLA

def synthetic_supply_chain(periods: int = 4, hours: int = 24, seed: int = 7) -> SupplyChainInstance:
    """Three candidate factories, two candidate warehouses, three stores, two products.

    ``periods`` production periods of ``hours`` hours each (one day by default), weighted to a year.
    """
    rng = np.random.default_rng(seed)
    K, N, S = ["F-East", "F-Mid", "F-West"], ["W-North", "W-South"], ["S1", "S2", "S3"]
    I, R = ["A", "B"], ["labour_h", "machine_h"]
    J, T = periods, hours
    P = range(1, J + 1)
    techs = illustrative_technologies()
    quality = {"F-East": {"wind": 0.8, "pv": 1.2}, "F-Mid": {"wind": 1.1, "pv": 1.0},
               "F-West": {"wind": 1.3, "pv": 0.9}, "W-North": {"wind": 1.2, "pv": 0.8},
               "W-South": {"wind": 0.8, "pv": 1.2}}
    sites = K + N

    def per_period(series):
        return series.reshape(J, T)

    mean = {(i, j, s): float(rng.integers(300, 601)) * (1.0 if i == "A" else 0.6) for i in I for j in P for s in S}
    std = {key: round(0.12 * v, 1) for key, v in mean.items()}
    total_peak = max(sum(mean[i, j, s] * 1.3 for i in I for s in S) for j in P)
    use = {("A", k, "labour_h"): 1.0 for k in K} | {("B", k, "labour_h"): 1.5 for k in K} \
        | {("A", k, "machine_h"): 0.5 for k in K} | {("B", k, "machine_h"): 0.8 for k in K}
    avail = {(j, k, r): round(total_peak * (1.6 if r == "labour_h" else 0.9) * f, 0)
             for j in P for k, f in zip(K, (0.55, 0.45, 0.5)) for r in R}
    hours_per_year = J * T
    return SupplyChainInstance(
        factories=K, warehouses=N, stores=S, products=I, resources=R, technologies=techs,
        storage=illustrative_storage(), truck=illustrative_truck(), periods=J, hours=T,
        factory_fixed_cost_usd={"F-East": 2.4e6, "F-Mid": 2.0e6, "F-West": 2.6e6},
        warehouse_fixed_cost_usd={"W-North": 0.9e6, "W-South": 1.1e6},
        factory_load_mw={k: 1.0 for k in K},
        warehouse_load_mw={n: 0.5 for n in N},
        production_cost_usd_per_item={("A", "F-East"): 52.0, ("A", "F-Mid"): 48.0, ("A", "F-West"): 50.0,
                                       ("B", "F-East"): 78.0, ("B", "F-Mid"): 82.0, ("B", "F-West"): 75.0},
        energy_mwh_per_item={(i, k): (0.02 if i == "A" else 0.03) for i in I for k in K},
        holding_cost_usd_per_item_period={"A": 0.3, "B": 0.5},
        backorder_cost_usd_per_item_period={"A": 6.0, "B": 9.0},
        ship_cost_fw_usd_per_item_km={"A": 0.02, "B": 0.03},
        ship_cost_ws_usd_per_item_km={"A": 0.03, "B": 0.04},
        product_weight_kg={"A": 5.0, "B": 8.0},
        resource_use=use, resource_available=avail,
        warehouse_capacity_items={(i, n): 4000.0 for i in I for n in N},
        demand_mean_items=mean, demand_std_items=std, service_level=0.9,
        distance_fw_km={(k, n): float(rng.integers(100, 601)) for k in K for n in N},
        distance_ws_km={(n, s): float(rng.integers(100, 601)) for n in N for s in S},
        trips_fw_per_hour={(k, n): 1.0 / 24 for k in K for n in N},
        trips_ws_per_hour={(n, s): 1.0 / 24 for n in N for s in S},
        max_capacity_mw={(x, g): 5.0 for x in sites for g in techs},
        cf={(x, g): per_period(synthetic_cf(rng, J * T, g, quality[x][g])) for x in sites for g in techs},
        buy_price_usd_per_mwh={x: per_period(tou_price(J * T, 85.0, 45.0)) for x in sites},
        sell_price_usd_per_mwh={x: np.full((J, T), 40.0) for x in sites},
        big_m=1.0e4,
        horizon_weight=HOURS_PER_YEAR / hours_per_year,
        meta={"description": "synthetic example; invented numbers"},
    ).validate()


def synthetic_cf_scenarios() -> Tuple[Dict[str, float], Dict[str, float]]:
    """Three capacity-factor scenarios as scale factors of the base series, and their probabilities."""
    return {"low": 0.8, "medium": 1.0, "high": 1.2}, {"low": 0.15, "medium": 0.6, "high": 0.25}


# --------------------------------------------------------------------------------------------- VPP

def synthetic_vpp(hours: int = 24, seed: int = 11, deviation: float = 0.06) -> VPPInstance:
    """A day of a VPP whose heat demand keeps the CHP running.

    The realized (delivery-day) series have up to ``deviation`` less renewable output and up to
    ``deviation``/2 more electrical load than the forecasts; the heat load is as forecast. Every
    shortfall can then be bought as imbalance energy. (With fixed day-ahead sales, a realized
    surplus has to go into the battery; see the spill extension in docs/formulation.md.)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(hours)
    hod = t % 24
    load = 75 + 25 * np.sin(np.pi * (hod - 7) / 14).clip(0) + rng.normal(0, 2, hours)
    heat = 70 + 15 * np.cos(np.pi * (hod - 3) / 12) + rng.normal(0, 2, hours)
    ren = (20 + 30 * np.clip(np.sin(np.pi * (hod - 6) / 12), 0, None) + rng.normal(0, 3, hours).clip(-5, 5)).clip(0, None)
    price = 30 + 25 * np.clip(np.sin(np.pi * (hod - 6) / 16), 0, None)
    return VPPInstance(
        load_mw=load, heat_load_mw=heat, renewable_mw=ren,
        day_ahead_price_usd_per_mwh=price, imbalance_price_usd_per_mwh=np.full(hours, 110.0),
        fuel_cost_usd_per_mwh=35.0, storage_cost_usd_per_mwh=1.5, heat_to_power=0.75,
        chp_min_mw=20.0, chp_max_mw=140.0, battery_min_mwh=0.0, battery_max_mwh=80.0,
        thermal_min_mwh=0.0, thermal_max_mwh=100.0,
        load_realized_mw=load * (1 + rng.uniform(0, deviation / 2, hours)),
        heat_load_realized_mw=heat.copy(),
        renewable_realized_mw=ren * (1 - rng.uniform(0, deviation, hours)),
    ).validate()
