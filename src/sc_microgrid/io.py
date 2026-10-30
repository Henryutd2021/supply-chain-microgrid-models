"""Read and write instance directories: ``instance.json`` plus, for hourly models, ``hourly.csv``.

The formats are described in docs/data_pipeline.md. Reading validates everything: unknown or
missing keys, unit-carrying names, complete hourly grids, finite values and ranges. Errors name
the file and the entry.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd

from .instances import (FixedChargeInstance, Storage, SupplyChainInstance, Technology, Truck, VPPInstance,
                        WarehouseInstance)

Instance = Union[FixedChargeInstance, WarehouseInstance, SupplyChainInstance, VPPInstance]
MODELS = {"fixed_charge": FixedChargeInstance, "wmla": WarehouseInstance, "fmla": SupplyChainInstance, "vpp": VPPInstance}

# key name -> number of index levels (0 = scalar or list); levels listed in INT_LEVELS hold period numbers
FIXED_CHARGE_KEYS = {"sites": 0, "stores": 0, "fixed_cost_usd": 1, "capacity_items": 1, "demand_items": 1,
                     "ship_cost_usd_per_item_km": 2, "distance_km": 2}
WMLA_KEYS = {"sites": 0, "stores": 0, "hours": 0, "fixed_cost_usd": 1, "capacity_items": 1, "demand_items": 1,
             "ship_cost_usd_per_item_km": 2, "distance_km": 2, "site_load_mw": 1, "product_weight_kg": 0,
             "max_capacity_mw": 2, "big_m": 0, "horizon_weight": 0}
FMLA_KEYS = {"factories": 0, "warehouses": 0, "stores": 0, "products": 0, "resources": 0, "periods": 0, "hours": 0,
             "factory_fixed_cost_usd": 1, "warehouse_fixed_cost_usd": 1, "factory_load_mw": 1, "warehouse_load_mw": 1,
             "production_cost_usd_per_item": 2, "energy_mwh_per_item": 2, "holding_cost_usd_per_item_period": 1,
             "backorder_cost_usd_per_item_period": 1, "ship_cost_fw_usd_per_item_km": 1,
             "ship_cost_ws_usd_per_item_km": 1, "product_weight_kg": 1, "resource_use": 3, "resource_available": 3,
             "warehouse_capacity_items": 2, "demand_mean_items": 3, "demand_std_items": 3, "service_level": 0,
             "distance_fw_km": 2, "distance_ws_km": 2, "trips_fw_per_hour": 2, "trips_ws_per_hour": 2,
             "max_capacity_mw": 2, "big_m": 0, "horizon_weight": 0}
INT_LEVELS = {"resource_available": (0,), "demand_mean_items": (1,), "demand_std_items": (1,)}
VPP_SCALARS = ["fuel_cost_usd_per_mwh", "storage_cost_usd_per_mwh", "heat_to_power", "chp_min_mw", "chp_max_mw",
               "battery_min_mwh", "battery_max_mwh", "thermal_min_mwh", "thermal_max_mwh",
               "battery_initial_mwh", "thermal_initial_mwh"]
VPP_SERIES = ["load_mw", "heat_load_mw", "renewable_mw", "day_ahead_price_usd_per_mwh", "imbalance_price_usd_per_mwh",
              "load_realized_mw", "heat_load_realized_mw", "renewable_realized_mw"]
MICROGRID_KEYS = {"technologies", "storage", "truck"}


# ------------------------------------------------------------------------------------------ helpers

def _nest(table: Dict, depth: int):
    if depth <= 1:
        return {str(k if not isinstance(k, tuple) else k[0]): float(v) for k, v in table.items()}
    out: Dict = {}
    for key, v in table.items():
        node = out
        for part in key[:-1]:
            node = node.setdefault(str(part), {})
        node[str(key[-1])] = float(v)
    return out


def _flat(name: str, obj, depth: int, int_levels: Tuple[int, ...] = ()) -> Dict:
    def conv(level, k):
        if level in int_levels:
            try:
                return int(k)
            except ValueError:
                raise ValueError(f"{name}: period key {k!r} is not an integer") from None
        return k

    if not isinstance(obj, dict):
        raise ValueError(f"{name}: expected a JSON object")
    if depth == 1:
        return {conv(0, k): v for k, v in obj.items()}
    out = {}

    def walk(node, prefix):
        level = len(prefix)
        if level == depth:
            out[tuple(prefix)] = node
            return
        if not isinstance(node, dict):
            raise ValueError(f"{name}: expected {depth} levels of keys, found a value at {prefix}")
        for k, v in node.items():
            walk(v, prefix + [conv(level, k)])
    walk(obj, [])
    return out


def _check_keys(path: Path, data: dict, expected: Iterable[str]) -> None:
    expected = set(expected) | {"model"}
    unknown = sorted(set(data) - expected)
    if unknown:
        raise ValueError(f"{path}: unknown keys {unknown} (names carry units; see docs/data_pipeline.md)")
    optional = {"big_m", "horizon_weight", "battery_initial_mwh", "thermal_initial_mwh"}
    missing = sorted(expected - set(data) - optional)
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")


def _microgrid_to_json(inst) -> dict:
    return {"technologies": {g: asdict(t) for g, t in inst.technologies.items()},
            "storage": asdict(inst.storage), "truck": asdict(inst.truck)}


def _microgrid_from_json(path: Path, data: dict) -> dict:
    try:
        techs = {g: Technology(**t) for g, t in data["technologies"].items()}
        return {"technologies": techs, "storage": Storage(**data["storage"]), "truck": Truck(**data["truck"])}
    except TypeError as exc:
        raise ValueError(f"{path}: technologies, storage or truck: {exc}") from None


def _read_hourly(path: Path, index: List[str], columns: List[str]) -> pd.DataFrame:
    if not path.exists():
        raise ValueError(f"{path}: file not found")
    # site names stay text exactly as written ("001", "NA"); empty or non-numeric values in the
    # numeric columns are then reported by the check below instead of becoming NaN
    df = pd.read_csv(path, dtype={"site": str}, keep_default_na=False)
    missing = [c for c in index + columns if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    unknown = [c for c in df.columns if c not in index + columns]
    if unknown:
        raise ValueError(f"{path}: unknown columns {unknown}")
    if df.duplicated(index).any():
        raise ValueError(f"{path}: duplicate rows for {df[df.duplicated(index)][index].iloc[0].tolist()}")
    for c in columns:
        if not pd.api.types.is_numeric_dtype(df[c]) or not np.all(np.isfinite(df[c].to_numpy(dtype=float))):
            raise ValueError(f"{path}: column {c} must hold finite numbers")
    return df


# -------------------------------------------------------------------------------------------- write

def write_instance(inst: Instance, directory: Union[str, Path]) -> Path:
    """Write ``instance.json`` (and ``hourly.csv``) to ``directory``; returns the directory."""
    inst.validate()
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    if isinstance(inst, FixedChargeInstance):
        data = {"model": "fixed_charge", "sites": inst.sites, "stores": inst.stores}
        for k, depth in FIXED_CHARGE_KEYS.items():
            if depth:
                data[k] = _nest(getattr(inst, k), depth)
    elif isinstance(inst, WarehouseInstance):
        data = {"model": "wmla", **_microgrid_to_json(inst)}
        for k, depth in WMLA_KEYS.items():
            v = getattr(inst, k)
            data[k] = _nest(v, depth) if depth else v
        rows = []
        for n in inst.sites:
            for t in range(inst.hours):
                r = {"hour": t + 1, "site": n}
                r.update({f"cf_{g}": inst.cf[n, g][t] for g in inst.technologies})
                r.update({"buy_price_usd_per_mwh": inst.buy_price_usd_per_mwh[n][t],
                          "sell_price_usd_per_mwh": inst.sell_price_usd_per_mwh[n][t],
                          "operating_share": inst.operating_share[n][t]})
                rows.append(r)
        pd.DataFrame(rows).to_csv(d / "hourly.csv", index=False)
    elif isinstance(inst, SupplyChainInstance):
        data = {"model": "fmla", **_microgrid_to_json(inst)}
        for k, depth in FMLA_KEYS.items():
            v = getattr(inst, k)
            data[k] = _nest(v, depth) if depth else v
        rows = []
        for x in inst.sites:
            for j in range(inst.periods):
                for t in range(inst.hours):
                    r = {"period": j + 1, "hour": t + 1, "site": x}
                    r.update({f"cf_{g}": inst.cf[x, g][j, t] for g in inst.technologies})
                    r.update({"buy_price_usd_per_mwh": inst.buy_price_usd_per_mwh[x][j, t],
                              "sell_price_usd_per_mwh": inst.sell_price_usd_per_mwh[x][j, t]})
                    rows.append(r)
        pd.DataFrame(rows).to_csv(d / "hourly.csv", index=False)
    elif isinstance(inst, VPPInstance):
        data = {"model": "vpp", **{k: getattr(inst, k) for k in VPP_SCALARS}}
        df = pd.DataFrame({"hour": np.arange(1, inst.hours + 1), **{k: getattr(inst, k) for k in VPP_SERIES}})
        df.to_csv(d / "hourly.csv", index=False)
    else:
        raise TypeError(f"unsupported instance type {type(inst).__name__}")
    (d / "instance.json").write_text(json.dumps(data, indent=1) + "\n")
    return d


# --------------------------------------------------------------------------------------------- read

def read_instance(directory: Union[str, Path]) -> Instance:
    """Read and validate an instance directory written by :func:`write_instance` or by hand."""
    d = Path(directory)
    path = d / "instance.json"
    if not path.exists():
        raise ValueError(f"{path}: file not found")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not valid JSON ({exc})") from None
    model = data.get("model")
    if model not in MODELS:
        raise ValueError(f"{path}: 'model' must be one of {sorted(MODELS)}, got {model!r}")
    try:
        if model == "fixed_charge":
            _check_keys(path, data, FIXED_CHARGE_KEYS)
            kw = {k: (_flat(k, data[k], depth) if depth else data[k]) for k, depth in FIXED_CHARGE_KEYS.items()}
            return FixedChargeInstance(**kw).validate()
        if model == "vpp":
            _check_keys(path, data, VPP_SCALARS)
            df = _read_hourly(d / "hourly.csv", ["hour"], VPP_SERIES)
            if df["hour"].tolist() != list(range(1, len(df) + 1)):
                raise ValueError(f"{d / 'hourly.csv'}: hours must run 1, 2, ..., T in order")
            kw = {k: data[k] for k in VPP_SCALARS if k in data}
            kw.update({k: df[k].to_numpy(dtype=float) for k in VPP_SERIES})
            return VPPInstance(**kw).validate()
        keys = WMLA_KEYS if model == "wmla" else FMLA_KEYS
        _check_keys(path, data, set(keys) | MICROGRID_KEYS)
        kw = {k: (_flat(k, data[k], depth, INT_LEVELS.get(k, ())) if depth else data[k])
              for k, depth in keys.items() if k in data}
        kw.update(_microgrid_from_json(path, data))
        techs = list(kw["technologies"])
        cf_cols = [f"cf_{g}" for g in techs]
        hourly = d / "hourly.csv"
        if model == "wmla":
            df = _read_hourly(hourly, ["hour", "site"], cf_cols + ["buy_price_usd_per_mwh", "sell_price_usd_per_mwh",
                                                                    "operating_share"])
            T, sites = int(data["hours"]), [str(s) for s in data["sites"]]
            _check_grid(hourly, df, {"site": sites, "hour": range(1, T + 1)})
            df = df.astype({"site": str}).sort_values(["site", "hour"])
            by = {n: g for n, g in df.groupby("site")}
            kw["cf"] = {(n, g): by[n][f"cf_{g}"].to_numpy(float) for n in sites for g in techs}
            kw["buy_price_usd_per_mwh"] = {n: by[n]["buy_price_usd_per_mwh"].to_numpy(float) for n in sites}
            kw["sell_price_usd_per_mwh"] = {n: by[n]["sell_price_usd_per_mwh"].to_numpy(float) for n in sites}
            kw["operating_share"] = {n: by[n]["operating_share"].to_numpy(float) for n in sites}
            return WarehouseInstance(**kw).validate()
        df = _read_hourly(hourly, ["period", "hour", "site"], cf_cols + ["buy_price_usd_per_mwh", "sell_price_usd_per_mwh"])
        J, T = int(data["periods"]), int(data["hours"])
        sites = [str(s) for s in data["factories"]] + [str(s) for s in data["warehouses"]]
        _check_grid(hourly, df, {"site": sites, "period": range(1, J + 1), "hour": range(1, T + 1)})
        df = df.astype({"site": str}).sort_values(["site", "period", "hour"])
        by = {x: g for x, g in df.groupby("site")}
        kw["cf"] = {(x, g): by[x][f"cf_{g}"].to_numpy(float).reshape(J, T) for x in sites for g in techs}
        kw["buy_price_usd_per_mwh"] = {x: by[x]["buy_price_usd_per_mwh"].to_numpy(float).reshape(J, T) for x in sites}
        kw["sell_price_usd_per_mwh"] = {x: by[x]["sell_price_usd_per_mwh"].to_numpy(float).reshape(J, T) for x in sites}
        return SupplyChainInstance(**kw).validate()
    except ValueError as exc:
        msg = str(exc)
        raise ValueError(msg if str(d) in msg else f"{d}: {msg}") from None


def _check_grid(path: Path, df: pd.DataFrame, levels: Dict[str, Iterable]) -> None:
    """Every combination of the listed index values must appear exactly once (duplicates are checked before)."""
    expected = 1
    for col, values in levels.items():
        values = [str(v) if col == "site" else v for v in values]
        got = set(df[col].astype(str) if col == "site" else df[col])
        missing, extra = set(values) - got, got - set(values)
        if missing or extra:
            raise ValueError(f"{path}: column {col}: missing {sorted(missing)[:5]}, unexpected {sorted(extra)[:5]}")
        expected *= len(values)
    if len(df) != expected:
        raise ValueError(f"{path}: expected {expected} rows (complete grid), found {len(df)}")
