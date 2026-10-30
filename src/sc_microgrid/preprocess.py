"""Generic helpers that turn raw hourly records into model inputs (docs/data_pipeline.md).

These are standard textbook steps, not the procedures of any particular study:

* :func:`wind_cf` - capacity factor from wind speed through a generic power curve, with an
  optional power-law height adjustment;
* :func:`pv_cf` - capacity factor from plane-of-array irradiance with a linear temperature derate;
* :func:`to_periods` - cut an hourly series into J periods of T hours (e.g. 52 weeks of 168 h);
* :func:`demand_statistics` - mean and standard deviation of demand per product, period and store;
* :func:`tou_profile` - an hourly time-of-use price profile.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import math

import numpy as np
import pandas as pd


def _scalar(name: str, value: float, positive: bool = False) -> float:
    v = float(value)
    if not math.isfinite(v) or (positive and v <= 0):
        raise ValueError(f"{name} must be finite{' and positive' if positive else ''}, got {value!r}")
    return v


def wind_cf(speed_m_s, cut_in: float = 3.0, rated: float = 12.0, cut_out: float = 25.0,
            measured_height_m: float | None = None, hub_height_m: float | None = None, shear_exponent: float = 1 / 7) -> np.ndarray:
    """Capacity factor of a generic turbine: 0 below cut-in, cubic rise to 1 at rated speed, 1 up to cut-out, 0 above.

    With both heights given, speeds are first scaled by (hub / measured) ** shear_exponent.
    """
    v = np.asarray(speed_m_s, dtype=float)
    if not np.all(np.isfinite(v)) or np.any(v < 0):
        raise ValueError("wind speeds must be finite and non-negative")
    cut_in, rated, cut_out = (_scalar(n, x) for n, x in (("cut_in", cut_in), ("rated", rated), ("cut_out", cut_out)))
    if not 0 <= cut_in < rated < cut_out:
        raise ValueError("need 0 <= cut_in < rated < cut_out")
    if (measured_height_m is None) != (hub_height_m is None):
        raise ValueError("give both measured_height_m and hub_height_m, or neither")
    if measured_height_m is not None:
        mh, hh = _scalar("measured_height_m", measured_height_m, True), _scalar("hub_height_m", hub_height_m, True)
        alpha = _scalar("shear_exponent", shear_exponent)
        if alpha < 0:
            raise ValueError("shear_exponent must be non-negative")
        v = v * (hh / mh) ** alpha
    cf = np.where(v < cut_in, 0.0, (v ** 3 - cut_in ** 3) / (rated ** 3 - cut_in ** 3))
    cf = np.where(v >= rated, 1.0, cf)
    return np.where(v >= cut_out, 0.0, cf)


def pv_cf(irradiance_w_m2, ambient_temp_c=None, temp_coeff_per_c: float = -0.004, noct_c: float = 45.0,
          reference_w_m2: float = 1000.0) -> np.ndarray:
    """Capacity factor of a PV array: irradiance / 1000 W/m2, derated by cell temperature.

    Cell temperature follows the NOCT approximation T_cell = T_amb + (NOCT - 20) / 800 * G.
    Without ambient temperatures the derate is skipped. The result is capped to [0, 1].
    """
    g = np.asarray(irradiance_w_m2, dtype=float)
    if not np.all(np.isfinite(g)) or np.any(g < 0):
        raise ValueError("irradiance must be finite and non-negative")
    reference_w_m2 = _scalar("reference_w_m2", reference_w_m2, True)
    temp_coeff_per_c, noct_c = _scalar("temp_coeff_per_c", temp_coeff_per_c), _scalar("noct_c", noct_c)
    cf = g / reference_w_m2
    if ambient_temp_c is not None:
        ta = np.asarray(ambient_temp_c, dtype=float)
        if ta.shape != g.shape or not np.all(np.isfinite(ta)):
            raise ValueError("ambient temperatures must be finite and match the irradiance series")
        t_cell = ta + (noct_c - 20.0) / 800.0 * g
        cf = cf * (1 + temp_coeff_per_c * (t_cell - 25.0))
    return np.clip(cf, 0.0, 1.0)


def to_periods(series, periods: int, hours: int, start: int = 0) -> np.ndarray:
    """Hours start .. start + J*T - 1 of an hourly series as a (J, T) array (e.g. 52 x 168 of a year)."""
    a = np.asarray(series, dtype=float)
    need = start + periods * hours
    if a.ndim != 1 or a.shape[0] < need:
        raise ValueError(f"need a 1-D series with at least {need} hours, got shape {a.shape}")
    return a[start:need].reshape(periods, hours)


def demand_statistics(records: pd.DataFrame, product: str = "product", period: str = "period", store: str = "store",
                      quantity: str = "quantity") -> Tuple[dict, dict]:
    """Mean and standard deviation of demand per (product, period, store) from repeated observations.

    ``records`` holds one row per observation (e.g. one row per product, store, week of the year
    and historical year); returns two dicts keyed (product, period, store) for
    :class:`~sc_microgrid.instances.SupplyChainInstance`.
    """
    for c in (product, period, store, quantity):
        if c not in records.columns:
            raise ValueError(f"records need a column {c!r}")
    g = records.groupby([product, period, store])[quantity]
    mean, std = g.mean(), g.std(ddof=1).fillna(0.0)
    key = lambda k: (str(k[0]), int(k[1]), str(k[2]))  # noqa: E731
    return {key(k): float(v) for k, v in mean.items()}, {key(k): float(v) for k, v in std.items()}


def tou_profile(hours: int, peak: float, offpeak: float, peak_hours: Iterable[int] = range(8, 20)) -> np.ndarray:
    """Hourly prices: ``peak`` in the listed hours of the day (0-23), ``offpeak`` otherwise; hour 1 = 0:00-1:00."""
    peak_set = set(int(h) for h in peak_hours)
    hod = np.arange(hours) % 24
    return np.array([peak if h in peak_set else offpeak for h in hod], dtype=float)
