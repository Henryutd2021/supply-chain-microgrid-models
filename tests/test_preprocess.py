import numpy as np
import pandas as pd
import pytest

from sc_microgrid.preprocess import demand_statistics, pv_cf, to_periods, tou_profile, wind_cf


def test_wind_cf_power_curve():
    cf = wind_cf([0, 2.9, 3.0, 7.5, 12.0, 20.0, 25.0, 30.0])
    assert cf[0] == cf[1] == cf[2] == 0.0
    assert 0 < cf[3] < 1
    assert cf[4] == cf[5] == 1.0
    assert cf[6] == cf[7] == 0.0
    assert np.all(np.diff(wind_cf(np.linspace(3, 12, 50))) > 0)
    higher = wind_cf([6.0], measured_height_m=10, hub_height_m=80)
    assert higher[0] > wind_cf([6.0])[0]
    with pytest.raises(ValueError):
        wind_cf([-1.0])
    with pytest.raises(ValueError):
        wind_cf([5.0], measured_height_m=10)
    with pytest.raises(ValueError):
        wind_cf([5.0], measured_height_m=float("nan"), hub_height_m=80)
    with pytest.raises(ValueError):
        pv_cf([500.0], reference_w_m2=-1000)


def test_pv_cf():
    assert np.allclose(pv_cf([0, 500, 1000, 1200]), [0, 0.5, 1.0, 1.0])
    hot = pv_cf([800.0], ambient_temp_c=[40.0])
    cold = pv_cf([800.0], ambient_temp_c=[-10.0])     # cell at 15 C
    assert hot[0] < 0.8 < cold[0]


def test_to_periods_and_tou():
    a = to_periods(np.arange(8760), 52, 168)
    assert a.shape == (52, 168) and a[1, 0] == 168
    with pytest.raises(ValueError):
        to_periods(np.arange(100), 1, 168)
    p = tou_profile(48, 90.0, 40.0, range(10, 22))
    assert p[9] == 40.0 and p[10] == 90.0 and p[34] == 90.0


def test_demand_statistics():
    rec = pd.DataFrame({"product": ["A"] * 4, "period": [1, 1, 2, 2], "store": ["S1"] * 4, "quantity": [10, 14, 20, 20]})
    mean, std = demand_statistics(rec)
    assert mean[("A", 1, "S1")] == 12 and std[("A", 2, "S1")] == 0
    assert std[("A", 1, "S1")] == pytest.approx(np.std([10, 14], ddof=1))
