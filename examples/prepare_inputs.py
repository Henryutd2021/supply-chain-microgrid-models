"""From raw hourly weather to capacity factors and a model instance (docs/data_pipeline.md).

Synthetic "weather records" stand in for real ones: hourly wind speed at 10 m, irradiance and
ambient temperature for each site. They are converted with the generic helpers in
sc_microgrid.preprocess, cut into periods, put into a synthetic supply-chain instance and written
as an instance directory that `sc-microgrid run` can solve.
"""

import sys
from pathlib import Path

import numpy as np

from sc_microgrid import read_instance, synthetic_supply_chain, write_instance
from sc_microgrid.preprocess import pv_cf, to_periods, wind_cf

out = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/prepared_case")
inst = synthetic_supply_chain(periods=4, hours=24)
rng = np.random.default_rng(1)
hours = inst.periods * inst.hours
hod = np.arange(hours) % 24
for site in inst.sites:
    speed_10m = np.clip(rng.weibull(2.0, hours) * rng.uniform(4, 7), 0, None)          # m/s
    ghi = 950 * np.clip(np.sin(np.pi * (hod - 6) / 12), 0, None) * rng.uniform(0.5, 1, hours)  # W/m2
    temp = 15 + 10 * np.sin(np.pi * (hod - 9) / 12)                                       # deg C
    inst.cf[site, "wind"] = to_periods(wind_cf(speed_10m, measured_height_m=10, hub_height_m=80), inst.periods, inst.hours)
    inst.cf[site, "pv"] = to_periods(pv_cf(ghi, temp), inst.periods, inst.hours)
inst.validate()
write_instance(inst, out)
back = read_instance(out)
print(f"wrote {out}: {len(back.sites)} sites, {back.periods} periods x {back.hours} h; "
      f"mean wind CF {np.mean([back.cf[x, 'wind'].mean() for x in back.sites]):.3f}, "
      f"mean PV CF {np.mean([back.cf[x, 'pv'].mean() for x in back.sites]):.3f}")
print(f"solve it with:  sc-microgrid run {out} --out-dir outputs/prepared_run")
