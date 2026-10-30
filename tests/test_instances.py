import numpy as np
import pytest

from sc_microgrid import synthetic_fixed_charge, synthetic_supply_chain, synthetic_vpp, synthetic_warehouses


def test_synthetic_instances_validate():
    for inst in (synthetic_fixed_charge(), synthetic_warehouses(hours=24), synthetic_supply_chain(), synthetic_vpp()):
        assert inst.validate() is inst


@pytest.mark.parametrize("change, message", [
    (lambda i: i.fixed_cost_usd.update({"DC-A": -1.0}), "must be >= 0"),
    (lambda i: i.fixed_cost_usd.update({"DC-A": float("nan")}), "finite"),
    (lambda i: i.demand_items.pop("S1"), "missing entries"),
    (lambda i: i.distance_km.update({("DC-X", "S1"): 1.0}), "unknown entries"),
    (lambda i: setattr(i, "sites", ["DC-A", "DC-A", "DC-C"]), "unique"),
    (lambda i: i.capacity_items.update({k: 1.0 for k in i.capacity_items}), "below total demand"),
])
def test_fixed_charge_validation(change, message):
    inst = synthetic_fixed_charge()
    change(inst)
    with pytest.raises(ValueError, match=message):
        inst.validate()


def test_supply_chain_validation():
    inst = synthetic_supply_chain()
    inst.cf["F-East", "pv"] = inst.cf["F-East", "pv"] * 2 + 1
    with pytest.raises(ValueError, match="above 1"):
        inst.validate()
    inst = synthetic_supply_chain()
    inst.cf["F-East", "pv"] = inst.cf["F-East", "pv"][:, :-1]
    with pytest.raises(ValueError, match="shape"):
        inst.validate()
    inst = synthetic_supply_chain()
    inst.service_level = 1.0
    with pytest.raises(ValueError, match="strictly"):
        inst.validate()
    inst = synthetic_supply_chain()
    inst.warehouses = ["F-East", "W-South"]
    with pytest.raises(ValueError):
        inst.validate()
    inst = synthetic_supply_chain()
    inst.technologies["pv"].capital_recovery_factor = 1.5
    with pytest.raises(ValueError, match="capital_recovery_factor"):
        inst.validate()


def test_warehouse_and_vpp_validation():
    inst = synthetic_warehouses(hours=24)
    inst.buy_price_usd_per_mwh["W-North"] = np.full(24, np.inf)
    with pytest.raises(ValueError, match="NaN or infinite"):
        inst.validate()
    v = synthetic_vpp()
    v.chp_min_mw = v.chp_max_mw + 1
    with pytest.raises(ValueError, match="exceeds"):
        v.validate()
