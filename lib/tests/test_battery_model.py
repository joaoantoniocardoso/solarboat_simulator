from lib.battery_model import Battery

import numpy as np


def test_battery_cycle():
    battery = Battery(
        soc_0=np.float64(0.5),
        minimum_soc=np.float64(0.0),
        efficiency=np.float64(1.0),
        maximum_energy=np.float64(1.0),
        maximum_power=np.float64(1e9),
    )

    dt = np.float64(0.5)
    power = np.float64(1.0)
    soc_before_cycle = battery.soc

    for i in range(10):
        # Charge
        assert battery.solve(dt, power) == power

        # Discharge
        assert battery.solve(dt, -power) == -power

        # SOC before and after the cycle should be the same
        assert soc_before_cycle == battery.soc


def test_battery_energy_limits():
    maximum_energy = np.float64(1.0)
    soc_0 = np.float64(1.0)
    minimum_soc = np.float64(0.5)
    battery = Battery(
        soc_0=soc_0,
        minimum_soc=minimum_soc,
        efficiency=np.float64(1.0),
        maximum_energy=maximum_energy,
        maximum_power=np.float64(1e9),
    )

    dt = np.float64(1.0)

    for i in range(10):
        # Discharges half of the battery in one second
        power = (soc_0 - minimum_soc) * maximum_energy / (dt / np.float64(3600))
        assert battery.solve(dt, -power) == -power
        assert battery.soc == minimum_soc

        for i in range(10):
            # Should fail to discharge the remaining of the battery
            assert battery.solve(dt, -power) == np.float64(0.0)
            assert battery.soc == minimum_soc

        # Now if we recharge it, should go back to the initial state
        assert battery.solve(dt, power) == power
        assert battery.soc == soc_0

        for i in range(10):
            # And finally, it should not charge beyond full
            assert battery.solve(dt, power) == np.float64(0.0)
            assert battery.soc == soc_0


def test_battery_power_limit():
    maximum_power = np.float64(5.0)
    battery = Battery(
        soc_0=np.float64(0.5),
        minimum_soc=np.float64(0.0),
        efficiency=np.float64(1.0),
        maximum_energy=np.float64(1.0),
        maximum_power=maximum_power,
    )

    dt = np.float64(1.0)
    power = np.float64(10.0)

    for i in range(10):
        # Out of the bounds of maximum power should return a limited power
        assert battery.solve(dt, power) == maximum_power
        assert battery.solve(dt, -power) == -maximum_power


def test_battery_charge():
    battery = Battery(
        soc_0=np.float64(0.0),
        minimum_soc=np.float64(0.0),
        efficiency=np.float64(1.0),
        maximum_energy=np.float64(1.0),
        maximum_power=np.float64(1e9),
    )

    duration = np.float64(0.5)  # in hours
    dt = duration * np.float64(3600)  # in seconds
    power = battery.maximum_energy / duration

    assert battery.solve(dt, power) == power
    assert battery.soc == np.float64(1.0)


def test_battery_discharge():
    battery = Battery(
        soc_0=np.float64(1.0),
        minimum_soc=np.float64(0.0),
        efficiency=np.float64(1.0),
        maximum_energy=np.float64(1.0),
        maximum_power=np.float64(1e9),
    )

    duration = np.float64(0.5)  # in hours
    dt = duration * np.float64(3600)  # in seconds
    power = -battery.maximum_energy / duration

    assert battery.solve(dt, power) == power
    assert battery.soc == battery.minimum_soc
