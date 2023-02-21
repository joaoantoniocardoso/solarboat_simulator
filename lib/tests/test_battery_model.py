from lib.battery_model import Battery


def test_battery_cycle():
    battery = Battery(
        soc_0=0.5,
        minimum_soc=0.0,
        efficiency=1.0,
        maximum_energy=1.0,
        maximum_power=1e9,
    )

    dt = 0.5
    power = 1.0
    soc_before_cycle = battery.soc

    for i in range(10):
        # Charge
        assert battery.solve(dt, power) == power

        # Discharge
        assert battery.solve(dt, -power) == -power

        # SOC before and after the cycle should be the same
        assert soc_before_cycle == battery.soc


def test_battery_energy_limits():
    maximum_energy = 1.0
    soc_0 = 1.0
    minimum_soc = 0.5
    battery = Battery(
        soc_0=soc_0,
        minimum_soc=minimum_soc,
        efficiency=1.0,
        maximum_energy=maximum_energy,
        maximum_power=1e9,
    )

    dt = 1.0

    # Discharges half of the battery in one second
    power = (soc_0 - minimum_soc) * maximum_energy / (dt / 3600)
    assert battery.solve(dt, -power) == -power
    assert battery.soc == minimum_soc

    # Should fail to discharge the remaining of the battery
    assert battery.solve(dt, -power) == 0.0
    assert battery.soc == minimum_soc

    # Now if we recharge it, should go back to the initial state
    assert battery.solve(dt, power) == power
    assert battery.soc == soc_0

    # And finally, it should not charge beyond full
    assert battery.solve(dt, power) == 0.0
    assert battery.soc == soc_0


def test_battery_power_limit():
    maximum_power = 5.0
    battery = Battery(
        soc_0=0.5,
        minimum_soc=0.0,
        efficiency=1.0,
        maximum_energy=1.0,
        maximum_power=maximum_power,
    )

    dt = 1.0
    power = 10.0

    # Out of the bounds of maximum power should return a limited power
    assert battery.solve(dt, power) == maximum_power
    assert battery.solve(dt, -power) == -maximum_power


def test_battery_charge():
    battery = Battery(
        soc_0=0.0,
        minimum_soc=0.0,
        efficiency=1.0,
        maximum_energy=1.0,
        maximum_power=1e9,
    )

    charge_duration = 0.5  # in hours
    dt = charge_duration * 3600  # in seconds
    power = battery.maximum_energy / charge_duration

    assert battery.solve(dt, power) == power

    assert battery.soc == 1.0
