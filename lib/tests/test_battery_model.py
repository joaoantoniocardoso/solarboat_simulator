import numpy as np
from hypothesis import given, example, settings, strategies

import lib.battery_model as battery_model


@given(
    dt=strategies.floats(
        min_value=1e-9,
        allow_infinity=False,
    ),
    minimum_soc=strategies.floats(min_value=0.0, max_value=0.5, exclude_max=True),
    maximum_power=strategies.floats(min_value=0.0, max_value=1e6, exclude_min=True),
)
def test_battery_cycle(
    dt: np.float64,
    minimum_soc: np.float64,
    maximum_power: np.float64,
):
    battery = battery_model.Battery(
        soc_0=np.float64(0.5),
        minimum_soc=minimum_soc,
        efficiency=np.float64(1.0),
        maximum_energy=np.float64(1.0),
        maximum_power=maximum_power,
    )

    power = (
        battery.maximum_energy
        * (battery.soc - battery.minimum_soc)
        / np.float64(dt / 3600)
    )

    soc_before_cycle = battery.soc
    energy_before_cycle = battery.energy

    for _ in range(10):
        # Charge
        assert np.isclose(
            battery.solve(dt, power), np.min([power, battery.maximum_power])
        )

        # Discharge
        assert np.isclose(
            battery.solve(dt, -power), -np.min([power, battery.maximum_power])
        )

        # SOC and Energy before and after the cycle should be the same
        assert np.isclose(battery.soc, soc_before_cycle)
        assert np.isclose(battery.energy, energy_before_cycle)


@given(
    minimum_soc=strategies.floats(min_value=0.0, max_value=0.5, exclude_max=True),
    maximum_energy=strategies.floats(min_value=1.0, max_value=1e5),
)
@settings(deadline=None)
def test_battery_energy_limits(
    minimum_soc: np.float64,
    maximum_energy: np.float64,
):
    battery = battery_model.Battery(
        soc_0=np.float64(1.0),
        minimum_soc=minimum_soc,
        efficiency=np.float64(1.0),
        maximum_energy=maximum_energy,
        maximum_power=np.float64(1e9),
    )

    dt = np.float64(0.5)

    soc_before_cycle = battery.soc
    energy_before_cycle = battery.energy

    p_atol = 1e-6  # 1µW of absolute tolerance for power

    for _ in range(10):
        power = (
            battery.maximum_energy
            * (battery.soc - battery.minimum_soc)
            / np.float64(dt / 3600)
        )

        # Completely discharges the battery
        assert np.isclose(battery.solve(dt, -power), -power)
        assert np.isclose(battery.soc, battery.minimum_soc)
        assert np.isclose(battery.energy, battery.minimum_energy)

        for _ in range(10):
            # Should fail to discharge the remaining of the battery
            assert np.isclose(battery.solve(dt, -power), np.float64(0.0), atol=p_atol)
            assert np.isclose(battery.soc, battery.minimum_soc)
            assert np.isclose(battery.energy, battery.minimum_energy)

        # Now if we recharge it, should go back to the initial state
        assert np.isclose(battery.solve(dt, power), power)
        assert np.isclose(battery.soc, soc_before_cycle)
        assert np.isclose(battery.energy, energy_before_cycle)

        for _ in range(10):
            # And finally, it should not charge beyond full
            assert np.isclose(battery.solve(dt, power), np.float64(0.0), atol=p_atol)
            assert np.isclose(battery.soc, battery.maximum_soc)
            assert np.isclose(battery.energy, battery.maximum_energy)


@given(
    dt=strategies.floats(
        min_value=1e-9,  # 1ns
        max_value=1e6,
    ),
    minimum_soc=strategies.floats(min_value=0.0, max_value=0.5, exclude_max=True),
    maximum_power=strategies.floats(min_value=0.0, max_value=1e6, exclude_min=True),
    power=strategies.floats(min_value=0.0, max_value=1e6),
)
def test_battery_power_limit(
    dt: np.float64,
    minimum_soc: np.float64,
    maximum_power: np.float64,
    power: np.float64,
):
    battery = battery_model.Battery(
        soc_0=np.float64(0.5),
        minimum_soc=minimum_soc,
        efficiency=np.float64(1.0),
        maximum_energy=np.float64(1e9),
        maximum_power=maximum_power,
    )

    for _ in range(10):
        # Out of the bounds of maximum power should return a limited power
        assert np.isclose(battery.solve(dt, power), np.min([power, maximum_power]))
        assert np.isclose(battery.solve(dt, -power), -np.min([power, maximum_power]))


def test_battery_charge():
    battery = battery_model.Battery(
        soc_0=np.float64(0.0),
        minimum_soc=np.float64(0.0),
        efficiency=np.float64(1.0),
        maximum_energy=np.float64(1.0),
        maximum_power=np.float64(1e9),
    )

    duration = np.float64(0.5)  # in hours
    dt = duration * np.float64(3600)  # in seconds
    power = battery.maximum_energy / duration

    assert np.isclose(battery.solve(dt, power), power)
    assert np.isclose(battery.soc, np.float64(1.0))


def test_battery_discharge():
    battery = battery_model.Battery(
        soc_0=np.float64(1.0),
        minimum_soc=np.float64(0.0),
        efficiency=np.float64(1.0),
        maximum_energy=np.float64(1.0),
        maximum_power=np.float64(1e9),
    )

    duration = np.float64(0.5)  # in hours
    dt = duration * np.float64(3600)  # in seconds
    power = -battery.maximum_energy / duration

    assert np.isclose(battery.solve(dt, power), power)
    assert np.isclose(battery.soc, battery.minimum_soc)


@given(
    dt=strategies.floats(
        min_value=1e-9,  # 1ns
        max_value=1e6,
    ),
    efficiency=strategies.floats(min_value=0.0, max_value=1.0),
)
def test_battery_efficiency(dt: np.float64, efficiency: np.float64):
    battery = battery_model.Battery(
        soc_0=np.float64(1.0),
        minimum_soc=np.float64(0.0),
        efficiency=efficiency,
        maximum_energy=np.float64(1.0),
        maximum_power=np.float64(1e9),
    )

    dt = np.float64(3600)

    # Half of the power to charge or discharge the battery if efficiency were 1
    power = (
        battery.maximum_energy
        * (battery.soc - battery.minimum_soc)
        / np.float64(dt / 3600)
    ) / np.float64(2)

    battery_energy_lost = (1 - battery.efficiency) * power * (dt / 3600)

    # Discharges the battery. When discharging, an efficiency lower
    # than 1 means we remove more energy than we deliver, or that it
    # takes less power to discharge the battery than it would be
    # otherwise.

    battery_energy_expected = (
        battery.energy - (power * (dt / 3600)) - battery_energy_lost
    )
    battery_soc_expected = battery_energy_expected / battery.maximum_energy

    assert np.isclose(battery.solve(dt, -power), -power)
    assert np.isclose(battery.soc, battery_soc_expected)
    assert np.isclose(battery.energy, battery_energy_expected)

    # Now, we charge it. When charging, an efficiency lower than 1 means we
    # need to give more energy than what it stores, or that it takes more
    # power to charge than it would be otherwise.

    battery_energy_expected = (
        battery.energy + (power * (dt / 3600)) - battery_energy_lost
    )
    battery_soc_expected = battery_energy_expected / battery.maximum_energy

    assert np.isclose(battery.solve(dt, power), power)
    assert np.isclose(battery.soc, battery_soc_expected)
    assert np.isclose(battery.energy, battery_energy_expected)
