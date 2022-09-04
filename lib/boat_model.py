import numpy as np
from dataclasses import dataclass
from numpy import float64, ndarray, dtype

from lib.utils import naive_power, naive_energy


@dataclass
class Panel:
    efficiency: float
    area: float
    mppt_maximum_power: float

    def run(self, irradiation: float) -> float:
        output_power = irradiation * self.area * self.efficiency
        if output_power > self.mppt_maximum_power:
            output_power = self.mppt_maximum_power
        return output_power


@dataclass
class Motor:
    maximum_power: float

    def run(self, throttle: float) -> float:
        throttle = np.clip(throttle, 0, 1)

        input_power = throttle * self.maximum_power
        if input_power > self.maximum_power:
            input_power = self.maximum_power
        return input_power


@dataclass(init=False)
class Battery:
    efficiency: float
    energy: float
    soc: float
    minimum_soc: float
    maximum_energy: float
    minimum_energy: float
    maximum_power: float

    def __init__(
        self,
        soc_0: float,
        minimum_soc: float,
        efficiency: float,
        maximum_energy: float,
        maximum_power: float,
    ):
        self.efficiency = efficiency
        self.soc = soc_0
        self.minimum_soc = minimum_soc
        self.energy = soc_0 * maximum_energy
        self.maximum_energy = maximum_energy
        self.minimum_energy = maximum_energy * minimum_soc
        self.maximum_power = maximum_power

    def _charge(self, dt: float, power: float) -> float:
        energy = naive_energy(power, dt, timebase=3600)
        self.energy += energy * self.efficiency

        if self.energy > self.maximum_energy:
            exceeded_energy = self.energy - self.maximum_energy
            self.energy -= exceeded_energy
            exceeded_power = naive_power(exceeded_energy, dt, timebase=3600)
            return power - exceeded_power

        return power

    def _discharge(self, dt: float, power: float) -> float:
        energy = naive_energy(power, dt, timebase=3600)
        self.energy -= energy * self.efficiency

        if self.energy < self.minimum_energy:
            exceeded_energy = self.minimum_energy - self.energy
            self.energy += exceeded_energy
            exceeded_power = naive_power(exceeded_energy, dt, timebase=3600)
            return power - exceeded_power

        return power

    def run(self, dt: float, target_power: float) -> float:
        power = 0.
        if target_power > 0:
            power = self._charge(dt, abs(target_power))
        else:
            power = -self._discharge(dt, abs(target_power))

        self.soc = self.energy / self.maximum_energy
        return power


@dataclass
class Other:
    power: float


@dataclass
class Boat:
    panel: Panel
    motor: Motor
    battery: Battery
    circuits: Other

    def run(self, dt: float, irradiation: float, motor_throttle: float):
        # TODO: Create exeption types and throw then in case of battery under or over voltage. This battery exception might be implemented as a BMS model, which could be disabled.
        # TODO: Create some way to programatically inject an exception, to simulate catastrophic events like crashes, which could take the boat off the race.

        # Step #1 - solve for battery:
        target_circuits_power = self.circuits.power
        target_pv_power = self.panel.run(irradiation)
        target_motor_power = self.motor.run(motor_throttle)
        target_battery_power = (
            target_pv_power - target_motor_power - target_circuits_power
        )
        actual_battery_power = self.battery.run(dt, target_battery_power)

        # Step #2 - solve for base circuits
        # if target_circuits_power > actual_battery_power:
        #     raise Exception("There is no power to keep the basic boat's circuits running!")
        actual_circuits_power = target_circuits_power

        # Step #3 - solve for pv:
        actual_pv_power = (
            actual_battery_power + target_motor_power + actual_circuits_power
        )
        if actual_pv_power > target_pv_power:
            actual_pv_power = target_pv_power

        # Step #4 - solve for motor:
        actual_motor_power = (
            actual_pv_power - actual_battery_power - actual_circuits_power
        )
        if actual_motor_power > target_motor_power:
            actual_motor_power = target_motor_power

        return (
            self.battery.soc,
            actual_pv_power,
            actual_motor_power,
            actual_battery_power,
            target_pv_power,
            target_motor_power,
            target_battery_power,
            motor_throttle,
        )
