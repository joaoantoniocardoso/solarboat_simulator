from bdb import effective
import numpy as np

from dataclasses import dataclass
from .boat_data import BoatOutputData

from lib.utils import naive_power, naive_energy


@dataclass
class Panel:
    efficiency: float
    area: float
    maximum_output_power: float

    def solve_output(self, irradiation: float) -> float:
        input_power = irradiation * self.area

        output_power = input_power * self.efficiency
        if output_power > self.maximum_output_power:
            output_power = self.maximum_output_power

        return output_power


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

    def solve(self, dt: float, target_power: float) -> float:
        power = 0.0
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
class ESC:
    efficiency: float
    maximum_input_power: float

    def solve_input(self, throttle: float) -> float:
        throttle = np.clip(throttle, 0, 1)

        input_power = throttle * self.maximum_input_power
        if input_power > self.maximum_input_power:
            input_power = self.maximum_input_power

        return input_power

    def solve_output(self, input_power) -> float:
        output_power = input_power * self.efficiency
        return output_power


@dataclass
class Motor:
    efficiency: float
    maximum_input_power: float

    def solve_input(self, input_power: float) -> float:
        if input_power > self.maximum_input_power:
            input_power = self.maximum_input_power

        return input_power

    def solve_output(self, input_power) -> float:
        output_power = input_power * self.efficiency
        return output_power


@dataclass
class Propulsion:
    efficiency: float
    maximum_input_power: float

    def solve_input(self, input_power: float) -> float:
        if input_power > self.maximum_input_power:
            input_power = self.maximum_input_power

        return input_power

    def solve_output(self, input_power) -> float:
        output_power = input_power * self.efficiency
        return output_power


@dataclass
class Hull:
    speed_over_power_constant: float

    def solve_output(self, propulsion_power: float) -> float:
        return propulsion_power * self.speed_over_power_constant


@dataclass
class Boat:
    panel: Panel
    battery: Battery
    circuits: Other
    esc: ESC
    motor: Motor
    propulsion: Propulsion
    hull: Hull

    def run(self, dt: float, irradiation: float, motor_throttle: float):
        # TODO: Create exeption types and throw then in case of battery under or over voltage. This battery exception might be implemented as a BMS model, which could be disabled.
        # TODO: Create some way to programatically inject an exception, to simulate catastrophic events like crashes, which could take the boat off the race.

        # Step #1 - solve for battery:
        target_circuits_input_power = self.circuits.power
        target_pv_output_power = self.panel.solve_output(irradiation)
        target_esc_input_power = self.propulsion.solve_input(
            self.motor.solve_input(self.esc.solve_input(motor_throttle))
        )
        target_battery_power = (
            target_pv_output_power
            - target_esc_input_power
            - target_circuits_input_power
        )
        actual_battery_power = self.battery.solve(dt, target_battery_power)

        # Step #2 - solve for base circuits
        # if target_circuits_power > actual_battery_power:
        #     raise Exception("There is no power to keep the basic boat's circuits running!")
        actual_circuits_input_power = target_circuits_input_power

        # Step #3 - solve for pv:
        actual_pv_output_power = (
            actual_battery_power + target_esc_input_power + actual_circuits_input_power
        )
        if actual_pv_output_power > target_pv_output_power:
            actual_pv_output_power = target_pv_output_power

        # Step #4 - solve for motor:
        actual_esc_input_power = (
            actual_pv_output_power - actual_battery_power - actual_circuits_input_power
        )
        if actual_esc_input_power > target_esc_input_power:
            actual_esc_input_power = target_esc_input_power

        # Step #5 - propagate the power that moves the boat:
        actual_esc_output_power = self.esc.solve_output(actual_esc_input_power)
        actual_motor_output_power = self.motor.solve_output(actual_esc_output_power)
        actual_propulsive_output_power = self.propulsion.solve_output(
            actual_motor_output_power
        )
        actual_hull_speed = self.hull.solve_output(actual_propulsive_output_power)

        return BoatOutputData(
            pv_output_power=np.array([actual_pv_output_power]),
            battery_stored_energy=np.array([self.battery.energy]),
            battery_soc=np.array([self.battery.soc]),
            battery_output_power=np.array([actual_battery_power]),
            esc_input_power=np.array([actual_esc_input_power]),
            esc_output_power=np.array([actual_esc_output_power]),
            motor_output_power=np.array([actual_esc_output_power]),
            propulsive_output_power=np.array([actual_propulsive_output_power]),
            hull_speed=np.array([actual_hull_speed]),
            pv_target_power=np.array([target_pv_output_power]),
            esc_target_power=np.array([target_battery_power]),
            battery_target_power=np.array([target_esc_input_power]),
            motor_target_throttle=np.array([motor_throttle]),
        )
