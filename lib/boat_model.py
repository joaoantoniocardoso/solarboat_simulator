from dataclasses import dataclass
from typeguard import typechecked

from lib.boat_data import BoatOutputData
from lib.panel_model import Panel
from lib.battery_model import Battery
from lib.esc_model import ESC
from lib.motor_model import Motor
from lib.propulsion_model import Propulsion
from lib.hull_model import Hull


class BoatError(Exception):
    """Exception raised for erros during boat operation.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message: str) -> None:
        self.message = (message,)
        super().__init__(self.message)


# TODO: This battery exceptions might be implemented as a BMS model, which could be disabled.
class BatteryOverVoltageError(BoatError):
    pass


class BatteryUnderVoltageError(BoatError):
    pass


@dataclass
class Other:
    power: float


@dataclass
class Boat:
    panel: Panel
    battery: Battery
    circuits: Other
    esc: ESC
    motor: Motor
    propulsion: Propulsion
    hull: Hull

    @typechecked
    def run(
        self, dt: float, irradiation: float, motor_throttle: float
    ) -> BoatOutputData:
        # TODO: Create some way to programatically inject an exception, to simulate catastrophic
        # events like crashes, which could take the boat off the race.

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
            pv_output_power=actual_pv_output_power,
            battery_stored_energy=self.battery.energy,
            battery_soc=self.battery.soc,
            battery_output_power=actual_battery_power,
            esc_input_power=actual_esc_input_power,
            esc_output_power=actual_esc_output_power,
            motor_output_power=actual_esc_output_power,
            propulsive_output_power=actual_propulsive_output_power,
            hull_speed=actual_hull_speed,
            pv_target_power=target_pv_output_power,
            esc_target_power=target_battery_power,
            battery_target_power=target_esc_input_power,
            motor_target_throttle=motor_throttle,
        )
