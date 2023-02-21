from dataclasses import dataclass
from typeguard import typechecked

from lib.boat_data import BoatOutputData
from lib.panel_model import Panel
from lib.battery_model import Battery
from lib.esc_model import ESC
from lib.motor_model import Motor
from lib.propulsion_model import Propulsion
from lib.hull_model import Hull


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

        esc_input_target_power = self.esc.solve_input(throttle=motor_throttle)

        requested_output_target_power = self.circuits.power + esc_input_target_power

        pv_output_target_power = self.panel.solve_output(irradiation=irradiation)

        battery_charge_target_power = (
            pv_output_target_power - requested_output_target_power
        )
        battery_charge_power = self.battery.solve(dt, battery_charge_target_power)

        pv_output_power = min(
            pv_output_target_power, battery_charge_power + requested_output_target_power
        )

        esc_input_power = pv_output_power - battery_charge_power - self.circuits.power

        esc_output_power = self.esc.solve_output(esc_input_power)
        motor_input_power = self.motor.solve_input(esc_output_power)
        motor_output_power = self.motor.solve_output(motor_input_power)
        propulsive_input_power = self.propulsion.solve_input(motor_output_power)
        propulsive_output_power = self.propulsion.solve_output(propulsive_input_power)
        hull_speed = self.hull.solve_output(propulsive_output_power)

        return BoatOutputData(
            pv_output_power=pv_output_power,
            battery_stored_energy=self.battery.energy,
            battery_soc=self.battery.soc,
            battery_output_power=battery_charge_power,
            esc_input_power=esc_input_power,
            esc_output_power=esc_output_power,
            motor_output_power=motor_output_power,
            propulsive_output_power=propulsive_output_power,
            hull_speed=hull_speed,
            pv_target_power=pv_output_target_power,
            esc_target_power=esc_input_target_power,
            battery_target_power=battery_charge_target_power,
            motor_target_throttle=motor_throttle,
        )
