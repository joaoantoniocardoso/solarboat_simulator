import numpy as np

from dataclasses import dataclass
from typeguard import typechecked

import lib.boat_data as boat_data
import lib.boat_error as boat_error


@dataclass
class Other:
    power: np.float64


@dataclass
class Boat:
    import lib.panel_model as panel_model
    import lib.battery_model as battery_model
    import lib.esc_model as esc_model
    import lib.motor_model as motor_model
    import lib.propulsion_model as propulsion_model
    import lib.hull_model as hull_model
    import lib.boat_error as boat_error

    name: str
    panel: panel_model.Panel
    battery: battery_model.Battery
    circuits: Other
    esc: esc_model.ESC
    motor: motor_model.Motor
    propulsion: propulsion_model.Propulsion
    hull: hull_model.Hull
    status: boat_error.BoatError = boat_error.BoatError.NORMAL

    @typechecked
    def solve(
        self, dt: np.float64, irradiation: np.float64, motor_throttle: np.float64
    ) -> boat_data.BoatOutputData:
        # TODO: Create some way to programatically inject an exception, to simulate catastrophic
        # events like crashes, which could take the boat off the race.

        esc_input_target_power: np.float64 = self.esc.solve_input(motor_throttle)

        requested_output_target_power: np.float64 = (
            self.circuits.power + esc_input_target_power
        )

        pv_output_target_power: np.float64 = self.panel.solve_output(irradiation)

        battery_charge_target_power: np.float64 = (
            pv_output_target_power - requested_output_target_power
        )
        battery_charge_power: np.float64 = self.battery.solve(
            dt, battery_charge_target_power
        )
        if (
            battery_charge_target_power < battery_charge_power
            or self.circuits.power > -battery_charge_power
        ):
            self.status = boat_error.BoatError.OUT_OF_ENERGY

        pv_output_power: np.float64 = np.min(
            [
                pv_output_target_power,
                battery_charge_power + requested_output_target_power,
            ]
        )

        esc_input_power: np.float64 = np.min(
            [
                esc_input_target_power,
                pv_output_power - battery_charge_power - self.circuits.power,
            ]
        )

        esc_output_power = self.esc.solve_output(esc_input_power)
        motor_input_power = self.motor.solve_input(esc_output_power)
        motor_output_power = self.motor.solve_output(motor_input_power)
        propulsive_input_power = self.propulsion.solve_input(motor_output_power)
        propulsive_output_power = self.propulsion.solve_output(propulsive_input_power)
        hull_speed = self.hull.solve_output(propulsive_output_power)

        # if self.status is not BoatError.NORMAL:
        #     return boat_data.BoatOutputData(
        #         pv_output_power=np.float64(0.0),
        #         battery_stored_energy=self.battery.energy,
        #         battery_soc=self.battery.soc,
        #         battery_output_power=np.float64(0.0),
        #         esc_input_power=np.float64(0.0),
        #         esc_output_power=np.float64(0.0),
        #         motor_output_power=np.float64(0.0),
        #         propulsive_output_power=np.float64(0.0),
        #         hull_speed=np.float64(0.0),
        #         pv_target_power=pv_output_target_power,
        #         esc_target_power=esc_input_target_power,
        #         battery_target_power=battery_charge_target_power,
        #         motor_target_throttle=np.float64(0.0),
        #     )

        return boat_data.BoatOutputData(
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
