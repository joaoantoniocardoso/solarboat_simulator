from dataclasses import dataclass

import numpy as np
from typeguard import typechecked

import lib.boat_data as boat_data
import lib.boat_error as boat_error


@dataclass
class Other:
    power: np.float64


@dataclass
class Boat:
    import lib.battery_model as battery_model
    import lib.boat_error as boat_error
    import lib.esc_model as esc_model
    import lib.hull_model as hull_model
    import lib.motor_model as motor_model
    import lib.mppt_model as mppt_model
    import lib.panel_model as panel_model
    import lib.transmission_model as transmission_model

    name: str
    panel: panel_model.Panel
    mppt: mppt_model.MPPT
    battery: battery_model.Battery
    circuits: Other
    esc: esc_model.ESC
    motor: motor_model.Motor
    transmission: transmission_model.Transmission
    hull: hull_model.Hull
    status: boat_error.BoatError = boat_error.BoatError.NORMAL

    @typechecked
    def solve(
        self, dt: np.float64, irradiation: np.float64, motor_throttle: np.float64
    ) -> boat_data.BoatOutputData:
        # TODO: Create some way to programatically inject an exception, to simulate catastrophic
        # events like crashes, which could take the boat off the race.

        # 1. Compute the requested ESC and battery output power
        esc_requested_input_power = self.esc.solve_input(motor_throttle)
        passive_power_consumption = self.circuits.power
        battery_requested_output_power = (
            passive_power_consumption + esc_requested_input_power
        )

        # 2. Compute the available system input power
        pv_available_input_power = self.panel.solve_input(irradiation)
        pv_available_output_power = self.panel.solve_output(pv_available_input_power)
        mppt_available_output_power = self.mppt.solve_output(pv_available_output_power)

        # 3. Compute actual delivered/available battery power
        battery_target_charge_power = (
            mppt_available_output_power - battery_requested_output_power
        )
        battery_output_power = self.battery.solve(dt, battery_target_charge_power)

        # 4. If there's not enough power, change boat status
        if (
            battery_target_charge_power < battery_output_power
            or self.circuits.power > battery_output_power
        ):
            self.status = boat_error.BoatError.OUT_OF_ENERGY

        # 5. Compute actual system input power
        mppt_output_power = np.min(
            [
                mppt_available_output_power,
                battery_output_power + battery_requested_output_power,
            ]
        )
        mppt_eta = self.mppt.efficiency(pv_available_input_power)
        if mppt_eta == 0:
            pv_output_power = pv_available_output_power
        else:
            pv_output_power = np.min(
                [
                    pv_available_output_power,
                    mppt_output_power / mppt_eta,
                ]
            )
        pv_input_power = pv_output_power / self.panel.efficiency

        # 6. Compute actual ESC output power
        esc_input_power = np.min(
            [
                esc_requested_input_power,
                pv_output_power - battery_output_power - self.circuits.power,
            ]
        )

        # 7. Compute system output power
        esc_output_power = self.esc.solve_output(esc_input_power)
        motor_input_power = self.motor.solve_input(esc_output_power)
        motor_output_power = self.motor.solve_output(motor_input_power)
        transmission_input_power = self.transmission.solve_input(motor_output_power)
        transmission_output_power = self.transmission.solve_output(
            transmission_input_power
        )
        hull_speed = self.hull.solve_output(transmission_output_power)

        # TODO: return empty data when some erro had happened
        # if self.status is not BoatError.NORMAL:
        #     return boat_data.BoatOutputData(
        #         battery_stored_energy=self.battery.energy,
        #         battery_soc=self.battery.soc,
        #         ...
        #     )

        return boat_data.BoatOutputData(
            pv_input_power=pv_input_power,
            pv_output_power=pv_output_power,
            mppt_input_power=pv_output_power,
            mppt_output_power=mppt_output_power,
            battery_output_power=battery_output_power,
            esc_input_power=esc_input_power,
            esc_output_power=esc_output_power,
            motor_input_power=motor_input_power,
            motor_output_power=motor_output_power,
            transmission_input_power=transmission_input_power,
            transmission_output_power=transmission_output_power,
            hull_speed=hull_speed,
            motor_throttle=motor_throttle,
            battery_stored_energy=self.battery.energy,
            battery_soc=self.battery.soc,
        )
