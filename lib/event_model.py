import numpy as np

from dataclasses import dataclass
from numpy import float64, int64
from pandas import Timestamp

from lib.boat_data import BoatInputData, BoatOutputData
from lib.boat_model import Boat
from lib.energy_controller_model import EnergyController


@dataclass
class EventResult:
    name: str
    input_data: BoatInputData
    output_data: BoatOutputData


@dataclass
class Event:
    name: str
    description: str
    # route: list[tuple[float, float]]
    start: Timestamp
    end: Timestamp

    def run(
        self, input_data: BoatInputData, boat: Boat, energy_controller: EnergyController
    ) -> EventResult:
        # Transform time vector to seconds
        t = input_data.time.astype(int64)
        t = (t - t[0]) * 1e-9

        output_data = BoatOutputData(
            battery_output_energy=np.zeros(t.size, dtype=float64),
            pv_output_power=np.zeros(t.size, dtype=float64),
            motor_output_power=np.zeros(t.size, dtype=float64),
            battery_output_power=np.zeros(t.size, dtype=float64),
            pv_target_power=np.zeros(t.size, dtype=float64),
            motor_target_power=np.zeros(t.size, dtype=float64),
            battery_target_power=np.zeros(t.size, dtype=float64),
            motor_target_throttle=np.zeros(t.size, dtype=float64),
        )

        dt: int64 = t[1] - t[0]
        for k in range(t.size):
            if k > 0:
                dt = t[k] - t[k - 1]

            control = energy_controller.run(
                dt=float(dt),
                input_data=BoatInputData(
                    time=input_data.time[k], poa=input_data.poa[k]
                ),
                output_data=BoatOutputData(
                    battery_output_energy=output_data.battery_output_energy[k],
                    pv_output_power=output_data.pv_output_power[k],
                    motor_output_power=output_data.motor_output_power[k],
                    battery_output_power=output_data.battery_output_power[k],
                    pv_target_power=output_data.pv_target_power[k],
                    motor_target_power=output_data.motor_target_power[k],
                    battery_target_power=output_data.battery_target_power[k],
                    motor_target_throttle=output_data.motor_target_throttle[k],
                ),
                boat=boat,
            )

            y = boat.run(float(dt), input_data.poa[k], control)

            output_data.battery_output_energy[k] = y[0]
            output_data.pv_output_power[k] = y[1]
            output_data.motor_output_power[k] = y[2]
            output_data.battery_output_power[k] = y[3]
            output_data.pv_target_power[k] = y[4]
            output_data.motor_target_power[k] = y[5]
            output_data.battery_target_power[k] = y[6]
            output_data.motor_target_throttle[k] = y[7]

        return EventResult(
            name=self.name, input_data=input_data, output_data=output_data
        )
