from numpy.typing import NDArray
from numpy import float64, timedelta64

from dataclasses import dataclass


@dataclass
class BoatInputData:
    time: NDArray[timedelta64]  # type: ignore
    poa: NDArray[float64]


@dataclass
class BoatOutputData:
    pv_output_power: NDArray[float64]
    battery_stored_energy: NDArray[float64]
    battery_soc: NDArray[float64]
    battery_output_power: NDArray[float64]
    esc_input_power: NDArray[float64]
    esc_output_power: NDArray[float64]
    motor_output_power: NDArray[float64]
    propulsive_output_power: NDArray[float64]
    hull_speed: NDArray[float64]
    pv_target_power: NDArray[float64]
    esc_target_power: NDArray[float64]
    battery_target_power: NDArray[float64]
    motor_target_throttle: NDArray[float64]
