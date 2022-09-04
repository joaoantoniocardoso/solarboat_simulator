from dataclasses import dataclass
from numpy.typing import NDArray
from numpy import float64, timedelta64


@dataclass
class BoatInputData:
    time: NDArray[timedelta64]  # type: ignore
    poa: NDArray[float64]


@dataclass
class BoatOutputData:
    battery_output_energy: NDArray[float64]
    pv_output_power: NDArray[float64]
    motor_output_power: NDArray[float64]
    battery_output_power: NDArray[float64]
    pv_target_power: NDArray[float64]
    motor_target_power: NDArray[float64]
    battery_target_power: NDArray[float64]
    motor_target_throttle: NDArray[float64]
