from numpy import datetime64

from strictly_typed_pandas.dataset import DataSet

from dataclasses import dataclass


@dataclass
class BoatInputData:
    time: datetime64
    poa: float


BoatInputDataSet = DataSet[BoatInputData]


@dataclass
class BoatOutputData:
    pv_output_power: float
    battery_stored_energy: float
    battery_soc: float
    battery_output_power: float
    esc_input_power: float
    esc_output_power: float
    motor_output_power: float
    propulsive_output_power: float
    hull_speed: float
    pv_target_power: float
    esc_target_power: float
    battery_target_power: float
    motor_target_throttle: float


BoatOutputDataSet = DataSet[BoatOutputData]
