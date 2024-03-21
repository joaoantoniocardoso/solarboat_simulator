from dataclasses import dataclass

import numpy as np
from numpy import datetime64

from strictly_typed_pandas.dataset import DataSet


@dataclass
class BoatInputData:
    time: datetime64
    poa: np.float64


BoatInputDataSet = DataSet[BoatInputData]


@dataclass
class BoatOutputData:
    pv_input_power: np.float64
    pv_output_power: np.float64
    mppt_input_power: np.float64
    mppt_output_power: np.float64
    battery_output_power: np.float64
    esc_input_power: np.float64
    esc_output_power: np.float64
    motor_input_power: np.float64
    motor_output_power: np.float64
    transmission_input_power: np.float64
    transmission_output_power: np.float64
    hull_speed: np.float64
    motor_throttle: np.float64
    battery_stored_energy: np.float64
    battery_soc: np.float64


BoatOutputDataSet = DataSet[BoatOutputData]
