from abc import ABC, abstractmethod
from dataclasses import dataclass
from typeguard import typechecked

import numpy as np
from numpy import datetime64

from strictly_typed_pandas.dataset import DataSet

from lib.utils import integrate
from lib.boat_model import Boat
from lib.boat_data import BoatInputData, BoatOutputData
from lib.event_model import EventInputData, EventResultData, RaceStatus


class EnergyController(ABC):
    @typechecked
    @abstractmethod
    def before_event_start(
        self,
        boat: Boat,
        event: EventInputData,
    ) -> None:
        ...

    @typechecked
    @abstractmethod
    def run(
        self,
        dt: np.float64,
        k: int,
        input_data: BoatInputData,
        output_data: BoatOutputData,
        event_result: EventResultData,
        boat: Boat,
        event: EventInputData,
    ) -> np.float64:
        ...


@typechecked
class ConstantPowerController(EnergyController):
    def __init__(self, constant: np.float64):
        self.constant = constant

    def run(
        self,
        dt: np.float64,
        k: int,
        input_data: BoatInputData,
        output_data: BoatOutputData,
        event_result: EventResultData,
        boat: Boat,
        event: EventInputData,
    ) -> np.float64:
        if event_result.status != RaceStatus.STARTED:
            return np.float64(0.0)

        return self.constant


@dataclass
class PredictionData:
    time: datetime64
    poa: np.float64
    poa10: np.float64
    poa90: np.float64


PredictionDataSet = DataSet[PredictionData]


class AverageController(EnergyController):
    @typechecked
    def __init__(self, prediction_dataset: PredictionDataSet, overdrive: np.float64):
        self.overdrive: np.float64 = overdrive
        self.prediction_dataset: PredictionDataSet = prediction_dataset
        self.event_power: np.float64 = np.float64(0.0)
        self.event_name: str = ""
        self.event_poa_avg: np.float64 = np.float64(0.0)
        self.event_energy: np.float64 = np.float64(0.0)
        self.competition_poa_avg: np.float64 = np.float64(0.0)
        self.competition_energy: np.float64 = np.float64(0.0)
        self.duration: np.float64 = np.float64(0.0)  # in hours
        self.event_battery_avg_power: np.float64 = np.float64(0.0)

    @typechecked
    def before_event_start(
        self,
        boat: Boat,
        event: EventInputData,
    ) -> None:
        self.event_power = np.float64(0.0)
        self.event_name = event.name
        self.event_poa_avg = np.float64(0.0)
        self.event_energy = np.float64(0.0)
        self.competition_poa_avg = np.float64(0.0)
        self.competition_energy = np.float64(0.0)
        self.duration = np.float64(0.0)
        self.event_battery_avg_power = np.float64(0.0)

    @typechecked
    def run(
        self,
        dt: np.float64,
        k: int,
        input_data: BoatInputData,
        output_data: BoatOutputData,
        event_result: EventResultData,
        boat: Boat,
        event: EventInputData,
    ) -> np.float64:
        if k == 0:
            # Compute time and duration
            self.duration = np.float64(
                (event.end - event.start).to_numpy().astype(np.float64) * 1e-9 / 3600.0
            )

            # Competition data TODO: use competiton time limits here, because the data might be bigger than the competition
            pred_competition_df = self.prediction_dataset.to_dataframe().set_index(
                "time"
            )
            self.competition_poa_avg = pred_competition_df.mean()["poa"]
            self.competition_energy = integrate(pred_competition_df)["poa"][-1]

            # Event data
            pred_event_df = pred_competition_df.query(
                "time >= @event.start & time <= @event.end"
            )

            self.event_poa_avg = pred_event_df.mean()["poa"]
            self.event_energy = integrate(pred_event_df)["poa"][-1]

            event_solar_avg_power = boat.panel.solve_output(
                irradiation=self.event_poa_avg
            )

            self.event_battery_avg_power = np.min(
                [
                    (boat.battery.energy - boat.battery.minimum_energy) / self.duration,
                    boat.battery.maximum_power,
                ]
            )

            self.event_power = event_solar_avg_power + self.event_battery_avg_power

            print(f"{self.event_name=}")
            print(f"{self.duration=}")
            print(f"{self.competition_energy=}")
            print(f"{self.competition_poa_avg=}")
            print(f"{self.event_battery_avg_power=}")
            print(f"{self.event_energy=}")
            print(f"{self.event_poa_avg=}")
            print(f"{self.event_power=}")
            print("-" * 80, end="\n\n")

        elif event_result.status != RaceStatus.STARTED:
            return np.float64(0.0)

        pred_poa: np.float64 = (
            self.prediction_dataset.to_dataframe().set_index("time")["poa"].iloc[k]
        )
        target_power = (
            boat.panel.solve_output(irradiation=pred_poa) + self.event_battery_avg_power
        )

        p_motor = boat.motor.solve_input(target_power)
        throttle = p_motor / boat.motor.maximum_input_power

        return throttle * self.overdrive
