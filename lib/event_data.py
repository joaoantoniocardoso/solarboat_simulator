import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typeguard import typechecked

from pandas import Timestamp, Timedelta
from strictly_typed_pandas.dataset import DataSet

import lib.boat_data as boat_data
import lib.event_data as event_data
import lib.event_error as event_error


@dataclass
class EventResultData:
    distance: np.float64
    elapsed_time: np.timedelta64
    status: int


EventResultDataSet = DataSet[EventResultData]


class EventGoal(ABC):
    @abstractmethod
    def accomplished(self, event_result: EventResultData) -> bool:
        ...


@dataclass
class FixedLapsGoal(EventGoal):
    total_laps: int
    lap_distance: np.float64
    event_duration: Timedelta

    @typechecked
    def accomplished(self, event_result: event_data.EventResultData) -> bool:
        completed_laps = int(event_result.distance // self.lap_distance)

        goal_accomplished = completed_laps >= self.total_laps

        if (not goal_accomplished) and (
            event_result.elapsed_time > self.event_duration
        ):
            raise event_error.EventGoalFailed(
                f"Time over: {Timedelta(event_result.elapsed_time)} > {self.event_duration}\n {self}"
                + f"\n {event_result}"
            )

        return goal_accomplished


@dataclass
class EventInputData:
    name: str
    description: str
    goal: EventGoal
    # route: list[tuple[np.float64, np.float64]]
    start: Timestamp
    end: Timestamp


@dataclass
class EventOutputData:
    name: str
    event_result: EventResultDataSet
    input_data: boat_data.BoatInputDataSet
    output_data: boat_data.BoatOutputDataSet


# TODO: Maybe separate the race status (started or finished) from the boat status (dns, dnf, racing)
class RaceStatus:
    DNS = 0
    STARTED = 1
    FINISHED = 2
    DNF = 3

    @staticmethod
    def to_str(status: int) -> str | None:
        if status == RaceStatus.DNS:
            return "DNS"
        elif status == RaceStatus.STARTED:
            return "STARTED"
        elif status == RaceStatus.FINISHED:
            return "FINISHED"
        elif status == RaceStatus.DNF:
            return "DNF"
        return None
