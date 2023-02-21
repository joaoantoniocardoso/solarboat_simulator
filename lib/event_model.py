import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typeguard import typechecked

from numpy import int64
from pandas import DataFrame, Timestamp, Timedelta
from strictly_typed_pandas.dataset import DataSet


from lib.boat_data import (
    BoatInputData,
    BoatInputDataSet,
    BoatOutputData,
    BoatOutputDataSet,
)
from lib.boat_error import BoatError


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


@dataclass
class EventResultData:
    distance: float
    elapsed_time: np.timedelta64
    status: int


EventResultDataSet = DataSet[EventResultData]


class EventError(Exception):
    """Exception raised for erros during event operation.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message: str) -> None:
        self.message = (message,)
        super().__init__(self.message)


class EventGoalFailed(EventError):
    pass


@dataclass
class EventOutputData:
    name: str
    event_result: EventResultDataSet
    input_data: BoatInputDataSet
    output_data: BoatOutputDataSet


class EventGoal(ABC):
    @abstractmethod
    def accomplished(self, event_result: EventResultData) -> bool:
        ...


@dataclass
class FixedLapsGoal(EventGoal):
    total_laps: int
    lap_distance: float
    total_time: Timedelta
    _completed_laps: int = 0

    @typechecked
    def accomplished(self, event_result: EventResultData) -> bool:
        self._completed_laps = int(event_result.distance // self.lap_distance)

        completed = self._completed_laps >= self.total_laps

        if (not completed) and (event_result.elapsed_time > self.total_time):
            raise EventGoalFailed(
                f"Time over: {Timedelta(event_result.elapsed_time)} > {self.total_time}\n {self}"
                + f"\n {event_result}"
            )

        return completed


@dataclass
class EventInputData:
    name: str
    description: str
    goal: EventGoal
    # route: list[tuple[float, float]]
    start: Timestamp
    end: Timestamp


@dataclass
class Event:
    from lib.boat_model import Boat
    from lib.energy_controller_model import EnergyController

    data: EventInputData

    @typechecked
    def run(
        self,
        boat_input_data: BoatInputDataSet,
        boat: Boat,
        energy_controller: EnergyController,
    ) -> EventOutputData:
        # Transform time vector to seconds
        t = boat_input_data.time.to_numpy().astype(int64)
        t = (t - t[0]) * 1e-9

        output_data = np.full(
            shape=t.size,
            fill_value=BoatOutputData(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            dtype=BoatOutputData,
        )

        event_result = np.full(
            shape=t.size,
            fill_value=EventResultData(
                0, Timedelta(0).to_timedelta64(), RaceStatus.DNS
            ),
            dtype=EventResultData,
        )

        dt: int64 = t[1] - t[0]
        for k in range(t.size):
            k_old = max(0, k - 1)

            if k > 0:
                dt = t[k] - t[k_old]

            status = RaceStatus.DNS
            try:
                control = energy_controller.run(
                    dt=float(dt),
                    input_data=BoatInputData(**boat_input_data.iloc[k].to_dict()),
                    output_data=output_data[k_old],
                    event_result=event_result[k_old],
                    boat=boat,
                    event=self.data,
                )

                output_data[k] = boat.run(
                    float(dt), boat_input_data.iloc[k].poa, control
                )

                if self.data.goal.accomplished(event_result=event_result[k_old]):
                    status = RaceStatus.FINISHED
                else:
                    status = RaceStatus.STARTED

            except (BoatError, EventGoalFailed) as e:
                old_status = event_result[k_old].status
                status = RaceStatus.DNF
                if old_status != RaceStatus.DNF:
                    print(
                        f"Boat out of the race, status: {RaceStatus.to_str(old_status)}"
                        + f" => {RaceStatus.to_str(status)}. Reason: {e}"
                    )

            distance = output_data[k].hull_speed * dt
            elapsed_time = Timedelta(
                boat_input_data.iloc[k].time - boat_input_data.iloc[0].time
            ).to_timedelta64()

            event_result[k] = EventResultData(
                distance=event_result[k_old].distance + distance,
                elapsed_time=event_result[k_old].elapsed_time + elapsed_time,
                status=status,
            )

        output_data = DataFrame(list(output_data)).pipe(BoatOutputDataSet)

        event_result = DataFrame(
            list(
                event_result,
            )
        ).pipe(EventResultDataSet)

        return EventOutputData(
            name=self.data.name,
            input_data=boat_input_data,
            output_data=output_data,
            event_result=event_result,
        )
