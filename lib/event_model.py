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


class RaceStatus:
    DNS = 0
    STARTED = 1
    FINISHED = 2
    DNF = 3


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
        if event_result.elapsed_time > self.total_time:
            raise Exception

        self._completed_laps = event_result.distance // self.lap_distance  # type: ignore
        return self._completed_laps >= self.total_laps


@dataclass
class Event:
    from lib.boat_model import Boat
    from lib.energy_controller_model import EnergyController

    name: str
    description: str
    goal: EventGoal
    # route: list[tuple[float, float]]
    start: Timestamp
    end: Timestamp

    @typechecked
    def run(
        self,
        input_data: BoatInputDataSet,
        boat: Boat,
        energy_controller: EnergyController,
    ) -> EventOutputData:
        # Transform time vector to seconds
        t = input_data.time.to_numpy().astype(int64)
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
                    input_data=BoatInputData(**input_data.iloc[k].to_dict()),
                    output_data=output_data[k_old],
                    event_result=event_result[k_old],
                    boat=boat,
                )

                output_data[k] = boat.run(float(dt), input_data.iloc[k].poa, control)

                if self.goal.accomplished(event_result=event_result[k]):
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
                input_data.iloc[k].time - input_data.iloc[0].time
            ).to_timedelta64()

            event_result[k] = EventResultData(
                distance=distance,
                elapsed_time=elapsed_time,
                status=status,
            )

        output_data = DataFrame(list(output_data)).pipe(BoatOutputDataSet)

        event_result = DataFrame(
            list(
                event_result,
            )
        ).pipe(EventResultDataSet)

        return EventOutputData(
            name=self.name,
            input_data=input_data,
            output_data=output_data,
            event_result=event_result,
        )
