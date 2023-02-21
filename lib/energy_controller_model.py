from abc import ABC, abstractmethod
from typeguard import typechecked

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
        dt: float,
        k: int,
        input_data: BoatInputData,
        output_data: BoatOutputData,
        event_result: EventResultData,
        boat: Boat,
        event: EventInputData,
    ) -> float:
        ...
