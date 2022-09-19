from abc import ABC, abstractmethod
from typeguard import typechecked


class EnergyController(ABC):
    from lib.boat_model import Boat
    from lib.boat_data import BoatInputData, BoatOutputData
    from lib.event_model import EventResultData

    @typechecked
    @abstractmethod
    def run(
        self,
        dt: float,
        input_data: BoatInputData,
        output_data: BoatOutputData,
        event_result: EventResultData,
        boat: Boat,
    ) -> float:
        ...
