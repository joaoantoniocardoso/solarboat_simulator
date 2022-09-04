from abc import ABC, abstractmethod

from lib.boat_model import Boat
from lib.boat_data import BoatInputData, BoatOutputData


class EnergyController(ABC):
    @abstractmethod
    def run(
        self,
        dt: float,
        input_data: BoatInputData,
        output_data: BoatOutputData,
        boat: Boat,
    ) -> float:
        ...
