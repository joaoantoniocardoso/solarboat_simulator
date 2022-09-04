import numpy as np
from numpy.typing import NDArray
from numpy import datetime64
from pandas import Timestamp
from dataclasses import dataclass

from lib.boat_model import Boat
from lib.boat_data import BoatInputData
from lib.event_model import Event, EventResult
from lib.energy_controller_model import EnergyController


@dataclass
class CompetitionResult:
    name: str
    results: list[EventResult]


@dataclass
class Competition:
    name: str
    events: list[Event]

    def run(
        self, input_data: BoatInputData, boat: Boat, energy_controller: EnergyController
    ) -> CompetitionResult:
        competition_start: datetime64 = Timestamp(self.events[0].start).to_datetime64()
        competition_end: datetime64 = Timestamp(self.events[-1].end).to_datetime64()

        if input_data.time.size != input_data.poa.size:
            raise ValueError(
                "All data must have the same length as the given time array."
            )
        if input_data.time[0] > competition_start:
            raise ValueError("Given data can't start after the first event's start.")
        if input_data.time[-1] < competition_end:
            raise ValueError("Given data can't end before the first event's end.")

        # Select the competition simulation input data
        competition_time_selection: NDArray[np.bool_] = (
            input_data.time >= competition_start
        ) & (input_data.time <= competition_end)
        input_data = BoatInputData(
            time=input_data.time[competition_time_selection],
            poa=input_data.poa[competition_time_selection],
        )

        results: list[EventResult] = []

        for event in self.events:
            # Select the event simulation input data
            event_start: datetime64 = Timestamp(event.start).to_datetime64()
            event_end: datetime64 = Timestamp(event.end).to_datetime64()
            selection: NDArray[np.bool_] = (input_data.time >= event_start) & (
                input_data.time <= event_end
            )
            event_input_data = BoatInputData(
                time=input_data.time[event_time_selection],
                poa=input_data.poa[event_time_selection],
            )
            results.append(
                event.run(event_input_data, boat, energy_controller=energy_controller)
            )

        return CompetitionResult(name=self.name, results=results)
