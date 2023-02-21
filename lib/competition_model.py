from dataclasses import dataclass
from typeguard import typechecked

from pandas import Timestamp

from lib.boat_data import BoatInputDataSet
from lib.event_model import EventOutputData


@dataclass
class CompetitionResult:
    name: str
    results: list[EventOutputData]


@dataclass
class Competition:
    from lib.boat_model import Boat
    from lib.energy_controller_model import EnergyController
    from lib.event_model import Event

    name: str
    events: list[Event]

    @typechecked
    def run(
        self,
        input_data: BoatInputDataSet,
        boat: Boat,
        energy_controller: EnergyController,
    ) -> CompetitionResult:
        competition_start: Timestamp = self.events[0].data.start
        competition_end: Timestamp = self.events[-1].data.end

        self._check_input(input_data, competition_start, competition_end)

        # Select the competition simulation input data
        input_data = input_data[
            (input_data.time >= competition_start)
            & (input_data.time <= competition_end)
        ].pipe(BoatInputDataSet)

        results: list[EventOutputData] = []

        for event in self.events:
            # Select the event simulation input data
            event_start: Timestamp = event.data.start
            event_end: Timestamp = event.data.end
            event_input_data: BoatInputDataSet = input_data[
                (input_data.time >= event_start) & (input_data.time <= event_end)
            ].pipe(BoatInputDataSet)
            results.append(
                event.run(
                    boat_input_data=event_input_data,
                    boat=boat,
                    energy_controller=energy_controller,
                )
            )

        return CompetitionResult(name=self.name, results=results)

    @typechecked
    def _check_input(
        self,
        input_data: BoatInputDataSet,
        competition_start: Timestamp,
        competition_end: Timestamp,
    ):
        if input_data.iloc[0].time > competition_start:
            print(input_data.iloc[0].time, competition_start)
            raise ValueError("Given data can't start after the first event's start")
        if input_data.iloc[-1].time < competition_end:
            print(input_data.iloc[-1].time, competition_end)
            raise ValueError("Given data can't end before the first event's end.")
