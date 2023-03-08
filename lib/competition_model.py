from dataclasses import dataclass
from typeguard import typechecked


from pandas import Timestamp

import lib.boat_data as boat_data
import lib.event_data as event_data


@dataclass
class CompetitionResult:
    name: str
    results: list[list[event_data.EventOutputData]]


@dataclass
class Competition:
    import lib.event_model as event_model

    name: str
    events: list[event_model.Event]

    @typechecked
    def solve(
        self,
        input_data: boat_data.BoatInputDataSet,
        controlled_boats: list[event_model.ControlledBoat],
    ) -> CompetitionResult:
        competition_start: Timestamp = self.events[0].data.start
        competition_end: Timestamp = self.events[-1].data.end

        self._check_input(input_data, competition_start, competition_end)

        # Select the competition simulation input data
        input_data = input_data[
            (input_data.time >= competition_start)
            & (input_data.time <= competition_end)
        ].pipe(boat_data.BoatInputDataSet)

        results: list[list[event_data.EventOutputData]] = []

        for event in self.events:
            # Select the event simulation input data
            event_start: Timestamp = event.data.start
            event_end: Timestamp = event.data.end
            event_input_data: boat_data.BoatInputDataSet = input_data[
                (input_data.time >= event_start) & (input_data.time <= event_end)
            ].pipe(boat_data.BoatInputDataSet)
            results.append(
                event.solve(
                    boat_input_data=event_input_data,
                    controlled_boats=controlled_boats,
                )
            )

        return CompetitionResult(name=self.name, results=results)

    @typechecked
    def _check_input(
        self,
        input_data: boat_data.BoatInputDataSet,
        competition_start: Timestamp,
        competition_end: Timestamp,
    ):
        if input_data.iloc[0].time > competition_start:
            print(input_data.iloc[0].time, competition_start)
            raise ValueError("Given data can't start after the first event's start")
        if input_data.iloc[-1].time < competition_end:
            print(input_data.iloc[-1].time, competition_end)
            raise ValueError("Given data can't end before the first event's end.")
