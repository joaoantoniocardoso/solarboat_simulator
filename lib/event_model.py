import numpy as np

from dataclasses import dataclass
from typeguard import typechecked

from pandas import Timedelta


import lib.boat_data as boat_data
import lib.event_data as event_data
import lib.event_error as event_error


@dataclass
class ControlledBoat:
    import lib.boat_model as boat_model
    import lib.energy_controller_model as energy_controller_model

    boat: boat_model.Boat
    energy_controller: energy_controller_model.EnergyController


@dataclass
class Event:
    data: event_data.EventInputData

    @typechecked
    def solve(
        self,
        boat_input_data: boat_data.BoatInputDataSet,
        controlled_boats: list[ControlledBoat],
    ) -> dict[str, event_data.EventOutputData]:  # boat_name: data
        # Transform time vector to seconds
        t = boat_input_data.time.to_numpy().astype(np.float64)
        t = (t - t[0]) * 1e-9

        event_output_data: dict[str, event_data.EventOutputData] = {}

        for controlled_boat in controlled_boats:
            boat_output_data = np.full(
                shape=t.size,
                fill_value=boat_data.BoatOutputData(
                    *[np.float64(0)]
                    * len(boat_data.BoatOutputData.__dataclass_fields__.keys())
                ),
                dtype=boat_data.BoatOutputData,
            )

            event_result = np.full(
                shape=t.size,
                fill_value=event_data.EventResultData(
                    np.float64(0),
                    Timedelta(0).to_timedelta64(),
                    event_data.RaceStatus.DNS,
                ),
                dtype=event_data.EventResultData,
            )

            controlled_boat.energy_controller.before_event_start(
                boat=controlled_boat.boat, event=self.data
            )

            status = event_data.RaceStatus.DNS
            dt: np.float64 = t[1] - t[0]
            for k in range(t.size):
                k_old = max(0, k - 1)

                if k > 0:
                    dt = t[k] - t[k_old]

                try:
                    control = controlled_boat.energy_controller.solve(
                        dt=dt,
                        k=k,
                        input_data=boat_data.BoatInputData(
                            **boat_input_data.iloc[k].to_dict()
                        ),
                        output_data=boat_output_data[k_old],
                        event_result=event_result[k_old],
                        boat=controlled_boat.boat,
                        event=self.data,
                    )

                    boat_output_data[k] = controlled_boat.boat.solve(
                        dt, boat_input_data.iloc[k].poa, control
                    )

                    if self.data.goal.accomplished(event_result=event_result[k_old]):
                        status = event_data.RaceStatus.FINISHED
                    else:
                        status = event_data.RaceStatus.STARTED

                    if status == event_data.RaceStatus.STARTED and k == t.size - 1:
                        raise event_error.EventGoalFailed("Race has ended")

                except event_error.EventGoalFailed as e:
                    old_status = event_result[k_old].status
                    status = event_data.RaceStatus.DNF
                    if old_status != event_data.RaceStatus.DNF:
                        print(
                            f"boat_model.Boat out of the race, status: {event_data.RaceStatus.to_str(old_status)}"
                            + f" => {event_data.RaceStatus.to_str(status)}. Reason: {e}"
                        )

                distance = boat_output_data[k].hull_speed * dt
                elapsed_time = Timedelta(
                    boat_input_data.iloc[k].time - boat_input_data.iloc[0].time
                ).to_timedelta64()

                event_result[k] = event_data.EventResultData(
                    distance=event_result[k_old].distance + distance,
                    elapsed_time=elapsed_time,
                    status=status,
                )

            output_data_dataset = boat_data.BoatOutputDataSet(list(boat_output_data))
            event_result_dataset = event_data.EventResultDataSet(list(event_result))

            event_output_data[controlled_boat.boat.name] = event_data.EventOutputData(
                name=self.data.name,
                input_data=boat_input_data,
                output_data=output_data_dataset,
                event_result=event_result_dataset,
            )

        return event_output_data
