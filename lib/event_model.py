import numpy as np

from dataclasses import dataclass
from numpy import int64
from pandas import Timestamp, DataFrame

from lib.boat_data import (
    BoatInputDataSet,
    BoatOutputData,
    BoatOutputDataSet,
)
from lib.boat_model import Boat
from lib.energy_controller_model import EnergyController


@dataclass
class EventResult:
    name: str
    input_data: BoatInputDataSet
    output_data: BoatOutputDataSet


@dataclass
class Event:
    name: str
    description: str
    # route: list[tuple[float, float]]
    start: Timestamp
    end: Timestamp

    def run(
        self,
        input_data: BoatInputDataSet,
        boat: Boat,
        energy_controller: EnergyController,
    ) -> EventResult:
        # Transform time vector to seconds
        t = input_data.time.to_numpy().astype(int64)
        t = (t - t[0]) * 1e-9

        output_data = np.zeros(t.size, dtype=BoatOutputData)

        dt: int64 = t[1] - t[0]
        for k in range(t.size):
            if k > 0:
                dt = t[k] - t[k - 1]

            control = energy_controller.run(
                dt=float(dt),
                input_data=input_data.iloc[k],
                output_data=output_data[k],
                boat=boat,
            )

            output_data[k] = boat.run(float(dt), input_data.iloc[k].poa, control)

            """ TODO list:
                - [ ] Calcular distância do barco
                - [ ] Criar objetivo e restrições da prova, suportando diferentes tipos de provas:
                    - [ ] Prova com tempo máximo, distância fixa. Exemplo: prova curta
                    - [ ] Prova com tempo fixo, distância variável. Exemplo: prova do piloto
                - [ ] Criar variável para monitorar o estado para o objetivo da prova.
                    - Started{time},
                    - DoNotStarted{time, reason}
                    - Finished{time}
                    - DoNotFinished{time, reason}
                - [ ] Se alguma excessão ocorrer com o controller.run ou boat.run, modificar o
                estado do objetivo da prova.
                - [ ] O Controlador deve monitorar o estado do objetivo da prova e saber quando
                deve iniciar a recarga das baterias.
            """
        return EventResult(
            name=self.name,
            input_data=input_data,
            output_data=DataFrame(list(output_data)).pipe(BoatOutputDataSet),
        )
