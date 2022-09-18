import numpy as np

from dataclasses import dataclass
from numpy import float64, int64
from pandas import Timestamp

from lib.boat_data import BoatInputData, BoatOutputData
from lib.boat_model import Boat
from lib.energy_controller_model import EnergyController


@dataclass
class EventResult:
    name: str
    input_data: BoatInputData
    output_data: BoatOutputData


@dataclass
class Event:
    name: str
    description: str
    # route: list[tuple[float, float]]
    start: Timestamp
    end: Timestamp

    def run(
        self, input_data: BoatInputData, boat: Boat, energy_controller: EnergyController
    ) -> EventResult:
        # Transform time vector to seconds
        t = input_data.time.astype(int64)
        t = (t - t[0]) * 1e-9

        output_data = BoatOutputData(
            pv_output_power=np.zeros(t.size, dtype=float64),
            battery_stored_energy=np.zeros(t.size, dtype=float64),
            battery_output_power=np.zeros(t.size, dtype=float64),
            esc_input_power=np.zeros(t.size, dtype=float64),
            motor_output_power=np.zeros(t.size, dtype=float64),
            propulsive_output_power=np.zeros(t.size, dtype=float64),
            hull_speed=np.zeros(t.size, dtype=float64),
            pv_target_power=np.zeros(t.size, dtype=float64),
            battery_target_power=np.zeros(t.size, dtype=float64),
            esc_target_power=np.zeros(t.size, dtype=float64),
            motor_target_throttle=np.zeros(t.size, dtype=float64),
            battery_soc=np.zeros(t.size, dtype=float64),
            esc_output_power=np.zeros(t.size, dtype=float64),
        )

        dt: int64 = t[1] - t[0]
        for k in range(t.size):
            if k > 0:
                dt = t[k] - t[k - 1]

            control = energy_controller.run(
                dt=float(dt),
                input_data=BoatInputData(
                    time=input_data.time[k], poa=input_data.poa[k]
                ),
                output_data=BoatOutputData(
                    pv_output_power=output_data.pv_output_power[k],
                    battery_stored_energy=output_data.battery_stored_energy[k],
                    battery_output_power=output_data.battery_output_power[k],
                    esc_input_power=output_data.esc_input_power[k],
                    motor_output_power=output_data.motor_output_power[k],
                    propulsive_output_power=output_data.propulsive_output_power[k],
                    hull_speed=output_data.hull_speed[k],
                    pv_target_power=output_data.pv_target_power[k],
                    battery_target_power=output_data.battery_target_power[k],
                    esc_target_power=output_data.esc_target_power[k],
                    motor_target_throttle=output_data.motor_target_throttle[k],
                    battery_soc=output_data.battery_soc[k],
                    esc_output_power=output_data.esc_output_power[k],
                ),
                boat=boat,
            )

            y = boat.run(float(dt), input_data.poa[k], control)

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

            output_data.pv_output_power[k] = y.pv_output_power
            output_data.battery_stored_energy[k] = y.battery_stored_energy
            output_data.battery_output_power[k] = y.battery_output_power
            output_data.esc_input_power[k] = y.esc_input_power
            output_data.motor_output_power[k] = y.motor_output_power
            output_data.propulsive_output_power[k] = y.propulsive_output_power
            output_data.hull_speed[k] = y.hull_speed
            output_data.pv_target_power[k] = y.pv_target_power
            output_data.battery_target_power[k] = y.battery_target_power
            output_data.esc_target_power[k] = y.esc_target_power
            output_data.motor_target_throttle[k] = y.motor_target_throttle
            output_data.battery_soc[k] = y.battery_soc
            output_data.esc_output_power[k] = y.esc_output_power

        return EventResult(
            name=self.name, input_data=input_data, output_data=output_data
        )
