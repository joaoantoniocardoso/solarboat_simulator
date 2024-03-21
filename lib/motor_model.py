from dataclasses import dataclass

import numpy as np
from typeguard import typechecked

debug = print


@dataclass
class Motor:
    maximum_current: np.float64
    operating_voltage: np.float64
    armature_resistence: np.float64
    viscous_friction_constant: np.float64
    motor_torque_constant: np.float64

    @typechecked
    def efficiency(self, input_power: np.float64) -> np.float64:
        # Clipping because zero is a pole for this model
        P_E = np.clip(input_power, a_min=0.1, a_max=None)  # type: ignore

        # Parameters from the model
        R_A = self.armature_resistence
        B_M = self.viscous_friction_constant
        K_T = self.motor_torque_constant
        V_A = self.operating_voltage

        # eta function broken into parts to be easier to digest
        i_a = P_E / V_A
        eta_p1 = -(B_M * R_A**2 * i_a) / (K_T**2 * V_A)
        eta_p2 = 2 * B_M * R_A / K_T**2
        eta_p3 = -(B_M * V_A) / (K_T**2 * i_a)
        eta_p4 = -(R_A * i_a) / V_A
        eta = 1 + eta_p1 + eta_p2 + eta_p3 + eta_p4

        # Efficiency can never be below 0 of above 1
        return np.clip(eta, a_min=0, a_max=1)  # type: ignore

    @typechecked
    def solve_input(self, input_power: np.float64) -> np.float64:
        if input_power > self.maximum_current * self.operating_voltage:
            input_power = self.maximum_current * self.operating_voltage
        elif input_power < 0:
            debug(f"[Motor] input_power < 0: {input_power=}")
            input_power = np.float64(0)

        return input_power

    @typechecked
    def solve_output(self, input_power: np.float64) -> np.float64:
        if input_power < 0:
            debug(f"[Motor] input_power < 0: {input_power=}")
            input_power = np.float64(0)

        # Computes the Output Power
        return input_power * self.efficiency(input_power)
