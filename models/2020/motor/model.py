import numpy as np
import control as ct


class SteadyStateMotor:
    """Motor model assuming steady-state behavior (no dynamics)."""

    @classmethod
    def initial_state(cls, X0: dict, U0, params: dict):
        return np.array([])

    @classmethod
    def _outputs(cls, t, x, u, params: dict):
        motor_R_A = params["motor_R_A"]
        motor_B = params["motor_B"]
        motor_K_Q = params["motor_K_Q"]
        motor_K_V = params["motor_K_V"]

        motor_v = u[0]
        motor_q_load = u[1]

        motor_i = (motor_B * motor_v + motor_K_V * motor_q_load) / (
            motor_B * motor_R_A + motor_K_Q * motor_K_V
        )
        motor_w = (motor_K_Q * motor_v - motor_R_A * motor_q_load) / (
            motor_B * motor_R_A + motor_K_Q * motor_K_V
        )

        return np.array([motor_i, motor_w])

    @classmethod
    def build(cls, params: dict):
        return ct.NonlinearIOSystem(
            None,
            cls._outputs,
            name="motor",
            inputs=("motor_v", "motor_q_load"),
            outputs=("motor_i", "motor_w"),
            params=params,
        )


class DynamicMotor:
    """Motor model including electrical and mechanical dynamics."""

    @classmethod
    def initial_state(cls, X0: dict, U0, params: dict):
        motor_R_A = params["motor_R_A"]
        motor_B = params["motor_B"]
        motor_K_Q = params["motor_K_Q"]
        motor_K_V = params["motor_K_V"]

        motor_v = U0[0]
        motor_q_load = U0[1]

        motor_i = (motor_B * motor_v + motor_K_V * motor_q_load) / (
            motor_B * motor_R_A + motor_K_Q * motor_K_V
        )
        motor_w = (motor_K_Q * motor_v - motor_R_A * motor_q_load) / (
            motor_B * motor_R_A + motor_K_Q * motor_K_V
        )

        return np.array([motor_i, motor_w])

    @classmethod
    def _update(cls, t, x, u, params: dict):
        motor_R_A = params["motor_R_A"]
        motor_L_A = params["motor_L_A"]
        motor_B = params["motor_B"]
        motor_I_r = params["motor_I_r"]
        motor_I_r_load = params["motor_I_r_load"]
        motor_K_Q = params["motor_K_Q"]
        motor_K_V = params["motor_K_V"]

        motor_i = x[0]
        motor_w = x[1]

        motor_v = u[0]
        motor_q_load = u[1]

        d_motor_i = (-motor_K_V * motor_w - motor_R_A * motor_i + motor_v) / motor_L_A
        d_motor_w = (-motor_B * motor_w + motor_K_Q * motor_i - motor_q_load) / (
            motor_I_r + motor_I_r_load
        )

        return np.array([d_motor_i, d_motor_w])

    @classmethod
    def build(cls, params: dict):
        return ct.NonlinearIOSystem(
            cls._update,
            None,
            name="motor",
            states=("motor_i", "motor_w"),
            inputs=("motor_v", "motor_q_load"),
            outputs=("motor_i", "motor_w"),
            params=params,
        )
