import numpy as np
import control as ct


class Generation:
    @classmethod
    def initial_state(cls, X0: dict, U0, params: dict):
        # Stateless model; allow caller to pass None or empty state.
        return None

    @classmethod
    def _outputs(cls, t, x, u, params: dict):
        # Params
        pv_S, pv_eta, mppt_eta = (
            params["pv_S"],
            params["pv_eta"],
            params["mppt_eta"],
        )

        # Inputs
        pv_g = u[0]  # Total incident irradiance [W/m^2]
        mppt_v_out = u[1]  # MPPT Output Voltage (battery) [V]
        mppt_d = u[2]  # MPPT Duty Cycle

        # Output equations
        pv_p_out = pv_g * pv_S * pv_eta
        mppt_i_out = mppt_eta * pv_p_out / mppt_v_out
        mppt_i_in = mppt_i_out * (mppt_d / (1 - mppt_d))
        mppt_v_in = pv_p_out / mppt_i_in

        return np.array([mppt_i_out, mppt_i_in, mppt_v_in])

    @classmethod
    def build(cls, params: dict):
        return ct.NonlinearIOSystem(
            None,
            cls._outputs,
            name="generation",
            states=None,
            inputs=("pv_g", "mppt_v_out", "mppt4_d"),
            outputs=("mppt_i_out", "mppt_i_in", "mppt_v_in"),
            params=params,
        )
