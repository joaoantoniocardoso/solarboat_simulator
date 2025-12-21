import control as ct
import numpy as np

from utils.models import eval_poly, lut_interp


class Battery:
    @classmethod
    def initial_state(cls, X0: dict, U0, params: dict):
        batt_R_0 = params["batt_R_0"]
        batt_k_V_OC_coeffs = params["batt_k_V_OC_coeffs"]
        batt_N_S = params["batt_N_S"]

        batt_z_0 = X0["batt_z"]
        batt_v_0 = X0["batt_v"]

        batt_i_0 = U0[0]

        batt_ocv_0 = eval_poly(batt_k_V_OC_coeffs, batt_z_0)

        v_C_0 = (batt_ocv_0 - batt_R_0 * batt_i_0 - batt_v_0 / batt_N_S) / 2.0

        batt_v_C_1_0 = v_C_0
        batt_v_C_2_0 = v_C_0

        return np.array([batt_z_0, batt_v_C_1_0, batt_v_C_2_0])

    @classmethod
    def _update(cls, t, x, u, params: dict):
        batt_eta = params["batt_eta"]
        batt_Q = params["batt_Q"]
        batt_R_1 = params["batt_R_1"]
        batt_C_1 = params["batt_C_1"]
        batt_R_2 = params["batt_R_2"]
        batt_C_2 = params["batt_C_2"]

        batt_i = u[0]

        batt_v_C_1 = x[1]
        batt_v_C_2 = x[2]

        d_batt_z = -batt_eta * batt_i / batt_Q
        d_batt_v_C_1 = -(batt_v_C_1 / (batt_R_1 * batt_C_1)) + (batt_i / batt_C_1)
        d_batt_v_C_2 = -(batt_v_C_2 / (batt_R_2 * batt_C_2)) + (batt_i / batt_C_2)

        return np.array([d_batt_z, d_batt_v_C_1, d_batt_v_C_2])

    @classmethod
    def _outputs(cls, t, x, u, params: dict):
        batt_N_S = params["batt_N_S"]
        batt_k_V_OC_coeffs = params["batt_k_V_OC_coeffs"]
        batt_R_0 = params["batt_R_0"]

        batt_i = u[0]

        batt_z = x[0]
        batt_v_C_1 = x[1]
        batt_v_C_2 = x[2]

        batt_ocv = eval_poly(batt_k_V_OC_coeffs, batt_z)
        batt_v = batt_N_S * (batt_ocv - batt_v_C_1 - batt_v_C_2 - batt_R_0 * batt_i)

        return np.array([batt_v, batt_z, batt_ocv])

    @classmethod
    def build(cls, params: dict):
        return ct.NonlinearIOSystem(
            cls._update,
            cls._outputs,
            name="battery",
            states=("batt_z", "batt_v_C_1", "batt_v_C_2"),
            inputs=("batt_i",),
            outputs=("batt_v", "batt_z", "batt_ocv"),
            params=params,
        )


class Battery1RC:
    @classmethod
    def initial_state(cls, X0: dict, U0, params: dict):
        batt_R_1_coeffs = params["batt_R_1_coeffs"]

        batt_z_0 = X0["batt_z"]
        batt_i_0 = U0[0]

        batt_R_1 = eval_poly(batt_R_1_coeffs, batt_z_0)
        batt_v_C_1_0 = batt_i_0 * batt_R_1

        return np.array([batt_z_0, batt_v_C_1_0])

    @classmethod
    def _update(cls, t, x, u, params: dict):
        batt_eta = params["batt_eta"]
        batt_Q = params["batt_Q"]
        batt_R_1_coeffs = params["batt_R_1_coeffs"]
        batt_C_1_coeffs = params["batt_C_1_coeffs"]

        batt_i = u[0]
        batt_z = x[0]
        batt_v_C_1 = x[1]

        batt_R_1 = eval_poly(batt_R_1_coeffs, batt_z)
        batt_C_1 = eval_poly(batt_C_1_coeffs, batt_z)

        d_batt_z = -batt_eta * batt_i / batt_Q
        d_batt_v_C_1 = -(batt_v_C_1 / (batt_R_1 * batt_C_1)) + (batt_i / batt_C_1)

        return np.array([d_batt_z, d_batt_v_C_1])

    @classmethod
    def _outputs(cls, t, x, u, params: dict):
        batt_N_S = params["batt_N_S"]
        batt_k_V_OC_coeffs = params["batt_k_V_OC_coeffs"]
        batt_R_0_coeffs = params["batt_R_0_coeffs"]

        batt_i = u[0]
        batt_z = x[0]
        batt_v_C_1 = x[1]

        batt_R_0 = eval_poly(batt_R_0_coeffs, batt_z)
        batt_ocv = eval_poly(batt_k_V_OC_coeffs, batt_z)
        batt_v = batt_N_S * (batt_ocv - batt_v_C_1 - batt_R_0 * batt_i)

        return np.array([batt_v, batt_z, batt_ocv])

    @classmethod
    def build(cls, params: dict):
        return ct.NonlinearIOSystem(
            cls._update,
            cls._outputs,
            name="battery",
            states=("batt_z", "batt_v_C_1"),
            inputs=("batt_i",),
            outputs=("batt_v", "batt_z", "batt_ocv"),
            params=params,
        )


class Battery1RCGassing:
    @staticmethod
    def _gassing_params(params: dict):
        I_g0 = params["batt_I_g0"]
        a_g = params["batt_a_g"]
        E_g = params["batt_E_g"]
        max_frac = params["batt_I_gas_max_frac"]
        return I_g0, a_g, E_g, max_frac

    @classmethod
    def _gassing_current(cls, v: float, i: float, params: dict) -> float:
        I_g0, a_g, E_g, max_frac = cls._gassing_params(params)

        if I_g0 <= 0 or a_g <= 0:
            return 0.0
        if i >= 0.0 or v <= E_g:
            return 0.0

        eta_g = v - E_g
        i_mag = I_g0 * np.exp(eta_g / a_g)
        i_gas = -i_mag

        max_i_gas = max_frac * abs(i)
        if abs(i_gas) > max_i_gas:
            i_gas = -max_i_gas
        return i_gas

    @classmethod
    def initial_state(cls, X0: dict, U0, params: dict):
        batt_R_1 = params["batt_R_1"]
        batt_k_V_OC_coeffs = params["batt_k_V_OC_coeffs"]
        batt_R_0 = params["batt_R_0"]

        batt_z_0 = X0["batt_z"]
        batt_i_0 = U0[0]

        batt_ocv_0 = lut_interp(batt_k_V_OC_coeffs, batt_z_0)

        v_guess = batt_ocv_0 - batt_R_0 * batt_i_0
        batt_i_gas_0 = cls._gassing_current(v_guess, batt_i_0, params)
        batt_i_main_0 = batt_i_0 - batt_i_gas_0

        batt_v_C_1_0 = batt_i_main_0 * batt_R_1

        return np.array([batt_z_0, batt_v_C_1_0])

    @classmethod
    def _update(cls, t, x, u, params: dict):
        batt_eta = params["batt_eta"]
        batt_Q = params["batt_Q"]
        batt_R_1 = params["batt_R_1"]
        batt_C_1 = params["batt_C_1"]
        batt_k_V_OC_coeffs = params["batt_k_V_OC_coeffs"]
        batt_R_0 = params["batt_R_0"]

        batt_i = u[0]
        batt_z = x[0]
        batt_v_C_1 = x[1]

        batt_ocv = lut_interp(batt_k_V_OC_coeffs, batt_z)
        batt_v = batt_ocv - batt_v_C_1 - batt_R_0 * batt_i
        batt_i_gas = cls._gassing_current(batt_v, batt_i, params)
        batt_i_main = batt_i - batt_i_gas

        d_batt_z = -batt_eta * batt_i_main / batt_Q
        d_batt_v_C_1 = -(batt_v_C_1 / (batt_R_1 * batt_C_1)) + (batt_i_main / batt_C_1)

        return np.array([d_batt_z, d_batt_v_C_1])

    @classmethod
    def _outputs(cls, t, x, u, params: dict):
        batt_k_V_OC_coeffs = params["batt_k_V_OC_coeffs"]
        batt_R_0 = params["batt_R_0"]

        batt_i = u[0]
        batt_z = x[0]
        batt_v_C_1 = x[1]

        batt_ocv = lut_interp(batt_k_V_OC_coeffs, batt_z)
        batt_v = batt_ocv - batt_v_C_1 - batt_R_0 * batt_i

        return np.array([batt_v, batt_z, batt_ocv])

    @classmethod
    def build(cls, params: dict):
        return ct.NonlinearIOSystem(
            cls._update,
            cls._outputs,
            name="battery",
            states=("batt_z", "batt_v_C_1"),
            inputs=("batt_i",),
            outputs=("batt_v", "batt_z", "batt_ocv"),
            params=params,
        )


class Battery1RCDeepGas:
    @staticmethod
    def _R0_from_soc(batt_z, params):
        R0_base = params["batt_R_0"]
        R0_deep = params.get("batt_R_0_deep", 0.0)
        if R0_deep == 0.0:
            return R0_base
        z_deep = params.get("batt_z_deep", 0.1)
        beta = params.get("batt_beta_deep", 20.0)
        s = 1.0 / (1.0 + np.exp(beta * (batt_z - z_deep)))
        return R0_base + R0_deep * s

    @staticmethod
    def _deep_discharge_extra_drop(batt_z, batt_i, params):
        k_deep = params.get("batt_k_deep", 0.0)
        if k_deep == 0.0:
            return 0.0
        z_deep = params.get("batt_z_deep", 0.0)
        depth = max(0.0, z_deep - batt_z)
        return k_deep * depth * abs(batt_i)

    @staticmethod
    def _gamma_gas(batt_z, params):
        z_gas = params.get("batt_z_gas", 1.0)
        alpha = params.get("batt_alpha_gas", 50.0)
        return 1.0 / (1.0 + np.exp(-alpha * (batt_z - z_gas)))

    @staticmethod
    def _V_over_from_q(batt_q_over, params):
        if batt_q_over <= 0.0:
            return 0.0
        V_ov_max = params.get("batt_V_ov_max", 0.0)
        q_ref = params.get("batt_q_ov_ref", 1.0)
        if V_ov_max == 0.0:
            return 0.0
        return V_ov_max * batt_q_over / (q_ref + batt_q_over)

    @classmethod
    def initial_state(cls, X0: dict, U0, params: dict):
        i0 = float(U0[0])
        z0 = float(X0["batt_z"])

        R1 = params["batt_R_1"]
        R2 = params["batt_R_2"]
        v_C1_0 = float(X0.get("batt_v_C_1", i0 * R1))
        v_C2_0 = float(X0.get("batt_v_C_2", i0 * R2))

        if "batt_v" in X0:
            v_pack = float(X0["batt_v"])
            n_s = params["batt_N_S"]
            v_block = v_pack / n_s

            ocv0 = lut_interp(params["batt_ocv_lut"], z0)
            R0eff = cls._R0_from_soc(z0, params)
            Vdeep = cls._deep_discharge_extra_drop(z0, i0, params)

            Vov_needed = v_block - (ocv0 - v_C1_0 - v_C2_0 - R0eff * i0 + Vdeep)

            Vov_max = params.get("batt_V_ov_max", 0.0)
            qref = params.get("batt_q_ov_ref", 1.0)

            if Vov_max <= 0.0:
                q_over_0 = 0.0
            else:
                if Vov_needed <= 0.0:
                    q_over_0 = 0.0
                elif Vov_needed >= Vov_max:
                    q_over_0 = 10.0 * qref
                else:
                    q_over_0 = (Vov_needed / (Vov_max - Vov_needed)) * qref
        else:
            q_over_0 = float(X0.get("batt_q_over", 0.0))

        return np.array([z0, v_C1_0, v_C2_0, q_over_0], dtype=float)

    @classmethod
    def _update(cls, t, x, u, params: dict):
        batt_Q = params["batt_Q"]
        batt_R_1 = params["batt_R_1"]
        batt_R_2 = params["batt_R_2"]
        batt_C_1 = params["batt_C_1"]
        batt_C_2 = params["batt_C_2"]
        eta_dis = params["batt_eta_dis"]
        eta_chg = params["batt_eta_chg"]

        batt_i = float(u[0])

        batt_z = float(x[0])
        batt_v_C_1 = float(x[1])
        batt_v_C_2 = float(x[2])
        batt_q_over = float(x[3])

        d_batt_v_C_1 = -(batt_v_C_1 / (batt_R_1 * batt_C_1)) + (batt_i / batt_C_1)
        d_batt_v_C_2 = -(batt_v_C_2 / (batt_R_2 * batt_C_2)) + (batt_i / batt_C_2)

        k_ov_in = params.get("batt_k_ov_in", 0.0)
        tau_ov = params.get("batt_tau_ov", 3600.0)

        d_batt_q_over = -batt_q_over / tau_ov

        if batt_i >= 0.0:
            d_batt_z = -eta_dis * batt_i / batt_Q
        else:
            gamma_gas = cls._gamma_gas(batt_z, params)
            i_soc = batt_i * (1.0 - gamma_gas)
            i_ov = (-batt_i) * gamma_gas
            d_batt_z = -eta_chg * i_soc / batt_Q
            d_batt_q_over += k_ov_in * i_ov

        return np.array([d_batt_z, d_batt_v_C_1, d_batt_v_C_2, d_batt_q_over])

    @classmethod
    def _outputs(cls, t, x, u, params: dict):
        batt_N_S = params["batt_N_S"]
        batt_ocv_lut = params["batt_ocv_lut"]

        batt_i = float(u[0])
        batt_z = float(x[0])
        batt_v_C_1 = float(x[1])
        batt_v_C_2 = float(x[2])
        batt_q_over = float(x[3])

        batt_ocv = lut_interp(batt_ocv_lut, batt_z)
        batt_R_0_eff = cls._R0_from_soc(batt_z, params)
        V_deep = cls._deep_discharge_extra_drop(batt_z, batt_i, params)
        V_over = cls._V_over_from_q(batt_q_over, params)

        batt_v_block = (
            batt_ocv + V_over - V_deep - batt_v_C_1 - batt_v_C_2 - batt_R_0_eff * batt_i
        )

        batt_v = batt_N_S * batt_v_block

        return np.array([batt_v, batt_z, batt_ocv])

    @classmethod
    def build(cls, params: dict):
        return ct.NonlinearIOSystem(
            cls._update,
            cls._outputs,
            name="battery",
            states=("batt_z", "batt_v_C_1", "batt_v_C_2", "batt_q_over"),
            inputs=("batt_i",),
            outputs=("batt_v", "batt_z", "batt_ocv"),
            params=params,
        )


Battery2RC = Battery
