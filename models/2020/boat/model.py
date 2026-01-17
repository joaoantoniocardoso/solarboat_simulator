import math

import control as ct
import numpy as np

from utils.models import eval_poly


class SolarBoat:
    @classmethod
    def initial_state(cls, X0: dict, U0, params: dict):
        if "batt_z" not in X0 or "batt_v" not in X0:
            raise ValueError(
                "X0 must contain 'batt_z' and 'batt_v' for SolarBoat.initial_state."
            )

        batt_z_0 = X0["batt_z"]
        batt_v_0 = X0["batt_v"]
        hull_u_0 = X0.get("hull_u", 0.0)

        # Inputs
        esc_d_0 = U0[1] if len(U0) > 1 else 0.0

        batt_R_0 = params["batt_R_0"]
        batt_k_V_OC_coeffs = params["batt_k_V_OC_coeffs"]
        batt_N_S = params["batt_N_S"]

        # [ASSUMPTION]: At t=0, battery current is 0 A when computing RC initial voltages.
        batt_i_0 = 0.0
        batt_ocv_0 = eval_poly(batt_k_V_OC_coeffs, batt_z_0)

        v_C_0 = (batt_ocv_0 - batt_R_0 * batt_i_0 - batt_v_0 / batt_N_S) / 2.0
        batt_v_C_1_0 = v_C_0
        batt_v_C_2_0 = v_C_0

        # [ASSUMPTION]: Motor starts from rest with zero current.
        motor_i_0 = 0.0
        motor_w_0 = 0.0

        return np.array(
            [
                batt_v_C_1_0,
                batt_v_C_2_0,
                batt_z_0,
                motor_i_0,
                motor_w_0,
                hull_u_0,
            ]
        )

    @classmethod
    def _common(cls, t, x, u, params: dict):
        # Parameters:
        esc_F_s = params["esc_F_s"]
        esc_V_ds_ov = params["esc_V_ds_ov"]
        esc_R_ds_on = params["esc_R_ds_on"]
        esc_E_on = params["esc_E_on"]
        esc_E_off = params["esc_E_off"]
        esc_V_F = params["esc_V_F"]
        esc_r_D = params["esc_r_D"]
        esc_Q_rr = params["esc_Q_rr"]
        batt_R_0 = params["batt_R_0"]
        batt_N_S = params["batt_N_S"]
        batt_k_V_OC_coeffs = params["batt_k_V_OC_coeffs"]
        mppts_n = params["mppts_n"]
        mppt_eta = params["mppt_eta"]
        pv_eta = params["pv_eta"]
        pv_S = params["pv_S"]
        oth_p_in = params["oth_p_in"]

        # Inputs:
        pv_g = u[0]
        esc_d = u[1]

        # States:
        batt_v_C_1 = x[0]
        batt_v_C_2 = x[1]
        batt_z = x[2]
        motor_i = x[3]

        # Outputs:
        eps = 1e-9
        batt_v_oc = max(eval_poly(batt_k_V_OC_coeffs, batt_z), eps)

        batt_v = (
            0.5
            * math.sqrt(batt_N_S)
            * math.sqrt(
                batt_N_S
                * (
                    -batt_R_0 * esc_F_s * esc_Q_rr * esc_V_ds_ov
                    - batt_R_0 * esc_F_s * esc_Q_rr
                    - batt_R_0 * esc_d * motor_i
                    - batt_v_C_1
                    - batt_v_C_2
                    + batt_v_oc
                )
                ** 2
                - 4 * batt_R_0 * esc_E_off * esc_F_s * esc_V_ds_ov
                - 4 * batt_R_0 * esc_E_off * esc_F_s
                - 4 * batt_R_0 * esc_E_on * esc_F_s * esc_V_ds_ov
                - 4 * batt_R_0 * esc_E_on * esc_F_s
                - 4 * batt_R_0 * esc_R_ds_on * esc_d * motor_i**2
                + 4 * batt_R_0 * esc_V_F * esc_d * motor_i
                - 4 * batt_R_0 * esc_V_F * motor_i
                + 4 * batt_R_0 * esc_r_D * esc_d * motor_i**2
                - 4 * batt_R_0 * esc_r_D * motor_i**2
                + 4 * batt_R_0 * mppts_n * pv_S * mppt_eta * pv_eta * pv_g
                - 4 * batt_R_0 * oth_p_in
            )
            - 0.5 * batt_N_S * batt_R_0 * esc_F_s * esc_Q_rr * esc_V_ds_ov
            - 0.5 * batt_N_S * batt_R_0 * esc_F_s * esc_Q_rr
            - 0.5 * batt_N_S * batt_R_0 * esc_d * motor_i
            - 0.5 * batt_N_S * batt_v_C_1
            - 0.5 * batt_N_S * batt_v_C_2
            + 0.5 * batt_N_S * batt_v_oc
        )

        batt_v_safe = max(batt_v, eps)
        mppts_i_out = mppts_n * pv_S * mppt_eta * pv_eta * pv_g / batt_v_safe
        esc_i_in = (
            esc_F_s * esc_Q_rr * (esc_V_ds_ov + 1) * batt_v
            + esc_F_s * (esc_E_off + esc_E_on) * (esc_V_ds_ov + 1)
            + esc_R_ds_on * esc_d * motor_i**2
            - esc_V_F * (esc_d - 1) * motor_i
            - esc_r_D * (esc_d - 1) * motor_i**2
            + esc_d * motor_i * batt_v
        ) / batt_v_safe

        return batt_v, mppts_i_out, esc_i_in

    @classmethod
    def _update(cls, t, x, u, params: dict):
        # Parameters:
        rho_water = params["rho_water"]
        prop_D = params["prop_D"]
        prop_I_r = params["prop_I_r"]
        prop_k_T_coeffs = params["prop_k_T_coeffs"]
        prop_k_Q_coeffs = params["prop_k_Q_coeffs"]
        rho_air = params["rho_air"]
        hull_S_water = params["hull_S_water"]
        hull_S_air = params["hull_S_air"]
        hull_C_T = params["hull_C_T"]
        hull_M = params["hull_M"]
        hull_M_a = params["hull_M_a"]
        prop_eta_R = params["prop_eta_R"]
        hull_W = params["hull_W"]
        hull_T_ded = params["hull_T_ded"]
        trans_k = params["trans_k"]
        trans_I_r_out = params["trans_I_r_out"]
        trans_I_r_in = params["trans_I_r_in"]
        trans_eta = params["trans_eta"]
        motor_R_A = params["motor_R_A"]
        motor_L_A = params["motor_L_A"]
        motor_K_Q = params["motor_K_Q"]
        motor_K_V = params["motor_K_V"]
        motor_I_r = params["motor_I_r"]
        motor_B = params["motor_B"]
        batt_R_1 = params["batt_R_1"]
        batt_C_1 = params["batt_C_1"]
        batt_R_2 = params["batt_R_2"]
        batt_C_2 = params["batt_C_2"]
        batt_Q = params["batt_Q"]
        batt_eta = params["batt_eta"]
        oth_p_in = params["oth_p_in"]

        # Inputs:
        esc_d = u[1]

        # States:
        batt_v_C_1 = x[0]
        batt_v_C_2 = x[1]
        motor_i = x[3]
        motor_w = x[4]
        hull_u = x[5]

        batt_v, mppts_i_out, esc_i_in = cls._common(t, x, u, params)

        # Auxiliars:
        eps = 1e-9
        oth_i_in = oth_p_in / max(batt_v, eps)
        batt_i = esc_i_in - mppts_i_out + oth_i_in

        motor_I_r_load = trans_I_r_in + trans_k**2 * (prop_I_r + trans_I_r_out)
        motor_v = esc_d * batt_v

        prop_n = 0.5 * trans_k * motor_w / math.pi
        prop_J_max = params.get("prop_J_max", 1.5)
        prop_j = (1 - hull_W) * hull_u / (prop_D * (prop_n + eps))
        prop_j = max(0.0, min(prop_j, prop_J_max))

        prop_k_t_ow = max(0.0, eval_poly(prop_k_T_coeffs, prop_j))
        prop_k_q_ow = max(0.0, eval_poly(prop_k_Q_coeffs, prop_j))

        prop_t = prop_D**4 * rho_water * prop_k_t_ow * prop_n**2
        prop_q = prop_D**5 * rho_water * prop_k_q_ow * prop_n**2 / max(prop_eta_R, eps)

        hull_r = (
            0.5
            * hull_C_T
            * (hull_S_air * rho_air + hull_S_water * rho_water)
            * hull_u**2
        )

        prop_t_e = (1.0 - hull_T_ded) * prop_t
        motor_q_load = trans_k * prop_q / max(trans_eta, eps)

        d_batt_v_C_1 = (batt_R_1 * batt_i - batt_v_C_1) / (batt_C_1 * batt_R_1)
        d_batt_v_C_2 = (batt_R_2 * batt_i - batt_v_C_2) / (batt_C_2 * batt_R_2)
        d_batt_z = -batt_eta * batt_i / batt_Q
        d_motor_i = (-motor_K_V * motor_w - motor_R_A * motor_i + motor_v) / motor_L_A
        d_motor_w = (-motor_B * motor_w + motor_K_Q * motor_i - motor_q_load) / (
            motor_I_r + motor_I_r_load
        )
        d_hull_u = (prop_t_e - hull_r) / (hull_M + hull_M_a)

        return [
            d_batt_v_C_1,
            d_batt_v_C_2,
            d_batt_z,
            d_motor_i,
            d_motor_w,
            d_hull_u,
        ]

    @classmethod
    def _outputs(cls, t, x, u, params: dict):
        # States:
        motor_w = x[4]
        hull_u = x[5]
        esc_d = u[1]

        # Outputs:
        batt_v, mppts_i_out, esc_i_in = cls._common(t, x, u, params)
        return [batt_v, mppts_i_out, motor_w, esc_i_in, esc_d, hull_u]

    @classmethod
    def build(cls, params: dict):
        return ct.NonlinearIOSystem(
            cls._update,
            cls._outputs,
            name="solarboat",
            inputs=["pv_g", "esc_d"],
            states=[
                "batt_v_C_1",
                "batt_v_C_2",
                "batt_z",
                "motor_i",
                "motor_w",
                "hull_u",
            ],
            outputs=["batt_v", "mppts_i_out", "motor_w", "esc_i_in", "esc_d", "hull_u"],
            params=params,
        )


class SolarBoatInternalSignalModel:
    @staticmethod
    def _safe_div(num: float, den: float, eps: float = 1e-9) -> float:
        if abs(den) <= eps:
            return 0.0
        return num / den

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(x, hi))

    @classmethod
    def initial_state(cls, X0: dict, U0, params: dict):
        if "batt_z" not in X0 or "batt_v" not in X0:
            raise ValueError(
                "X0 must contain 'batt_z' and 'batt_v' for SolarBoatInternalSignalModel.initial_state."
            )

        # Required initial conditions
        batt_z_0 = X0["batt_z"]
        batt_v_0 = X0["batt_v"]

        # Params
        batt_R_0 = params["batt_R_0"]
        batt_k_V_OC_coeffs = params["batt_k_V_OC_coeffs"]
        batt_N_S = params["batt_N_S"]
        oth_p_in = params["oth_p_in"]

        eps = 1e-9

        # U_obs order: batt_v, pv_g, mppts_i_out, motor_w, esc_i_in, esc_d, hull_u
        mppts_i_out_0 = U0[2] if len(U0) > 2 else 0.0
        esc_i_in_0 = U0[4] if len(U0) > 4 else 0.0

        oth_i_in_0 = oth_p_in / max(batt_v_0, eps)
        batt_i_0 = esc_i_in_0 - mppts_i_out_0 + oth_i_in_0

        batt_ocv_0 = eval_poly(batt_k_V_OC_coeffs, batt_z_0)
        v_C_0 = (batt_ocv_0 - batt_R_0 * batt_i_0 - batt_v_0 / batt_N_S) / 2.0

        batt_v_C_1_0 = v_C_0
        batt_v_C_2_0 = v_C_0
        motor_i_0 = X0.get("motor_i", 0.0)

        return np.array([batt_v_C_1_0, batt_v_C_2_0, batt_z_0, motor_i_0])

    @classmethod
    def _update(cls, t, x, u, params: dict):
        # Params
        batt_R_1 = params["batt_R_1"]
        batt_C_1 = params["batt_C_1"]
        batt_R_2 = params["batt_R_2"]
        batt_C_2 = params["batt_C_2"]
        batt_eta = params["batt_eta"]
        batt_Q = params["batt_Q"]
        motor_R_A = params["motor_R_A"]
        motor_L_A = params["motor_L_A"]
        motor_K_V = params["motor_K_V"]
        oth_p_in = params["oth_p_in"]

        # Inputs
        batt_v = u[0]
        mppts_i_out = u[2]
        motor_w = u[3]
        esc_i_in = u[4]
        esc_d = u[5]

        # States
        batt_v_C_1 = x[0]
        batt_v_C_2 = x[1]
        batt_z = x[2]
        motor_i = x[3]

        # Auxiliars:
        eps = 1e-9
        oth_i_in = oth_p_in / max(batt_v, eps)
        batt_i = esc_i_in - mppts_i_out + oth_i_in
        motor_v = esc_d * batt_v

        d_batt_v_C_1 = (batt_R_1 * batt_i - batt_v_C_1) / (batt_C_1 * batt_R_1)
        d_batt_v_C_2 = (batt_R_2 * batt_i - batt_v_C_2) / (batt_C_2 * batt_R_2)
        d_batt_z = -batt_eta * batt_i / batt_Q
        d_motor_i = (-motor_K_V * motor_w - motor_R_A * motor_i + motor_v) / motor_L_A

        return np.array([d_batt_v_C_1, d_batt_v_C_2, d_batt_z, d_motor_i], float)

    @classmethod
    def _outputs(cls, t, x, u, params: dict):
        # Params
        rho_water = params["rho_water"]
        rho_air = params["rho_air"]
        prop_D = params["prop_D"]
        prop_eta_R = params["prop_eta_R"]
        prop_k_T_coeffs = params["prop_k_T_coeffs"]
        prop_k_Q_coeffs = params["prop_k_Q_coeffs"]
        hull_S_water = params["hull_S_water"]
        hull_S_air = params["hull_S_air"]
        hull_C_T = params["hull_C_T"]
        hull_W = params["hull_W"]
        hull_T_ded = params["hull_T_ded"]
        trans_k = params["trans_k"]
        trans_eta = params["trans_eta"]
        motor_R_A = params["motor_R_A"]
        motor_B = params["motor_B"]
        esc_F_s = params["esc_F_s"]
        esc_V_ds_ov = params["esc_V_ds_ov"]
        esc_R_ds_on = params["esc_R_ds_on"]
        esc_E_on = params["esc_E_on"]
        esc_E_off = params["esc_E_off"]
        esc_V_F = params["esc_V_F"]
        esc_r_D = params["esc_r_D"]
        esc_Q_rr = params["esc_Q_rr"]
        batt_k_V_OC_coeffs = params["batt_k_V_OC_coeffs"]
        pv_S = params["pv_S"]
        pv_eta = params["pv_eta"]
        pvs_n = params["pvs_n"]
        mppts_n = params["mppts_n"]
        mppt_eta = params["mppt_eta"]
        oth_p_in = params["oth_p_in"]

        # Inputs
        batt_v = u[0]
        pv_g = u[1]
        mppts_i_out = u[2]
        motor_w = u[3]
        esc_i_in = u[4]
        esc_d_in = u[5]
        hull_u_in = u[6]

        # States
        batt_v_C_1 = x[0]
        batt_v_C_2 = x[1]
        batt_z = x[2]
        motor_i = x[3]

        # Auxiliars:
        eps = 1e-9
        batt_v_safe = max(batt_v, eps)
        esc_d = cls._clamp(esc_d_in, 0.0, 1.0)
        hull_u = max(hull_u_in, 0.0)

        oth_i_in = oth_p_in / batt_v_safe
        batt_i = esc_i_in - mppts_i_out + oth_i_in
        batt_v_oc = eval_poly(batt_k_V_OC_coeffs, batt_z)
        batt_p_out = batt_i * batt_v
        batt_p_in = -batt_i * batt_v

        pv_p_in = pv_S * pv_g
        pv_p_out = pv_eta * pv_p_in

        # PVs/MPPTs special case (aggregated behavior):
        # Keep the bookkeeping consistent with the previous implementation and the
        # notebook export: MPPT output is observable (mppts_i_out, batt_v), while
        # PV-side power is back-computed from efficiencies.

        # Base PV array power (still reported for reference)
        pvs_p_in = pvs_n * pv_p_in
        pvs_p_out = pvs_n * pv_p_out

        # MPPTs (aggregate)
        mppt_v_out = batt_v
        mppt_i_out = cls._safe_div(mppts_i_out, mppts_n, eps)
        mppt_p_out = mppt_i_out * mppt_v_out
        mppt_p_in = cls._safe_div(mppt_p_out, mppt_eta, eps)

        # Choose a consistent MPPT input voltage to close equations.
        # Prior code did not model PV I/V, so we keep v_in = v_out.
        mppt_v_in = mppt_v_out
        mppt_i_in = cls._safe_div(mppt_p_in, max(mppt_v_in, eps), eps)

        mppts_p_out = mppts_i_out * batt_v
        mppts_p_in = cls._safe_div(mppts_p_out, mppt_eta, eps)
        mppts_eta = cls._safe_div(mppts_p_out, mppts_p_in, eps)

        # Back-compute PVs from MPPT input power (not from irradiance model).
        pvs_p_out = mppts_p_in
        pvs_p_in = cls._safe_div(pvs_p_out, max(pv_eta, eps), eps)
        pvs_eta = cls._safe_div(pvs_p_out, pvs_p_in, eps)
        pvs_i_out = cls._safe_div(pvs_p_out, batt_v_safe, eps)

        mppts_i_out_y = mppts_i_out

        esc_v_in = batt_v
        esc_v_out = esc_d * esc_v_in
        esc_i_out = motor_i
        esc_p_in = esc_i_in * esc_v_in
        esc_p_out = esc_i_out * esc_v_out

        esc_i_M_RMS = math.sqrt(max(esc_d, 0.0)) * esc_i_out
        esc_i_D_RMS = math.sqrt(max(1.0 - esc_d, 0.0)) * esc_i_out
        esc_i_D_AVG = (1.0 - esc_d) * esc_i_out

        esc_v_ds_sw = (esc_V_ds_ov + 1.0) * esc_v_in
        esc_p_M_cond = esc_R_ds_on * esc_i_M_RMS**2
        esc_p_D_cond = esc_V_F * esc_i_D_AVG + esc_r_D * esc_i_D_RMS**2
        esc_p_M_sw = esc_F_s * (esc_E_off + esc_E_on) * esc_v_ds_sw / batt_v_safe
        esc_p_D_sw = esc_F_s * esc_Q_rr * esc_v_ds_sw
        esc_p_loss = esc_p_D_cond + esc_p_D_sw + esc_p_M_cond + esc_p_M_sw

        esc_eta = cls._safe_div(esc_p_out, esc_p_in, eps)

        motor_v = esc_d * batt_v

        prop_w = trans_k * motor_w
        prop_n = 0.5 * trans_k * motor_w / math.pi
        prop_u = (1.0 - hull_W) * hull_u

        prop_j = (1.0 - hull_W) * hull_u / (prop_D * (prop_n + eps))
        prop_j = max(0.0, min(prop_j, params.get("prop_J_max", 1.5)))

        prop_j_ow = prop_j
        prop_k_t_ow = max(0.0, eval_poly(prop_k_T_coeffs, prop_j_ow))
        prop_k_q_ow = max(0.0, eval_poly(prop_k_Q_coeffs, prop_j_ow))

        prop_t_ow = prop_D**4 * rho_water * prop_k_t_ow * prop_n**2
        prop_q_ow = prop_D**5 * rho_water * prop_k_q_ow * prop_n**2

        prop_t = prop_t_ow
        prop_q = prop_q_ow / max(prop_eta_R, eps)

        prop_u_ow = prop_D * prop_j_ow * prop_n
        prop_p_in_ow = prop_w * prop_q_ow
        prop_p_out_ow = prop_t_ow * prop_u_ow
        prop_eta_ow = cls._safe_div(
            0.5 * prop_j_ow * prop_k_t_ow, math.pi * prop_k_q_ow, eps
        )

        prop_p_in = prop_w * prop_q
        prop_p_out = prop_t * prop_u
        prop_eta = cls._safe_div(prop_p_out, prop_p_in, eps)

        motor_q_load = trans_k * prop_q / max(trans_eta, eps)
        motor_p_in = motor_i * motor_v
        motor_p_out = motor_w * motor_q_load
        motor_p_loss = motor_B * motor_w**2 + motor_R_A * motor_i**2
        motor_eta = cls._safe_div(motor_p_out, motor_p_in, eps)

        trans_w_in = motor_w
        trans_w_out = trans_k * trans_w_in
        trans_q_in = motor_q_load
        trans_q_out = prop_q
        trans_p_in = trans_w_in * trans_q_in
        trans_p_out = trans_w_out * trans_q_out

        hull_r_t = (
            0.5
            * hull_C_T
            * (hull_S_air * rho_air + hull_S_water * rho_water)
            * hull_u**2
        )
        hull_t = (1.0 - hull_T_ded) * prop_t
        hull_p_in = hull_t * hull_u
        hull_p_out = hull_r_t * hull_u
        hull_eta = cls._safe_div(hull_p_out, hull_p_in, eps)

        return np.array(
            [
                batt_i,
                batt_v_C_1,
                batt_v_C_2,
                batt_z,
                batt_v_oc,
                batt_p_out,
                batt_p_in,
                pv_p_in,
                pv_p_out,
                pv_eta,
                pvs_p_in,
                pvs_p_out,
                pvs_i_out,
                mppt_p_in,
                mppt_p_out,
                mppt_v_in,
                mppt_v_out,
                mppt_i_in,
                mppt_i_out,
                mppt_eta,
                mppts_p_in,
                mppts_p_out,
                mppts_i_out_y,
                esc_v_in,
                esc_v_out,
                esc_i_in,
                esc_i_out,
                esc_p_in,
                esc_p_out,
                esc_eta,
                esc_p_loss,
                esc_p_M_cond,
                esc_p_D_cond,
                esc_p_M_sw,
                esc_p_D_sw,
                esc_i_M_RMS,
                esc_i_D_RMS,
                esc_i_D_AVG,
                motor_i,
                motor_v,
                motor_w,
                motor_q_load,
                motor_p_in,
                motor_p_out,
                motor_p_loss,
                motor_eta,
                trans_w_in,
                trans_w_out,
                trans_q_in,
                trans_q_out,
                trans_p_in,
                trans_p_out,
                trans_eta,
                prop_u,
                prop_u_ow,
                prop_n,
                prop_w,
                prop_t,
                prop_q,
                prop_p_in,
                prop_p_out,
                prop_eta,
                prop_j,
                prop_t_ow,
                prop_q_ow,
                prop_p_in_ow,
                prop_p_out_ow,
                prop_eta_ow,
                hull_u,
                hull_t,
                hull_r_t,
                hull_p_in,
                hull_p_out,
                hull_eta,
                oth_p_in,
                oth_i_in,
            ],
            float,
        )

    @classmethod
    def build(cls, params: dict):
        return ct.NonlinearIOSystem(
            cls._update,
            cls._outputs,
            name="solarboat_full",
            inputs=[
                "batt_v",
                "pv_g",
                "mppts_i_out",
                "motor_w",
                "esc_i_in",
                "esc_d",
                "hull_u",
            ],
            states=["batt_v_C_1", "batt_v_C_2", "batt_z", "motor_i"],
            outputs=[
                "batt_i",
                "batt_v_C_1",
                "batt_v_C_2",
                "batt_z",
                "batt_v_oc",
                "batt_p_out",
                "batt_p_in",
                "pv_p_in",
                "pv_p_out",
                "pv_eta",
                "pvs_p_in",
                "pvs_p_out",
                "pvs_i_out",
                "mppt_p_in",
                "mppt_p_out",
                "mppt_v_in",
                "mppt_v_out",
                "mppt_i_in",
                "mppt_i_out",
                "mppt_eta",
                "mppts_p_in",
                "mppts_p_out",
                "mppts_i_out",
                "esc_v_in",
                "esc_v_out",
                "esc_i_in",
                "esc_i_out",
                "esc_p_in",
                "esc_p_out",
                "esc_eta",
                "esc_p_loss",
                "esc_p_M_cond",
                "esc_p_D_cond",
                "esc_p_M_sw",
                "esc_p_D_sw",
                "esc_i_M_RMS",
                "esc_i_D_RMS",
                "esc_i_D_AVG",
                "motor_i",
                "motor_v",
                "motor_w",
                "motor_q_load",
                "motor_p_in",
                "motor_p_out",
                "motor_p_loss",
                "motor_eta",
                "trans_w_in",
                "trans_w_out",
                "trans_q_in",
                "trans_q_out",
                "trans_p_in",
                "trans_p_out",
                "trans_eta",
                "prop_u",
                "prop_u_ow",
                "prop_n",
                "prop_w",
                "prop_t",
                "prop_q",
                "prop_p_in",
                "prop_p_out",
                "prop_eta",
                "prop_j",
                "prop_t_ow",
                "prop_q_ow",
                "prop_p_in_ow",
                "prop_p_out_ow",
                "prop_eta_ow",
                "hull_u",
                "hull_t",
                "hull_r_t",
                "hull_p_in",
                "hull_p_out",
                "hull_eta",
                "oth_p_in",
                "oth_i_in",
            ],
            params=params,
        )
