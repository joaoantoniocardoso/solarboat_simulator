import math

import control as ct
import numpy as np

from utils.models import eval_poly


def _safe_div(num: float, den: float, eps: float = 1e-9) -> float:
    if abs(den) <= eps:
        return 0.0
    return num / den


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(x, hi))


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

        oth_i_in = oth_p_in / max(batt_v, eps)
        batt_i = esc_i_in - mppts_i_out + oth_i_in

        return batt_v, batt_i, mppts_i_out, esc_i_in

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

        batt_v, batt_i, mppts_i_out, esc_i_in = cls._common(t, x, u, params)

        # Auxiliars:
        eps = 1e-9

        prop_n = 0.5 * trans_k * motor_w / math.pi
        prop_j = (1.0 - hull_W) * hull_u / (prop_D * (prop_n + eps))
        prop_j = _clamp(prop_j, 0.0, float(params.get("prop_J_max", 1.5)))
        prop_k_t_ow = max(0.0, eval_poly(prop_k_T_coeffs, prop_j))
        prop_k_q_ow = max(0.0, eval_poly(prop_k_Q_coeffs, prop_j))
        prop_t = prop_D**4 * rho_water * prop_k_t_ow * prop_n**2
        prop_q = prop_D**5 * rho_water * prop_k_q_ow * prop_n**2 / max(prop_eta_R, eps)
        prop_t_e = (1.0 - hull_T_ded) * prop_t
        hull_r = (
            0.5
            * hull_C_T
            * (hull_S_air * rho_air + hull_S_water * rho_water)
            * hull_u**2
        )
        d_hull_u = (prop_t_e - hull_r) / (hull_M + hull_M_a)

        motor_v = esc_d * batt_v
        motor_q_load = trans_k * prop_q / max(trans_eta, eps)
        motor_I_r_load = trans_I_r_in + trans_k**2 * (prop_I_r + trans_I_r_out)

        d_batt_v_C_1 = (batt_R_1 * batt_i - batt_v_C_1) / (batt_C_1 * batt_R_1)
        d_batt_v_C_2 = (batt_R_2 * batt_i - batt_v_C_2) / (batt_C_2 * batt_R_2)
        d_batt_z = -batt_eta * batt_i / batt_Q
        d_motor_i = (-motor_K_V * motor_w - motor_R_A * motor_i + motor_v) / motor_L_A
        d_motor_w = (-motor_B * motor_w + motor_K_Q * motor_i - motor_q_load) / (
            motor_I_r + motor_I_r_load
        )

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

        # Outputs:
        batt_v, batt_i, mppts_i_out, esc_i_in = cls._common(t, x, u, params)
        return [batt_v, batt_i, mppts_i_out, motor_w, esc_i_in, hull_u]

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
            outputs=[
                "batt_v",
                "batt_i",
                "mppts_i_out",
                "motor_w",
                "esc_i_in",
                "hull_u",
            ],
            params=params,
        )


class SolarBoatFull:
    @classmethod
    def initial_state(cls, X0: dict, U0, params: dict):
        return SolarBoat.initial_state(X0=X0, U0=U0, params=params)

    @classmethod
    def _update(cls, t, x, u, params: dict):
        return SolarBoat._update(t, x, u, params)

    @classmethod
    def _outputs(cls, t, x, u, params: dict):
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
        batt_k_V_OC_coeffs = params["batt_k_V_OC_coeffs"]
        pv_S = params["pv_S"]
        pv_eta = params["pv_eta"]
        mppts_n = int(params["mppts_n"])
        pvs_n = params.get("pvs_n", mppts_n)
        mppt_eta = params["mppt_eta"]
        oth_p_in = params["oth_p_in"]

        pv_g = float(u[0])
        esc_d = float(u[1])

        batt_v_C_1 = float(x[0])
        batt_v_C_2 = float(x[1])
        batt_z = float(x[2])
        motor_i = float(x[3])
        motor_w = float(x[4])
        hull_u = float(x[5])

        eps = 1e-9
        esc_d = _clamp(esc_d, 0.0, 1.0)
        hull_u = max(hull_u, 0.0)

        batt_v, batt_i, mppts_i_out, esc_i_in = SolarBoat._common(t, x, u, params)
        batt_v_safe = max(batt_v, eps)

        oth_i_in = oth_p_in / batt_v_safe
        batt_v_oc = eval_poly(batt_k_V_OC_coeffs, batt_z)
        batt_p_out = batt_i * batt_v
        batt_p_in = -batt_i * batt_v

        pv_p_in = pv_S * pv_g
        pv_p_out = pv_eta * pv_p_in
        pvs_p_in = float(pvs_n) * pv_p_in
        pvs_p_out = float(pvs_n) * pv_p_out
        pvs_i_out = _safe_div(pvs_p_out, batt_v_safe, eps)

        mppts_p_in = pvs_p_out
        mppts_p_out = float(mppt_eta) * mppts_p_in
        mppts_eta = _safe_div(mppts_p_out, mppts_p_in, eps)
        mppts_i_out_y = mppts_i_out

        mppt_p_in = _safe_div(mppts_p_in, mppts_n, eps)
        mppt_p_out = _safe_div(mppts_p_out, mppts_n, eps)
        mppt_v_in = batt_v
        mppt_v_out = batt_v
        mppt_i_in = _safe_div(mppt_p_in, batt_v_safe, eps)
        mppt_i_out = _safe_div(mppt_p_out, batt_v_safe, eps)

        mppt_rows: list[tuple[float, float, float, float, float, float]] = []
        for _ in range(mppts_n):
            i_in = mppt_i_in
            v_in = mppt_v_in
            p_in = mppt_p_in
            v_out = mppt_v_out
            p_out = mppt_p_out
            i_out = _safe_div(p_out, batt_v_safe, eps)
            mppt_rows.append((i_in, v_in, p_in, v_out, i_out, p_out))

        esc_v_in = batt_v
        esc_v_out = esc_d * esc_v_in
        esc_i_out = motor_i
        esc_p_in = esc_i_in * esc_v_in
        esc_p_out = esc_i_out * esc_v_out

        esc_F_s = params["esc_F_s"]
        esc_V_ds_ov = params["esc_V_ds_ov"]
        esc_R_ds_on = params["esc_R_ds_on"]
        esc_E_on = params["esc_E_on"]
        esc_E_off = params["esc_E_off"]
        esc_V_F = params["esc_V_F"]
        esc_r_D = params["esc_r_D"]
        esc_Q_rr = params["esc_Q_rr"]

        esc_i_M_RMS = math.sqrt(max(esc_d, 0.0)) * esc_i_out
        esc_i_D_RMS = math.sqrt(max(1.0 - esc_d, 0.0)) * esc_i_out
        esc_i_D_AVG = (1.0 - esc_d) * esc_i_out

        esc_v_ds_sw = (esc_V_ds_ov + 1.0) * esc_v_in
        esc_p_M_cond = esc_R_ds_on * esc_i_M_RMS**2
        esc_p_D_cond = esc_V_F * esc_i_D_AVG + esc_r_D * esc_i_D_RMS**2
        esc_p_M_sw = esc_F_s * (esc_E_off + esc_E_on) * esc_v_ds_sw / batt_v_safe
        esc_p_D_sw = esc_F_s * esc_Q_rr * esc_v_ds_sw
        esc_p_loss = esc_p_D_cond + esc_p_D_sw + esc_p_M_cond + esc_p_M_sw

        esc_eta = _safe_div(esc_p_out, esc_p_in, eps)

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
        prop_eta_ow = _safe_div(
            0.5 * prop_j_ow * prop_k_t_ow, math.pi * prop_k_q_ow, eps
        )

        prop_p_in = prop_w * prop_q
        prop_p_out = prop_t * prop_u
        prop_eta = _safe_div(prop_p_out, prop_p_in, eps)

        motor_q_load = trans_k * prop_q / max(trans_eta, eps)
        motor_p_in = motor_i * motor_v
        motor_p_out = motor_w * motor_q_load
        motor_p_loss = motor_B * motor_w**2 + motor_R_A * motor_i**2
        motor_eta = _safe_div(motor_p_out, motor_p_in, eps)

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
        hull_eta = _safe_div(hull_p_out, hull_p_in, eps)

        return np.array(
            [
                batt_v,
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
                pv_g,
                *[v for r in mppt_rows for v in r],
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
                prop_k_t_ow,
                prop_k_q_ow,
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
            inputs=["pv_g", "esc_d"],
            states=SolarBoat.build(params).state_labels,
            outputs=SolarBoatInternalSignalModel.build(params).output_labels,
            params=params,
        )


class SolarBoatInternalSignalModel:
    @classmethod
    def _mppt_input_labels(cls, params: dict) -> list[str]:
        mppts_n = int(params["mppts_n"])
        labels: list[str] = []
        for i in range(1, mppts_n + 1):
            labels += [f"mppt{i}_i_in", f"mppt{i}_v_in"]
        return labels

    @classmethod
    def _mppt_output_labels(cls, params: dict) -> list[str]:
        mppts_n = int(params["mppts_n"])
        labels: list[str] = []
        for i in range(1, mppts_n + 1):
            labels += [
                f"mppt{i}_i_in",
                f"mppt{i}_v_in",
                f"mppt{i}_p_in",
                f"mppt{i}_v_out",
                f"mppt{i}_i_out",
                f"mppt{i}_p_out",
            ]
        return labels

    @classmethod
    def _esc_i_in(
        cls, batt_v: float, motor_i: float, esc_d: float, params: dict
    ) -> float:
        eps = 1e-9
        batt_v_safe = max(float(batt_v), eps)
        esc_d = _clamp(float(esc_d), 0.0, 1.0)

        esc_F_s = params["esc_F_s"]
        esc_V_ds_ov = params["esc_V_ds_ov"]
        esc_R_ds_on = params["esc_R_ds_on"]
        esc_E_on = params["esc_E_on"]
        esc_E_off = params["esc_E_off"]
        esc_V_F = params["esc_V_F"]
        esc_r_D = params["esc_r_D"]
        esc_Q_rr = params["esc_Q_rr"]

        batt_v_term = float(batt_v)
        motor_i_term = float(motor_i)

        num = (
            esc_F_s * esc_Q_rr * (esc_V_ds_ov + 1.0) * batt_v_term
            + esc_F_s * (esc_E_off + esc_E_on) * (esc_V_ds_ov + 1.0)
            + esc_R_ds_on * esc_d * motor_i_term**2
            - esc_V_F * (esc_d - 1.0) * motor_i_term
            - esc_r_D * (esc_d - 1.0) * motor_i_term**2
            + esc_d * motor_i_term * batt_v_term
        )
        return num / batt_v_safe

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

        esc_d_0 = _clamp(U0[1] if len(U0) > 1 else 0.0, 0.0, 1.0)

        mppts_n = int(params["mppts_n"])
        mppt_inputs = U0[3 : 3 + 2 * mppts_n]

        mppts_p_in_0 = 0.0
        mppts_i_out_0 = 0.0
        for i in range(mppts_n):
            i_in = mppt_inputs[2 * i] if len(mppt_inputs) > 2 * i else 0.0
            v_in = mppt_inputs[2 * i + 1] if len(mppt_inputs) > 2 * i + 1 else 0.0
            p_in = float(i_in) * float(v_in)
            p_out = float(params["mppt_eta"]) * p_in
            mppts_p_in_0 += p_in
            mppts_i_out_0 += _safe_div(p_out, max(batt_v_0, eps), eps)

        motor_w_0 = X0.get("motor_w", 0.0)
        motor_i_0 = X0.get("motor_i", 0.0)
        esc_i_in_0 = cls._esc_i_in(batt_v_0, motor_i_0, esc_d_0, params)

        oth_i_in_0 = oth_p_in / max(batt_v_0, eps)
        batt_i_0 = esc_i_in_0 - mppts_i_out_0 + oth_i_in_0

        batt_ocv_0 = eval_poly(batt_k_V_OC_coeffs, batt_z_0)
        v_C_0 = (batt_ocv_0 - batt_R_0 * batt_i_0 - batt_v_0 / batt_N_S) / 2.0

        batt_v_C_1_0 = v_C_0
        batt_v_C_2_0 = v_C_0
        motor_i_0 = X0.get("motor_i", 0.0)

        hull_u_0 = X0.get("hull_u", 0.0)
        return np.array(
            [batt_v_C_1_0, batt_v_C_2_0, batt_z_0, motor_i_0, motor_w_0, hull_u_0],
            float,
        )

    @classmethod
    def _update(cls, t, x, u, params: dict):
        # Params
        batt_C_1 = params["batt_C_1"]
        batt_C_2 = params["batt_C_2"]
        batt_eta = params["batt_eta"]
        batt_Q = params["batt_Q"]
        batt_R_1 = params["batt_R_1"]
        batt_R_2 = params["batt_R_2"]
        hull_C_T = params["hull_C_T"]
        hull_M = params["hull_M"]
        hull_M_a = params["hull_M_a"]
        hull_S_air = params["hull_S_air"]
        hull_S_water = params["hull_S_water"]
        hull_T_ded = params["hull_T_ded"]
        hull_W = params["hull_W"]
        motor_B = params["motor_B"]
        motor_I_r = params["motor_I_r"]
        motor_K_Q = params["motor_K_Q"]
        motor_K_V = params["motor_K_V"]
        motor_L_A = params["motor_L_A"]
        motor_R_A = params["motor_R_A"]
        mppt_eta = params["mppt_eta"]
        mppts_n = int(params["mppts_n"])
        oth_p_in = params["oth_p_in"]
        prop_D = params["prop_D"]
        prop_eta_R = params["prop_eta_R"]
        prop_I_r = params["prop_I_r"]
        prop_k_Q_coeffs = params["prop_k_Q_coeffs"]
        prop_k_T_coeffs = params["prop_k_T_coeffs"]
        rho_air = params["rho_air"]
        rho_water = params["rho_water"]
        trans_eta = params["trans_eta"]
        trans_I_r_in = params["trans_I_r_in"]
        trans_I_r_out = params["trans_I_r_out"]
        trans_k = params["trans_k"]

        # Inputs
        batt_v = u[0]
        esc_d = u[1]
        mppt_inputs = u[2 : 3 + 2 * int(mppts_n)]

        # States
        batt_v_C_1 = x[0]
        batt_v_C_2 = x[1]
        batt_z = x[2]
        motor_i = x[3]
        motor_w = x[4]
        hull_u = x[5]

        # Auxiliars:
        eps = 1e-9
        batt_v_safe = max(batt_v, eps)

        mppts_p_in = 0.0
        mppts_i_out = 0.0
        for i in range(mppts_n):
            i_in = float(mppt_inputs[2 * i]) if len(mppt_inputs) > 2 * i else 0.0
            v_in = (
                float(mppt_inputs[2 * i + 1]) if len(mppt_inputs) > 2 * i + 1 else 0.0
            )
            p_in = i_in * v_in
            p_out = float(mppt_eta) * p_in
            mppts_p_in += p_in
            mppts_i_out += _safe_div(p_out, batt_v_safe, eps)

        esc_i_in = cls._esc_i_in(batt_v, motor_i, esc_d, params)
        oth_i_in = oth_p_in / batt_v_safe
        batt_i = esc_i_in - mppts_i_out + oth_i_in

        prop_n = 0.5 * trans_k * motor_w / math.pi
        prop_j = (1.0 - hull_W) * hull_u / (prop_D * (prop_n + eps))
        prop_j = _clamp(prop_j, 0.0, float(params.get("prop_J_max", 1.5)))
        prop_k_t_ow = max(0.0, eval_poly(prop_k_T_coeffs, prop_j))
        prop_k_q_ow = max(0.0, eval_poly(prop_k_Q_coeffs, prop_j))
        prop_t = prop_D**4 * rho_water * prop_k_t_ow * prop_n**2
        prop_q = prop_D**5 * rho_water * prop_k_q_ow * prop_n**2 / max(prop_eta_R, eps)
        prop_t_e = (1.0 - hull_T_ded) * prop_t
        hull_r = (
            0.5
            * hull_C_T
            * (hull_S_air * rho_air + hull_S_water * rho_water)
            * hull_u**2
        )
        d_hull_u = (prop_t_e - hull_r) / (hull_M + hull_M_a)

        motor_v = esc_d * batt_v
        motor_q_load = trans_k * prop_q / max(trans_eta, eps)
        motor_I_r_load = trans_I_r_in + trans_k**2 * (prop_I_r + trans_I_r_out)

        d_batt_v_C_1 = (batt_R_1 * batt_i - batt_v_C_1) / (batt_C_1 * batt_R_1)
        d_batt_v_C_2 = (batt_R_2 * batt_i - batt_v_C_2) / (batt_C_2 * batt_R_2)
        d_batt_z = -batt_eta * batt_i / batt_Q
        d_motor_i = (-motor_K_V * motor_w - motor_R_A * motor_i + motor_v) / motor_L_A
        d_motor_w = (-motor_B * motor_w + motor_K_Q * motor_i - motor_q_load) / (
            motor_I_r + motor_I_r_load
        )

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
        mppts_n = params["mppts_n"]
        pvs_n = params.get("pvs_n", mppts_n)  # Consider 1:1 mapping
        mppt_eta = params["mppt_eta"]
        oth_p_in = params["oth_p_in"]

        # Inputs
        batt_v = u[0]
        esc_d = u[1]
        mppt_inputs = u[2 : 3 + 2 * int(mppts_n)]

        # States
        batt_v_C_1 = x[0]
        batt_v_C_2 = x[1]
        batt_z = x[2]
        motor_i = x[3]
        motor_w = x[4]
        hull_u = x[5]

        # Auxiliars:
        eps = 1e-9
        batt_v_safe = max(batt_v, eps)
        esc_d = _clamp(esc_d, 0.0, 1.0)

        # MPPTs (aggregate)
        mppts_p_in = 0.0
        mppts_p_out = 0.0
        mppts_i_out = 0.0
        mppt_rows: list[tuple[float, float, float, float, float, float]] = []

        for i in range(int(mppts_n)):
            i_in = float(mppt_inputs[2 * i]) if len(mppt_inputs) > 2 * i else 0.0
            v_in = (
                float(mppt_inputs[2 * i + 1])
                if len(mppt_inputs) > 2 * i + 1
                else batt_v
            )
            p_in = i_in * v_in
            p_out = float(mppt_eta) * p_in
            v_out = batt_v
            i_out = _safe_div(p_out, batt_v_safe, eps)
            mppt_rows.append((i_in, v_in, p_in, v_out, i_out, p_out))
            mppts_p_in += p_in
            mppts_p_out += p_out
            mppts_i_out += i_out

        total_mppt_i_in = sum(r[0] for r in mppt_rows)
        total_mppt_v_in = sum(r[1] for r in mppt_rows)

        mppt_v_out = batt_v
        mppt_i_in = _safe_div(total_mppt_i_in, mppts_n, eps)
        mppt_v_in = _safe_div(total_mppt_v_in, mppts_n, eps)
        mppt_p_in = _safe_div(mppts_p_in, mppts_n, eps)
        mppt_p_out = _safe_div(mppts_p_out, mppts_n, eps)
        mppt_i_out = _safe_div(mppts_i_out, mppts_n, eps)

        mppts_eta = _safe_div(mppts_p_out, mppts_p_in, eps)

        pvs_p_out = mppts_p_in
        pvs_p_in = _safe_div(pvs_p_out, max(float(pv_eta), eps), eps)
        pvs_eta = _safe_div(pvs_p_out, pvs_p_in, eps)
        pvs_i_out = _safe_div(pvs_p_out, batt_v_safe, eps)

        pv_g = _safe_div(pvs_p_in, float(pvs_n) * float(pv_S), eps)
        pv_p_in = pv_S * pv_g
        pv_p_out = pv_eta * pv_p_in

        esc_i_in = cls._esc_i_in(batt_v, motor_i, esc_d, params)
        oth_i_in = oth_p_in / batt_v_safe
        batt_i = esc_i_in - mppts_i_out + oth_i_in

        batt_v_oc = eval_poly(batt_k_V_OC_coeffs, batt_z)
        batt_p_out = batt_i * batt_v
        batt_p_in = -batt_i * batt_v

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

        esc_eta = _safe_div(esc_p_out, esc_p_in, eps)

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
        prop_eta_ow = _safe_div(
            0.5 * prop_j_ow * prop_k_t_ow, math.pi * prop_k_q_ow, eps
        )

        prop_p_in = prop_w * prop_q
        prop_p_out = prop_t * prop_u
        prop_eta = _safe_div(prop_p_out, prop_p_in, eps)

        motor_q_load = trans_k * prop_q / max(trans_eta, eps)
        motor_p_in = motor_i * motor_v
        motor_p_out = motor_w * motor_q_load
        motor_p_loss = motor_B * motor_w**2 + motor_R_A * motor_i**2
        motor_eta = _safe_div(motor_p_out, motor_p_in, eps)

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
        hull_eta = _safe_div(hull_p_out, hull_p_in, eps)

        return np.array(
            [
                batt_v,
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
                pv_g,
                *[v for r in mppt_rows for v in r],
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
                prop_k_t_ow,
                prop_k_q_ow,
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
            name="solarboat_internal_signals",
            inputs=["batt_v", "esc_d", *cls._mppt_input_labels(params)],
            states=[
                "batt_v_C_1",
                "batt_v_C_2",
                "batt_z",
                "motor_i",
                "motor_w",
                "hull_u",
            ],
            outputs=[
                "batt_v",
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
                "pv_g",
                *cls._mppt_output_labels(params),
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
                "prop_k_t_ow",
                "prop_k_q_ow",
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
