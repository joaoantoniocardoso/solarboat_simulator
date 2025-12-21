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

        # Required initial conditions
        batt_z_0 = X0["batt_z"]
        batt_v_0 = X0["batt_v"]

        # Optional: initial hull speed
        hull_u_0 = X0.get("hull_u", 0.0)

        # Inputs
        # pv_g_0 = U0[0] if len(U0) > 0 else 0.0
        pilot_d_0 = U0[1] if len(U0) > 1 else 0.0

        # --- Battery RC states ---------------------------------------------
        # Params
        batt_R_0 = params["batt_R_0"]
        batt_k_V_OC_coeffs = params["batt_k_V_OC_coeffs"]
        batt_N_S = params["batt_N_S"]

        # [ASSUMPTION]: At t=0, battery current is 0 A when computing RC initial voltages.
        batt_i_0 = 0.0

        # OCV at initial SoC
        batt_ocv_0 = eval_poly(batt_k_V_OC_coeffs, batt_z_0)

        # v_C1_0 = v_C2_0 = v_C_0 so that terminal voltage matches batt_v_0:
        # batt_v_0 / batt_N_S = batt_ocv_0 - 2*v_C_0 - batt_R_0 * batt_i_0
        v_C_0 = (batt_ocv_0 - batt_R_0 * batt_i_0 - batt_v_0 / batt_N_S) / 2.0

        batt_v_C_1_0 = v_C_0
        batt_v_C_2_0 = v_C_0

        # --- Propulsion states ---------------------------------------------
        # [ASSUMPTION]: Motor starts from rest with zero current, and ESC duty equals pilot input.
        motor_i_0 = 0.0
        motor_w_0 = 0.0
        esc_d_0 = pilot_d_0

        return np.array(
            [
                batt_v_C_1_0,
                batt_v_C_2_0,
                batt_z_0,
                motor_i_0,
                motor_w_0,
                esc_d_0,
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
        # pilot_d = u[1]

        # States:
        batt_v_C_1 = x[0]
        batt_v_C_2 = x[1]
        batt_z = x[2]
        motor_i = x[3]
        # motor_w = x[4]
        esc_d = x[5]
        # hull_u = x[6]

        # Outputs:
        eps = 1e-6
        batt_v_oc = max(eval_poly(batt_k_V_OC_coeffs, batt_z), eps)
        batt_v = (
            (1 / 2)
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
            - 1 / 2 * batt_N_S * batt_R_0 * esc_F_s * esc_Q_rr * esc_V_ds_ov
            - 1 / 2 * batt_N_S * batt_R_0 * esc_F_s * esc_Q_rr
            - 1 / 2 * batt_N_S * batt_R_0 * esc_d * motor_i
            - 1 / 2 * batt_N_S * batt_v_C_1
            - 1 / 2 * batt_N_S * batt_v_C_2
            + (1 / 2) * batt_N_S * batt_v_oc
        )
        mppts_i_out = mppts_n * pv_S * mppt_eta * pv_eta * pv_g / batt_v
        esc_i_in = (
            esc_F_s * esc_Q_rr * (esc_V_ds_ov + 1) * batt_v
            + esc_F_s * (esc_E_off + esc_E_on) * (esc_V_ds_ov + 1)
            + esc_R_ds_on * esc_d * motor_i**2
            - esc_V_F * (esc_d - 1) * motor_i
            - esc_r_D * (esc_d - 1) * motor_i**2
            + esc_d * motor_i * batt_v
        ) / batt_v

        return [batt_v, mppts_i_out, esc_i_in]

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
        esc_tau_rise = params["esc_tau_rise"]
        esc_tau_fall = params["esc_tau_fall"]
        batt_R_1 = params["batt_R_1"]
        batt_C_1 = params["batt_C_1"]
        batt_R_2 = params["batt_R_2"]
        batt_C_2 = params["batt_C_2"]
        batt_Q = params["batt_Q"]
        batt_eta = params["batt_eta"]
        oth_p_in = params["oth_p_in"]

        # Inputs:
        # pv_g = u[0]
        pilot_d = u[1]

        # States:
        batt_v_C_1 = x[0]
        batt_v_C_2 = x[1]
        # batt_z = x[2]
        motor_i = x[3]
        motor_w = x[4]
        esc_d = x[5]
        hull_u = x[6]

        batt_v, mppts_i_out, esc_i_in = cls._common(t, x, u, params)

        # Auxiliars:
        eps = 1e-6
        oth_i_in = oth_p_in / batt_v
        batt_i = esc_i_in - mppts_i_out + oth_i_in
        motor_I_r_load = trans_I_r_in + trans_k**2 * (prop_I_r + trans_I_r_out)
        motor_v = esc_d * batt_v
        prop_n = (1 / 2) * trans_k * motor_w / math.pi
        prop_j = max(0, min((1 - hull_W) * hull_u / (prop_D * (prop_n + eps)), 1.5))
        prop_k_t_ow = max(0, eval_poly(prop_k_T_coeffs, prop_j))
        prop_k_q_ow = max(0, eval_poly(prop_k_Q_coeffs, prop_j))
        prop_t = prop_D**4 * rho_water * prop_k_t_ow * prop_n**2
        prop_q = prop_D**5 * rho_water * prop_k_q_ow * prop_n**2 / prop_eta_R
        hull_r_t = (
            (1 / 2)
            * hull_C_T
            * (hull_S_air * rho_air + hull_S_water * rho_water)
            * hull_u**2
        )
        hull_r = -hull_r_t / (hull_T_ded - 1)
        motor_q_load = trans_k * prop_q / trans_eta

        d_batt_v_C_1 = (batt_R_1 * batt_i - batt_v_C_1) / (batt_C_1 * batt_R_1)
        d_batt_v_C_2 = (batt_R_2 * batt_i - batt_v_C_2) / (batt_C_2 * batt_R_2)
        d_batt_z = -batt_eta * batt_i / batt_Q
        d_motor_i = (-motor_K_V * motor_w - motor_R_A * motor_i + motor_v) / motor_L_A
        d_motor_w = (-motor_B * motor_w + motor_K_Q * motor_i - motor_q_load) / (
            motor_I_r + motor_I_r_load
        )
        d_esc_d = (
            ((-esc_d + pilot_d) / esc_tau_rise)
            if (esc_d < pilot_d)
            else ((-esc_d + pilot_d) / esc_tau_fall)
        )
        d_hull_u = (-hull_r + prop_t) / (hull_M + hull_M_a)

        return [
            d_batt_v_C_1,
            d_batt_v_C_2,
            d_batt_z,
            d_motor_i,
            d_motor_w,
            d_esc_d,
            d_hull_u,
        ]

    @classmethod
    def _outputs(cls, t, x, u, params: dict):
        # States:
        # batt_v_C_1 = x[0]
        # batt_v_C_2 = x[1]
        # batt_z = x[2]
        # motor_i = x[3]
        motor_w = x[4]
        esc_d = x[5]
        hull_u = x[6]

        # Outputs:
        batt_v, mppts_i_out, esc_i_in = cls._common(t, x, u, params)

        return [batt_v, mppts_i_out, motor_w, esc_i_in, esc_d, hull_u]

    @classmethod
    def build(cls, params: dict):
        return ct.NonlinearIOSystem(
            cls._update,
            cls._outputs,
            name="solarboat",
            inputs=["pv_g", "pilot_d"],
            states=[
                "batt_v_C_1",
                "batt_v_C_2",
                "batt_z",
                "motor_i",
                "motor_w",
                "esc_d",
                "hull_u",
            ],
            outputs=["batt_v", "mppts_i_out", "motor_w", "esc_i_in", "esc_d", "hull_u"],
            params=params,
        )


class SolarBoatInternalSignalModel:
    @classmethod
    def initial_state(cls, X0: dict, U0, params: dict):
        if "batt_z" not in X0 or "batt_v" not in X0:
            raise ValueError(
                "X0 must contain 'batt_z' and 'batt_v' for SolarBoat.initial_state."
            )

        # Required initial conditions
        batt_z_0 = X0["batt_z"]
        batt_v_0 = X0["batt_v"]

        # Optional: initial hull speed
        hull_u_0 = X0.get("hull_u", 0.0)

        # Inputs
        # pv_g_0 = U0[0] if len(U0) > 0 else 0.0
        pilot_d_0 = U0[1] if len(U0) > 1 else 0.0

        # --- Battery RC states ---------------------------------------------
        # Params
        batt_R_0 = params["batt_R_0"]
        batt_k_V_OC_coeffs = params["batt_k_V_OC_coeffs"]
        batt_N_S = params["batt_N_S"]

        # [ASSUMPTION]: At t=0, battery current is 0 A when computing RC initial voltages.
        batt_i_0 = 0.0

        # OCV at initial SoC
        batt_ocv_0 = eval_poly(batt_k_V_OC_coeffs, batt_z_0)

        # v_C1_0 = v_C2_0 = v_C_0 so that terminal voltage matches batt_v_0:
        # batt_v_0 / batt_N_S = batt_ocv_0 - 2*v_C_0 - batt_R_0 * batt_i_0
        v_C_0 = (batt_ocv_0 - batt_R_0 * batt_i_0 - batt_v_0 / batt_N_S) / 2.0

        batt_v_C_1_0 = v_C_0
        batt_v_C_2_0 = v_C_0

        # --- Propulsion states ---------------------------------------------
        # [ASSUMPTION]: Motor starts from rest with zero current, and ESC duty equals pilot input.
        motor_i_0 = 0.0
        esc_d_0 = pilot_d_0

        return np.array(
            [
                batt_v_C_1_0,
                batt_v_C_2_0,
                batt_z_0,
                motor_i_0,
                esc_d_0,
                hull_u_0,
            ]
        )

    @classmethod
    def _aux(cls, t, x, u, params: dict):
        # Params
        hull_C_T = params["hull_C_T"]
        hull_S_air = params["hull_S_air"]
        hull_S_water = params["hull_S_water"]
        hull_T_ded = params["hull_T_ded"]
        hull_W = params["hull_W"]
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
        batt_k_V_OC_coeffs = params["batt_k_V_OC_coeffs"]

        # Inputs
        # pv_g = u[0]
        # pilot_d = u[1]
        batt_v = u[2]
        mppts_i_out = u[3]
        motor_w = u[4]
        esc_i_in = u[5]

        # States
        # batt_v_C_1 = x[0]
        # batt_v_C_2 = x[1]
        batt_z = x[2]
        # motor_i = x[3]
        esc_d = x[4]
        hull_u = x[5]

        eps = 1e-6
        batt_v_oc = max(eval_poly(batt_k_V_OC_coeffs, batt_z), eps)
        oth_i_in = oth_p_in / batt_v
        batt_i = esc_i_in - mppts_i_out + oth_i_in
        motor_I_r_load = trans_I_r_in + trans_k**2 * (prop_I_r + trans_I_r_out)
        motor_v = esc_d * batt_v
        prop_n = (1 / 2) * trans_k * motor_w / math.pi
        prop_j = max(0, min((1 - hull_W) * hull_u / (prop_D * (prop_n + eps)), 1.5))
        prop_k_t_ow = max(0, eval_poly(prop_k_T_coeffs, prop_j))
        prop_k_q_ow = max(0, eval_poly(prop_k_Q_coeffs, prop_j))
        prop_t = prop_D**4 * rho_water * prop_k_t_ow * prop_n**2
        prop_q = prop_D**5 * rho_water * prop_k_q_ow * prop_n**2 / prop_eta_R
        hull_r_t = (
            (1 / 2)
            * hull_C_T
            * (hull_S_air * rho_air + hull_S_water * rho_water)
            * hull_u**2
        )
        hull_r = -hull_r_t / (hull_T_ded - 1)
        motor_q_load = trans_k * prop_q / trans_eta

        return np.array(
            [
                batt_v_oc,
                oth_i_in,
                batt_i,
                motor_I_r_load,
                motor_v,
                prop_n,
                prop_j,
                prop_k_t_ow,
                prop_k_q_ow,
                prop_t,
                prop_q,
                hull_r_t,
                hull_r,
                motor_q_load,
            ],
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
        esc_tau_fall = params["esc_tau_fall"]
        esc_tau_rise = params["esc_tau_rise"]
        hull_M = params["hull_M"]
        hull_M_a = params["hull_M_a"]
        # motor_B = params["motor_B"]
        # motor_I_r = params["motor_I_r"]
        # motor_K_Q = params["motor_K_Q"]
        motor_K_V = params["motor_K_V"]
        motor_L_A = params["motor_L_A"]
        motor_R_A = params["motor_R_A"]

        # Inputs
        # pv_g = u[0]
        pilot_d = u[1]
        # batt_v = u[2]
        # mppts_i_out = u[3]
        motor_w = u[4]
        # esc_i_in = u[5]

        # States
        batt_v_C_1 = x[0]
        batt_v_C_2 = x[1]
        # batt_z = x[2]
        motor_i = x[3]
        esc_d = x[4]
        # hull_u = x[5]

        # Auxiliars:
        A = cls._aux(t, x, u, params)
        # batt_v_oc = A[0]
        # oth_i_in = A[1]
        batt_i = A[2]
        # motor_I_r_load = A[3]
        motor_v = A[4]
        # prop_n = A[5]
        # prop_j = A[6]
        # prop_k_t_ow = A[7]
        # prop_k_q_ow = A[8]
        prop_t = A[9]
        # prop_q = A[10]
        # hull_r_t = A[11]
        hull_r = A[12]
        # motor_q_load = A[13]

        d_batt_v_C_1 = (batt_R_1 * batt_i - batt_v_C_1) / (batt_C_1 * batt_R_1)
        d_batt_v_C_2 = (batt_R_2 * batt_i - batt_v_C_2) / (batt_C_2 * batt_R_2)
        d_batt_z = -batt_eta * batt_i / batt_Q
        d_motor_i = (-motor_K_V * motor_w - motor_R_A * motor_i + motor_v) / motor_L_A
        d_esc_d = (
            ((-esc_d + pilot_d) / esc_tau_rise)
            if (esc_d < pilot_d)
            else ((-esc_d + pilot_d) / esc_tau_fall)
        )
        d_hull_u = (-hull_r + prop_t) / (hull_M + hull_M_a)

        return np.array(
            [
                d_batt_v_C_1,
                d_batt_v_C_2,
                d_batt_z,
                d_motor_i,
                d_esc_d,
                d_hull_u,
            ],
            float,
        )

    @classmethod
    def _outputs(cls, t, x, u, params: dict):
        # Params
        batt_eta = params["batt_eta"]
        hull_T_ded = params["hull_T_ded"]
        hull_W = params["hull_W"]
        mppt_eta = params["mppt_eta"]
        mppts_n = params["mppts_n"]
        oth_p_in = params["oth_p_in"]
        pv_eta = params["pv_eta"]
        pv_S = params["pv_S"]
        trans_eta = params["trans_eta"]
        trans_k = params["trans_k"]

        # Inputs
        pv_g = u[0]
        # pilot_d = u[1]
        batt_v = u[2]
        mppts_i_out = u[3]
        motor_w = u[4]
        esc_i_in = u[5]

        # States
        # batt_v_C_1 = x[0]
        # batt_v_C_2 = x[1]
        batt_z = x[2]
        motor_i = x[3]
        esc_d = x[4]
        hull_u = x[5]

        # Auxiliars:
        aux = cls._aux(t, x, u, params)
        batt_v_oc = aux[0]
        oth_i_in = aux[1]
        batt_i = aux[2]
        # motor_I_r_load = aux[3]
        motor_v = aux[4]
        # prop_n = aux[5]
        prop_j = aux[6]
        prop_k_t_ow = aux[7]
        prop_k_q_ow = aux[8]
        prop_t = aux[9]
        prop_q = aux[10]
        hull_r_t = aux[11]
        hull_r = aux[12]
        motor_q_load = aux[13]

        # ESC
        esc_i_out = motor_i
        esc_v_out = motor_v
        esc_v_in = batt_v
        esc_p_in = esc_i_in * esc_v_in
        esc_p_out = esc_i_out * esc_v_out
        esc_eta = np.ma.masked_invalid(esc_p_out / esc_p_in)

        # Motor
        motor_p_in = motor_v * motor_i
        motor_p_out = motor_w * motor_q_load
        motor_eta = np.ma.masked_invalid(motor_p_out / motor_p_in)

        # Prop
        prop_w = trans_k * motor_w
        prop_p_in = prop_w * prop_q
        prop_u = hull_u * (1 - hull_W)
        prop_p_out = prop_t * prop_u
        prop_eta = np.ma.masked_invalid(prop_p_out / prop_p_in)

        # Trans
        trans_w_in = motor_w
        trans_w_out = prop_w
        trans_q_in = motor_q_load
        trans_q_out = prop_q
        trans_p_in = trans_w_in * trans_q_in
        trans_p_out = trans_w_out * trans_q_out
        trans_eta = np.ma.masked_invalid(trans_p_out / trans_p_in)

        # Hull
        hull_r_t = hull_r * (1 - hull_T_ded)
        hull_p_out = hull_r_t * hull_u
        hull_t = prop_t
        hull_p_in = hull_t * hull_u
        hull_eta = np.ma.masked_invalid(hull_p_out / hull_p_in)

        # Battery
        batt_p = batt_i * batt_v

        # PVs
        pvs_p_in = pv_g * pv_S * mppts_n
        pvs_p_out = pvs_p_in * pv_eta
        pvs_eta = np.ma.masked_invalid(pvs_p_out / pvs_p_in)

        # MPPTs
        mppts_v_out = batt_v
        mppts_p_in = pvs_p_out
        mppts_p_out = mppt_eta * mppts_p_in
        mppts_eta = np.ma.masked_invalid(mppts_p_out / mppts_p_in)

        return np.array(
            [
                # PVs
                pv_g,
                pvs_p_in,
                pvs_p_out,
                pvs_eta,
                # MPPTs
                mppts_i_out,
                mppts_v_out,
                mppts_p_in,
                mppts_p_out,
                mppts_eta,
                # Battery
                batt_v,
                batt_v_oc,
                batt_i,
                batt_p,
                batt_z,
                batt_eta,
                # ESC
                esc_d,
                esc_v_in,
                esc_v_out,
                esc_i_in,
                esc_i_out,
                esc_p_in,
                esc_p_out,
                esc_eta,
                # Motor
                motor_v,
                motor_i,
                motor_w,
                motor_q_load,
                motor_p_in,
                motor_p_out,
                motor_eta,
                # Transmission
                trans_w_in,
                trans_w_out,
                trans_q_in,
                trans_q_out,
                trans_p_in,
                trans_p_out,
                trans_eta,
                # Propeller
                prop_w,
                prop_q,
                prop_t,
                prop_u,
                prop_k_t_ow,
                prop_k_q_ow,
                prop_j,
                prop_p_in,
                prop_p_out,
                prop_eta,
                # Hull
                hull_t,
                hull_r,
                hull_r_t,
                hull_u,
                hull_p_in,
                hull_p_out,
                hull_eta,
                # Others
                oth_i_in,
                oth_p_in,
            ],
            float,
        )

    @classmethod
    def build(cls, params: dict):
        return ct.NonlinearIOSystem(
            cls._update,
            cls._outputs,
            name="solarboat_full",
            inputs=["pv_g", "pilot_d", "batt_v", "mppts_i_out", "motor_w", "esc_i_in"],
            states=["batt_v_C_1", "batt_v_C_2", "batt_z", "motor_i", "esc_d", "hull_u"],
            outputs=[
                # PVs
                "pv_g",
                # *[f'pv_{i}_{s}' for i in range(1, solar_boat_params['mppts_n']+1) for s in ['g', 'i', 'v', 'p_in', 'p_out', 'eta']]
                "pvs_p_in",
                "pvs_p_out",
                "pvs_eta",
                # MPPTs
                # *[f'mppt_{i}_{s}' for i in range(1, solar_boat_params['mppts_n']+1) for s in ['d', 'i_in', 'i_out', 'v_in', 'v_out', 'p_in', 'p_out', 'eta']]
                "mppts_i_out",
                "mppts_v_out",
                "mppts_p_in",
                "mppts_p_out",
                "mppts_eta",
                # Battery
                "batt_v",
                "batt_v_oc",
                "batt_i",
                "batt_p",
                "batt_z",
                "batt_eta",
                # ESC
                "esc_d",
                "esc_v_in",
                "esc_v_out",
                "esc_i_in",
                "esc_i_out",
                "esc_p_in",
                "esc_p_out",
                "esc_eta",
                # Motor
                "motor_v",
                "motor_i",
                "motor_w",
                "motor_q_load",
                "motor_p_in",
                "motor_p_out",
                "motor_eta",
                # Transmission
                "trans_w_in",
                "trans_w_out",
                "trans_q_in",
                "trans_q_out",
                "trans_p_in",
                "trans_p_out",
                "trans_eta",
                # Propeller
                "prop_w",
                "prop_q",
                "prop_t",
                "prop_u",
                "prop_k_t_ow",
                "prop_k_q_ow",
                "prop_j",
                "prop_p_in",
                "prop_p_out",
                "prop_eta",
                # Hull
                "hull_t",
                "hull_r",
                "hull_r_t",
                "hull_u",
                "hull_p_in",
                "hull_p_out",
                "hull_eta",
                # Others
                "oth_i_in",
                "oth_p_in",
            ],
            params=params,
        )
