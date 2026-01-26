import math

import control as ct
import numpy as np
from scipy.optimize import fsolve
from scipy.special import expit  # noqa: F401  # kept for parity with notebook imports

from utils.models import eval_poly


class Propulsion:
    @classmethod
    def _aux(cls, t, x, u, params: dict):
        # Params
        rho_water = params["rho_water"]
        prop_D = params["prop_D"]
        prop_I_r = params["prop_I_r"]
        prop_k_T_coeffs = params["prop_k_T_coeffs"]
        prop_k_Q_coeffs = params["prop_k_Q_coeffs"]
        rho_air = params["rho_air"]
        hull_S_water = params["hull_S_water"]
        hull_S_air = params["hull_S_air"]
        hull_C_T = params["hull_C_T"]
        prop_eta_R = params["prop_eta_R"]
        hull_W = params["hull_W"]
        hull_T_ded = params["hull_T_ded"]
        trans_k = params["trans_k"]
        trans_I_r_out = params["trans_I_r_out"]
        trans_I_r_in = params["trans_I_r_in"]
        trans_eta = params["trans_eta"]
        prop_J_max = params.get("prop_J_max", 1.5)
        prop_j_eps = params.get("prop_j_eps", 1e-3)

        # Inputs
        batt_v = u[0]  # Battery voltage [V]
        esc_d = u[1]  # ESC duty-cycle [unitless]

        # States
        # motor_i = x[0]  # Motor current [A]
        motor_w = x[1]  # Motor angular speed [rad/s]
        hull_u = x[2]  # Hull speed [m/s]

        motor_I_r_load = trans_I_r_in + trans_k**2 * (prop_I_r + trans_I_r_out)
        motor_v = esc_d * batt_v
        prop_n = (1 / 2) * trans_k * motor_w / math.pi

        # Advance ratio for open-water curves:
        # J = V_A/(nD) with V_A = V(1-w) (Birk 2019 Eq. 32.4, 41.7).
        prop_j = max(
            0, min((1 - hull_W) * hull_u / (prop_D * prop_n + prop_j_eps), prop_J_max)
        )

        # Open-water coefficients evaluated at J (Wageningen curves / polynomials).
        prop_k_t_ow = max(0, eval_poly(prop_k_T_coeffs, prop_j))
        prop_k_q_ow = max(0, eval_poly(prop_k_Q_coeffs, prop_j))

        # [ASSUMPTION] Open-water thrust approximation (thrust identity method):
        # T ≈ T0(V_A, n) = ρ n^2 D^4 K_T(J).
        prop_t = prop_D**4 * rho_water * prop_k_t_ow * prop_n**2

        # Relative rotative efficiency correction on torque/power:
        # η_R = η_B/η_O (Birk 2019 Eq. 32.11) ⇒ Q ≈ Q0/η_R.
        prop_q = prop_D**5 * rho_water * prop_k_q_ow * prop_n**2 / prop_eta_R

        # Bare-hull (towed) resistance model R_T(V).
        hull_r_t = (
            (1 / 2)
            * hull_C_T
            * (hull_S_air * rho_air + hull_S_water * rho_water)
            * hull_u**2
        )

        # Effective thrust available to overcome towed resistance:
        # Birk (2019), Eq. (32.2): T(1 - t) = R_T.
        prop_t_e = (1.0 - hull_T_ded) * prop_t

        motor_q_load = trans_k * prop_q / trans_eta

        return (
            motor_I_r_load,
            motor_v,
            motor_q_load,
            prop_j,
            prop_k_q_ow,
            prop_k_t_ow,
            prop_q,
            prop_t,
            prop_t_e,
            hull_u,
            hull_r_t,
        )

    @classmethod
    def initial_state(cls, X0: dict, U0, params: dict):
        # X0 may include known initial states; here we solve for all
        X_guess = [50, 1000, 0]  # [motor_i, motor_w, hull_u]

        motor_B = params["motor_B"]
        motor_K_Q = params["motor_K_Q"]
        motor_K_V = params["motor_K_V"]
        motor_R_A = params["motor_R_A"]

        def equations(X):
            motor_i_0, motor_w_0, hull_u_0 = X

            temp_x = np.array([motor_i_0, motor_w_0, hull_u_0])
            temp_u = U0
            y = cls._outputs(0, temp_x, temp_u, params)
            motor_v_0 = y[0]  # Initial Motor voltage [V]
            motor_q_load_0 = y[3]  # Initial Motor torque [Nm]

            motor_i_0_calc = (motor_B * motor_v_0 + motor_K_V * motor_q_load_0) / (
                motor_B * motor_R_A + motor_K_Q * motor_K_V
            )
            motor_w_0_calc = (motor_K_Q * motor_v_0 - motor_R_A * motor_q_load_0) / (
                motor_B * motor_R_A + motor_K_Q * motor_K_V
            )
            hull_u_0_calc = 0.0  # assume start from rest

            return [
                motor_i_0 - motor_i_0_calc,
                motor_w_0 - motor_w_0_calc,
                hull_u_0 - hull_u_0_calc,
            ]

        solution = fsolve(equations, X_guess, xtol=1e-10)
        motor_i_0, motor_w_0, hull_u_0 = solution
        return np.array([motor_i_0, motor_w_0, hull_u_0])

    @classmethod
    def _update(cls, t, x, u, params: dict):
        hull_M = params["hull_M"]
        hull_M_a = params["hull_M_a"]
        motor_R_A = params["motor_R_A"]
        motor_L_A = params["motor_L_A"]
        motor_K_Q = params["motor_K_Q"]
        motor_K_V = params["motor_K_V"]
        motor_I_r = params["motor_I_r"]
        motor_B = params["motor_B"]

        motor_i = x[0]  # Motor current [A]
        motor_w = x[1]  # Motor angular speed [rad/s]
        # hull_u = x[2]  # Hull speed [m/s]

        A = cls._aux(t, x, u, params)
        motor_I_r_load = A[0]
        motor_v = A[1]
        motor_q_load = A[2]
        prop_t_e = A[8]
        hull_r_t = A[10]

        d_motor_i = (-motor_K_V * motor_w - motor_R_A * motor_i + motor_v) / motor_L_A
        d_motor_w = (-motor_B * motor_w + motor_K_Q * motor_i - motor_q_load) / (
            motor_I_r + motor_I_r_load
        )

        # Surge dynamics: (M + M_a) dV/dt = T_E - R_T
        d_hull_u = (prop_t_e - hull_r_t) / (hull_M + hull_M_a)

        return np.array([d_motor_i, d_motor_w, d_hull_u])

    @classmethod
    def _outputs(cls, t, x, u, params: dict):
        trans_k = params["trans_k"]
        esc_F_s = params["esc_F_s"]
        esc_V_ds_ov = params["esc_V_ds_ov"]
        esc_R_ds_on = params["esc_R_ds_on"]
        esc_E_on = params["esc_E_on"]
        esc_E_off = params["esc_E_off"]
        esc_V_F = params["esc_V_F"]
        esc_R_D = params["esc_R_D"]
        esc_Q_rr = params["esc_Q_rr"]

        batt_v = u[0]  # Battery voltage [V]
        esc_d = u[1]  # ESC duty-cycle [unitless]

        motor_i = x[0]  # Motor current [A]
        motor_w = x[1]  # Motor angular speed [rad/s]
        hull_u = x[2]  # Hull speed [m/s]

        A = cls._aux(t, x, u, params)
        motor_v = A[1]
        motor_q_load = A[2]
        prop_j = A[3]
        prop_k_q_ow = A[4]
        prop_k_t_ow = A[5]
        prop_q = A[6]
        prop_t = A[7]
        prop_t_e = A[8]
        hull_u = A[9]
        hull_r_t = A[10]

        prop_w = trans_k * motor_w

        if batt_v <= 1e-6:
            esc_i_in = 0.0
        else:
            esc_i_in = (
                esc_F_s * esc_Q_rr * (esc_V_ds_ov + 1) * batt_v
                + esc_F_s * (esc_E_off + esc_E_on) * (esc_V_ds_ov + 1)
                + esc_R_ds_on * esc_d * motor_i**2
                - esc_V_F * (esc_d - 1) * motor_i
                - esc_R_D * (esc_d - 1) * motor_i**2
                + esc_d * motor_i * batt_v
            ) / batt_v
        esc_i_in = max(0, esc_i_in)  # Clamp: this is a 1st-quadrant-only chopper

        return np.array(
            [
                motor_v,  # motor terminal voltage [V]
                motor_w,  # motor angular speed [rad/s]
                motor_i,  # motor current [A]
                motor_q_load,  # motor load torque [N·m]
                esc_i_in,  # ESC input current (DC bus) [A]
                prop_w,  # Propeller angular speed [rad/s]
                prop_j,  # J = V_A/(n D), V_A=V(1-w) (Birk 2019 Eq. 32.4, 41.7)
                prop_k_q_ow,  # open-water K_Q(J) (torque uses η_R correction)
                prop_k_t_ow,  # open-water K_T(J) used as thrust approximation
                prop_q,  # prop torque [N·m]
                prop_t,  # prop thrust [N]
                prop_t_e,  # effective thrust available to overcome towed resistance
                hull_u,  # hull speed [m/s]
                hull_r_t,  # towed resistance R_T [N]
            ]
        )

    @classmethod
    def build(cls, params: dict):
        return ct.NonlinearIOSystem(
            cls._update,
            cls._outputs,
            name="propulsion",
            states=("motor_i", "motor_w", "hull_u"),
            inputs=("batt_v", "esc_d"),
            outputs=(
                "motor_v",
                "motor_w",
                "motor_i",
                "motor_q_load",
                "esc_i_in",
                "prop_w",
                "prop_j",
                "prop_k_q",
                "prop_k_t",
                "prop_q",
                "prop_t",
                "prop_t_e",
                "hull_u",
                "hull_r_t",
            ),
            params=params,
        )


def get_hull_areas(hull_M, hull_cog_x, hull_S_total=8.2379):
    import sys

    sys.path.append("../")
    from hull.hydrostatic_hull import HydrostaticHull

    assert hull_S_total > 0

    hull = HydrostaticHull(cog_x=hull_cog_x, disp_mass=hull_M, total_area=hull_S_total)
    hull_S_water = hull.wet_surface_area()
    hull_S_air = hull_S_total - hull_S_water

    assert hull.is_in_valid_range()
    assert hull_S_water > 0
    assert hull_S_air > 0

    return (hull_S_water, hull_S_air)
