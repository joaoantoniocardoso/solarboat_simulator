import csv
import numpy as np
import pandas as pd

from utils.data import load_df
from utils.models import eval_poly


def _estimate_bseries_poly_coeffs(prop_PD, prop_AEA0, prop_Z, Rn=2e7):
    """Returns K_T(J) and K_Q(J) polynomial coefficients with Reynolds correction.

    Wageningen B-series open-water propeller performance charts (https://doi.org/10.5281/zenodo.8352831)
    """

    def load_terms(filepath):
        terms = []
        with open(filepath, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                C, s, t, u, v = map(float, row)
                terms.append([C, int(s), int(t), int(u), int(v)])
        return np.array(terms)

    def polynomial_in_J(terms, PD, AEA0, Z):
        max_s = int(np.max(terms[:, 1]))
        coeffs = np.zeros(max_s + 1)
        for C, s, t, u, v in terms:
            s = int(s)
            coeffs[s] += C * (PD ** int(t)) * (AEA0 ** int(u)) * (Z ** int(v))
        return coeffs

    def _delta_KT_coeffs(PD, AEA0, Z, Rn):
        K = np.log10(Rn) - 0.301
        c0 = +0.000353485
        c1 = (
            -0.00478125 * AEA0 * PD
            + 0.0000954 * K * Z * AEA0 * PD
            + 0.0000032049 * K * Z**2 * AEA0 * PD**3
        )
        c2 = (
            -0.00333758 * AEA0
            + 0.000257792 * K**2 * AEA0
            + 0.0000643192 * K * PD**6
            - 0.0000110636 * K**2 * PD**6
            - 0.0000276305 * K**2 * Z * AEA0
        )
        return np.array([c0, c1, c2])

    def _delta_KQ_coeffs(PD, AEA0, Z, Rn):
        K = np.log10(Rn) - 0.301
        c0 = (
            -0.000591412
            + 0.00696898 * PD
            - 0.0000666654 * Z * PD**6
            + 0.0160818 * AEA0**2
            - 0.000938091 * K * PD
            - 0.00059593 * K * PD**2
            + 0.0000782099 * K**2 * PD**2
            + 0.0000230171 * K * Z * PD**6
            - 0.00000184341 * K**2 * Z * PD**6
            - 0.00400252 * K * AEA0**2
            + 0.000220915 * K**2 * AEA0**2
        )
        c1 = -0.00000088528 * K**2 * Z * AEA0 * PD
        c2 = +0.0000052199 * K * Z * AEA0
        return np.array([c0, c1, c2])

    def _add_delta(base_coeffs, delta_coeffs):
        n = max(len(base_coeffs), len(delta_coeffs))
        out = np.zeros(n)
        out[: len(base_coeffs)] = base_coeffs
        out[: len(delta_coeffs)] += delta_coeffs
        return out

    KT_TERMS = load_terms("data/kt_terms_wageningen_b_series.csv")
    KQ_TERMS = load_terms("data/kq_terms_wageningen_b_series.csv")

    base_KT = polynomial_in_J(KT_TERMS, prop_PD, prop_AEA0, prop_Z)
    base_KQ = polynomial_in_J(KQ_TERMS, prop_PD, prop_AEA0, prop_Z)

    dKT = _delta_KT_coeffs(prop_PD, prop_AEA0, prop_Z, Rn)
    dKQ = _delta_KQ_coeffs(prop_PD, prop_AEA0, prop_Z, Rn)

    prop_k_T_coeffs = _add_delta(base_KT, dKT)
    prop_k_Q_coeffs = _add_delta(base_KQ, dKQ)

    return prop_k_T_coeffs, prop_k_Q_coeffs


def _estimate_prop_chord_07(prop_D, prop_hub_D, prop_AEA0, prop_Z):
    prop_R = prop_D / 2.0
    prop_hub_R = prop_hub_D / 2.0
    prop_A0 = np.pi * prop_R**2
    prop_AE = prop_AEA0 * prop_A0
    prop_A_blade = prop_AE / prop_Z
    prop_L_radial = prop_R - prop_hub_R
    prop_c_mean = prop_A_blade / prop_L_radial
    prop_c_07 = prop_c_mean
    return {
        "prop_R": prop_R,
        "prop_hub_R": prop_hub_R,
        "prop_A0": prop_A0,
        "prop_AE": prop_AE,
        "prop_A_blade": prop_A_blade,
        "prop_L_radial": prop_L_radial,
        "prop_c_mean": prop_c_mean,
        "prop_c_07": prop_c_07,
    }


def _estimate_prop_I_r(
    prop_blade_thickness,
    prop_mat_rho,
    prop_AE,
    prop_Z,
    prop_hub_R,
    prop_L_radial,
    prop_total_m,
):
    prop_blades_total_m = prop_mat_rho * prop_AE * prop_blade_thickness
    prop_blade_m = prop_blades_total_m / prop_Z
    prop_hub_m = prop_total_m - prop_blades_total_m
    if prop_hub_m < 0:
        return None

    J_hub = 0.5 * prop_hub_m * prop_hub_R**2
    J_blade = (1.0 / 3.0) * prop_blade_m * prop_L_radial**2
    J_total = J_hub + prop_Z * J_blade

    return {
        "prop_blade_thickness": prop_blade_thickness,
        "prop_mat_rho": prop_mat_rho,
        "prop_blades_total_m": prop_blades_total_m,
        "prop_blade_m": prop_blade_m,
        "prop_hub_m": prop_hub_m,
        "prop_L_radial": prop_L_radial,
        "J_hub": J_hub,
        "J_blade": J_blade,
        "J_total": J_total,
    }


def _estimate_wake_fraction(
    hull_u, hull_L, hull_B, hull_M, hull_C_TB, water_rho, g=9.81
):
    """Wake fraction using Taylor relation from Molland et al. (Ship Resistance and Propulsion, 2017)."""
    hull_D = hull_M * g
    hull_disp_vol = hull_D / (water_rho * g)
    hull_T = hull_disp_vol / (hull_C_TB * hull_L * hull_B)
    hull_Fn = hull_u / np.sqrt(g * hull_L)
    hull_W = 0.5 * hull_C_TB - 0.05
    return {"hull_W": hull_W, "hull_Fn": hull_Fn, "hull_T": hull_T}


def _estimate_thrust_deduction(hull_W, k_R=0.6):
    """Thrust deduction and hull efficiency per Molland et al. for small craft."""
    hull_T_ded = k_R * hull_W
    hull_eta_H = (1.0 - hull_T_ded) / (1.0 - hull_W)
    return {"hull_T_ded": hull_T_ded, "hull_k_R": k_R, "hull_eta_H": hull_eta_H}


def _round_re_to_2e_power(Re):
    x = int(round(np.log10(Re / 2.0)))
    return 2.0 * 10.0**x


def _estimate_hull_C_T_steady(df, params):
    rho_w = params["rho_water"]
    D = params["prop_D"]
    tded = params["hull_T_ded"]
    S_eff = params["hull_S_air"] * params["rho_air"] + params["hull_S_water"] * rho_w

    prop_w = df["motor_w"] * params["trans_k"]
    # lambdas / KT
    prop_n = prop_w / (2 * np.pi)
    lam = df["hull_u"] / (prop_n * D + 1e-9)
    KT = eval_poly(params["prop_k_T_coeffs"], lam)  # your Horner eval

    T_prop = rho_w * (prop_n**2) * (D**4) * KT
    T_eff = (1.0 - tded) * T_prop

    # numerical du/dt (for steady filter only)
    dt = np.mean(np.diff(df.index.to_numpy(dtype=float)))  # if index is time
    du = np.gradient(df["hull_u"].to_numpy(), dt)
    du_abs = np.abs(du)

    mask = (df["hull_u"] > 0.5) & (du_abs < 0.05) & np.isfinite(T_eff)
    u_sel = df["hull_u"].to_numpy()[mask]
    Te_sel = T_eff.to_numpy()[mask]

    hull_C_T_i = 2.0 * Te_sel / (u_sel**2 * S_eff)
    hull_C_T_i = hull_C_T_i[np.isfinite(hull_C_T_i)]

    if hull_C_T_i.size == 0:
        raise RuntimeError("No steady points found for hull_C_T estimation")

    return float(np.average(hull_C_T_i))


def _estimate_hull_C_T_dynamic(df, params):
    rho_w = params["rho_water"]
    D = params["prop_D"]
    tded = params["hull_T_ded"]
    m = params["hull_M"]
    S_eff = params["hull_S_air"] * params["rho_air"] + params["hull_S_water"] * rho_w

    prop_w = df["motor_w"] * params["trans_k"]
    prop_n = prop_w / (2 * np.pi)
    lam = df["hull_u"] / (prop_n * D + 1e-9)
    KT = eval_poly(params["prop_k_T_coeffs"], lam)

    T_prop = rho_w * (prop_n**2) * (D**4) * KT
    T_eff = (1.0 - tded) * T_prop

    # du/dt
    dt = np.mean(np.diff(df.index.to_numpy(dtype=float)))
    du = np.gradient(df["hull_u"].to_numpy(), dt)

    y = T_eff.to_numpy() - m * du
    x = 0.5 * S_eff * (df["hull_u"].to_numpy() ** 2)

    mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x = x[mask]
    y = y[mask]

    hull_C_T = np.dot(x, y) / np.dot(x, x)
    return float(hull_C_T)


def _estimate_hull_C_T(params):
    df = load_boat_data("../../../models/2020/boat_data_50ms.csv")

    hull_C_T1 = _estimate_hull_C_T_steady(df, params)
    hull_C_T2 = _estimate_hull_C_T_dynamic(df, params)
    hull_C_T = 0.5 * (hull_C_T1 + hull_C_T2)

    return hull_C_T


def estimate_initial_values(
    motor_rpm,
    batt_v,
    esc_i_in,
    motor_eta,
    esc_eta,
    trans_eta,
    trans_k,
    hull_u,
    hull_L,
    hull_B,
    hull_M,
    hull_C_TB,
    hull_S_air,
    hull_S_water,
    prop_D,
    prop_Z,
    prop_hub_D,
    prop_AEA0,
    prop_PD,
    prop_total_m,
    prop_blade_thickness,
    prop_mat_rho,
    prop_c_07=None,
    water_rho=1000.0,
    air_rho=1.0,
    water_mu=1e-3,
):
    motor_n = motor_rpm / 60.0
    motor_w = 2.0 * np.pi * motor_n
    prop_rpm = motor_rpm * trans_k
    prop_n = prop_rpm / 60.0
    prop_w = 2.0 * np.pi * prop_n

    wake = _estimate_wake_fraction(hull_u, hull_L, hull_B, hull_M, hull_C_TB, water_rho)
    hull_W = wake["hull_W"]
    hull_Fn = wake["hull_Fn"]
    hull_T = wake["hull_T"]

    prop_va = hull_u * (1.0 - hull_W)
    prop_I_ra = prop_va / (prop_n * prop_D)

    td = _estimate_thrust_deduction(hull_W, k_R=0.6)
    hull_T_ded = td["hull_T_ded"]
    hull_k_R = td["hull_k_R"]
    hull_eta_H = td["hull_eta_H"]

    geom = _estimate_prop_chord_07(prop_D, prop_hub_D, prop_AEA0, prop_Z)
    prop_R = geom["prop_R"]
    prop_hub_R = geom["prop_hub_R"]
    prop_A0 = geom["prop_A0"]
    prop_AE = geom["prop_AE"]
    prop_A_blade = geom["prop_A_blade"]
    prop_L_radial = geom["prop_L_radial"]
    prop_c_mean = geom["prop_c_mean"]
    prop_c_07_est = geom["prop_c_07"]
    if prop_c_07 is None:
        prop_c_07 = prop_c_07_est

    prop_r_07 = 0.7 * prop_R
    prop_Vtheta_07 = prop_w * prop_r_07
    prop_Vrel_07 = np.sqrt(prop_va**2 + prop_Vtheta_07**2)

    prop_Re_07 = water_rho * prop_Vrel_07 * prop_c_07 / water_mu
    prop_Re_07_rounded = _round_re_to_2e_power(prop_Re_07)

    prop_k_T_coeffs, prop_k_Q_coeffs = _estimate_bseries_poly_coeffs(
        prop_PD=prop_PD,
        prop_AEA0=prop_AEA0,
        prop_Z=prop_Z,
        Rn=prop_Re_07,
    )

    def _eval_poly(coeffs, x):
        result = 0.0
        for c in reversed(coeffs[1:]):
            result = (result + c) * x
        return result + coeffs[0]

    prop_k_T = _eval_poly(prop_k_T_coeffs, prop_I_ra)
    prop_k_Q = _eval_poly(prop_k_Q_coeffs, prop_I_ra)

    prop_T = water_rho * (prop_n**2) * (prop_D**4) * prop_k_T
    prop_Q = water_rho * (prop_n**2) * (prop_D**5) * prop_k_Q
    prop_P_shaft = 2.0 * np.pi * prop_n * prop_Q
    prop_eta_open = (
        prop_I_ra * prop_k_T / (2.0 * np.pi * prop_k_Q) if prop_k_Q != 0.0 else None
    )

    esc_P_in = batt_v * esc_i_in
    esc_P_out = esc_P_in * esc_eta
    motor_P_shaft = esc_P_out * motor_eta
    trans_P_out = motor_P_shaft * trans_eta

    system_eta_elec_to_prop = prop_P_shaft / esc_P_in if esc_P_in > 0 else None
    system_eta_available_to_prop = (
        prop_P_shaft / trans_P_out if trans_P_out > 0 else None
    )

    J_est = _estimate_prop_I_r(
        prop_blade_thickness=prop_blade_thickness,
        prop_mat_rho=prop_mat_rho,
        prop_AE=prop_AE,
        prop_Z=prop_Z,
        prop_hub_R=prop_hub_R,
        prop_L_radial=prop_L_radial,
        prop_total_m=prop_total_m,
    )

    hull_C_T = _estimate_hull_C_T(
        {
            "rho_water": water_rho,
            "rho_air": air_rho,
            "prop_D": prop_D,
            "hull_T_ded": hull_T_ded,
            "hull_S_air": hull_S_air,
            "hull_S_water": hull_S_water,
            "trans_k": trans_k,
            "prop_k_T_coeffs": prop_k_T_coeffs,
            "hull_M": hull_M,
        }
    )

    return {
        "motor_rpm": motor_rpm,
        "motor_n": motor_n,
        "motor_w": motor_w,
        "motor_v": batt_v,
        "motor_i": esc_i_in,
        "motor_eta": motor_eta,
        "esc_eta": esc_eta,
        "trans_eta": trans_eta,
        "trans_k": trans_k,
        "esc_P_in": esc_P_in,
        "esc_P_out": esc_P_out,
        "motor_P_shaft": motor_P_shaft,
        "trans_P_out": trans_P_out,
        "system_eta_elec_to_prop": system_eta_elec_to_prop,
        "system_eta_available_to_prop": system_eta_available_to_prop,
        "hull_u": hull_u,
        "hull_L": hull_L,
        "hull_B": hull_B,
        "hull_M": hull_M,
        "hull_C_TB": hull_C_TB,
        "hull_S_air": hull_S_air,
        "hull_S_water": hull_S_water,
        "hull_C_T": hull_C_T,
        "hull_T": hull_T,
        "hull_Fn": hull_Fn,
        "hull_W": hull_W,
        "hull_T_ded": hull_T_ded,
        "hull_k_R": hull_k_R,
        "hull_eta_H": hull_eta_H,
        "prop_rpm": prop_rpm,
        "prop_n": prop_n,
        "prop_w": prop_w,
        "prop_va": prop_va,
        "prop_I_ra": prop_I_ra,
        "prop_D": prop_D,
        "prop_R": prop_R,
        "prop_Z": prop_Z,
        "prop_hub_D": prop_hub_D,
        "prop_hub_R": prop_hub_R,
        "prop_A0": prop_A0,
        "prop_AE": prop_AE,
        "prop_A_blade": prop_A_blade,
        "prop_L_radial": prop_L_radial,
        "prop_c_mean": prop_c_mean,
        "prop_c_07": prop_c_07,
        "prop_r_07": prop_r_07,
        "prop_Vtheta_07": prop_Vtheta_07,
        "prop_Vrel_07": prop_Vrel_07,
        "prop_Re_07": prop_Re_07,
        "prop_Re_07_rounded": prop_Re_07_rounded,
        "prop_k_T_coeffs": prop_k_T_coeffs,
        "prop_k_Q_coeffs": prop_k_Q_coeffs,
        "prop_k_T": prop_k_T,
        "prop_k_Q": prop_k_Q,
        "prop_T": prop_T,
        "prop_Q": prop_Q,
        "prop_P_shaft": prop_P_shaft,
        "prop_eta_open": prop_eta_open,
        "prop_I_r": J_est["J_total"] if J_est is not None else None,
        "prop_I_r_details": J_est,
        "air_rho": air_rho,
        "water_rho": water_rho,
        "water_mu": water_mu,
    }


def load_boat_data(filepath: str):
    rename_columns = {
        "Battery Pack Voltage": "batt_v",
        "Battery Output Current": "batt_i_out",
        "ESC Duty Cycle": "pilot_d",
        "Motor Angular Speed": "motor_w",
        "MPPTs Input Power": "mppts_p_in",
    }
    df = load_df(
        filename=filepath,
        start=None,
        end=None,
        resample_rule="1s",
        rename_columns=rename_columns,
        print_columns=False,
        iqr_threshold=None,
        cutoff_freq=None,
        sampling_rate=1,
        order=1,
    )
    if "hull_u" not in df.columns:

        def boat_speed_from_motor_v(motor_v, a=0.54340307 / 3.6):
            return a * motor_v

        df["hull_u"] = boat_speed_from_motor_v(df["batt_v"] * df["pilot_d"])

    return df
