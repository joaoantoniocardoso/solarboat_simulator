"""Propeller + hull interaction helper utilities.

This module implements:
- Wageningen B-series open-water polynomials for ``K_T(J)`` and ``K_Q(J)`` with an
  additive Reynolds-number correction.
- Simple empirical wake fraction / thrust deduction / hull efficiency relations.
- Data-driven estimation of an effective hull total-resistance coefficient ``C_T``.

References are documented at the function level.

Important: this file is the ground-truth implementation for the thesis text. The
annotations below aim to document equations, parameter meaning, and SI units
without changing numerical behavior.
"""

import csv
import numpy as np
import pandas as pd

from utils.data import load_df
from utils.models import eval_poly


def _estimate_bseries_poly_coeffs(prop_PD, prop_AEA0, prop_Z, Rn=2e7):
    """Compute Wageningen B-series ``K_T(J)`` and ``K_Q(J)`` polynomial coefficients.

    Parameters
    - prop_PD: float
        Pitch-to-diameter ratio ``P/D`` [-].
    - prop_AEA0: float
        Expanded blade area ratio ``AE/A0`` [-].
    - prop_Z: int
        Number of blades ``Z`` [-].
    - Rn: float
        Reynolds number used by the Reynolds correction term [-].

        [ASSUMPTION] This ``Rn`` is intended to represent the Reynolds number at the
        blade section near 0.75R (as in Oosterveld & van Oossanen, 1975) or a close
        proxy (this code later uses ``Re`` computed at 0.7R).

    Returns
    - (prop_k_T_coeffs, prop_k_Q_coeffs): tuple[np.ndarray, np.ndarray]
        Arrays of coefficients (ascending powers) such that:

        - ``K_T(J) = Σ_{s=0..S} prop_k_T_coeffs[s] * J^s``
        - ``K_Q(J) = Σ_{s=0..S} prop_k_Q_coeffs[s] * J^s``

        where ``J`` is the advance coefficient [-].

    Equations
    1) Base Wageningen B-series polynomial form (Table 5):

       ``K_T = Σ C * J^s * (P/D)^t * (AE/A0)^u * Z^v``
       ``K_Q = Σ C * J^s * (P/D)^t * (AE/A0)^u * Z^v``

       This function rearranges the above into a polynomial-in-``J`` by collecting
       all terms that share the same exponent ``s``.

       The local CSV term lists in ``data/kt_terms_wageningen_b_series.csv``
       and ``data/kq_terms_wageningen_b_series.csv`` correspond to Table 5.

    2) Reynolds-number correction (Table 6):

       The correction is implemented as an additive polynomial in ``J``:

       ``ΔK_T(J) = c0 + c1*J + c2*J^2``
       ``ΔK_Q(J) = c0 + c1*J + c2*J^2``

       where ``K = log10(Rn) - 0.301`` is used in the coefficients.

       The returned coefficient arrays already include the additive correction:

       ``K_T = K_T,base + ΔK_T`` and ``K_Q = K_Q,base + ΔK_Q``.

    References
    - `oosterveld1975`
        M.W.C. Oosterveld; P. van Oossanen (1975), "Further Computer-Analyzed Data
        of the Wageningen B-Screw Series".

        - Table 5 (printed page 257; PDF page 6): coefficients/terms for the
          Wageningen B-series ``K_T`` and ``K_Q`` polynomials at ``Rn = 2*10^6``.
        - Table 6 (printed page 257; PDF page 7): polynomials for Reynolds-number
          effect (above ``Rn = 2*10^6``) on ``K_T`` and ``K_Q``.
    """

    def load_terms(filepath):
        """Load polynomial terms from CSV.

        CSV format: columns ``C, s, t, u, v``.

        Each term means:
        ``C * J^s * (P/D)^t * (AE/A0)^u * Z^v``

        All quantities are dimensionless.
        """

        terms = []
        with open(filepath, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                C, s, t, u, v = map(float, row)
                terms.append([C, int(s), int(t), int(u), int(v)])
        return np.array(terms)

    def polynomial_in_J(terms, PD, AEA0, Z):
        """Collect term list into an ascending-power polynomial in ``J``."""

        max_s = int(np.max(terms[:, 1]))
        coeffs = np.zeros(max_s + 1)
        for C, s, t, u, v in terms:
            s = int(s)
            coeffs[s] += C * (PD ** int(t)) * (AEA0 ** int(u)) * (Z ** int(v))
        return coeffs

    def _delta_KT_coeffs(PD, AEA0, Z, Rn):
        """Reynolds correction polynomial coefficients for ``ΔK_T(J)``.

        Implements Table 6 from Oosterveld & van Oossanen (1975) as:
        ``ΔK_T(J) = c0 + c1*J + c2*J^2``.

        Reference location: Table 6 (printed page 257; PDF page 7).

        Note: the table uses ``(log Rn - 0.301)`` where ``log`` is base 10.
        """

        # Table 6 uses K = log10(Rn) - 0.301
        K = np.log10(Rn) - 0.301

        # Constant term for ΔK_T(J)
        c0 = +0.000353485

        # Coefficient of J in ΔK_T(J)
        c1 = (
            -0.00478125 * AEA0 * PD
            + 0.0000954 * K * Z * AEA0 * PD
            + 0.0000032049 * K * Z**2 * AEA0 * PD**3
        )

        # Coefficient of J^2 in ΔK_T(J)
        c2 = (
            -0.00333758 * AEA0
            + 0.000257792 * K**2 * AEA0
            + 0.0000643192 * K * PD**6
            - 0.0000110636 * K**2 * PD**6
            - 0.0000276305 * K**2 * Z * AEA0
        )

        return np.array([c0, c1, c2])

    def _delta_KQ_coeffs(PD, AEA0, Z, Rn):
        """Reynolds correction polynomial coefficients for ``ΔK_Q(J)``.

        Implements Table 6 from Oosterveld & van Oossanen (1975) as:
        ``ΔK_Q(J) = c0 + c1*J + c2*J^2``.

        Reference location: Table 6 (printed page 257; PDF page 7).

        Note: the table uses ``(log Rn - 0.301)`` where ``log`` is base 10.
        """

        K = np.log10(Rn) - 0.301

        # c0 corresponds to the constant term in J.
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

        # Coefficient of J (Table 6)
        c1 = -0.00000088528 * K**2 * Z * AEA0 * PD

        # Coefficient of J^2 (Table 6)
        c2 = +0.0000052199 * K * Z * AEA0

        return np.array([c0, c1, c2])

    def _add_delta(base_coeffs, delta_coeffs):
        """Add a (possibly shorter) delta polynomial to base polynomial."""

        n = max(len(base_coeffs), len(delta_coeffs))
        out = np.zeros(n)
        out[: len(base_coeffs)] = base_coeffs
        out[: len(delta_coeffs)] += delta_coeffs
        return out

    # Table 5 term lists (dimensionless coefficients and exponents).
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
    """Estimate propeller chord length at radius ~0.7R.

    Parameters
    - prop_D: float
        Propeller diameter ``D`` [m].
    - prop_hub_D: float
        Hub diameter [m].
    - prop_AEA0: float
        Expanded blade area ratio ``AE/A0`` [-].
    - prop_Z: int
        Number of blades ``Z`` [-].

    Returns
    A dict with:
    - prop_R: float
        Propeller radius ``R = D/2`` [m].
    - prop_hub_R: float
        Hub radius [m].
    - prop_A0: float
        Disk area ``A0 = πR²`` [m²].
    - prop_AE: float
        Expanded blade area ``AE = (AE/A0)*A0`` [m²].
    - prop_A_blade: float
        Expanded area per blade [m²].
    - prop_L_radial: float
        Radial span length from hub to tip [m].
    - prop_c_mean: float
        Mean chord estimate [m].
    - prop_c_07: float
        Chord estimate used at 0.7R [m].

    Equations
    - ``A0 = π (D/2)²``
    - ``AE = (AE/A0) * A0``
    - ``A_blade = AE / Z``
    - ``L_radial = R - R_hub``
    - ``c_mean = A_blade / L_radial``

    Assumptions
    - [ASSUMPTION] Uses ``c_0.7R ≈ c_mean`` (uniform chord distribution surrogate).

    References
    - None (geometric bookkeeping).
    """

    prop_R = prop_D / 2.0
    prop_hub_R = prop_hub_D / 2.0
    prop_A0 = np.pi * prop_R**2
    prop_AE = prop_AEA0 * prop_A0
    prop_A_blade = prop_AE / prop_Z
    prop_L_radial = prop_R - prop_hub_R
    prop_c_mean = prop_A_blade / prop_L_radial

    # [ASSUMPTION] Mean chord used as chord at 0.7R.
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
    """Estimate propeller polar moment of inertia about the shaft axis.

    Parameters
    - prop_blade_thickness: float
        Blade thickness [m].
    - prop_mat_rho: float
        Propeller material density [kg/m³].
    - prop_AE: float
        Expanded blade area (total over all blades) [m²].
    - prop_Z: int
        Number of blades [-].
    - prop_hub_R: float
        Hub radius [m].
    - prop_L_radial: float
        Blade radial span (tip radius minus hub radius) [m].
    - prop_total_m: float
        Total propeller mass [kg].

    Returns
    - dict | None
        Returns ``None`` if computed hub mass is negative.

        Otherwise returns a dict with:
        - prop_blades_total_m: [kg]
        - prop_blade_m: [kg]
        - prop_hub_m: [kg]
        - J_hub: hub inertia [kg·m²]
        - J_blade: single-blade inertia [kg·m²]
        - J_total: total inertia [kg·m²]

    Equations
    - Blade mass estimate (volume = area * thickness):
      ``m_blades = rho_mat * AE * t``

    - Hub mass: ``m_hub = m_total - m_blades``

    - Hub inertia (solid disk approximation):
      ``J_hub = 0.5 * m_hub * R_hub²``

    - Blade inertia (slender rod about one end):
      ``J_blade = (1/3) * m_blade * L_radial²``

    - Total inertia:
      ``J_total = J_hub + Z * J_blade``

    Assumptions
    - Hub is a solid disk.
    - Each blade behaves like a slender rod about its root.

    References
    - None (basic rigid-body approximations).
    """

    prop_blades_total_m = prop_mat_rho * prop_AE * prop_blade_thickness
    prop_blade_m = prop_blades_total_m / prop_Z
    prop_hub_m = prop_total_m - prop_blades_total_m
    if prop_hub_m < 0:
        return None

    # [ASSUMPTION] Hub inertia as solid disk: J = (1/2) m R^2.
    J_hub = 0.5 * prop_hub_m * prop_hub_R**2

    # [ASSUMPTION] Blade inertia as slender rod about one end: J = (1/3) m L^2.
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
    hull_u, hull_L, hull_B, hull_M, hull_C_B, water_rho, g=9.81
):
    """Estimate wake fraction using Taylor relation (empirical).

    Parameters
    - hull_u: float
        Ship speed through water ``V`` [m/s].
    - hull_L: float
        Length (assumed LWL / characteristic length) ``L`` [m].
    - hull_B: float
        Breadth ``B`` [m].
    - hull_M: float
        Mass ``M`` [kg].
    - hull_C_B: float
        Block coefficient ``C_B`` [-].
    - water_rho: float
        Water density ``ρ`` [kg/m³].
    - g: float
        Gravitational acceleration [m/s²].

    Returns
    - dict
        - hull_W: wake fraction ``w_T`` [-]
        - hull_Fn: Froude number ``Fn`` [-]
        - hull_T: draft ``T`` [m] (computed from displacement proxy)

    Equations
    - Displacement weight: ``Δ = M g`` [N]
    - Displacement volume: ``∇ = Δ / (ρ g)`` [m³]
    - Draft proxy from ``∇ = C_B L B T``:
      ``T = ∇ / (C_B L B)`` [m]

    - Froude number:
      ``Fn = V / sqrt(g L)`` [-]

    - Taylor wake fraction regression (Molland et al.):
      ``w_T = 0.5*C_B - 0.05`` [-]

    References
    - `mollandShipResistancePropulsion2017`
        Molland, Turnock & Hudson (2017), 2nd ed.
        - Eq. (8.15): ``w_T = 0.50 C_B - 0.05`` (book page 161; PDF page 192).

    Assumptions / validity
    - Empirical relation; hull-form and propulsor arrangement dependent.
      Applied here as a first estimate for this hull.
    """

    hull_D = hull_M * g
    hull_disp_vol = hull_D / (water_rho * g)

    # Draft proxy: ∇ = C_B L B T
    hull_T = hull_disp_vol / (hull_C_B * hull_L * hull_B)

    hull_Fn = hull_u / np.sqrt(g * hull_L)

    # Taylor wake fraction relation.
    hull_W = 0.5 * hull_C_B - 0.05

    return {"hull_W": hull_W, "hull_Fn": hull_Fn, "hull_T": hull_T}


def _estimate_thrust_deduction(hull_W, k_R=0.6):
    """Estimate thrust deduction and hull efficiency from wake fraction.

    Parameters
    - hull_W: float
        Wake fraction ``w_T`` [-].
    - k_R: float
        Rudder factor ``k_R`` [-].

        Molland et al. note ``k_R`` varies roughly from 0.5 (thin rudders) to 0.7
        (thick rudders). This function defaults to ``k_R = 0.6``.

    Returns
    - dict
        - hull_T_ded: thrust deduction factor ``t`` [-]
        - hull_k_R: ``k_R`` [-]
        - hull_eta_H: hull efficiency ``η_H`` [-]

    Equations
    - Thrust deduction regression:
      ``t = k_R * w_T``

    - Hull efficiency definition:
      ``η_H = (1 - t) / (1 - w_T)``

    References
    - `mollandShipResistancePropulsion2017`
        Molland, Turnock & Hudson (2017), 2nd ed.
        - Eq. (8.18): ``t = k_R * w_T`` (book page 163; PDF page 194).
        - Eq. (16.2): ``η_H = (1 - t)/(1 - w_T)`` (book page 395; PDF page 426).

    Assumptions / validity
    - [ASSUMPTION] ``k_R=0.6`` is used as a generic mid-range estimate.
    """

    hull_T_ded = k_R * hull_W
    hull_eta_H = (1.0 - hull_T_ded) / (1.0 - hull_W)
    return {"hull_T_ded": hull_T_ded, "hull_k_R": k_R, "hull_eta_H": hull_eta_H}


def _round_re_to_2e_power(Re):
    """Round a Reynolds number to the nearest ``2*10^x``.

    Parameters
    - Re: float
        Reynolds number [-].

    Returns
    - float
        Rounded Reynolds number [-] of the form ``2*10^x``.

    Notes
    - Used to map ``Re`` to reference magnitudes often used in propeller charts.
      In this codebase, the rounded value is returned for reporting but is not used
      directly in the Wageningen correction calculation.
    """

    x = int(round(np.log10(Re / 2.0)))
    return 2.0 * 10.0**x


def _estimate_hull_C_T_steady(df, params):
    """Estimate a steady-state effective hull total-resistance coefficient ``C_T``.

    Parameters
    - df: pandas.DataFrame
        Must contain:
        - ``df["motor_w"]``: motor angular speed [rad/s]
        - ``df["hull_u"]``: hull speed [m/s]

        ``df.index`` is treated as time [s] when computing ``du/dt``.
        [ASSUMPTION] If the index is not in seconds, the steady mask will be wrong.
    - params: dict
        Must contain:
        - ``rho_water`` [kg/m³]
        - ``rho_air`` [kg/m³]
        - ``prop_D`` [m]
        - ``hull_T_ded`` thrust deduction ``t`` [-]
        - ``hull_S_air`` [m²]
        - ``hull_S_water`` [m²]
        - ``trans_k`` gear ratio ``n_prop/n_motor`` [-] (as used below)
        - ``prop_k_T_coeffs`` coefficients for ``K_T(J)`` [-]

    Returns
    - float
        Estimated ``C_T`` [-].

    Equations
    - Propeller rev/s: ``n = ω/(2π)``
    - [ASSUMPTION] Uses ``J = V/(nD)`` (ship speed), not ``V_A``.

    - Thrust from open-water coefficient:
      ``T = ρ n² D⁴ K_T`` [N]

    - Effective thrust on hull (interaction):
      ``T_eff = (1 - t) T`` [N]

    - Effective drag model (combined air + water):
      ``T_eff ≈ 0.5 * C_T * V² * S_eff``

      where ``S_eff = ρ_air*S_air + ρ_water*S_water`` has units [kg/m].

      Solving for ``C_T`` gives:
      ``C_T = 2*T_eff / (V² * S_eff)``.

    References
    - ``K_T`` definition is standard in propeller theory.
      (No additional reference used here; this step is a data-fit for this project.)

    Assumptions / validity
    - [ASSUMPTION] Single combined ``C_T`` applied to combined air+water effective area.
    - Steady-point filter uses thresholds ``V > 0.5`` and ``|du/dt| < 0.05``.
    """

    rho_w = params["rho_water"]
    D = params["prop_D"]
    tded = params["hull_T_ded"]
    S_eff = params["hull_S_air"] * params["rho_air"] + params["hull_S_water"] * rho_w

    prop_w = df["motor_w"] * params["trans_k"]

    # Advance coefficient proxy (ship speed): J = V / (nD)
    prop_n = prop_w / (2 * np.pi)
    lam = df["hull_u"] / (prop_n * D + 1e-9)

    KT = eval_poly(params["prop_k_T_coeffs"], lam)

    T_prop = rho_w * (prop_n**2) * (D**4) * KT
    T_eff = (1.0 - tded) * T_prop

    dt = np.mean(np.diff(df.index.to_numpy(dtype=float)))
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
    """Estimate an effective hull total-resistance coefficient ``C_T`` with dynamics.

    Parameters
    - df: pandas.DataFrame
        Must contain ``motor_w`` [rad/s] and ``hull_u`` [m/s].
        ``df.index`` is treated as time [s].
    - params: dict
        Must contain the same fields as in ``_estimate_hull_C_T_steady`` plus:
        - ``hull_M`` [kg]

    Returns
    - float
        Estimated ``C_T`` [-].

    Equations
    Starting from:

    ``T_eff - R = M * dV/dt``

    with ``R = 0.5 * C_T * V² * S_eff`` and using ``T_eff = (1-t)T``.

    Rearranged into a linear regression:

    - ``y = T_eff - M * dV/dt``
    - ``x = 0.5 * S_eff * V²``

    Estimate ``C_T`` via least squares through the origin:
    ``C_T = (x·y) / (x·x)``.

    Assumptions
    - Same modeling assumptions as the steady estimator (single combined air+water term).
    """

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
    """Compute an effective hull total-resistance coefficient ``C_T``.

    Parameters
    - params: dict
        Passed through to the steady/dynamic estimators.

    Returns
    - float
        ``C_T`` [-].

    Notes
    - Loads a hard-coded boat dataset path. This helper is intended for the
      propeller parameter initialization notebook/workflow.
    - Final estimate is the arithmetic mean of the steady and dynamic estimates.
    """

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
    hull_C_B,
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
    """Estimate initial propulsive/hull parameters and derived quantities.

    This function is a convenience wrapper used by the propeller notebook to:
    - apply wake/thrust-deduction relations,
    - compute advance coefficient and open-water ``K_T, K_Q``,
    - compute thrust/torque/power estimates,
    - compute efficiency bookkeeping,
    - compute a project-specific effective hull resistance coefficient ``C_T``.

    Parameters (SI units unless noted)
    Rotational / drivetrain
    - motor_rpm: float
        Motor rotational speed [rev/min].
    - motor_eta: float
        Motor efficiency [-].
    - trans_eta: float
        Transmission efficiency [-].
    - trans_k: float
        Speed ratio ``n_prop / n_motor`` [-].

    Electrical
    - batt_v: float
        Battery pack voltage [V].
    - esc_i_in: float
        ESC input current [A].
    - esc_eta: float
        ESC efficiency [-].

    Hull
    - hull_u: float
        Ship speed through water ``V`` [m/s].
    - hull_L: float
        Characteristic length (assumed LWL) [m].
    - hull_B: float
        Breadth [m].
    - hull_M: float
        Mass [kg].
    - hull_C_B: float
        Used as block coefficient ``C_B`` [-].
    - hull_S_air: float
        Projected area used for air drag term [m²].
    - hull_S_water: float
        Wetted area used for water drag term [m²].

    Propeller geometry / mass
    - prop_D: float
        Diameter ``D`` [m].
    - prop_Z: int
        Number of blades ``Z`` [-].
    - prop_hub_D: float
        Hub diameter [m].
    - prop_AEA0: float
        Expanded blade area ratio ``AE/A0`` [-].
    - prop_PD: float
        Pitch-to-diameter ratio ``P/D`` [-].
    - prop_total_m: float
        Total propeller mass [kg].
    - prop_blade_thickness: float
        Blade thickness [m].
    - prop_mat_rho: float
        Material density [kg/m³].
    - prop_c_07: float | None
        Blade chord length at 0.7R used for Reynolds estimate [m]. If ``None``,
        uses the estimate from ``_estimate_prop_chord_07``.

    Fluids
    - water_rho: float
        Water density [kg/m³].
    - air_rho: float
        Air density [kg/m³].
    - water_mu: float
        Water dynamic viscosity ``μ`` [Pa·s = kg/(m·s)].

    Returns
    - dict
        Contains both inputs and derived quantities. Key outputs include:

        - ``hull_W``: wake fraction ``w_T`` [-]
        - ``hull_T_ded``: thrust deduction factor ``t`` [-]
        - ``hull_eta_H``: hull efficiency ``η_H`` [-]
        - ``prop_va``: advance speed ``V_A = V(1-w_T)`` [m/s]
        - ``prop_I_ra``: advance coefficient ``J = V_A/(nD)`` [-]
        - ``prop_k_T``, ``prop_k_Q``: open-water coefficients ``K_T``, ``K_Q`` [-]
        - ``prop_T``: thrust ``T = ρ n² D⁴ K_T`` [N]
        - ``prop_Q``: torque ``Q = ρ n² D⁵ K_Q`` [N·m]
        - ``prop_P_shaft``: shaft power ``P = 2π n Q`` [W]
        - ``prop_eta_open``: open-water efficiency ``η_0 = (J K_T)/(2π K_Q)`` [-]
        - ``prop_I_r``: estimated inertia about shaft axis [kg·m²]

        Additional bookkeeping keys are returned exactly as implemented.

    References
    - `mollandShipResistancePropulsion2017`
        Wake / thrust deduction / hull efficiency:
        - Eq. (8.15) (book p.161; PDF p.192)
        - Eq. (8.18) (book p.163; PDF p.194)
        - Eq. (16.2) (book p.395; PDF p.426)

    - `oosterveld1975`
        ``K_T``/``K_Q`` polynomials and Reynolds correction:
        - Table 5 (printed p.257; PDF p.6)
        - Table 6 (printed p.257; PDF p.7)
        - Eq. (4): definition of ``K_T`` (printed p.255; PDF p.4)
        - Eq. (10): Reynolds number definition at 0.75R (printed p.255; PDF p.4)

        [ASSUMPTION] The mapping of the code's ``prop_Re_07`` (0.7R) to the paper's
        ``Rn_0.75R`` is treated as a close proxy.
    """

    motor_n = motor_rpm / 60.0
    motor_w = 2.0 * np.pi * motor_n
    prop_rpm = motor_rpm * trans_k
    prop_n = prop_rpm / 60.0
    prop_w = 2.0 * np.pi * prop_n

    wake = _estimate_wake_fraction(hull_u, hull_L, hull_B, hull_M, hull_C_B, water_rho)
    hull_W = wake["hull_W"]
    hull_Fn = wake["hull_Fn"]
    hull_T = wake["hull_T"]

    # Advance speed (effective inflow at prop): V_A = V (1 - w)
    prop_va = hull_u * (1.0 - hull_W)

    # Advance coefficient J = V_A / (n D)
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

    # Reynolds number estimate at 0.7R.
    # Oosterveld & van Oossanen define Rn at 0.75R (Eq. 10); this uses 0.7R.
    prop_r_07 = 0.7 * prop_R
    prop_Vtheta_07 = prop_w * prop_r_07
    prop_Vrel_07 = np.sqrt(prop_va**2 + prop_Vtheta_07**2)

    # Reynolds number (dimensionless): Re = ρ V c / μ
    prop_Re_07 = water_rho * prop_Vrel_07 * prop_c_07 / water_mu

    prop_Re_07_rounded = _round_re_to_2e_power(prop_Re_07)

    prop_k_T_coeffs, prop_k_Q_coeffs = _estimate_bseries_poly_coeffs(
        prop_PD=prop_PD,
        prop_AEA0=prop_AEA0,
        prop_Z=prop_Z,
        Rn=prop_Re_07,
    )

    def _eval_poly(coeffs, x):
        """Evaluate polynomial with coefficients in ascending powers."""

        result = 0.0
        for c in reversed(coeffs[1:]):
            result = (result + c) * x
        return result + coeffs[0]

    prop_k_T = _eval_poly(prop_k_T_coeffs, prop_I_ra)
    prop_k_Q = _eval_poly(prop_k_Q_coeffs, prop_I_ra)

    # Thrust and torque definitions (standard):
    #   K_T = T / (ρ n^2 D^4)
    #   K_Q = Q / (ρ n^2 D^5)
    prop_T = water_rho * (prop_n**2) * (prop_D**4) * prop_k_T
    prop_Q = water_rho * (prop_n**2) * (prop_D**5) * prop_k_Q

    prop_P_shaft = 2.0 * np.pi * prop_n * prop_Q

    # Open-water efficiency:
    #   η_0 = J K_T / (2π K_Q)
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
        "hull_C_B": hull_C_B,
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
    """Load and preprocess boat dataset used by hull ``C_T`` estimation.

    Parameters
    - filepath: str
        CSV path.

    Returns
    - pandas.DataFrame
        Dataframe with standardized columns (when present):
        - ``batt_v``: battery voltage [V]
        - ``batt_i_out``: battery output current [A]
        - ``pilot_d``: ESC duty cycle [-]
        - ``motor_w``: motor angular speed [rad/s]
        - ``mppts_p_in``: MPPT input power [W]
        - ``hull_u``: hull speed [m/s]

    Assumptions
    - If ``hull_u`` is not present, it is approximated from ``batt_v * pilot_d``.
      [ASSUMPTION] This is a project-specific surrogate relationship.

    Notes
    - Confirmed: the raw data column "Motor Angular Speed" is in [rad/s].
    """

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
            """Project-specific surrogate mapping from voltage to boat speed.

            Parameters
            - motor_v: float or array-like
                Voltage surrogate [V].
            - a: float
                Conversion coefficient with units [m/(s·V)].

            Returns
            - float or array-like
                Estimated speed [m/s].

            Assumptions
            - [ASSUMPTION] Linear mapping is valid in the dataset regime.
            """

            return a * motor_v

        df["hull_u"] = boat_speed_from_motor_v(df["batt_v"] * df["pilot_d"])

    return df
