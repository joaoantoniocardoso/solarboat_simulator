"""Holtrop(-Mennen) calm-water resistance (displacement ships).

This module implements the Holtrop regression formulas as presented in:
- Molland, Turnock & Hudson (2017), Ship Resistance and Propulsion, Cambridge.
  - Total resistance decomposition: Eq. (10.24)
  - Wave resistance: Eq. (10.25)–(10.27)
  - Waterline entrance angle: Eq. (10.28)
  - Length of run: Eq. (10.29a)
  - Bulb and transom terms: Eq. (10.30)–(10.33)
  - Correlation allowance (model–ship correlation): Eq. (10.34)
- Form factor regression (Holtrop): Molland et al. (2017), Eq. (4.23)–(4.24)
- ITTC-1957 friction line: Molland et al. (2017), Eq. (4.15)

Scope
-----
- Resistance only (no propulsion factors).
- Designed for use with meshmagick-derived hydrostatics in this repo.

Important
---------
Many Holtrop inputs (bulb, transom, appendages, projected air area) may be unknown
for the solar boat hull. Those terms are supported but default to zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class HoltropInputs:
    # Principal dimensions
    LWL: float  # [m]
    B: float  # [m]
    T: float  # [m] (mean draft)
    TA: float  # [m] draft aft
    TF: float  # [m] draft forward
    volume: float  # [m^3] displacement volume (∇)

    # Coefficients
    CP: float
    CM: float
    CWP: float
    LCB_percent: float  # [% LWL], positive forward of midship

    # Wetted surface
    S: float  # [m^2]

    # Optional geometry for extra terms
    AT: float = 0.0  # [m^2] immersed transom area at rest
    ABT: float = 0.0  # [m^2] transverse area of bulbous bow
    hB: float = 0.0  # [m] center height of ABT above keel

    # Stern shape parameter
    Cstern: float = 0.0

    # Environment
    rho: float = 1023.0  # [kg/m^3]
    nu: float = 1.0e-6  # [m^2/s]
    g: float = 9.80665

    def validate(self) -> None:
        vals = {
            "LWL": self.LWL,
            "B": self.B,
            "T": self.T,
            "TA": self.TA,
            "TF": self.TF,
            "volume": self.volume,
            "CP": self.CP,
            "CM": self.CM,
            "CWP": self.CWP,
            "S": self.S,
            "rho": self.rho,
            "nu": self.nu,
        }
        for k, v in vals.items():
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f"Invalid {k}={v}")


def ittc57_cf(Re: np.ndarray) -> np.ndarray:
    """ITTC-1957 model–ship correlation line (Molland 2017, Eq. 4.15)."""

    Re = np.asarray(Re, dtype=float)
    Re = np.maximum(Re, 1.0)
    return 0.075 / (np.log10(Re) - 2.0) ** 2


def froude_number(V: np.ndarray, LWL: float, g: float) -> np.ndarray:
    return np.asarray(V, dtype=float) / np.sqrt(g * LWL)


def reynolds_number(V: np.ndarray, LWL: float, nu: float) -> np.ndarray:
    return np.asarray(V, dtype=float) * LWL / nu


def length_of_run(inputs: HoltropInputs) -> float:
    """Length of run LR (Molland 2017, Eq. 4.24 / Eq. 10.29a)."""

    CP = float(inputs.CP)
    LCB = float(inputs.LCB_percent)

    denom = 4.0 * CP - 1.0
    if denom <= 0:
        raise ValueError(f"Invalid Holtrop LR denominator: 4*CP-1={denom} (CP={CP})")

    LR = float(inputs.LWL * (1.0 - CP + 0.06 * CP * LCB / denom))
    if not np.isfinite(LR) or LR <= 0:
        raise ValueError(f"Invalid Holtrop length of run LR={LR} (CP={CP}, LCB%={LCB})")

    return LR


def entrance_half_angle_deg(inputs: HoltropInputs, LR: float) -> float:
    """Half angle of entrance iE in degrees (Molland 2017, Eq. 10.28)."""

    L = inputs.LWL
    B = inputs.B
    CP = inputs.CP
    CWP = inputs.CWP
    LCB = inputs.LCB_percent
    V = inputs.volume

    a = -(
        (L / B) ** 0.80856
        * (1.0 - CWP) ** 0.30484
        * (1.0 - CP - 0.0225 * LCB) ** 0.6367
        * (LR / B) ** 0.34574
        * (100.0 * V / (L**3)) ** 0.16302
    )
    return float(1.0 + 89.0 * np.exp(a))


def form_factor_1k(inputs: HoltropInputs, LR: float) -> float:
    """Form factor (1+k) regression (Molland 2017, Eq. 4.23)."""

    if not np.isfinite(LR) or LR <= 0:
        raise ValueError(f"Invalid LR for form factor: LR={LR}")

    L = float(inputs.LWL)
    B = float(inputs.B)
    T = float(inputs.T)
    V = float(inputs.volume)
    CP = float(inputs.CP)

    if (1.0 - CP) <= 0:
        raise ValueError(f"Invalid CP for form factor: CP={CP} (requires CP<1)")

    one_k = 0.93 + 0.487118 * (1.0 + 0.011 * inputs.Cstern) * (B / L) ** 1.06806 * (
        T / L
    ) ** 0.46106 * (L / LR) ** 0.121563 * (L**3 / V) ** 0.36486 * (1.0 - CP) ** (
        -0.604247
    )
    return float(one_k)


def wave_resistance(
    inputs: HoltropInputs, V: np.ndarray, *, Fr: Optional[np.ndarray] = None
) -> np.ndarray:
    """Wave resistance RW (Molland 2017, Eq. 10.25–10.27)."""

    V = np.asarray(V, dtype=float)
    if Fr is None:
        Fr = froude_number(V, inputs.LWL, inputs.g)

    L = inputs.LWL
    B = inputs.B
    T = inputs.T
    TF = inputs.TF
    CM = inputs.CM
    CP = inputs.CP
    vol = inputs.volume

    LR = length_of_run(inputs)
    iE = entrance_half_angle_deg(inputs, LR)

    # Table coefficients
    BL = B / L
    if BL <= 0.11:
        c7 = 0.229577 * BL ** (1.0 / 3.0)
    elif BL <= 0.25:
        c7 = BL
    else:
        c7 = 0.5 - 0.0625 * (L / B)

    c1 = 2223105.0 * c7**3.78613 * (T / B) ** 1.07961 * (90.0 - iE) ** (-1.37565)

    # bulb / transom factors inside c3 and c5
    ABT = float(inputs.ABT)
    hB = float(inputs.hB)
    AT = float(inputs.AT)

    denom = B * T * (0.31 * np.sqrt(max(ABT, 0.0)) + TF - hB)
    if denom <= 0 or ABT <= 0:
        c3 = 0.0
    else:
        c3 = 0.56 * (ABT**1.5) / denom

    c2 = float(np.exp(-1.89 * np.sqrt(max(c3, 0.0))))

    c5 = 1.0
    if AT > 0:
        c5 = float(1.0 - 0.8 * AT / (B * T * CM))

    # c15
    L3_over_V = (L**3) / vol
    if L3_over_V <= 512.0:
        c15 = -1.69385
    elif L3_over_V <= 1726.91:
        c15 = -1.69385 + (L / (vol ** (1.0 / 3.0)) - 8.0) / 2.36
    else:
        c15 = 0.0

    # c16
    if CP <= 0.8:
        c16 = 8.07981 * CP - 13.8673 * CP**2 + 6.984388 * CP**3
    else:
        c16 = 1.73014 - 0.7067 * CP

    d = -0.9

    # lambda
    if (L / B) <= 12.0:
        lam = 1.446 * CP - 0.03 * (L / B)
    else:
        lam = 1.446 * CP - 0.36

    # m1, m3
    m1 = (
        0.0140407 * (L / T)
        - 1.75254 * (vol ** (1.0 / 3.0) / L)
        - 4.79323 * (B / L)
        - c16
    )
    m3 = -7.2035 * (B / L) ** 0.326869 * (T / B) ** 0.605375

    def m4(fr: np.ndarray) -> np.ndarray:
        fr = np.asarray(fr, dtype=float)
        return 0.4 * c15 * np.exp(-0.034 * fr ** (-3.29))

    rho = inputs.rho
    g = inputs.g

    def rw_a(fr: np.ndarray) -> np.ndarray:
        fr = np.asarray(fr, dtype=float)
        return (
            c1
            * c2
            * c5
            * vol
            * rho
            * g
            * np.exp(m1 * fr**d + m4(fr) * np.cos(lam * fr ** (-2.0)))
        )

    # c17 depends on CM and vol
    c17 = (
        6919.3 * CM ** (-1.3346) * (vol / (L**3)) ** 2.00977 * (L / B - 2.0) ** 1.40692
    )

    def rw_b(fr: np.ndarray) -> np.ndarray:
        fr = np.asarray(fr, dtype=float)
        return (
            c17
            * c2
            * c5
            * vol
            * rho
            * g
            * np.exp(m3 * fr**d + m4(fr) * np.cos(lam * fr ** (-2.0)))
        )

    RW = np.zeros_like(V, dtype=float)

    # evaluate piecewise
    mask_a = Fr <= 0.4
    mask_b = Fr > 0.55
    mask_mid = (~mask_a) & (~mask_b)

    if np.any(mask_a):
        RW[mask_a] = rw_a(Fr[mask_a])

    if np.any(mask_b):
        RW[mask_b] = rw_b(Fr[mask_b])

    if np.any(mask_mid):
        rwa_04 = float(rw_a(np.array([0.4]))[0])
        rwb_055 = float(rw_b(np.array([0.55]))[0])
        RW[mask_mid] = rwa_04 + (20.0 * Fr[mask_mid] - 8.0) * (rwb_055 - rwa_04) / 3.0

    return RW


def bulb_resistance(inputs: HoltropInputs, V: np.ndarray) -> np.ndarray:
    """Bulbous bow resistance RB (Molland 2017, Eq. 10.30–10.31)."""

    ABT = float(inputs.ABT)
    if ABT <= 0:
        return np.zeros_like(V, dtype=float)

    V = np.asarray(V, dtype=float)
    rho = inputs.rho
    g = inputs.g

    TF = inputs.TF
    hB = inputs.hB

    PB = 0.56 * np.sqrt(ABT) / (TF - 1.5 * hB)

    # Fri as given in Molland 2017 Eq. (10.31)
    Fri = V / np.sqrt(g * (TF - hB - 0.25 * np.sqrt(ABT)) + 0.15 * V**2)

    RB = (
        0.11
        * rho
        * g
        * (ABT**1.5)
        * np.exp(-3.0 * PB ** (-2.0))
        * (Fri**3)
        / (1.0 + Fri**2)
    )
    return RB


def transom_resistance(inputs: HoltropInputs, V: np.ndarray) -> np.ndarray:
    """Immersed transom resistance RTR (Molland 2017, Eq. 10.32–10.33)."""

    AT = float(inputs.AT)
    if AT <= 0:
        return np.zeros_like(V, dtype=float)

    V = np.asarray(V, dtype=float)
    rho = inputs.rho
    g = inputs.g

    B = inputs.B
    CWP = inputs.CWP

    FrT = V / np.sqrt(2.0 * g * AT / (B + B * CWP))
    c6 = np.where(FrT < 5.0, 0.2 * (1.0 - 0.2 * FrT), 0.0)
    return 0.5 * rho * V**2 * AT * c6


def correlation_allowance_CA(inputs: HoltropInputs, *, c2: float) -> float:
    """Correlation allowance coefficient CA (Molland 2017, Eq. 10.34)."""

    L = inputs.LWL
    CB = float(inputs.volume / (inputs.B * inputs.T * inputs.LWL))

    c4 = inputs.TF / L if (inputs.TF / L) <= 0.04 else 0.04

    # Molland 2017 Eq. (10.34)
    CA = (
        0.006 * (L + 100.0) ** (-0.16)
        - 0.00205
        + 0.003 * np.sqrt(L / 7.5) * (CB**4) * c2 * (0.04 - c4)
    )
    return float(CA)


def correlation_resistance(
    inputs: HoltropInputs, V: np.ndarray, *, c2: float
) -> np.ndarray:
    """Model–ship correlation resistance RA (Molland 2017, Eq. 10.34)."""

    V = np.asarray(V, dtype=float)
    CA = correlation_allowance_CA(inputs, c2=c2)
    return 0.5 * inputs.rho * inputs.S * V**2 * CA


def holtrop_total_resistance(
    inputs: HoltropInputs, V: np.ndarray
) -> Dict[str, np.ndarray]:
    """Compute Holtrop resistance components and total RT.

    Returns a dict with keys: RF, RAPP, RW, RB, RTR, RA, RT.
    """

    inputs.validate()
    V = np.asarray(V, dtype=float)

    Re = reynolds_number(V, inputs.LWL, inputs.nu)
    Cf = ittc57_cf(Re)

    RF = 0.5 * inputs.rho * V**2 * inputs.S * Cf

    LR = length_of_run(inputs)
    one_k = form_factor_1k(inputs, LR)

    # Wave resistance and c2 coupling
    # We reuse the c2 computed inside wave_resistance; recompute it here for CA.
    denom = (
        inputs.B
        * inputs.T
        * (0.31 * np.sqrt(max(inputs.ABT, 0.0)) + inputs.TF - inputs.hB)
    )
    if denom <= 0 or inputs.ABT <= 0:
        c3 = 0.0
    else:
        c3 = 0.56 * (inputs.ABT**1.5) / denom
    c2 = float(np.exp(-1.89 * np.sqrt(max(c3, 0.0))))

    RW = wave_resistance(inputs, V)
    RB = bulb_resistance(inputs, V)
    RTR = transom_resistance(inputs, V)
    RA = correlation_resistance(inputs, V, c2=c2)

    # Appendage resistance not modeled here (set to zero by default)
    RAPP = np.zeros_like(V, dtype=float)

    RT = RF * one_k + RAPP + RW + RB + RTR + RA

    return {
        "Re": Re,
        "Cf": Cf,
        "RF": RF,
        "one_k": np.full_like(V, one_k, dtype=float),
        "RW": RW,
        "RB": RB,
        "RTR": RTR,
        "RA": RA,
        "RAPP": RAPP,
        "RT": RT,
    }


def holtrop_inputs_from_hull_params(
    hull_params: Dict[str, float],
    *,
    rho: float,
    nu: float,
    g: float = 9.80665,
    Cstern: float = 0.0,
    AT: float = 0.0,
    ABT: float = 0.0,
    hB: float = 0.0,
    use_end_drafts: bool = False,
) -> HoltropInputs:
    LWL = float(hull_params["LWL"])
    B = float(hull_params.get("BWL", hull_params.get("B", np.nan)))
    volume = float(hull_params["disp_volume_m3"])

    CP = float(hull_params["CP"])
    CM = float(hull_params["CM"])
    CWP = float(hull_params["CWP"])
    LCB_percent = float(hull_params["lcb_percent"])

    S = float(
        hull_params.get(
            "wet_surface_area_interp", hull_params.get("wet_surface_area", np.nan)
        )
    )

    T = float(hull_params.get("draft_keel_m", hull_params.get("draft_mean_m", np.nan)))
    if not np.isfinite(T) or T <= 0:
        raise ValueError(f"Invalid draft in hull_params: T={T}")

    if use_end_drafts:
        TA = float(hull_params.get("draft_aft_m", np.nan))
        TF = float(hull_params.get("draft_fwd_m", np.nan))
        if not np.isfinite(TA) or TA <= 0:
            TA = T
        if not np.isfinite(TF) or TF <= 0:
            TF = T
    else:
        TA = T
        TF = T

    return HoltropInputs(
        LWL=LWL,
        B=B,
        T=T,
        TA=TA,
        TF=TF,
        volume=volume,
        CP=CP,
        CM=CM,
        CWP=CWP,
        LCB_percent=LCB_percent,
        S=S,
        AT=float(AT),
        ABT=float(ABT),
        hB=float(hB),
        Cstern=float(Cstern),
        rho=float(rho),
        nu=float(nu),
        g=float(g),
    )
