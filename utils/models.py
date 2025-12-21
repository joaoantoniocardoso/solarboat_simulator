import numpy as np


def eval_poly(coeffs, x):
    """Evaluate polynomial coefficients using Horner's method."""
    coeffs = list(coeffs)
    if len(coeffs) == 0:
        return 0.0
    if len(coeffs) == 1:
        return coeffs[0]

    result = 0.0
    for c in reversed(coeffs[1:]):
        result = result * x + c
    return result * x + coeffs[0]


def lut_interp(lut: np.array, x: float) -> float:
    """
    Linear interpolation on a 1D LUT where entries are evenly spaced from 0 to 1.
    """
    n = len(lut)
    if n == 0:
        raise ValueError("LUT cannot be empty.")
    if n == 1:
        return float(lut[0])

    pos = x * (n - 1)
    lower_index = int(np.floor(pos))

    if lower_index <= 0:
        lower_index = 0
    if lower_index >= n - 1:
        lower_index = n - 2

    interp_frac = pos - lower_index
    return float(
        lut[lower_index] * (1 - interp_frac) + lut[lower_index + 1] * interp_frac
    )
