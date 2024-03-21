from dataclasses import dataclass
from typeguard import typechecked

import numpy as np


@dataclass
class MPPT:
    """Simple MPPT model, described by its efficiency and maximum output power.

    Attributes:
    ----------
    efficiency (np.float64)
        MPPT Efficiency, between 0.0 and 1.0.
    maximum_output_power (np.float64)
        MPPT maximum output power, in watts.

    """

    efficiency_constant: np.float64
    maximum_output_power: np.float64

    @typechecked
    def solve_output(self, input_power: np.float64) -> np.float64:
        """Solves MPPT output power (in watts) for a given input power (in watts).

        Args:
            input_power (np.float64): input power (in watts)

        Returns:
            np.float64: MPPT output power (in watts)
        """
        if input_power <= 0.0:
            return np.float64(0.0)

        output_power = input_power * self.efficiency(input_power)
        if output_power > self.maximum_output_power:
            output_power = self.maximum_output_power

        return output_power

    @typechecked
    def efficiency(self, input_power: np.float64) -> np.float64:
        return input_power * self.efficiency_constant
