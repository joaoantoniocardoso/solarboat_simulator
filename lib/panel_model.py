from dataclasses import dataclass
from typeguard import typechecked

import numpy as np


@dataclass
class Panel:
    """Simple photovoltaic panel model, described by its efficiency, area and maximum output power.

    Attributes:
    ----------
    efficiency (np.float64)
        Panel Efficiency, between 0.0 and 1.0.
    area (np.float64)
        Panel surface area, in meters.
    maximum_output_power (np.float64)
        Panel maximum output power, in watts.

    """

    efficiency: np.float64
    area: np.float64
    maximum_output_power: np.float64

    @typechecked
    def solve_input(self, irradiation: np.float64) -> np.float64:
        """Solves panel input power (in watts) for a given irradiation (in watt per square meter).

        Args:
            irradiation (np.float64): Solar irradiation (in watt per square meter)

        Returns:
            np.float64: panel input power (in watts)
        """

        if irradiation <= 0.0:
            return np.float64(0.0)

        input_power = irradiation * self.area

        return input_power

    @typechecked
    def solve_output(self, input_power: np.float64) -> np.float64:
        """Solves panel output power (in watts) for a given input_power (in watts).

        Args:
            input_power (np.float64): Solar input power (in watts)

        Returns:
            np.float64: panel output power (in watts)
        """
        if input_power <= 0.0:
            return np.float64(0.0)

        output_power = input_power * self.efficiency
        if output_power > self.maximum_output_power:
            output_power = self.maximum_output_power

        return output_power
