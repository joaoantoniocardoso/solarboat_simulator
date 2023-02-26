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
    def solve_output(self, irradiation: np.float64) -> np.float64:
        """Solves panel output power (in watts) for a given irradiation (in watt per square meter).

        Args:
            irradiation (np.float64): Solar irradiation (in watt per square meter)

        Returns:
            np.float64: panel output power (in watts)
        """
        if irradiation <= 0.0:
            return np.float64(0.0)

        input_power = irradiation * self.area

        output_power = input_power * self.efficiency
        if output_power > self.maximum_output_power:
            output_power = self.maximum_output_power

        return output_power


class Panel2(Panel):
    """Example of how to create a wrap around the simple model with a more complex efficnecy"""

    @typechecked
    def solve_output(self, irradiation: np.float64) -> np.float64:
        self.efficiency = irradiation * 0.2 + 0.8

        return Panel.solve_output(self, irradiation)
