import numpy as np

from dataclasses import dataclass
from typeguard import typechecked


@dataclass
class ESC:
    """Simple ESC model, described by its efficiency and maximum output power.

    Attributes:
    ----------
    efficiency (float)
        ESC Efficiency, between 0.0 and 1.0.
    maximum_output_power (float)
        ESC maximum output power, in watts.

    """

    efficiency: float
    maximum_input_power: float

    @typechecked
    def solve_input(self, throttle: float) -> float:
        """Solves ESC input power (in watts) for a given throttle (percentage)

        Args:
            input_power (float): ESC throttle, between 0.0 and 1.0.

        Returns:
            float: ESC input power (in watts)
        """
        throttle = np.clip(throttle, 0, 1)

        input_power = throttle * self.maximum_input_power
        if input_power > self.maximum_input_power:
            input_power = self.maximum_input_power

        return input_power

    @typechecked
    def solve_output(self, input_power: float) -> float:
        """Solves ESC output power (in watts) for a given input power (in watts)

        Args:
            input_power (float): ESC input power (in watts)

        Returns:
            float: ESC output power (in watts)
        """
        output_power = input_power * self.efficiency
        return output_power
