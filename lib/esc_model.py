import numpy as np

from dataclasses import dataclass
from typeguard import typechecked


@dataclass
class ESC:
    """Simple ESC model, described by its efficiency and maximum output power.

    Attributes:
    ----------
    efficiency (np.float64)
        ESC Efficiency, between 0.0 and 1.0.
    maximum_output_power (np.float64)
        ESC maximum output power, in watts.

    """

    efficiency: np.float64
    maximum_input_power: np.float64

    @typechecked
    def solve_input(self, throttle: np.float64) -> np.float64:
        """Solves ESC input power (in watts) for a given throttle (percentage)

        Args:
            input_power (np.float64): ESC throttle, between 0.0 and 1.0.

        Returns:
            np.float64: ESC input power (in watts)
        """
        throttle = np.clip(throttle, 0, 1)

        input_power = throttle * self.maximum_input_power
        if input_power > self.maximum_input_power:
            input_power = self.maximum_input_power

        return input_power

    @typechecked
    def solve_output(self, input_power: np.float64) -> np.float64:
        """Solves ESC output power (in watts) for a given input power (in watts)

        Args:
            input_power (np.float64): ESC input power (in watts)

        Returns:
            np.float64: ESC output power (in watts)
        """
        output_power = input_power * self.efficiency
        return output_power
