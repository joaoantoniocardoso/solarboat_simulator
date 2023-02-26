from dataclasses import dataclass
from typeguard import typechecked

import numpy as np


@dataclass
class Hull:
    speed_over_power_constant: np.float64
    exponential_factor: np.float64

    @typechecked
    def solve_output(self, propulsion_power: np.float64) -> np.float64:
        speed = self.speed_over_power_constant * (
            propulsion_power**self.exponential_factor
        )

        return speed
