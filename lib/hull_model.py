from dataclasses import dataclass
from typeguard import typechecked

import numpy as np


@dataclass
class Hull:
    resistance_coefficient: np.float64

    @typechecked
    def solve_output(self, propulsion_power: np.float64) -> np.float64:
        speed = (self.resistance_coefficient * propulsion_power) ** (1 / 3)

        return speed
