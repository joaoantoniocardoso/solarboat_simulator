from dataclasses import dataclass
from typeguard import typechecked

import numpy as np

debug = print


@dataclass
class Motor:
    efficiency: np.float64
    maximum_input_power: np.float64

    @typechecked
    def solve_input(self, input_power: np.float64) -> np.float64:
        if input_power > self.maximum_input_power:
            input_power = self.maximum_input_power
        elif input_power < 0:
            debug(f"[Motor] input_power < 0: {input_power=}")
            input_power = np.float64(0)

        return input_power

    @typechecked
    def solve_output(self, input_power: np.float64) -> np.float64:
        if input_power < 0:
            debug(f"[Motor] input_power < 0: {input_power=}")
            input_power = np.float64(0)

        output_power = input_power * self.efficiency
        return output_power
