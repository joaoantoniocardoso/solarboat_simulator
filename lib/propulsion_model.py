from dataclasses import dataclass
from typeguard import typechecked


@dataclass
class Propulsion:
    efficiency: float
    maximum_input_power: float

    @typechecked
    def solve_input(self, input_power: float) -> float:
        if input_power > self.maximum_input_power:
            input_power = self.maximum_input_power

        return input_power

    @typechecked
    def solve_output(self, input_power: float) -> float:
        output_power = input_power * self.efficiency
        return output_power
