from dataclasses import dataclass
from typeguard import typechecked


debug = print


@dataclass
class Motor:
    efficiency: float
    maximum_input_power: float

    @typechecked
    def solve_input(self, input_power: float) -> float:
        if input_power > self.maximum_input_power:
            input_power = self.maximum_input_power
        elif input_power < 0:
            debug(f"[Motor] input_power < 0: {input_power=}")
            input_power = 0

        return input_power

    @typechecked
    def solve_output(self, input_power: float) -> float:
        if input_power < 0:
            debug(f"[Motor] input_power < 0: {input_power=}")
            input_power = 0

        output_power = input_power * self.efficiency
        return output_power