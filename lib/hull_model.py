from dataclasses import dataclass
from typeguard import typechecked


@dataclass
class Hull:
    speed_over_power_constant: float
    exponential_factor: float

    @typechecked
    def solve_output(self, propulsion_power: float) -> float:
        speed = self.speed_over_power_constant * (
            propulsion_power**self.exponential_factor
        )

        return speed
