from dataclasses import dataclass
from typeguard import typechecked


@dataclass
class Panel:
    """Simple photovoltaic panel model, described by its efficiency, area and maximum output power.

    Attributes:
    ----------
    efficiency (float)
        Panel Efficiency, between 0.0 and 1.0.
    area (float)
        Panel surface area, in meters.
    maximum_output_power (float)
        Panel maximum output power, in watts.

    """

    efficiency: float
    area: float
    maximum_output_power: float

    @typechecked
    def solve_output(self, irradiation: float) -> float:
        """Solves panel output power (in watts) for a given irradiation (in watt per square meter).

        Args:
            irradiation (float): Solar irradiation (in watt per square meter)

        Returns:
            float: panel output power (in watts)
        """
        if irradiation <= 0.0:
            return 0.0

        input_power = irradiation * self.area

        output_power = input_power * self.efficiency
        if output_power > self.maximum_output_power:
            output_power = self.maximum_output_power

        return output_power


class Panel2(Panel):
    """Example of how to create a wrap around the simple model with a more complex efficnecy"""

    @typechecked
    def solve_output(self, irradiation: float) -> float:
        self.efficiency = irradiation * 0.2 + 0.8

        return Panel.solve_output(self, irradiation)
