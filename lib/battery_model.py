import numpy as np

from dataclasses import dataclass
from typeguard import typechecked

from lib.utils import naive_power, naive_energy


@dataclass(init=False)
class Battery:
    """Simple battery model, described by its efficiency, energy, and SOC.

    Attributes:
    ----------
    soc_0 (float)
        Initial State Of Charge, between 0.0 and 1.0.
    minimum_soc (float)
        Minimum State Of Charge, between 0.0 and 1.0.
    efficiency (float)
        Charge and Discharge Efficiency, between 0.0 and 1.0.
    maximum_energy (float)
        Maximum energy to be stored in the battery
    maximum_power (float)
        Maximum power to be used during charge and discharge

    """

    efficiency: float
    energy: float
    soc: float
    minimum_soc: float
    maximum_energy: float
    minimum_energy: float
    maximum_power: float

    @typechecked
    def __init__(
        self,
        soc_0: float,
        minimum_soc: float,
        efficiency: float,
        maximum_energy: float,
        maximum_power: float,
    ):
        self.efficiency = efficiency
        self.soc = soc_0
        self.minimum_soc = minimum_soc
        self.energy = soc_0 * maximum_energy
        self.maximum_energy = maximum_energy
        self.minimum_energy = maximum_energy * minimum_soc
        self.maximum_power = maximum_power

    @typechecked
    def _charge(self, dt: float, power: float) -> float:
        energy = naive_energy(power=power, time=dt, timebase=3600)
        self.energy += energy * self.efficiency

        if self.energy > self.maximum_energy:
            exceeded_energy = self.energy - self.maximum_energy
            self.energy -= exceeded_energy
            exceeded_power = naive_power(exceeded_energy, dt, timebase=3600)
            return power - exceeded_power

        return power

    @typechecked
    def _discharge(self, dt: float, power: float) -> float:
        energy = naive_energy(power=power, time=dt, timebase=3600)
        self.energy -= energy * self.efficiency

        if self.energy < self.minimum_energy:
            exceeded_energy = self.minimum_energy - self.energy
            self.energy += exceeded_energy
            exceeded_power = naive_power(exceeded_energy, dt, timebase=3600)
            return power - exceeded_power

        return power

    @typechecked
    def solve(self, dt: float, target_power: float) -> float:
        """Solves battery output power for a given target (input) power.

        Args:
            dt (float): Duration of the event (in seconds)
            target_power (float): Target (input) power of the event

        Returns:
            float: battery output power (in watts)
        """
        power = 0.0
        if target_power > 0:
            if self.soc < 1:
                power = self._charge(dt, abs(target_power))
        else:
            if self.soc > 0:
                power = -self._discharge(dt, abs(target_power))

        if np.abs(power) > self.maximum_power:
            power = np.sign(power) * self.maximum_power

        self.soc = self.energy / self.maximum_energy
        return power
