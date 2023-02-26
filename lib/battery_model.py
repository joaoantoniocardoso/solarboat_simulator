import numpy as np

from dataclasses import dataclass
from typeguard import typechecked

from lib.utils import naive_power, naive_energy


@dataclass(init=False)
class Battery:
    """Simple battery model, described by its efficiency, energy, and SOC.

    Attributes:
    ----------
    soc_0 (np.float64)
        Initial State Of Charge, between 0.0 and 1.0.
    minimum_soc (np.float64)
        Minimum State Of Charge, between 0.0 and 1.0.
    efficiency (np.float64)
        Charge and Discharge Efficiency, between 0.0 and 1.0.
    maximum_energy (np.float64)
        Maximum energy to be stored in the battery
    maximum_power (np.float64)
        Maximum power to be used during charge and discharge

    """

    efficiency: np.float64
    energy: np.float64
    soc: np.float64
    minimum_soc: np.float64
    maximum_energy: np.float64
    minimum_energy: np.float64
    maximum_power: np.float64

    @typechecked
    def __init__(
        self,
        soc_0: np.float64,
        minimum_soc: np.float64,
        efficiency: np.float64,
        maximum_energy: np.float64,
        maximum_power: np.float64,
    ):
        self.efficiency = efficiency
        self.soc = soc_0
        self.minimum_soc = minimum_soc
        self.energy = soc_0 * maximum_energy
        self.maximum_energy = maximum_energy
        self.minimum_energy = maximum_energy * minimum_soc
        self.maximum_power = maximum_power

    @typechecked
    def _charge(self, dt: np.float64, power: np.float64) -> np.float64:
        energy = naive_energy(power=power, time=dt, timebase=np.float64(3600))
        self.energy += energy * self.efficiency

        if self.energy > self.maximum_energy:
            exceeded_energy = self.energy - self.maximum_energy
            self.energy -= exceeded_energy
            exceeded_power = naive_power(exceeded_energy, dt, timebase=np.float64(3600))
            return power - exceeded_power

        return power

    @typechecked
    def _discharge(self, dt: np.float64, power: np.float64) -> np.float64:
        energy = naive_energy(power=power, time=dt, timebase=np.float64(3600))
        self.energy -= energy * self.efficiency

        if self.energy < self.minimum_energy:
            exceeded_energy = self.minimum_energy - self.energy
            self.energy += exceeded_energy
            exceeded_power = naive_power(exceeded_energy, dt, timebase=np.float64(3600))
            return power - exceeded_power

        return power

    @typechecked
    def solve(self, dt: np.float64, target_power: np.float64) -> np.float64:
        """Solves battery output power for a given target (input) power.

        Args:
            dt (np.float64): Duration of the event (in seconds)
            target_power (np.float64): Target (input) power of the event

        Returns:
            np.float64: battery output power (in watts)
        """
        power = np.float64(0.0)
        if target_power > 0:
            if self.soc < 1:
                power = self._charge(dt, np.abs(target_power))
        else:
            if self.soc > 0:
                power = -self._discharge(dt, np.abs(target_power))

        if np.abs(power) > self.maximum_power:
            power = np.sign(power) * self.maximum_power

        self.soc = self.energy / self.maximum_energy
        return power
