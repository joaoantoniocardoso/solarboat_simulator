import numpy as np

from dataclasses import dataclass
from typeguard import typechecked
import warnings

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
    maximum_soc: np.float64
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
        self.maximum_soc = np.float64(1.0)
        self.energy = soc_0 * maximum_energy
        self.maximum_energy = maximum_energy
        self.minimum_energy = maximum_energy * minimum_soc
        self.maximum_power = maximum_power

    @typechecked
    def solve(self, dt: np.float64, target_power: np.float64) -> np.float64:
        """Solves battery output power for a given target (input) power.

        Args:
            dt (np.float64): Duration of the event (in seconds)
            target_power (np.float64): Target (input) power of the event

        Returns:
            np.float64: battery output power (in watts)
        """

        if dt < 1e-9:
            raise ValueError("Parameter 'dt' must be >= 1e-9.")

        if target_power == 0:
            return target_power

        target_delta_energy = naive_energy(
            power=target_power, time=dt, timebase=np.float64(3600)
        )

        lost_energy = naive_energy(
            power=np.abs(target_power) * (1 - self.efficiency),
            time=dt,
            timebase=np.float64(3600),
        )

        exceeded_energy = np.float64(0)
        if target_delta_energy > 0:
            exceeded_energy = np.max(
                [
                    np.float64(0),
                    target_delta_energy - (self.maximum_energy - self.energy),
                ]
            )
        else:
            exceeded_energy = np.min(
                [
                    np.float64(0),
                    (self.energy - self.minimum_energy) + target_delta_energy,
                ]
            )

        exceeded_power = naive_power(exceeded_energy, dt, timebase=np.float64(3600))

        power: np.float64 = target_power - exceeded_power

        if abs(power) > self.maximum_power:
            warnings.warn(
                "power greater than self.maximum_power, its value will be saturated to self.maximum_power"
            )
            power = np.sign(power) * self.maximum_power

        self.energy += (
            naive_energy(power=power, time=dt, timebase=np.float64(3600))
        ) - lost_energy
        self.soc = self.energy / self.maximum_energy

        return power
