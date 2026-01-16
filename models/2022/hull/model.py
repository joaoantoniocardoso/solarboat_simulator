from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Union

import numpy as np

ArrayLike = Union[float, int, np.ndarray]

CSV_NAME_2022 = "data/motor_voltage_speed_power_model_2022.csv"
DEFAULT_MOTOR_V_MIN = 0.0
DEFAULT_MOTOR_V_MAX = 60.0


def _load_power_model_csv(csv_path: Path) -> tuple[float, float, float, float, float]:
    if not csv_path.exists():
        raise FileNotFoundError(str(csv_path))

    data = np.genfromtxt(
        csv_path, delimiter=",", names=True, dtype=float, encoding=None
    )
    if getattr(data, "dtype", None) is None or data.dtype.names is None:
        raise ValueError(f"Invalid CSV format: {csv_path}")

    names = set(data.dtype.names)
    if not {"k", "a", "v0"}.issubset(names):
        raise ValueError(f"CSV must contain columns k,a,v0: {csv_path}")

    k = float(np.asarray(data["k"]).reshape(-1)[0])
    a = float(np.asarray(data["a"]).reshape(-1)[0])
    v0 = float(np.asarray(data["v0"]).reshape(-1)[0])

    if not (np.isfinite(k) and np.isfinite(a) and np.isfinite(v0)):
        raise ValueError(f"Non-finite parameters in CSV: {csv_path}")
    if k <= 0 or a <= 0 or v0 < 0:
        raise ValueError(f"Invalid parameters in CSV: {csv_path}")

    motor_v_min = DEFAULT_MOTOR_V_MIN
    motor_v_max = DEFAULT_MOTOR_V_MAX
    if "motor_v_min" in names:
        motor_v_min = float(np.asarray(data["motor_v_min"]).reshape(-1)[0])
    if "motor_v_max" in names:
        motor_v_max = float(np.asarray(data["motor_v_max"]).reshape(-1)[0])

    if not (np.isfinite(motor_v_min) and np.isfinite(motor_v_max)):
        raise ValueError(f"Non-finite motor_v range in CSV: {csv_path}")
    if motor_v_min < 0 or motor_v_max <= motor_v_min:
        raise ValueError(f"Invalid motor_v range in CSV: {csv_path}")

    return k, a, v0, motor_v_min, motor_v_max


@dataclass(frozen=True)
class MotorVoltageSpeedModel:
    k: float
    a: float
    v0: float
    motor_v_min: float = DEFAULT_MOTOR_V_MIN
    motor_v_max: float = DEFAULT_MOTOR_V_MAX

    def speed_kmh_from_motor_v(
        self, motor_v: ArrayLike, *, clip: bool = False
    ) -> np.ndarray:
        mv = np.asarray(motor_v, dtype=float)
        x = mv
        if clip:
            x = np.maximum(x, 0.0)
        xeff = np.maximum(x - self.v0, 0.0)
        return self.k * np.power(xeff, self.a)

    def speed_mps_from_motor_v(
        self, motor_v: ArrayLike, *, clip: bool = False
    ) -> np.ndarray:
        return self.speed_kmh_from_motor_v(motor_v, clip=clip) / 3.6

    def speed_kmh_from_duty_and_batt_v(
        self, esc_dt: ArrayLike, batt_v: ArrayLike, *, clip: bool = False
    ) -> np.ndarray:
        esc_dt_arr = np.asarray(esc_dt, dtype=float)
        batt_v_arr = np.asarray(batt_v, dtype=float)
        return self.speed_kmh_from_motor_v(esc_dt_arr * batt_v_arr, clip=clip)

    def motor_v_from_speed_kmh(self, speed_kmh: ArrayLike) -> np.ndarray:
        v = np.asarray(speed_kmh, dtype=float)
        out = np.full_like(v, np.nan, dtype=float)
        mask = np.isfinite(v) & (v >= 0)
        if np.any(mask):
            out[mask] = self.v0 + np.power(v[mask] / self.k, 1.0 / self.a)
        return out


@lru_cache(maxsize=1)
def get_2022_motor_voltage_speed_map() -> MotorVoltageSpeedModel:
    csv_path = Path(__file__).resolve().parent / CSV_NAME_2022
    k, a, v0, motor_v_min, motor_v_max = _load_power_model_csv(csv_path)
    return MotorVoltageSpeedModel(
        k=k, a=a, v0=v0, motor_v_min=motor_v_min, motor_v_max=motor_v_max
    )
