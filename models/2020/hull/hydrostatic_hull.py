import os
from typing import Dict

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


class HydrostaticHull:
    """
    Hydrostatic surrogate model initialized at a specific operating point
    (cog_x, disp_mass). Geometry (L, B) is treated like other interpolated
    quantities using the feather dataset, while total area is carried as
    a constant column.
    """

    def __init__(
        self,
        cog_x: float,
        disp_mass: float,
        total_area: float,
        feather_path: str = None,
    ):
        if feather_path is None:
            feather_path = os.path.join(
                os.path.dirname(__file__), "data", "meshmagick.feather"
            )

        self.cog_x = float(cog_x)
        self.disp_mass = float(disp_mass)
        self.total_area = float(total_area)

        self.df = self._load_dataset(feather_path)
        self._build_interpolators()

    def _load_dataset(self, path: str) -> pd.DataFrame:
        df = pd.read_feather(path)
        df = df[abs(df["angles_deg_y"]) < 10].reset_index(drop=True)
        return df

    def _build_interpolators(self):
        X = self.df[["cog_x", "disp_mass"]].values
        self._wet_interp = LinearNDInterpolator(
            X, self.df["wet_surface_area"].values, fill_value=np.nan
        )
        self._trim_interp = LinearNDInterpolator(
            X, self.df["angles_deg_y"].values, fill_value=np.nan
        )
        self._wet_nearest = NearestNDInterpolator(X, self.df["wet_surface_area"].values)
        self._trim_nearest = NearestNDInterpolator(X, self.df["angles_deg_y"].values)

        if "LWL" in self.df.columns:
            self._L_interp = LinearNDInterpolator(
                X, self.df["LWL"].values, fill_value=np.nan
            )
            self._L_nearest = NearestNDInterpolator(X, self.df["LWL"].values)
        elif "hull_L" in self.df.columns:
            self._L_interp = LinearNDInterpolator(
                X, self.df["hull_L"].values, fill_value=np.nan
            )
            self._L_nearest = NearestNDInterpolator(X, self.df["hull_L"].values)
        else:
            self._L_interp = self._L_nearest = None

        if "BWL" in self.df.columns:
            self._B_interp = LinearNDInterpolator(
                X, self.df["BWL"].values, fill_value=np.nan
            )
            self._B_nearest = NearestNDInterpolator(X, self.df["BWL"].values)
        elif "hull_B" in self.df.columns:
            self._B_interp = LinearNDInterpolator(
                X, self.df["hull_B"].values, fill_value=np.nan
            )
            self._B_nearest = NearestNDInterpolator(X, self.df["hull_B"].values)
        else:
            self._B_interp = self._B_nearest = None

        if "AWP" in self.df.columns:
            self._awp_interp = LinearNDInterpolator(
                X, self.df["AWP"].values, fill_value=np.nan
            )
            self._awp_nearest = NearestNDInterpolator(X, self.df["AWP"].values)
        else:
            self._awp_interp = self._awp_nearest = None

    def _interp_with_fallback(self, interp, nearest):
        x = np.array([[self.cog_x, self.disp_mass]])
        result = interp(x)[0]
        if np.isnan(result):
            result = nearest(x)[0]
        return float(result)

    def wet_surface_area(self) -> float:
        return self._interp_with_fallback(self._wet_interp, self._wet_nearest)

    def trim_angle(self) -> float:
        return self._interp_with_fallback(self._trim_interp, self._trim_nearest)

    def hull_length(self) -> float:
        if self._L_interp is None:
            return np.nan
        return self._interp_with_fallback(self._L_interp, self._L_nearest)

    def hull_beam(self) -> float:
        if self._B_interp is None:
            return np.nan
        return self._interp_with_fallback(self._B_interp, self._B_nearest)

    def waterplane_area(self) -> float:
        if self._awp_interp is None:
            return np.nan
        return self._interp_with_fallback(self._awp_interp, self._awp_nearest)

    def get_params(self) -> Dict[str, float]:
        return {
            "LWL": self.hull_length(),
            "BWL": self.hull_beam(),
            "AWP": self.waterplane_area(),
            "total_area": self.total_area,
            "wet_surface_area_interp": self.wet_surface_area(),
            "trim_angle_interp": self.trim_angle(),
            "cog_x": self.cog_x,
            "disp_mass": self.disp_mass,
        }

    def get_valid_range(self):
        return {
            "cog_x": (self.df["cog_x"].min(), self.df["cog_x"].max()),
            "disp_mass": (self.df["disp_mass"].min(), self.df["disp_mass"].max()),
        }

    def is_in_valid_range(self, margin: float = 0.0) -> bool:
        r = self.get_valid_range()
        return (
            r["cog_x"][0] - margin <= self.cog_x <= r["cog_x"][1] + margin
            and r["disp_mass"][0] - margin
            <= self.disp_mass
            <= r["disp_mass"][1] + margin
        )
