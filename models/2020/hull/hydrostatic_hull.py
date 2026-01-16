import os
from typing import Dict, Optional, Tuple

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
        feather_path: Optional[str] = None,
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
        df = df.loc[abs(df["angles_deg_y"]) < 10].reset_index(drop=True)
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

        def _maybe_build(
            field: str,
        ) -> Tuple[Optional[LinearNDInterpolator], Optional[NearestNDInterpolator]]:
            if field not in self.df.columns:
                return None, None
            interp = LinearNDInterpolator(X, self.df[field].values, fill_value=np.nan)
            nearest = NearestNDInterpolator(X, self.df[field].values)
            return interp, nearest

        self._L_interp, self._L_nearest = _maybe_build("LWL")
        if self._L_interp is None:
            self._L_interp, self._L_nearest = _maybe_build("hull_L")

        self._B_interp, self._B_nearest = _maybe_build("BWL")
        if self._B_interp is None:
            self._B_interp, self._B_nearest = _maybe_build("hull_B")

        self._awp_interp, self._awp_nearest = _maybe_build("AWP")

        self._disp_vol_interp, self._disp_vol_nearest = _maybe_build("disp_volume_m3")
        self._draft_mean_interp, self._draft_mean_nearest = _maybe_build("draft_mean_m")
        self._draft_aft_interp, self._draft_aft_nearest = _maybe_build("draft_aft_m")
        self._draft_fwd_interp, self._draft_fwd_nearest = _maybe_build("draft_fwd_m")
        self._draft_keel_interp, self._draft_keel_nearest = _maybe_build("draft_keel_m")

        self._lcb_interp, self._lcb_nearest = _maybe_build("lcb_percent")

        self._cb_interp, self._cb_nearest = _maybe_build("CB")
        self._cm_interp, self._cm_nearest = _maybe_build("CM")
        self._cp_interp, self._cp_nearest = _maybe_build("CP")
        self._cwp_interp, self._cwp_nearest = _maybe_build("CWP")
        self._amid_interp, self._amid_nearest = _maybe_build("A_mid_m2")

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

    def disp_volume_m3(self) -> float:
        if self._disp_vol_interp is None:
            return np.nan
        return self._interp_with_fallback(self._disp_vol_interp, self._disp_vol_nearest)

    def draft_mean_m(self) -> float:
        if self._draft_mean_interp is None:
            return np.nan
        return self._interp_with_fallback(
            self._draft_mean_interp, self._draft_mean_nearest
        )

    def draft_aft_m(self) -> float:
        if self._draft_aft_interp is None:
            return np.nan
        return self._interp_with_fallback(
            self._draft_aft_interp, self._draft_aft_nearest
        )

    def draft_fwd_m(self) -> float:
        if self._draft_fwd_interp is None:
            return np.nan
        return self._interp_with_fallback(
            self._draft_fwd_interp, self._draft_fwd_nearest
        )

    def draft_keel_m(self) -> float:
        if self._draft_keel_interp is None:
            return np.nan
        return self._interp_with_fallback(
            self._draft_keel_interp, self._draft_keel_nearest
        )

    def lcb_percent(self) -> float:
        if self._lcb_interp is None:
            return np.nan
        return self._interp_with_fallback(self._lcb_interp, self._lcb_nearest)

    def cb(self) -> float:
        if self._cb_interp is None:
            return np.nan
        return self._interp_with_fallback(self._cb_interp, self._cb_nearest)

    def cm(self) -> float:
        if self._cm_interp is None:
            return np.nan
        return self._interp_with_fallback(self._cm_interp, self._cm_nearest)

    def cp(self) -> float:
        if self._cp_interp is None:
            return np.nan
        return self._interp_with_fallback(self._cp_interp, self._cp_nearest)

    def cwp(self) -> float:
        if self._cwp_interp is None:
            return np.nan
        return self._interp_with_fallback(self._cwp_interp, self._cwp_nearest)

    def amid_m2(self) -> float:
        if self._amid_interp is None:
            return np.nan
        return self._interp_with_fallback(self._amid_interp, self._amid_nearest)

    def get_params(self) -> Dict[str, float]:
        return {
            "LWL": self.hull_length(),
            "BWL": self.hull_beam(),
            "AWP": self.waterplane_area(),
            "CWP": self.cwp(),
            "CB": self.cb(),
            "CM": self.cm(),
            "CP": self.cp(),
            "lcb_percent": self.lcb_percent(),
            "disp_volume_m3": self.disp_volume_m3(),
            "draft_mean_m": self.draft_mean_m(),
            "draft_aft_m": self.draft_aft_m(),
            "draft_fwd_m": self.draft_fwd_m(),
            "draft_keel_m": self.draft_keel_m(),
            "A_mid_m2": self.amid_m2(),
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
