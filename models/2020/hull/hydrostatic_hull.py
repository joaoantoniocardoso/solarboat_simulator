import pandas as pd
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import os

class HydrostaticHull:
    def __init__(self, feather_path=None):
        """
        Hydrostatic surrogate model based on meshmagick simulation data.

        Parameters
        ----------
        feather_path : str, optional
            Path to feather file with simulation data.
            If None, assumes 'data/meshmagick.feather' relative to this file.
        """
        if feather_path is None:
            feather_path = os.path.join(os.path.dirname(__file__), 'data', 'meshmagick.feather')

        self.df = self._load_dataset(feather_path)
        self._build_interpolators()

    def _load_dataset(self, path):
        """Load and filter simulation data."""
        df = pd.read_feather(path)
        # Keep only reasonable trim angles
        df = df[abs(df['angles_deg_y']) < 10].reset_index(drop=True)
        return df

    def _build_interpolators(self):
        """Build interpolation functions from simulation data."""
        X = self.df[['cog_x', 'disp_mass']].values
        wet_area_values = self.df['wet_surface_area'].values
        trim_values = self.df['angles_deg_y'].values

        # Primary interpolators (linear)
        self._wet_interp = LinearNDInterpolator(X, wet_area_values, fill_value=np.nan)
        self._trim_interp = LinearNDInterpolator(X, trim_values, fill_value=np.nan)

        # Fallback interpolators (nearest neighbor for extrapolation)
        self._wet_nearest = NearestNDInterpolator(X, wet_area_values)
        self._trim_nearest = NearestNDInterpolator(X, trim_values)

    def wet_surface_area(self, cog_x, disp_mass):
        """
        Get interpolated wetted surface area.

        Parameters
        ----------
        cog_x : float
            Longitudinal COG position [m] from transom
        disp_mass : float
            Displacement mass [kg]

        Returns
        -------
        float
            Wetted surface area [m²]
        """
        x = np.array([[cog_x, disp_mass]])
        result = self._wet_interp(x)[0]
        if np.isnan(result):
            result = self._wet_nearest(x)[0]
        return float(result)

    def trim_angle(self, cog_x, disp_mass):
        """
        Get interpolated trim angle.

        Parameters
        ----------
        cog_x : float
            Longitudinal COG position [m] from transom
        disp_mass : float
            Displacement mass [kg]

        Returns
        -------
        float
            Trim angle [degrees]
            Positive = bow up (stern down)
            Negative = bow down (stern up)
        """
        x = np.array([[cog_x, disp_mass]])
        result = self._trim_interp(x)[0]
        if np.isnan(result):
            result = self._trim_nearest(x)[0]
        return float(result)

    def get_valid_range(self):
        """
        Get the valid range of the training data.

        Returns
        -------
        dict
            Dictionary with 'cog_x' and 'disp_mass' ranges
        """
        return {
            'cog_x': (self.df['cog_x'].min(), self.df['cog_x'].max()),
            'disp_mass': (self.df['disp_mass'].min(), self.df['disp_mass'].max())
        }

    def is_in_valid_range(self, cog_x, disp_mass, margin=0.0):
        """
        Check if a point is within the valid training range.

        Parameters
        ----------
        cog_x : float
            Longitudinal COG position [m] from transom
        disp_mass : float
            Displacement mass [kg]
        margin : float, optional
            Extra margin to consider as valid (default: 0.0)

        Returns
        -------
        bool
            True if point is within valid range
        """
        cog_range = self.df['cog_x'].min() - margin, self.df['cog_x'].max() + margin
        mass_range = self.df['disp_mass'].min() - margin, self.df['disp_mass'].max() + margin

        return (cog_range[0] <= cog_x <= cog_range[1]) and (mass_range[0] <= disp_mass <= mass_range[1])
