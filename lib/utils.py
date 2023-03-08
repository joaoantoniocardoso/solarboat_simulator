import pandas as pd
import numpy as np
import scipy as sp
from pvlib import irradiance, location
import matplotlib.pyplot as plt
import seaborn as sns


def naive_power(
    energy: np.float64, time: np.float64, timebase: np.float64 = np.float64(1.0)
) -> np.float64:
    """energy [Wh], time [h|s], use timebase=1 if time in hours, or timebase=3600 if time in
    seconds. Time has a minimum value of 1e-9"""
    return energy / (time / timebase)


def naive_energy(
    power: np.float64, time: np.float64, timebase: np.float64 = np.float64(1.0)
) -> np.float64:
    """power [W], time [h|s], use timebase=1 if time in hours, or timebase=3600 if time in
    seconds. Time has a minimum value of 1e-9"""
    return power * (time / timebase)


def integrate(df: pd.DataFrame, time_constant: int = 3600) -> pd.DataFrame:
    """
    Integrates a datetime indexed dataframe relative to a time_constant (3600
    to return in hours)
    """
    if df.index.freq is None:  # type: ignore
        df.index.freq = pd.infer_freq(df.index)  # type: ignore
    if df.index.freq is None:  # type: ignore
        raise Exception("Failed infering frequency!")
    return df.apply(
        sp.integrate.cumtrapz,
        initial=0,
        dx=((df.index.freq.nanos * 1e-9) / time_constant),  # type: ignore
    )  # type: ignore


def mse(x0: np.float64, x1: np.float64) -> np.float64:
    """Mean Squared Error (MSE)"""
    return np.sum(x0 - x1) ** 2 / x0


def rmse(x0: np.float64, x1: np.float64) -> np.float64:
    """Root Mean Squared Error (RMSE)"""
    return np.sqrt(np.sum(x0 - x1) ** 2 / np.abs(x0))


def mae(x0: np.float64, x1: np.float64) -> np.float64:
    """Mean Absolute Error (MAE)"""
    return np.sum(np.abs(x0 - x1)) / x0


def simple_error(x0: np.float64, x1: np.float64) -> np.float64:
    """Simple error"""
    return (x0 - x1) / x0


def get_irradiance(site_location, tilt, surface_azimuth, weather_data):
    solar_position = site_location.get_solarposition(times=weather_data.index)

    POA_irradiance = irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=surface_azimuth,
        dni=weather_data["dni"],
        ghi=weather_data["ghi"],
        dhi=weather_data["dhi"],
        solar_zenith=solar_position["apparent_zenith"],
        solar_azimuth=solar_position["azimuth"],
    )

    return pd.DataFrame({"POA": POA_irradiance["poa_global"]})  # type: ignore


def open_forecast_file(forecast_file: str) -> pd.DataFrame:
    df = pd.read_csv(forecast_file).rename(
        columns={
            "period_end": "PeriodEnd",
            "air_temp": "AirTemp",
            "cloud_opacity": "CloudOpacity",
        }
    )

    sel = ~df["PeriodEnd"].str.endswith("Z")
    df.loc[sel, "PeriodEnd"] = df.loc[sel, "PeriodEnd"] + "T00:00:00Z"
    df.set_index("PeriodEnd", inplace=True)
    return df


def open_forecast_files(forecast_files: list[str], event: dict) -> pd.DataFrame:
    """Combine all forecasts, always keeping the range interval from the
    first file from the forecast_files, while using the latest updated
    forecast data, provided by the consecutive files from the forecast_files."""
    forecast_files = sorted(set(forecast_files))
    df = open_forecast_file(forecast_files[0])
    for i in range(1, len(forecast_files)):
        df_new = open_forecast_file(forecast_files[i])
        df.loc[df.index >= df_new.index[0]] = df_new  # type: ignore
    return df[event["time"]["start"] : event["time"]["end"]][:-1]


def plot_radiation_and_irradiance(
    ideal_data: pd.DataFrame, real_data: pd.DataFrame, site: location.Location
):
    plt.Figure()  # type: ignore
    ideal_data["poa"].plot(label="Clearsky Model")
    real_data["poa"].plot(label="Solcast Forecast")
    plt.fill_between(
        real_data.index,
        real_data["poa10"],  # type: ignore
        real_data["poa90"],  # type: ignore
        color="orange",
        alpha=0.3,
        label="10 to 90 percentile",
    )
    plt.ylabel("Irradiance ($W/m^2$)")
    plt.xlabel("Time ({})".format(site.tz))
    plt.title(
        "Hourly Irradiance for {} area (lat={}, lon={}, alt={})".format(
            site.name, site.latitude, site.longitude, site.altitude
        )
    )
    plt.legend()
    plt.show()

    plt.Figure()  # type: ignore
    ideal_data["Solar Energy"].plot(label="Clearsky Model")
    real_data["Solar Energy"].plot(label="Solcast Forecast")
    plt.fill_between(
        real_data.index,
        real_data["Solar Energy10"],  # type: ignore
        real_data["Solar Energy90"],  # type: ignore
        color="orange",
        alpha=0.3,
        label="10 to 90 percentile",
    )
    plt.ylabel("Energy ($Wh/m^2$)")
    plt.xlabel("Time ({})".format(site.tz))
    plt.title(
        "Hourly Sun Energy for {} area (lat={}, lon={}, alt={})".format(
            site.name, site.latitude, site.longitude, site.altitude
        )
    )
    plt.legend()
    plt.show()


def plot_energy_bars(
    ideal_data: pd.DataFrame, real_data: pd.DataFrame, site: location.Location
):
    plt.Figure()  # type: ignore

    sns.barplot(
        x=np.datetime_as_string(ideal_data.resample("D").mean().index.values, unit="D"),
        y=ideal_data.resample("D").mean()["poa"],
        label="Clearsky Model",
        color=sns.color_palette("light:b")[2],  # type: ignore
    )
    sns.barplot(
        x=np.datetime_as_string(real_data.resample("D").mean().index.values, unit="D"),
        y=real_data.resample("D").mean()["poa"],
        # yerr=[
        #     real_data.resample('D').mean()['poa10'],
        #     real_data.resample('D').mean()['poa90'],
        # ],
        label="Solcast Forecast",
        color=sns.color_palette("light:b")[5],  # type: ignore
    )
    plt.ylabel("Irradiance ($W/m^2$)")
    plt.xlabel("Time ({})".format(site.tz))
    plt.title(
        "Average Daily Irradiance for {} area (lat={}, lon={}, alt={})".format(
            site.name, site.latitude, site.longitude, site.altitude
        )
    )
    plt.legend()
    plt.show()
