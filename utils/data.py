from pytz import timezone
import pandas as pd
import numpy as np
import vaex


def zero_phase_lowpass_filter(dataframe, cutoff_freq, sampling_rate, order):
    """
    Applies a zero-phase lowpass filter to each column of a pandas DataFrame,
    handling NaN values by filtering only the continuous segments between NaNs.

    Parameters:
        dataframe (pd.DataFrame): The input DataFrame.
        cutoff_freq (float): The cutoff frequency of the lowpass filter (in Hz).
        sampling_rate (float): The sampling rate of the data (in Hz).
        order (int): The order of the filter.

    Returns:
        pd.DataFrame: A new DataFrame with the filtered data.
    """
    import numpy as np
    import pandas as pd
    from scipy.signal import butter, filtfilt

    # Design the Butterworth filter
    nyquist_freq = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)

    # Apply the filter to each column
    filtered_data = {}
    for column in dataframe.columns:
        # Extract the column data
        column_data = dataframe[column].copy()

        # Find segments of continuous data (between NaNs)
        nan_indices = np.where(column_data.isna())[0]
        segments = []
        start = 0
        for i in nan_indices:
            if start < i:
                segments.append(column_data.iloc[start:i])
            start = i + 1
        if start < len(column_data):
            segments.append(column_data.iloc[start:])

        # Filter each segment individually
        filtered_segments = []
        for segment in segments:
            if len(segment.dropna()) > len(b) * 3:  # Ensure segment is long enough
                filtered_segment = filtfilt(b, a, segment.dropna())
                filtered_segments.append(
                    pd.Series(filtered_segment, index=segment.dropna().index)
                )
            else:
                # Skip or handle short segments (keep as is or interpolate)
                filtered_segments.append(segment)

        # Combine the filtered segments back into a single Series
        filtered_data[column] = pd.concat(filtered_segments).sort_index()

    # Return a new DataFrame with the filtered data
    return pd.DataFrame(filtered_data, index=dataframe.index)


def process_df(
    df, start, end, resample_rule, iqr_threshold, cutoff_freq, sampling_rate, order
):
    if start is None:
        start = df.index[0]
    if end is None:
        end = df.index[-1]

    df = df.copy(deep=True).loc[(df.index >= start) & (df.index <= end),]

    if iqr_threshold:
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        df[
            ~(
                (df < (Q1 - iqr_threshold * IQR)) | (df > (Q3 + iqr_threshold * IQR))
            ).any(axis=1)
        ] = np.nan

    if resample_rule:
        df = df.resample(resample_rule).mean().interpolate(method="time", limit=1000)

    if cutoff_freq:
        df = zero_phase_lowpass_filter(df, cutoff_freq, sampling_rate, order)

    # Create the equally-spaced 't' index, used for control simulations
    dT = (
        df.index.diff().median().to_numpy().astype(np.float64) * 1e-9
    )  # simulation time in seconds
    lenT = len(df.index)
    df["t"] = np.linspace(
        0, lenT * dT, lenT, endpoint=False
    )  # Recreate the time array because of numerical issues from the index datetime to float transformation

    return df


def load_df(
    filename,
    start,
    end,
    resample_rule="1s",
    rename_columns={},
    print_columns=True,
    iqr_threshold=None,
    cutoff_freq=None,
    sampling_rate=1,
    order=2,
):
    df = vaex.from_csv(filename).to_pandas_df()
    df["timestamp"] = pd.DatetimeIndex(df["timestamp"])
    df = df.set_index("timestamp", drop=True)

    original_columns = df.columns

    df = df[rename_columns.keys()].rename(columns=rename_columns)

    if print_columns:
        print("original columns:", original_columns)
        print("renamed columns:", rename_columns)

    df = process_df(
        df, start, end, resample_rule, iqr_threshold, cutoff_freq, sampling_rate, order
    )

    return df


def get_events():
    tzinfo = timezone("America/Sao_Paulo")

    events = [
        {
            "name": "01.Curta",
            "start": pd.Timestamp(
                year=2020, month=1, day=29, hour=13, minute=51, second=59, tzinfo=tzinfo
            ),
            "end": pd.Timestamp(
                year=2020, month=1, day=29, hour=14, minute=20, second=25, tzinfo=tzinfo
            ),
        },
        {
            "name": "02.Longa",
            "start": pd.Timestamp(
                year=2020, month=1, day=30, hour=11, minute=16, second=45, tzinfo=tzinfo
            ),
            "end": pd.Timestamp(
                year=2020, month=1, day=30, hour=14, minute=55, second=30, tzinfo=tzinfo
            ),
        },
        {
            "name": "03.Revezamento",
            "start": pd.Timestamp(
                year=2020, month=1, day=31, hour=11, minute=23, second=23, tzinfo=tzinfo
            ),
            "end": pd.Timestamp(
                year=2020, month=1, day=31, hour=12, minute=27, second=23, tzinfo=tzinfo
            ),
        },
        {
            "name": "04.Curta",
            "start": pd.Timestamp(
                year=2020, month=2, day=1, hour=10, minute=15, second=15, tzinfo=tzinfo
            ),
            "end": pd.Timestamp(
                year=2020, month=2, day=1, hour=11, minute=37, second=28, tzinfo=tzinfo
            ),
        },
        {
            "name": "05.Curta",
            "start": pd.Timestamp(
                year=2020, month=2, day=1, hour=13, minute=15, second=9, tzinfo=tzinfo
            ),
            "end": pd.Timestamp(
                year=2020, month=2, day=1, hour=13, minute=32, second=55, tzinfo=tzinfo
            ),
        },
        {
            "name": "07.Sprint",
            "start": pd.Timestamp(
                year=2020, month=2, day=2, hour=11, minute=58, second=27, tzinfo=tzinfo
            ),
            "end": pd.Timestamp(
                year=2020, month=2, day=2, hour=11, minute=59, second=9, tzinfo=tzinfo
            ),
        },
    ]

    events = pd.DataFrame(events).set_index("name")
    events["start"] = events["start"].dt.tz_convert(None)
    events["end"] = events["end"].dt.tz_convert(None)

    return events, tzinfo


def get_sections():
    tzinfo = timezone("America/Sao_Paulo")

    events = [
        {
            "name": "Section A",
            "start": pd.Timestamp(
                year=2020, month=1, day=29, hour=13, minute=45, second=0, tzinfo=tzinfo
            ),
            "end": pd.Timestamp(
                year=2020, month=1, day=29, hour=16, minute=0, second=0, tzinfo=tzinfo
            ),
        },
        {
            "name": "Section B",
            "start": pd.Timestamp(
                year=2020, month=1, day=30, hour=10, minute=0, second=0, tzinfo=tzinfo
            ),
            "end": pd.Timestamp(
                year=2020, month=1, day=30, hour=17, minute=0, second=0, tzinfo=tzinfo
            ),
        },
        {
            "name": "Section C",
            "start": pd.Timestamp(
                year=2020, month=1, day=30, hour=21, minute=26, second=0, tzinfo=tzinfo
            ),
            "end": pd.Timestamp(
                year=2020, month=1, day=30, hour=23, minute=50, second=0, tzinfo=tzinfo
            ),
        },
        {
            "name": "Section D",
            "start": pd.Timestamp(
                year=2020, month=1, day=31, hour=10, minute=40, second=0, tzinfo=tzinfo
            ),
            "end": pd.Timestamp(
                year=2020, month=1, day=31, hour=17, minute=20, second=0, tzinfo=tzinfo
            ),
        },
        {
            "name": "Section E",
            "start": pd.Timestamp(
                year=2020, month=2, day=1, hour=13, minute=10, second=0, tzinfo=tzinfo
            ),
            "end": pd.Timestamp(
                year=2020, month=2, day=1, hour=16, minute=20, second=0, tzinfo=tzinfo
            ),
        },
        {
            "name": "Section F",
            "start": pd.Timestamp(
                year=2020, month=2, day=2, hour=9, minute=40, second=0, tzinfo=tzinfo
            ),
            "end": pd.Timestamp(
                year=2020, month=2, day=2, hour=12, minute=40, second=0, tzinfo=tzinfo
            ),
        },
    ]

    events = pd.DataFrame(events).set_index("name")
    events["start"] = events["start"].dt.tz_convert(None)
    events["end"] = events["end"].dt.tz_convert(None)

    return events, tzinfo


def export_dataframe_to_latex(filename, label, caption, df_steady_state_mean):
    """
    Generates a LaTeX table from a DataFrame of Sobol indices in steady state.

    Parameters:
        filename (str): The full file path.
        label (str): The label for the LaTeX table.
        caption (str): The caption for the LaTeX table.
        df_steady_state_mean (pd.DataFrame): The DataFrame containing the Sobol indices.

    Returns:
        str: A string representing the LaTeX code for the table.
    """

    # Define constants for voltage and motor load torque steps
    voltage_step = 36
    motor_load_torque_step = 10

    # Generate the LaTeX table
    table = (
        df_steady_state_mean.to_latex(
            index=True,
            caption=caption,
            label=label,
            float_format="%.2f",
            column_format="|".join(["l"] * (df_steady_state_mean.index.nlevels))
            + "|"
            + "|".join(["c"] * len(df_steady_state_mean.columns)),
            escape=True,
            decimal=",",
            position="h!",
            multicolumn=True,
        )
        .replace("\\toprule", "\\hline")
        .replace("\\bottomrule\n", "")
        .replace("\\midrule", "\\hline")
        .replace(
            "\\end{table}", "\\fonte{Elaboração Própria (\\the\\year)}\n\\end{table}"
        )
        .replace("\\caption", "\\centering\n\\caption")
    )

    if filename:
        with open(filename, "w") as f:
            f.write(table)

    return table
