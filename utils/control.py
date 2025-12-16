import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import control as ct


def get_steady_state_map(
    model_class,
    model_function,
    model_params: dict,
    sweep_bounds: dict,
    sweep_steps: int,
    dt: float,
    ss_len: int,
    ss_tail_len: int,
) -> pd.DataFrame:
    sys = model_class.build(model_params)
    Y_list = []

    # Validate that sweep_bounds covers all inputs
    input_labels = sys.input_labels
    if set(sweep_bounds.keys()) != set(input_labels):
        raise ValueError(
            f"sweep_bounds keys {list(sweep_bounds.keys())} "
            f"must exactly match system input labels {input_labels}."
        )

    # Time vector
    T = np.linspace(0, dt * (ss_len - 1), ss_len)

    # Generate sampled values per input (respecting system input order)
    input_ranges = []
    for label in input_labels:
        bounds = sweep_bounds[label]
        input_ranges.append(np.linspace(bounds["min"], bounds["max"], sweep_steps))

    # Create full meshgrid in system input order
    meshgrids = np.meshgrid(*input_ranges, indexing="ij")
    flattened_inputs = [
        grid.ravel() for grid in meshgrids
    ]  # List of 1D arrays, one per input

    n_points = len(flattened_inputs[0])
    base_X0 = np.zeros(len(sys.state_labels))

    for i in range(n_points):
        X0 = base_X0.copy()
        # Stack inputs in sys.input_labels order: shape (n_inputs, ss_len)
        U = np.vstack([inp_vals[i] * np.ones(ss_len) for inp_vals in flattened_inputs])
        Y = model_function(T, U, X0, **model_params)
        if "time" in Y.columns:
            Y = Y.drop(columns="time")
        Y_ss = Y.tail(ss_tail_len).mean().to_frame().T
        Y_list.append(Y_ss)

    return pd.concat(Y_list, ignore_index=True)


def step_response(sys, T, X0, input_values, **kwargs):
    # This does the same as ct.step_response but allow us to specify the value of each step

    frames = []
    for i, input_label in enumerate(sys.input_labels):
        # Inputs
        U = np.zeros((sys.ninputs, len(T)))
        U[i] = input_values[input_label]

        # Simulation
        frame = ct.input_output_response(sys, T=T, U=U, X0=X0, **kwargs).to_pandas()
        frame["trace_label"] = "From " + input_label

        frames.append(frame)

    return pd.concat(frames)


def plot_step_response_dataframe(df):
    grouped = df.groupby(level="trace_label")
    row_size = 1

    for trace_label, group in grouped:
        fig, axes = plt.subplots(
            len(group.columns),
            1,
            figsize=(6.4, len(group.columns) * row_size),
            sharex=True,
        )
        fig.suptitle(f"Trace: {trace_label}", fontsize=16)

        if len(group.columns) == 1:
            axes = [axes]

        for ax, (signal_name, signal_data) in zip(axes, group.items()):
            ax.plot(
                group.index.get_level_values("time"), signal_data, label=signal_name
            )
            ax.grid(True)
            ax.set_ylabel(signal_name)

        axes[-1].set_xlabel("Time")

        plt.show()
