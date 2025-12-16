import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import multiprocessing
import copy
from itertools import chain

from SALib import ProblemSpec

import sys

sys.path.append(".")

from .optimization import convert_to_model_params
from .plot import figsize


def describe_param_with_uniform_distribution(
    lower: float, upper: float, group: str = None
) -> dict:
    return dict(
        bounds=(lower, upper),
        dist="unif",
        group=group,
    )


def describe_param_with_log_uniform_distribution(
    lower: float, upper: float, group: str = None
) -> dict:
    return dict(
        bounds=(lower, upper),
        dist="logunif",
        group=group,
    )


def describe_param_with_triangular_distribution(
    lower: float, peak: float, upper: float, group: str = None
) -> dict:
    return dict(
        bounds=(lower, peak, upper),
        dist="triang",
        group=group,
    )


def describe_param_with_normal_distribution(
    mean: float, std: float, group: str = None
) -> dict:
    return dict(
        bounds=(mean, std),
        dist="norm",
        group=group,
    )


def describe_param_with_truncated_normal_distribution(
    lower: float, upper: float, mean: float, std: float, group: str = None
) -> dict:
    return dict(
        bounds=(lower, upper, mean, std),
        dist="truncnorm",
        group=group,
    )


def describe_param_with_log_normal_distribution() -> dict:
    return dict(
        bounds=(mean, std),
        dist="lognorm",
        group=group,
    )


def wrapped_model_function(
    factors: np.ndarray,
    model_function,
    model_class,
    model_params,
    input_params_names: list,
    T,
    U,
    X0,
    print_exceptions=True,
) -> np.ndarray:
    # Reimporting packages for multiprocessing
    # See: https://salib.readthedocs.io/en/latest/user_guide/wrappers.html#parallel-evaluation-and-analysis
    import numpy as np

    factors_len = factors.shape[0]
    model = model_class.build({})
    self_model_params = model_params

    results = np.empty(
        (factors_len, model.noutputs, len(T))
    )  # SA factors [axis 0] vs model outputs [axis 1] over time [axis 2]

    # Make all values -1 to begin with to indicate erroneous results
    # NOTE: This will bias analyses!
    results.fill(-1.0)

    for i in range(factors_len):
        values = factors[i, :]

        input_params = {k: v for k, v in zip(input_params_names, values)}
        model_params = self_model_params | convert_to_model_params(input_params)

        try:
            results[i, :, :] = (
                model_function(T=T, U=U, X0=X0, **model_params)[model.output_labels]
                .to_numpy()
                .T
            )

        except Exception as e:
            if print_exceptions:
                print(f"Exception from model_function: {e}")
            # Ignore runs that failed
            # Again, this will bias results...
            pass

    return results


def create_problem_spec_and_sample(
    params_description: dict,
    model_class,
    problem_kwargs: dict,
    sample_func,
    sample_kwargs: dict,
) -> ProblemSpec:

    model = model_class.build({})

    problem = ProblemSpec(
        dict(
            names=list(params_description.keys()),
            bounds=[p["bounds"] for p in params_description.values()],
            outputs=model.output_labels,
            dists=[p["dist"] for p in params_description.values()],
            groups=[p.get("group") or name for name, p in params_description.items()],
        )
        | problem_kwargs,
    )

    problem.sample(sample_func, **sample_kwargs)

    return problem


def plot_sampled_distribuitions(problem):
    fig = plt.figure(layout="constrained")

    for i, factor_name in enumerate(problem["names"]):
        plt.hist(problem._samples[:, i], bins=10, label=factor_name, alpha=0.3)
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    return fig


def evaluate_problem(
    problem,
    T,
    U,
    X0,
    model_function,
    model_class,
    model_params,
    nprocs,
    evaluate_kwargs,
):
    problem["T"] = T
    problem["X0"] = X0
    problem["U"] = U

    problem.evaluate_parallel(
        wrapped_model_function,
        model_function=model_function,
        model_class=model_class,
        model_params=model_params,
        input_params_names=problem["groups"],
        T=problem["T"],
        U=problem["U"],
        X0=problem["X0"],
        nprocs=nprocs,
        **evaluate_kwargs,
    )


def analyze_time_step(t_idx, problem, analyze_func, analyze_kwargs):
    # Since we (and SALib too) don't modify the samples, we can safely create
    # a shallow copy to have a new instance of ProblemSpec,
    # keeping the original data pointers, and then modifying only
    # the pointers we want:
    problem = copy.copy(problem)

    # Set results for the current time step
    problem.set_results(problem.results[:, :, t_idx])

    # Perform analysis
    problem.analyze(
        analyze_func,
        nprocs=1,  # Set to 1 since we are parallelizing `t`
        **analyze_kwargs,
    )

    df_list = []
    time = problem["T"][t_idx]

    for output_name in problem.analysis.keys():
        for df_si in problem.analysis[output_name].to_df():
            si = df_si.to_dict(orient="split")
            metric_name = si["columns"][0]

            for factor_idx, factor_name in enumerate(si["index"]):
                df_list.append(
                    {
                        "t": time,
                        "output": output_name,
                        "metric": metric_name,
                        "factor": str(factor_name).replace("'", ""),
                        "value": si["data"][factor_idx][0],
                        "conf": si["data"][factor_idx][1],
                    }
                )

    return df_list


def generate_args(problem, analyze_func, analyze_kwargs):
    for t_idx in range(problem.results.shape[-1]):
        yield (t_idx, problem, analyze_func, analyze_kwargs)


def analyze_problem(problem, analyze_func, analyze_kwargs, nprocs):
    # Analyze each time step in parallel
    print("pool")
    with multiprocessing.Pool(processes=nprocs) as pool:
        results = pool.starmap(
            analyze_time_step, generate_args(problem, analyze_func, analyze_kwargs)
        )

    # Convert the results
    print("chain")
    return pd.DataFrame(list(chain.from_iterable(results)))


def sobol_sensitivity_analysis_from_model_params_to_outputs(
    params_description: dict,
    T: np.array,
    U: np.array,
    X0: np.array,
    model_function,
    model_class,
    model_params,
    samples,
    calc_second_order,
    seed,
    nprocs,
    print_exceptions=True,
    problem_kwargs={},
    sample_kwargs={},
    evaluate_kwargs={},
    analyze_kwargs={},
) -> (ProblemSpec, pd.DataFrame):
    from SALib.sample.sobol import sample as sample_sobol
    from SALib.analyze.sobol import analyze as analyze_sobol

    # 1. Create the problem
    problem = create_problem_spec_and_sample(
        params_description=params_description,
        model_class=model_class,
        problem_kwargs=dict(
            calc_second_order=calc_second_order,
            seed=seed,
        )
        | problem_kwargs,
        sample_func=sample_sobol,
        sample_kwargs=dict(
            N=samples,
            calc_second_order=calc_second_order,
            seed=seed,
        )
        | sample_kwargs,
    )

    # 2. Evaluate the problem
    evaluate_problem(
        problem=problem,
        T=T,
        U=U,
        X0=X0,
        model_function=model_function,
        model_class=model_class,
        model_params=model_params,
        nprocs=nprocs,
        evaluate_kwargs=dict(
            print_exceptions=print_exceptions,
        )
        | evaluate_kwargs,
    )

    # 3. Analyze the problem
    df = analyze_problem(
        problem=problem,
        nprocs=nprocs,
        analyze_func=analyze_sobol,
        analyze_kwargs=dict(
            num_resamples=100,
            conf_level=0.95,
            seed=seed,
            calc_second_order=calc_second_order,
        )
        | analyze_kwargs,
    )

    # Ensure 'factor' column is properly formatted
    df["factor"] = df["factor"].astype("str").str.replace("'", "")

    indexes = ["output", "factor", "metric", "t"]
    df = (
        df.sort_values(by=indexes, ascending=[True, False, True, True])
        .set_index(indexes)
        .reset_index()
    )

    return df, problem


def plot_sensitivity_analysis(df: pd.DataFrame, output: str):
    df_st = (
        df.loc[(df["output"] == output) & (df["metric"] == "ST")][
            ["t", "factor", "value", "conf"]
        ]
        .set_index(["t", "factor"])
        .unstack()["value"]
    )
    df_s1 = (
        df.loc[(df["output"] == output) & (df["metric"] == "S1")][
            ["t", "factor", "value", "conf"]
        ]
        .set_index(["t", "factor"])
        .unstack()["value"]
    )

    fig, ax = plt.subplots(
        2, 1, figsize=figsize(), sharex=True, sharey=True, layout="constrained"
    )
    fig.suptitle("Sobol Indices over time")

    for factor in df_st.columns.unique():
        ax[0].plot(df_st.index, df_st[factor], label=factor)
        ax[0].set_ylabel("Total Order")

    for factor in df_s1.columns.unique():
        ax[1].plot(df_s1.index, df_s1[factor], label=factor)
        ax[1].set_ylabel("First Order")

    ax[-1].set_xlabel("Time [s]")

    return fig


def plot_sensitivity_analysis_heatmaps(df: pd.DataFrame, output: str):
    output_df = df.loc[df["output"] == output]

    # Pivot tables for heatmaps
    st_pivot = output_df.loc[output_df["metric"] == "ST"].pivot(
        index="factor", columns="t", values="value"
    )
    s1_pivot = output_df.loc[output_df["metric"] == "S1"].pivot(
        index="factor", columns="t", values="value"
    )

    factors = st_pivot.index.unique()

    if st_pivot.empty or s1_pivot.empty:
        print(f"Empty st_pivot or s1_pivot, skipping plots for output {output}!")
        return

    # Repeat the rows for each factor to increase pixel height
    factor_pixels_y = st_pivot.shape[1]
    st_pivot = st_pivot.reindex(st_pivot.index.repeat(factor_pixels_y))
    s1_pivot = s1_pivot.reindex(s1_pivot.index.repeat(factor_pixels_y))

    # Define the factor for repeating 't' axis
    factor_pixels_x = 10
    st_pivot = st_pivot.loc[:, st_pivot.columns.repeat(factor_pixels_x)]
    s1_pivot = s1_pivot.loc[:, s1_pivot.columns.repeat(factor_pixels_x)]

    (width, height) = figsize(subplots=(1, 2))
    height = (height / 1.5) + (height / 4) * (len(factors) / 2)

    # Create the figure with constrained layout
    fig, ax = plt.subplots(
        1,
        3,
        figsize=(width, height),
        gridspec_kw={"width_ratios": [1, 1, 0.05]},
        layout="constrained",
    )
    fig.suptitle(f"SA over time for the output: {output}")

    # Plot the Total Order sensitivity
    im0 = ax[0].imshow(
        st_pivot, cmap="coolwarm", origin="lower", interpolation="none", vmin=0, vmax=1
    )
    ax[0].set_title("Total Order")
    ax[0].set_xlabel("Time [s]")
    ax[0].set_ylabel("Sobol Indice")

    # Manually set xticks to correspond to the factors
    xticks_location = [0, len(st_pivot.columns) // 2, len(st_pivot.columns) - 1]
    ax[0].set_xticks(xticks_location)
    ax[0].set_xticklabels(st_pivot.columns[xticks_location])

    # Manually set yticks to correspond to the factors
    yticks_location = (
        np.arange(0, len(st_pivot), factor_pixels_y) + factor_pixels_y // 2
    )  # Position at the middle of each factor group
    ax[0].set_yticks(yticks_location)
    ax[0].set_yticklabels(factors)
    ax[0].tick_params(axis="y", which="both", left=False, right=False)
    ax[0].grid(False)

    # Plot the First Order sensitivity
    im1 = ax[1].imshow(
        s1_pivot, cmap="coolwarm", origin="lower", interpolation="none", vmin=0, vmax=1
    )
    ax[1].set_title("First Order")
    ax[1].set_xlabel("Time [s]")
    ax[1].set_yticks(yticks_location)
    ax[1].set_yticklabels("")  # Clear y-axis labels
    ax[1].grid(False)

    # Manually set xticks to correspond to the factors
    xticks_location = [0, len(s1_pivot.columns) // 2, len(st_pivot.columns) - 1]
    ax[1].set_xticks(xticks_location)
    ax[1].set_xticklabels(s1_pivot.columns[xticks_location])

    # Add a shared colorbar
    fig.colorbar(im1, cax=ax[2], label="Sensitivity Index")
    ax[2].tick_params(labelsize=10)
    ax[2].set_aspect(15)

    return fig


def get_region_mean(df: pd.DataFrame, t_start, t_end) -> pd.DataFrame:
    indexes = ["output", "factor", "metric"]
    return (
        df.loc[(df["t"] >= t_start) & (df["t"] <= t_end)]
        .set_index(["t"])
        .groupby(indexes)
        .mean()
        .reset_index()
        .sort_values(by=indexes, ascending=[True, False, True])
        .set_index(indexes)
    )


def plot_sensitivity_analysis_bars(df: pd.DataFrame, output: str):
    # NOTE: The dataframe df should be like from get_region_mean()

    df = df.reset_index(drop=False)

    # Filter data for the specified output
    df = df.loc[df["output"] == output]

    # Pivot for easier access
    df = df.pivot(index="factor", columns="metric", values=["value", "conf"])

    if df.empty:
        print(f"No data available for output '{output}'.")
        return

    # Extract categories and ensure alignment
    categories = df.index.get_level_values("factor")
    y = np.arange(len(categories)) / 2
    bar_width = 1 / len(categories)

    # Create the plot
    fig = plt.figure(figsize=figsize(), layout="constrained")
    fig.suptitle("Sobol Indice")

    # Plot horizontal bars for ST, S1, and S2
    plt.barh(y + (bar_width / 2), df["value"]["ST"], height=bar_width * 0.8, label="ST")
    plt.barh(y - (bar_width / 2), df["value"]["S1"], height=bar_width * 0.8, label="S1")
    if "S2" in df["value"]:
        plt.barh(y, df["value"]["S2"], height=bar_width * 0.8, label="S2", color="gray")

    # Add confidence intervals
    plt.errorbar(
        df["value"]["ST"],
        y + (bar_width / 2),
        xerr=df["conf"]["ST"],
        fmt="none",
        color="black",
        alpha=0.7,
        capsize=2.5,
    )
    plt.errorbar(
        df["value"]["S1"],
        y - (bar_width / 2),
        xerr=df["conf"]["S1"],
        fmt="none",
        color="black",
        alpha=0.7,
        capsize=2.5,
    )
    if "S2" in df["value"]:
        plt.errorbar(
            df["value"]["S2"],
            y,
            xerr=df["conf"]["S2"],
            fmt="none",
            color="black",
            alpha=0.7,
            capsize=2.5,
        )

    # Add labels and legend
    plt.yticks(y, categories)
    plt.xlabel("Sobol Indice")
    plt.ylabel("Factors")
    plt.xlim((0, 1))
    plt.legend()

    return fig


# def get_s2_matrix(df: pd.DataFrame, output: str) -> pd.DataFrame:
#     df = consolidate_sa_results(df)

#     df = df.loc[(df["Order"] == "S2") & (df["Output"] == output), ["Factor", "Mean"]]
#     df[["Factor 1", "Factor 2"]] = df["Factor"].apply(pd.Series)
#     df = df.drop(columns=["Factor"])[["Factor 1", "Factor 2", "Mean"]].reset_index(
#         drop=True
#     )

#     all_factors = df["Factor 1"].unique()
#     corr = df.pivot(index="Factor 1", columns="Factor 2", values="Mean").reindex(
#         index=all_factors, columns=all_factors
#     )

#     # Refill the rest of the matrix
#     corr[corr.isna()] = corr[~corr.isna()][::-1].T
#     corr = corr.fillna(1)

#     return corr


# def plot_sensitivity_analysis_second_order(df: pd.DataFrame, max_cols=2):
#     # Calculate the number of rows and columns needed
#     num_outputs = len(df["output"].unique())
#     rows = (
#         num_outputs + max_cols - 1
#     ) // max_cols  # Ceiling division to calculate rows
#     cols = min(num_outputs, max_cols)  # Ensure no more than `max_cols` columns

#     # Create the plot
#     width_inches, _ = figsize()  # IMPORTANT: Respect this width at all costs!!!
#     width_per_subplot = width_inches / cols
#     height_per_subplot = (
#         width_per_subplot  # Because it should be approximately a square
#     )
#     height_inches = height_per_subplot * rows

#     fig, ax = plt.subplots(
#         rows,
#         cols,
#         figsize=(width_inches, height_inches),
#         sharex=True,
#         sharey=True,
#         layout="constrained",
#     )
#     fig.suptitle(f"Second Order Sensitivity Analysis")

#     ax = ax.flatten()  # Flatten ax array in case there are more than one row

#     # Track index for axes (so we don't plot on an unused axis)
#     ax_idx = 0

#     for output in df["output"].unique():
#         df_s2 = get_s2_matrix(df, output)

#         if df_s2.empty:
#             continue  # Skip if the dataframe is empty

#         # Apply the mask and transformations
#         mask = np.zeros_like(df_s2, dtype=bool)
#         mask[np.triu_indices_from(mask)] = True
#         df_s2[mask] = np.nan
#         df_s2 = df_s2[::-1]

#         # Plot the correlation matrix
#         im = ax[ax_idx].imshow(
#             df_s2, cmap="coolwarm", origin="lower", interpolation="none"
#         )

#         # Set axis labels and titles
#         ax[ax_idx].set_title(output)

#         # Set xticks and yticks
#         ticks = np.arange(0, len(df_s2.columns), 1)  # Corrected tick calculation
#         ax[ax_idx].set_xticks(ticks + 0.5)  # Shift ticks slightly for better alignment
#         ax[ax_idx].set_xticklabels(df_s2.index, rotation=45, ha="right")
#         ax[ax_idx].set_yticks(ticks)
#         ax[ax_idx].set_yticklabels(df_s2.columns)

#         ax[ax_idx].tick_params(
#             axis="both", which="both", left=False, right=False, top=False, bottom=False
#         )
#         ax[ax_idx].grid(False)

#         # Add colorbar
#         vals = df_s2.to_numpy().ravel()
#         vals = vals[~np.isnan(vals)]  # Exclude NaN values from the range
#         vmin, vmax = vals.min(), vals.max()
#         ticks = np.linspace(vmin, vmax, min(3, len(df_s2.columns)), endpoint=True)

#         cbar = fig.colorbar(
#             im,
#             shrink=0.8,
#             aspect=10 / 0.7,
#             ticks=ticks,
#             format=mticker.FixedFormatter([f"{n:.2f}" for n in ticks]),
#             location="right",
#         )

#         # Move to the next axis for the next plot
#         ax_idx += 1

#     # Hide any empty subplots if the number of outputs is less than the grid size
#     for j in range(ax_idx, len(ax)):
#         ax[j].axis("off")

#     return fig
