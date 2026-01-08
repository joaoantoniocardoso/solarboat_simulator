import json

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
from pymoo.core.problem import ElementwiseProblem

import sys

sys.path.append(".")

from .plot import figsize


class MyOptimizationProblem(ElementwiseProblem):
    def __init__(
        self,
        model,
        model_function,
        training_data,
        training_data_columns,
        model_params,
        opt_params_bounds,
        input_initial_state,
        input_columns,
        data_scaler=RobustScaler,
        constraint_funcs=None,
        print_exceptions=False,
        variable_weights=None,
        **kwargs,
    ):
        assert all(
            "min" in pb and "max" in pb for pb in opt_params_bounds.values()
        ), "All opt_params_bounds must have 'min' and 'max' keys"

        self.model = model
        self.model_function = model_function
        self.model_params = model_params
        self.data_scaler = data_scaler
        self.training_data_columns = training_data_columns
        self.opt_param_names = list(opt_params_bounds.keys())
        self.input_columns = input_columns
        self.print_exceptions = print_exceptions
        self.constraint_funcs = constraint_funcs or []
        self.variable_weights = variable_weights

        if variable_weights is None:
            # default: all weights = 1
            self.variable_weights = np.ones(len(training_data_columns), dtype=float)

        else:
            # must be dict with keys = training_data_columns
            if not all(col in variable_weights for col in training_data_columns):
                missing = [
                    c for c in training_data_columns if c not in variable_weights
                ]
                raise ValueError(f"Missing weights for: {missing}")

            # convert to array following column order
            self.variable_weights = np.array(
                [variable_weights[col] for col in training_data_columns],
                dtype=float,
            )

        # ---- Normalize weights: sum → n_columns ----
        self.variable_weights /= self.variable_weights.sum() / len(
            self.variable_weights
        )

        # Model Time array
        dt = (training_data.index[1] - training_data.index[0]).total_seconds()
        self.T = np.ascontiguousarray(
            np.linspace(0, len(training_data) * dt, len(training_data), endpoint=False)
        )

        # Model Input data
        self.U = np.ascontiguousarray(
            training_data[self.input_columns].copy(deep=True).to_numpy().T
        )

        # Model Initial state
        self.input_initial_state = input_initial_state
        self.compute_initial_state = model.initial_state

        # Store scale ground-truth data
        self.training_data = training_data[self.training_data_columns].to_numpy()
        self.data_scaler = data_scaler().fit(self.training_data)
        self.training_data = self.scale_data(self.training_data)

        super().__init__(
            n_var=len(self.opt_param_names),  # Number of parameters to optimize
            n_obj=1,  # Single objective
            n_constr=len(self.constraint_funcs),  # Number of constraints
            xl=np.array(
                [p["min"] for p in opt_params_bounds.values()]
            ),  # Lower bounds for parameters
            xu=np.array(
                [p["max"] for p in opt_params_bounds.values()]
            ),  # Upper bounds for parameters
            **kwargs,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.objective_function(x, *args, **kwargs)

        if self.constraint_funcs:
            # Each constraint func should return a scalar ≤ 0 when satisfied
            g = [func(x) for func in self.constraint_funcs]
            out["G"] = np.array(g)

    def objective_function(self, x, *args, **kwargs):
        input_params = {k: x[i] for (i, k) in enumerate(self.opt_param_names)}
        model_params = self.model_params | convert_to_model_params(input_params)
        X0 = self.compute_initial_state(
            X0=self.input_initial_state, U0=self.U[:, 0], params=model_params
        )

        try:
            model_predictions = self.model_function(
                T=self.T, U=self.U, X0=X0, **model_params
            )[self.training_data_columns].to_numpy()
        except Exception:
            return 1e9

        scaled_model_predictions = self.scale_data(model_predictions)

        # --- Weighted MSE ----------------------------------
        errors = self.training_data - scaled_model_predictions
        weighted_errors = (errors**2) * self.variable_weights  # shape (N, n_cols)
        return weighted_errors.mean()

    def scale_data(self, data):
        """Helper to scale data"""
        return self.data_scaler.transform(data)

    def rescale_data(self, data):
        """Helper rescale data"""
        return self.data_scaler.inverse_transform(data)


def make_min_constraint(param_name: str, min_val: float, params_bounds: dict):
    """
    Create a constraint: param >= min_val

    Parameters
    ----------
    param_name : str
        Name of the parameter to constrain
    min_val : float
        Minimum allowed value
    params_bounds : dict
        Original params_bounds dict used in MyOptimizationProblem
        (needed to determine parameter ordering)
    """
    opt_param_names = list(params_bounds.keys())

    def constraint(x):
        params = {k: x[i] for i, k in enumerate(opt_param_names)}
        return min_val - params[param_name]  # <= 0 when satisfied

    return constraint


def make_max_constraint(param_name: str, max_val: float, params_bounds: dict):
    """
    Create a constraint: param <= max_val

    Parameters
    ----------
    param_name : str
        Name of the parameter to constrain
    max_val : float
        Maximum allowed value
    params_bounds : dict
        Original params_bounds dict used in MyOptimizationProblem
        (needed to determine parameter ordering)
    """
    opt_param_names = list(params_bounds.keys())

    def constraint(x):
        params = {k: x[i] for i, k in enumerate(opt_param_names)}
        return params[param_name] - max_val  # <= 0 when satisfied

    return constraint


def convert_to_model_params(input_params: dict):
    """Helper to convert between our input_params and the model input_params.

    Groups entries like 'batt_ocv_coeffs_0', 'batt_ocv_coeffs_1', ...
    and 'batt_R_1_lut_0', 'batt_R_1_lut_1', ... into lists under
    'batt_k_V_OC_coeffs' and 'batt_R_1_lut', respectively.
    """
    params = {}
    suffixes = ("coeffs", "lut")

    for suffix in suffixes:
        # Find all grouped parameter base names for this suffix
        base_keys = {
            key.split(f"{suffix}_")[0] + suffix
            for key in list(input_params.keys())
            if f"{suffix}_" in key
        }

        for base in base_keys:
            # Collect and sort by index: base_0, base_1, ...
            indexed_keys = [
                (int(k.rsplit("_", 1)[-1]), k)
                for k in list(input_params.keys())
                if k.startswith(f"{base}_")
            ]
            indexed_keys.sort(key=lambda x: x[0])

            params[base] = [input_params.pop(k) for _, k in indexed_keys]

    return input_params | params


def _json_default(value):
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": True,
            "dtype": str(value.dtype),
            "shape": value.shape,
            "data": value.tolist(),
        }
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _json_object_hook(obj):
    if obj.get("__ndarray__") is True:
        arr = np.array(obj["data"], dtype=obj["dtype"])
        shape = tuple(obj["shape"])
        if arr.shape != shape:
            arr = arr.reshape(shape)
        return arr
    return obj


def save_model_params_to_json(filepath: str, params: dict):
    """Save model parameters to a JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, sort_keys=True, default=_json_default)


def load_model_params_from_json(filepath: str) -> dict:
    """Load model parameters from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f, object_hook=_json_object_hook)


def estimate_polynomial_coefficient_bounds(
    x_interval, y_bounds, degree, grid_points=1000
):
    """
    Estimate bounds for each coefficient a_j (j = 0, ..., degree) of a polynomial
      p(x) = a_0 + a_1*x + ... + a_degree*x^degree
    subject to:
      y_min <= p(x) <= y_max  for all x in [x_min, x_max].

    The "for all x" constraint is approximated by enforcing the constraints on a dense grid.

    Parameters:
        x_interval : tuple (x_min, x_max)
        y_bounds   : tuple (y_min, y_max)
        degree     : int, degree of the polynomial (there are degree+1 coefficients)
        grid_points: int, number of grid points in [x_min, x_max] for discretization.

    Returns:
        bounds: list of tuples [(a_0_min, a_0_max), (a_1_min, a_1_max), ...]
    """
    from scipy.optimize import linprog

    x_min, x_max = x_interval
    y_min, y_max = y_bounds

    # Create a dense grid on the x interval.
    x_grid = np.linspace(x_min, x_max, grid_points)
    M = grid_points

    # Build the design matrix: each row corresponds to an x value.
    # Each row is [1, x, x^2, ..., x^degree]
    X = np.vstack([x_grid**i for i in range(degree + 1)]).T  # shape: (M, degree+1)

    # Set up inequality constraints.
    # For each x in x_grid, we require:
    #    a_0 + a_1*x + ... + a_degree*x^degree <= y_max   and
    #   -a_0 - a_1*x - ... - a_degree*x^degree <= -y_min  (i.e. p(x) >= y_min)
    A_upper = X  # p(x) <= y_max
    A_lower = -X  # -p(x) <= -y_min
    A_ub = np.vstack([A_upper, A_lower])  # Combined constraint matrix
    b_upper = np.full(M, y_max)
    b_lower = np.full(M, -y_min)
    b_ub = np.concatenate([b_upper, b_lower])

    # Prepare a list to hold the computed bounds.
    bounds_list = []

    # For each coefficient index j, solve two LPs: one to maximize a_j, one to minimize a_j.
    for j in range(degree + 1):
        # --- Maximize a_j:  maximize a_j  <=> minimize -a_j ---
        # Objective: minimize c^T a, where c[j] = -1 and all other entries are 0.
        c_max = np.zeros(degree + 1)
        c_max[j] = -1  # maximize a_j by minimizing -a_j.
        res_max = linprog(
            c=c_max,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=[(None, None)] * (degree + 1),
            method="highs",
        )
        if not res_max.success:
            raise RuntimeError(
                f"LP maximizing coefficient a_{j} failed: {res_max.message}"
            )
        a_j_max = res_max.x[j]

        # --- Minimize a_j: objective is to minimize a_j ---
        c_min = np.zeros(degree + 1)
        c_min[j] = 1
        res_min = linprog(
            c=c_min,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=[(None, None)] * (degree + 1),
            method="highs",
        )
        if not res_min.success:
            raise RuntimeError(
                f"LP minimizing coefficient a_{j} failed: {res_min.message}"
            )
        a_j_min = res_min.x[j]

        bounds_list.append((a_j_min, a_j_max))

    return bounds_list


def plot_optimization_params(
    result,
    params_bounds,
    scatter_alpha=0.5,
    scatter_size=5,
    line_color="k",
    line_width=1.5,
    max_columns=5,
    show=True,
):
    # Extract data from the result's history
    parameters = []
    generations = []
    optimal_parameters = []

    for gen_idx, entry in enumerate(result.history):
        X = entry.pop.get("X")  # Parameter values
        F = entry.pop.get("F")  # Objective values

        generations.extend(
            [gen_idx] * len(X)
        )  # Repeat the generation index for each individual
        parameters.extend(X)

        # Find the best parameter set in the current generation
        best_idx = np.argmin(F)
        optimal_parameters.append(X[best_idx])

    parameters = np.array(parameters)
    generations = np.array(generations)
    optimal_parameters = np.array(optimal_parameters)

    # Number of parameters
    parameters_names = list(params_bounds.keys())
    n_params = parameters.shape[1]

    # Determine subplot layout
    n_rows = (n_params + max_columns - 1) // max_columns  # Calculate rows needed
    n_cols = min(n_params, max_columns)

    # Create subplots
    this_figsize = figsize()
    if this_figsize[0] > n_cols * 2:
        this_figsize = (n_cols * 2, n_rows * 4)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        sharey=True,
        figsize=this_figsize,
        layout="constrained",
    )
    axes = np.array(axes).reshape(-1)  # Flatten axes for easier iteration

    for i, ax in enumerate(axes[:n_params]):
        # Scatter plot for parameter i
        ax.scatter(
            parameters[:, i],
            generations,
            alpha=scatter_alpha,
            s=scatter_size,
            label="Samples",
        )

        # Plot optimal parameter values as a black line
        ax.plot(
            optimal_parameters[:, i],
            np.arange(len(optimal_parameters)),
            color=line_color,
            lw=line_width,
            label="Optimal",
        )

        ax.set_title(parameters_names[i])
        ax.set_xlabel("Value")
        if i % n_cols == 0:
            ax.set_ylabel("Generation")  # Label only the first subplot in each row
        ax.grid(True)

    # Turn off unused subplots
    for ax in axes[n_params:]:
        ax.axis("off")

    # Adjust subplot spacing
    plt.gca().invert_yaxis()
    plt.subplots_adjust(wspace=0.3, hspace=0.4)  # Add spacing between subplots

    if show:
        plt.show()

    return fig


def plot_optimization_error(
    result, show=True, fill_percentiles=True, running_best=False
):
    n_evals = []
    best_vals = []
    median_vals = []
    p10_vals = []
    p90_vals = []

    for algo in result.history:
        n_evals.append(getattr(getattr(algo, "evaluator", None), "n_eval", np.nan))

        opt = getattr(algo, "opt", None)
        if opt is None:
            best_vals.append(np.nan)
            median_vals.append(np.nan)
            p10_vals.append(np.nan)
            p90_vals.append(np.nan)
            continue

        # safely obtain F
        try:
            F = opt.get("F")
        except TypeError:
            F = getattr(opt, "F", None)
        except Exception:
            F = None

        # safely obtain feasible
        try:
            feasible = opt.get("feasible")
        except TypeError:
            feasible = getattr(opt, "feasible", None)
        except Exception:
            feasible = None

        if F is None:
            best_vals.append(np.nan)
            median_vals.append(np.nan)
            p10_vals.append(np.nan)
            p90_vals.append(np.nan)
            continue

        F = np.asarray(F)
        # reduce to first objective (1-D)
        if F.ndim > 1:
            # common case: shape (n_ind, n_obj)
            F1 = F[:, 0].ravel()
        else:
            F1 = F.ravel()

        # Build feasible_mask as 1-D boolean array same length as F1
        if feasible is None:
            feasible_mask = np.ones_like(F1, dtype=bool)
        else:
            feasible_arr = np.asarray(feasible)
            # If feasible is 2D (e.g. per-constraint flags), combine with all(axis=1)
            if feasible_arr.ndim == 2:
                # treat a row as feasible only if all constraints satisfied
                feasible_mask = np.all(feasible_arr, axis=1).astype(bool)
            elif feasible_arr.ndim == 1:
                feasible_mask = feasible_arr.astype(bool)
            else:
                # unexpected shape: try to reduce to 1-D
                feasible_mask = feasible_arr.ravel()[: F1.shape[0]].astype(bool)

            # ensure same length as F1: if longer, trim; if shorter, pad with False
            if feasible_mask.shape[0] > F1.shape[0]:
                feasible_mask = feasible_mask[: F1.shape[0]]
            elif feasible_mask.shape[0] < F1.shape[0]:
                pad = np.zeros(F1.shape[0] - feasible_mask.shape[0], dtype=bool)
                feasible_mask = np.concatenate([feasible_mask, pad])

        # Now safe to index
        if feasible_mask.sum() == 0:
            best_vals.append(np.nan)
            median_vals.append(np.nan)
            p10_vals.append(np.nan)
            p90_vals.append(np.nan)
        else:
            sel = F1[feasible_mask]
            best_vals.append(float(np.nanmin(sel)))
            median_vals.append(float(np.nanmedian(sel)))
            p10_vals.append(float(np.nanpercentile(sel, 10)))
            p90_vals.append(float(np.nanpercentile(sel, 90)))

    # finalize arrays
    n_evals = np.array(n_evals).ravel()
    best_vals = np.array(best_vals, dtype=float)
    median_vals = np.array(median_vals, dtype=float)
    p10_vals = np.array(p10_vals, dtype=float)
    p90_vals = np.array(p90_vals, dtype=float)

    if running_best:
        best_vals_running = np.minimum.accumulate(
            np.where(np.isfinite(best_vals), best_vals, np.inf)
        )
        best_vals_running[best_vals_running == np.inf] = np.nan
        y_to_plot = best_vals_running
        label_best = "running best"
    else:
        y_to_plot = best_vals
        label_best = "best"

    fig = plt.figure(figsize=figsize())

    plt.plot(n_evals, y_to_plot, label=label_best, marker="o", markersize=3)
    plt.plot(n_evals, median_vals, label="median", linestyle="--", alpha=0.7)

    if fill_percentiles:
        finite_mask = np.isfinite(p10_vals) & np.isfinite(p90_vals)
        if finite_mask.any():
            plt.fill_between(
                n_evals[finite_mask],
                p10_vals[finite_mask],
                p90_vals[finite_mask],
                alpha=0.2,
                label="10-90 percentile",
            )

    plt.xlabel("Evaluations")
    plt.ylabel("Objective (MSE)")
    plt.title("Optimization progress")
    plt.grid(True)

    if show:
        plt.legend()
        plt.show()

    return fig


def compute_metrics(y_true: np.array, y_pred: np.array) -> dict:
    y_residuals = y_true - y_pred

    return dict(
        residuals=y_residuals,
        MSE=mean_squared_error(y_true, y_pred),
        MAE=mean_absolute_error(y_true, y_pred),
        RMSE=root_mean_squared_error(y_true, y_pred),
        AE=np.sum(y_residuals),
        R2=r2_score(y_true, y_pred),
    )


def plot_compare(df_true, df_pred, column: str, show=True):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=figsize(), gridspec_kw={"height_ratios": [2, 1]}
    )

    ax1.plot(df_pred.index, df_true[column], label="data")
    ax1.plot(df_pred.index, df_pred[column], label="model")
    ax1.set_title(f"Comparing {column}")
    if show:
        ax1.legend()

    (
        y_residuals,
        y_mean_squared_error,
        y_mean_absolute_error,
        y_root_mean_squared_error,
        y_accumulated_error,
        y_r2_score,
    ) = compute_metrics(df_true[column].to_numpy(), df_pred[column].to_numpy()).values()

    ax2.plot(df_pred.index, y_residuals, label="Residuals", color="gray")

    title = (
        f"MSE: {y_mean_squared_error:.2f},\t"
        f"MAE: {y_mean_absolute_error:.2f},\t"
        f"RMSE: {y_root_mean_squared_error:.2f},\t"
        f"AE: {y_accumulated_error:.2f},\t"
        f"R²: {y_r2_score:.2f}"
    )

    ax2.set_title(title, fontsize=12)
    if show:
        ax2.legend()

    ax1.set_ylabel("Value")

    ax2.set_ylabel("Error")
    ax2.set_xlabel("Time [s]")

    if show:
        plt.show()

    return fig
