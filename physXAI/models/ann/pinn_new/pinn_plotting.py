from pathlib import Path

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from physXAI.plotting.plotting import plot_training_history, subplots
from physXAI.preprocessing.training_data import TrainingDataGeneric


def _plot_history_metric(histroy: dict, metric_key: str, y_axis_title: str) -> go.Figure:

    if metric_key not in histroy:
        raise KeyError(f"The metric '{metric_key}' is not contained in the training history!")

    validation_key = f"val_{metric_key}"

    training_values = list(histroy[metric_key])
    validation_values = list(histroy.get(validation_key, []))

    training_name = f"Training {y_axis_title}"
    validation_name = f"Validation {y_axis_title}"

    data = {
        "Epochs": list(range(1, len(training_values) + 1)) + list(range(1, len(validation_values) + 1)),
        "Value": training_values + validation_values,
        "Type": [training_name] * len(training_values) + [validation_name] * len(validation_values),
    }

    frame = pd.DataFrame(data)

    color_map = {
        training_name: "green",
        validation_name: "blue",
    }

    figure = px.line(
        frame,
        x="Epochs",
        y="Value",
        color="Type",
        markers=True,
        color_discrete_map=color_map,
    )

    figure.update_layout(
        xaxis_title="Epochs",
        yaxis_title=y_axis_title,
    )

    return figure


def _add_weighted_loss_columns(history_frame: pd.DataFrame, physics_loss_weight: float, wall_dynamics_loss_weight: float, tabs_physics_loss_weight: float, use_tabs_physics_loss: bool):

    for prefix in ("", "val_"):
        physics_key = f"{prefix}physics_loss"

        if physics_key in history_frame:
            history_frame[f"{prefix}weighted_physics_loss"] = physics_loss_weight * history_frame[physics_key]

        wall_key = f"{prefix}wall_dynamics_loss"

        if wall_key in history_frame:
            history_frame[f"{prefix}weighted_wall_dynamics_loss"] = wall_dynamics_loss_weight * history_frame[wall_key]

        tabs_key = f"{prefix}tabs_physics_loss"

        if use_tabs_physics_loss and tabs_key in history_frame:
            history_frame[f"{prefix}weighted_tabs_physics_loss"] = tabs_physics_loss_weight * history_frame[tabs_key]


def _plot_weighted_loss_contributions(history_frame: pd.DataFrame, physics_loss_weight: float, wall_dynamics_loss_weight: float, tabs_physics_loss_weight: float, use_tabs_physics_loss: bool) -> go.Figure:

    if "val_prediction_loss" in history_frame:
        prefix = "val_"
        split_name = "validation"
    else:
        prefix = ""
        split_name = "Training"

    epochs = list(history_frame.index)

    series = [
        (
            "Prediction Loss", 
            history_frame[f"{prefix}prediction_loss"].tolist(), 
            "green",
        ),
        (
            f"{physics_loss_weight:g} x Physics Loss",
            (physics_loss_weight * history_frame[f"{prefix}physics_loss"]).tolist(),
            "blue",
        )
    ]

    wall_key = f"{prefix}wall_dynamics_loss"

    if wall_key in history_frame:
        series.append(
            (
                f"{wall_dynamics_loss_weight:g} x Wall Dynamics Loss",
                (wall_dynamics_loss_weight * history_frame[wall_key]).tolist(),
                "red",
            ),
        )

    tabs_key = f"{prefix}tabs_physics_loss"

    if use_tabs_physics_loss and tabs_key in history_frame:
        series.append(
            (
                f"{tabs_physics_loss_weight:g} x Tabs Physics Loss",
                (tabs_physics_loss_weight * history_frame[tabs_key]).tolist(),
                "orange",
            ),
        )

    series.append(
        (
            "Total Loss",
            history_frame[f"{prefix}loss"].tolist(),
            "black",
        ),
    )

    plot_data = {
        "Epochs": [],
        "Weighted Loss": [],
        "Type": [],
    }

    color_map = {}

    for name, values, color in series:
        plot_data["Epochs"].extend(epochs)
        plot_data["Weighted Loss"].extend(values)
        plot_data["Type"].extend([name] * len(values))
        color_map[name] = color

    figure = px.line(
        pd.DataFrame(plot_data),
        x="Epochs",
        y="Weighted Loss",
        color="Type",
        markers=True,
        color_discrete_map=color_map,
    )

    figure.update_traces(
        line=dict(dash="dash"),
        selector=dict(name="Total Loss"),
    )

    figure.update_layout(
        xaxis_title="Epochs",
        yaxis_title="Weighted Loss Contribution",
        legend_title_text=split_name
    )

    return figure


def plot_and_save_pinn_training_history(td: TrainingDataGeneric, training_model, output_directory, model_name: str) -> go.Figure:
    """
    
    """

    if td.training_record is None:
        raise ValueError("No training history is available!")

    history = td.training_record.history

    required_keys = {
        "loss",
        "prediction_loss",
        "physics_loss",
        "rmse",
    }

    missing_keys = required_keys.difference(history)

    if missing_keys:
        raise ValueError(f"The following history values are missing: {sorted(missing_keys)}!")

    physics_loss_weight = float(getattr(training_model, "physics_loss_weight", 1.0))

    wall_dynamics_loss_weight = float(getattr(training_model, "wall_dynamics_loss_weight", 0.0))

    use_tabs_physics_loss = bool(getattr(training_model, "use_tabs_physics_loss", False))

    tabs_physics_loss_weight = float(getattr(training_model, "tabs_physics_loss_weight", 0.0))

    history_frame = pd.DataFrame(history)

    history_frame.index = range(1, len(history_frame) + 1)
    history_frame.index.name = "Epochs"

    _add_weighted_loss_columns(
        history_frame=history_frame,
        physics_loss_weight=physics_loss_weight,
        wall_dynamics_loss_weight=wall_dynamics_loss_weight,
        use_tabs_physics_loss=use_tabs_physics_loss,
        tabs_physics_loss_weight=tabs_physics_loss_weight,
    )

    rmse_figure = plot_training_history(td)

    total_loss_figure = _plot_history_metric(
        histroy=history,
        metric_key="loss",
        y_axis_title="Total Loss",
    )

    physics_loss_figure = _plot_history_metric(
        histroy=history,
        metric_key="physics_loss",
        y_axis_title="Physics Loss",
    )

    figures = [
        {
            "title": "RMSE",
            "type": "scatter",
            "figure": rmse_figure,
        },
        {
            "title": "Total Loss",
            "type": "scatter",
            "figure": total_loss_figure,
        },

        {
            "title": "Physics Loss",
            "type": "scatter",
            "figure": physics_loss_figure,
        },
    ]

    if "wall_dynamics_loss" in history:
        wall_loss_figure = _plot_history_metric(
            histroy=history,
            metric_key="wall_dynamics_loss",
            y_axis_title="Wall Dynamics Loss",
        )

        figures.append(
            {
                "title": "Wall Dynamics Loss",
                "type": "scatter",
                "figure": wall_loss_figure,
            },
        )

    if use_tabs_physics_loss and "tabs_physics_loss" in history:
        tabs_loss_figure = _plot_history_metric(
            histroy=history,
            metric_key="tabs_physics_loss",
            y_axis_title="Tabs Physics Loss"
        )

        figures.append(
            {
                "title": "Tabs Physics Loss",
                "type": "scatter",
                "figure": tabs_loss_figure,
            }
        )

    weighted_loss_figure = (
        _plot_weighted_loss_contributions(
            history_frame=history_frame,
            physics_loss_weight=physics_loss_weight,
            wall_dynamics_loss_weight=wall_dynamics_loss_weight,
            use_tabs_physics_loss=use_tabs_physics_loss,
            tabs_physics_loss_weight=tabs_physics_loss_weight,
        )
    )

    figures.append(
        {
            "title": "Weighted Loss Contributions",
            "type": "scatter",
            "figure": weighted_loss_figure,
        },
    )



    

    display_name = model_name.replace("_", " ")

    complete_figure = subplots(
        f"PINN Training History: {display_name}",
        *figures,
    )

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    history_path = (output_directory / f"{model_name}_training_history.csv")

    figure_path = (output_directory / f"{model_name}_training_history.html")

    history_frame.to_csv(history_path)

    complete_figure.write_html(
        str(figure_path),
        include_plotlyjs=True,
        full_html=True,
    )

    selection_key = ("val_loss" if "val_loss" in history_frame else "loss")

    best_epoch = int(history_frame[selection_key].idxmin())

    best_loss = float(history_frame.loc[best_epoch, selection_key])

    print(f"\nModel: {model_name}")
    print(f"Trained epochs: {len(history_frame)}")
    print(f"Best epochs according to {selection_key}: {best_epoch}")
    print(f"Best {selection_key}: {best_loss:.6e}")
    print(f"Training history saved to: {history_path}")
    print(f"Interactive figure saved to: {figure_path}")

    return complete_figure


def plot_and_save_single_step_prediction(
        td: TrainingDataGeneric,
        output_directory,
        model_name: str,
        time_step: float,
        t_air_column: str = 'TAir',
        split: str = 'test',
        plot_start: int = 0,
        plot_steps: int | None = None,
        temperature_unit: str = 'K',
        show: bool = True,
) -> go.Figure:
    """
    
    """

    split_data = {
        "train": (td.X_train_single, td.y_train_single, td.y_train_pred_single),
        "val": (td.X_val_single, td.y_val_single, td.y_val_pred_single),
        "test": (td.X_test_single, td.y_test_single, td.y_test_pred_single),
    }

    if split not in split_data:
        raise ValueError("split must be 'train, 'val' or 'test'!")

    X, delta_true, delta_pred = split_data[split]

    if X is None or delta_true is None or delta_pred is None:
        raise ValueError(f"No predictions are available for split '{split}'!")

    X = np.asarray(X)
    delta_true = np.asarray(delta_true).reshape(-1)
    delta_pred = np.asarray(delta_pred).reshape(-1)

    if len(X) != len(delta_true) or len(delta_true) != len(delta_pred):
        raise ValueError("Inputs, targets and predictions have different lengths!")

    t_air_index = td.columns.index(t_air_column)
    t_air_k = X[:, t_air_index].reshape(-1)

    t_air_true_k1 = t_air_k + delta_true
    t_air_pred_k1 = t_air_k + delta_pred

    delta_error = delta_pred - delta_true

    time_hours = np.arange(len(delta_true), dtype=float) * float(time_step) / 3600.0

    results = pd.DataFrame(
        {
            "time_hours": time_hours,
            "t_air_k": t_air_k,
            "delta_t_air_true": delta_true,
            "delta_t_air_pred": delta_pred,
            "delta_t_air_error": delta_error,
            "t_air_true_k1": t_air_true_k1,
            "t_air_pred_k1": t_air_pred_k1,
        }
    )

    rmse = float(np.sqrt(np.mean(delta_error**2)))
    mae = float(np.mean(np.abs(delta_error)))
    bias = float(np.mean(delta_error))

    presistence_rmse = float(np.sqrt(np.mean(delta_true**2)))

    if presistence_rmse > 0:
        skill_vs_persistance = 1.0 - rmse / presistence_rmse
    else:
        skill_vs_persistance = np.nan

    summary = pd.DataFrame(
        [
            {
                "model": model_name,
                "split": split,
                "samples": len(results),
                "rmse_change_t_air": rmse,
                "mae_change_t_air": mae,
                "bias_change_t_air": bias,
                "persistence_rmse": presistence_rmse,
                "skill_vs_persistance": skill_vs_persistance,
            }
        ]
    )

    if plot_start < 0 or plot_start >= len(results):
        raise ValueError("plot_start is outside the available data!")

    if plot_steps is None:
        plot_end = len(results)
    else:
        plot_end = min(plot_start + plot_steps, len(results))

    plot_frame = results.iloc[plot_start:plot_end]

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=(
            "Single-step prediction of Change(TAir)",
            "Reconstructed Single-step room temperature"
        ),
    )

    figure.add_trace(
        go.Scatter(
            x=plot_frame["time_hours"],
            y=plot_frame["delta_t_air_true"],
            name="Measured Change(TAir)",
            line=dict(color="black"),
        ),
        row=1,
        col=1,
    )

    figure.add_trace(
        go.Scatter(
            x=plot_frame["time_hours"],
            y=plot_frame["delta_t_air_pred"],
            name="Predicted Change(TAir)",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )

    figure.add_trace(
        go.Scatter(
            x=plot_frame["time_hours"],
            y=np.zeros(len(plot_frame)),
            name="Persistence: Change(TAir) = 0",
            line=dict(color="gray", dash="dot"),
        ),
        row=1,
        col=1,
    )

    figure.add_trace(
        go.Scatter(
            x=plot_frame["time_hours"],
            y=plot_frame["t_air_true_k1"],
            name="Measured TAir(k+1)",
            line=dict(color="black"),
        ),
        row=2,
        col=1,
    )

    figure.add_trace(
        go.Scatter(
            x=plot_frame["time_hours"],
            y=plot_frame["t_air_pred_k1"],
            name="Predicted TAir(k+1)",
            line=dict(color="blue"),
        ),
        row=2,
        col=1,
    )

    figure.add_trace(
        go.Scatter(
            x=plot_frame["time_hours"],
            y=plot_frame["t_air_k"],
            name="Persistence: TAir(k)",
            line=dict(color="gray", dash="dot"),
        ),
        row=2,
        col=1,
    )

    figure.update_yaxes(
        title_text=f"Change(TAir) [{temperature_unit}]",
        row=1,
        col=1,
    )

    figure.update_yaxes(
        title_text=f"TAir [{temperature_unit}]",
        row=2,
        col=1,
    )

    figure.update_xaxes(
        title_text="Time since split start [h]",
        row=2,
        col=1,
    )

    figure.update_layout(
        title=f"Single-step evaluation: {model_name}",
        height=800,
        width=1250,
        hovermode="x unified",
    )

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    results_path = output_directory / f"{model_name}_{split}_single_step_prediction.csv"

    summary_path = output_directory / f"{model_name}_{split}_single_step_metrics.csv"

    figure_path = output_directory / f"{model_name}_{split}_single_step_prediction.html"

    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)

    figure.write_html(
        str(figure_path),
        include_plotlyjs=True,
        full_html=True,
    )

    print("\nSingle-step evaluation:")
    print(summary.to_string(index=False))
    print(f"Predictions saved to: {results_path}")
    print(f"Metrics saved to: {summary_path}")
    print(f"Figure saved to: {figure_path}")

    if show:
        figure.show()

    return figure