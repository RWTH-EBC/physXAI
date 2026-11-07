"""
Plots of training data, predictions and errors (plotly).

`model_report(td, title)` combines all plots of a trained model into one figure. Figures are not shown
by default (`show=False`); the model pipelines save them as html file next to the model instead.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from physXAI.preprocessing.training_data import TrainingDataGeneric

logger = logging.getLogger("physXAI")

SPLITS = ("train", "val", "test")
SPLIT_COLORS = {"train": "green", "val": "blue", "test": "red"}
_FILL = {"train": "rgba(0,128,0,0.15)", "val": "rgba(0,0,255,0.15)", "test": "rgba(255,0,0,0.15)"}


def subplots(header: str, *figures, show: bool = False) -> go.Figure:
    """
    Stacks figures into one figure with one row per figure.

    Args:
        header: Title of the figure.
        *figures: Dicts {'title': str, 'type': 'scatter' | 'table', 'figure': go.Figure}.
        show: Whether to open the figure (in the browser).
    """
    fig = make_subplots(
        rows=len(figures),
        cols=1,
        subplot_titles=[f["title"] for f in figures],
        specs=[[{"type": f["type"]}] for f in figures],
    )
    for i, fig_dict in enumerate(figures):
        figure = fig_dict["figure"]
        for trace in figure["data"]:
            fig.add_trace(trace, row=i + 1, col=1)
        if fig_dict["type"] == "scatter":
            fig.update_xaxes(title_text=figure.layout.xaxis.title.text, row=i + 1, col=1)
            fig.update_yaxes(title_text=figure.layout.yaxis.title.text, row=i + 1, col=1)
    fig.update_layout(title_text=header, height=350 * len(figures), width=1250)
    if show:
        fig.show()
    return fig


def save_figure(fig: go.Figure, path: str | Path) -> Path:
    """Saves a figure as html file (plotly is loaded from its CDN when the file is opened)."""
    path = Path(path).with_suffix(".html")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn")
    logger.info("Saved plot to %s", path)
    return path


# --- data access ---------------------------------------------------------------------------------
def _split_values(td: TrainingDataGeneric, split: str, output: int):
    """True and predicted values of one output of a split (None if the split is empty)."""
    y = getattr(td, f"y_{split}_single")
    prediction = getattr(td, f"y_{split}_pred_single")
    if y is None or prediction is None:
        return None
    y = np.asarray(y).reshape(len(y), -1)[:, output]
    prediction = np.asarray(prediction).reshape(len(prediction), -1)[:, output]
    return y, prediction


def _output_names(td: TrainingDataGeneric) -> list[str]:
    from physXAI.evaluation.metrics import output_names

    return output_names(td)


def binned_median(y: np.ndarray, prediction: np.ndarray, bins: int = 30):
    """
    Median (and quartiles) of the predictions in quantile bins of the true values.

    Returns:
        tuple: median of the true values per bin, median, 25 % and 75 % quantile of the predictions.
    """
    edges = np.unique(np.quantile(y, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        q25, q50, q75 = np.quantile(prediction, [0.25, 0.5, 0.75])
        return np.array([np.median(y)]), np.array([q50]), np.array([q25]), np.array([q75])
    index = np.clip(np.digitize(y, edges[1:-1]), 0, len(edges) - 2)
    x, median, low, high = [], [], [], []
    for b in range(len(edges) - 1):
        mask = index == b
        if mask.sum() < 3:
            continue
        x.append(np.median(y[mask]))
        q25, q50, q75 = np.quantile(prediction[mask], [0.25, 0.5, 0.75])
        median.append(q50)
        low.append(q25)
        high.append(q75)
    return np.array(x), np.array(median), np.array(low), np.array(high)


#: Maximum number of points shown per trace (statistics always use all values)
MAX_POINTS = 20000


def _thin(n: int, max_points: int = MAX_POINTS) -> np.ndarray:
    """Indices of at most `max_points` evenly spread of `n` points."""
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_points).astype(int))


# --- single plots --------------------------------------------------------------------------------
def plot_prediction_correlation(
    td: TrainingDataGeneric, output: int = 0, bins: int = 30, iqr: bool = True
) -> go.Figure:
    """
    Predicted over true values: small transparent points, the binned median of every split as bold line
    and (optionally) the interquartile range of the predictions per bin as band.

    Args:
        td: Evaluated training data.
        output: Index of the output.
        bins: Number of quantile bins of the median.
        iqr: Whether to show the 25 % - 75 % band of the predictions per bin.
    """
    fig = go.Figure()
    lowest, highest = np.inf, -np.inf
    for split in SPLITS:
        values = _split_values(td, split, output)
        if values is None:
            continue
        y, prediction = values
        lowest, highest = min(lowest, y.min()), max(highest, y.max())
        name, color = split.capitalize(), SPLIT_COLORS[split]
        shown = _thin(len(y))
        fig.add_trace(
            go.Scattergl(
                x=y[shown],
                y=prediction[shown],
                mode="markers",
                name=name,
                legendgroup=split,
                marker=dict(color=color, size=3, opacity=0.3),
            )
        )
        x, median, low, high = binned_median(y, prediction, bins)
        if iqr and len(x) > 1:
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([x, x[::-1]]),
                    y=np.concatenate([high, low[::-1]]),
                    fill="toself",
                    fillcolor=_FILL[split],
                    line=dict(width=0),
                    hoverinfo="skip",
                    name=f"{name} IQR",
                    legendgroup=split,
                    showlegend=False,
                )
            )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=median,
                mode="lines",
                name=f"{name} median",
                legendgroup=split,
                line=dict(color=color, width=4),
            )
        )
    if np.isfinite(lowest):
        fig.add_trace(
            go.Scatter(
                x=[lowest, highest],
                y=[lowest, highest],
                mode="lines",
                name="Ref",
                line=dict(color="black", dash="dash"),
            )
        )
    fig.update_layout(xaxis_title="True values", yaxis_title="Predicted values")
    return fig


def plot_predictions(td: TrainingDataGeneric, output: int = 0) -> go.Figure:
    """True values sorted by size (black) and the corresponding predictions per split."""
    truth, predictions, splits = [], [], []
    for split in SPLITS:
        values = _split_values(td, split, output)
        if values is not None:
            truth.append(values[0])
            predictions.append(values[1])
            splits.append(np.full(len(values[0]), split))
    truth, predictions, splits = np.concatenate(truth), np.concatenate(predictions), np.concatenate(splits)
    order = np.argsort(truth, kind="stable")
    truth, predictions, splits = truth[order], predictions[order], splits[order]
    index = np.arange(len(truth))
    shown = _thin(len(truth))
    index, truth, predictions, splits = index[shown], truth[shown], predictions[shown], splits[shown]
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(x=index, y=truth, mode="markers", name="True values", marker=dict(color="black", size=3))
    )
    for split in SPLITS:
        mask = splits == split
        if mask.any():
            fig.add_trace(
                go.Scattergl(
                    x=index[mask],
                    y=predictions[mask],
                    mode="markers",
                    name=split.capitalize(),
                    marker=dict(color=SPLIT_COLORS[split], size=3, opacity=0.5),
                )
            )
    fig.update_layout(xaxis_title="Sample", yaxis_title="Value")
    return fig


def plot_training_history(td: TrainingDataGeneric) -> go.Figure:
    """Training and validation error per epoch (also of a multi step training, if any)."""
    fig = go.Figure(layout=dict(xaxis_title="Epochs", yaxis_title="RMSE"))
    records = [("", td.training_record, "solid")]
    recursive = getattr(td, "recursive_training_record", None)
    if recursive is not None and recursive is not td.training_record:
        records.append((" (multi step)", recursive, "dot"))
    for suffix, record, dash in records:
        if record is None:  # e.g. closed form solutions
            continue
        history = record.history
        key = "rmse" if "rmse" in history else "loss"
        if key == "loss":
            fig.update_layout(yaxis_title="Loss")
        for prefix, name, color in (("", "Training", "green"), ("val_", "Validation", "blue")):
            values = history.get(prefix + key)
            if values:
                fig.add_trace(
                    go.Scatter(
                        x=np.arange(1, len(values) + 1),
                        y=values,
                        mode="lines+markers",
                        name=f"{name}{suffix}",
                        line=dict(color=color, dash=dash),
                    )
                )
    return fig


def plot_metrics_table(td: TrainingDataGeneric) -> go.Figure:
    """Table of the metrics, one row per output and metric."""
    metrics = td.metrics
    fig = go.Figure()
    by_output = metrics.by_output
    splits = [s for s in SPLITS if any(s in kpis for kpis in by_output.values())]
    columns = [[], []] + [[] for _ in splits]
    for name, kpis in by_output.items():
        for metric, label in (("rmse", "RMSE"), ("mae", "MAE"), ("bias", "Bias"), ("r2", "R2")):
            columns[0].append(name)
            columns[1].append(label)
            for j, split in enumerate(splits):
                value = kpis.get(split, {}).get(metric)
                columns[2 + j].append("" if value is None else f"{value:.4g}")
    fig.add_trace(
        go.Table(
            header=dict(values=["Output", "Metric", *[s.capitalize() for s in splits]]), cells=dict(values=columns)
        )
    )
    return fig


def plot_horizon_metrics(td, feature: str | None = None) -> go.Figure:
    """
    Error of the multi step prediction over the horizon per split: RMSE (solid), bias (dashed) and the
    band bias +- standard deviation of the error. The x axis is in hours if the time step is known.
    """
    metrics = td.horizon_metrics
    kpis_by_split = metrics.by_feature[feature] if feature is not None else metrics.kpis
    x = metrics.hours if metrics.hours is not None else metrics.steps
    fig = go.Figure()
    for split, kpis in kpis_by_split.items():
        color, name = SPLIT_COLORS.get(split, "black"), split.capitalize()
        bias = np.asarray(kpis["bias"])
        if "std" in kpis:
            std = np.asarray(kpis["std"])
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([x, x[::-1]]),
                    y=np.concatenate([bias + std, (bias - std)[::-1]]),
                    fill="toself",
                    fillcolor=_FILL.get(split, "rgba(0,0,0,0.1)"),
                    line=dict(width=0),
                    hoverinfo="skip",
                    name=f"{name} bias ± σ",
                    legendgroup=split,
                )
            )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=kpis["rmse"],
                mode="lines",
                name=f"{name} RMSE",
                legendgroup=split,
                line=dict(color=color, width=3),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x, y=bias, mode="lines", name=f"{name} bias", legendgroup=split, line=dict(color=color, dash="dash")
            )
        )
    fig.update_layout(
        xaxis_title="Prediction horizon [h]" if metrics.hours is not None else "Prediction step",
        yaxis_title=f"Error of {feature or metrics.feature}",
    )
    return fig


# --- report --------------------------------------------------------------------------------------
def model_report(td: TrainingDataGeneric, title: str, show: bool = False) -> go.Figure:
    """
    All plots of a trained model: per output the prediction correlation, the sorted predictions and (for
    multi step evaluations) the error over the horizon, followed by the training history and the metrics.

    Args:
        td: Evaluated training data.
        title: Title of the figure.
        show: Whether to open the figure.
    """
    names = _output_names(td)
    # PINN data hold further target columns next to the predicted outputs
    n_outputs = np.asarray(td.y_train_pred_single).reshape(len(td.y_train_pred_single), -1).shape[1]
    if len(names) != n_outputs:
        names = names[:n_outputs] if len(names) > n_outputs else [f"output_{j}" for j in range(n_outputs)]
    horizon = getattr(td, "horizon_metrics", None)
    figures = list()
    for j, name in enumerate(names):
        label = f" – {name}" if len(names) > 1 else ""
        figures.append(
            {"title": f"Prediction Correlation{label}", "type": "scatter", "figure": plot_prediction_correlation(td, j)}
        )
        figures.append({"title": f"Predictions Sorted{label}", "type": "scatter", "figure": plot_predictions(td, j)})
        if horizon is not None and name in horizon.by_feature:
            figures.append(
                {
                    "title": f"Prediction Error over the Horizon{label}",
                    "type": "scatter",
                    "figure": plot_horizon_metrics(td, name),
                }
            )
    if horizon is not None and not any(name in horizon.by_feature for name in names):
        figures.append(
            {"title": "Prediction Error over the Horizon", "type": "scatter", "figure": plot_horizon_metrics(td)}
        )
    if td.training_record is not None:
        figures.append({"title": "Training History", "type": "scatter", "figure": plot_training_history(td)})
    figures.append({"title": "Performance Metrics", "type": "table", "figure": plot_metrics_table(td)})
    shape = (td.physics or {}).get("shape_constraints")
    if shape:
        figures.append({"title": "Shape Constraints", "type": "table", "figure": plot_shape_constraints_table(shape)})
    return subplots(title, *figures, show=show)


def plot_shape_constraints_table(shape: dict) -> go.Figure:
    """
    Table of the shape constraints of a constrained least squares fit (`ModularANN(solver='qp', shape=...)`):
    certificate, maximum violation and the share of training samples where the constraint binds, followed by the
    train RMSE with and without the constraints (the cost of the physics).
    """
    rows = [[], [], [], [], []]
    for c in shape["constraints"]:
        rows[0].append(f"{c['type']} {c['column']} ({'+' if c['sign'] > 0 else '-'})")
        rows[1].append(c["certificate"])
        rows[2].append(f"{c['max_violation']:.2g}")
        rows[3].append("" if c["active_share_data"] is None else f"{c['active_share_data']:.1%}")
        rows[4].append(str(c["constraint_points"]))
    for label, value in (("Train RMSE (constrained)", shape["rmse_train"]),
                         ("Train RMSE (unconstrained)", shape["rmse_train_unconstrained"])):
        for k, cell in enumerate((label, f"{value:.4g}", "", "", "")):
            rows[k].append(cell)
    fig = go.Figure()
    fig.add_trace(go.Table(header=dict(values=["Constraint", "Certificate", "Max violation", "Binding on data",
                                               "Constraint points"]), cells=dict(values=rows)))
    return fig
