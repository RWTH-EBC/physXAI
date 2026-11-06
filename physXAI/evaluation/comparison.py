"""
Comparison of several models, each trained several times.

A single training run says little about a model: the split of the data and the initialization of the
weights change the errors, often more than the difference between two models. `repeat_training` trains
a model with several seeds and saves every run with `Run.save`. `ModelComparison` loads the saved runs
(or takes training data in memory) and compares the distributions of the errors:

- the single step metrics per output (RMSE, MAE, R2, bias),
- the errors over the prediction horizon (recursive single step models and RNNs), on a common horizon,
- plots of the distributions (box plots over the runs, median and band over the horizon).

Example:
    >>> run = Run("stored_data", "TAir_compare", overwrite=True)
    >>> repeat_training(file_path, prep, LinearRegressionModel(), seeds=5, run=run, name="linreg")
    >>> repeat_training(file_path, prep, ClassicalANNModel(epochs=50), seeds=5, run=run, name="ann")
    >>> comparison = ModelComparison.load(run.path)  # one model per sub directory
    >>> comparison.summary()
    >>> comparison.save(run)  # comparison.html and csv files

With the same seeds, all models use the same splits of the data, so their runs can be compared in pairs
(column `split_seed`).
"""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger("physXAI")

SPLITS = ("train", "val", "test")
TRAINING_DATA_SUFFIX = "training_data.json"

#: Colors of the models, in a fixed order (colorblind-safe categorical palette)
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


@dataclass
class RunRecord:
    """One trained model: its training data config (metrics) and, if available, the model and preprocessing configs."""

    model: str
    run: str
    training_data: dict
    model_config: dict | None = None
    preprocessing: dict | None = None
    path: str | None = None
    extra: dict = field(default_factory=dict)


# --- training ------------------------------------------------------------------------------------
def repeat_training(
    file_path: str,
    preprocessing,
    model,
    seeds: int | Iterable[int] = 5,
    run=None,
    name: str = "model",
    vary: Iterable[str] = ("split", "init"),
    plot: bool = False,
) -> ModelComparison:
    """
    Trains a model several times with different seeds.

    Args:
        file_path: Path to the data.
        preprocessing: The preprocessing. It is not changed, every run uses a copy from its config.
        model: The physXAI model. It is not changed, every run uses a copy.
        seeds: Number of runs (seeds 0, 1, ...) or the seeds. Use the same seeds for all compared models,
               then they are trained and tested on the same splits.
        run: Optional `Run`. Every run is saved into `<run>/<name>/seed_<seed>/` with `Run.save`.
        name: Name of the model (label in the comparison and name of the saved files).
        vary: What the seed changes: 'split' (the random split of the data) and / or 'init'
              (initialization of the weights and order of the batches).
        plot: Whether to save the report of every run (with `run`).

    Returns:
        ModelComparison: The runs of this model. Comparisons can be added: `cmp_a + cmp_b`.
    """
    seeds = list(range(seeds)) if isinstance(seeds, int) else list(seeds)
    vary = set(vary)
    unknown = vary - {"split", "init"}
    if unknown:
        raise ValueError(f"Unknown entries in 'vary': {sorted(unknown)}, options are 'split' and 'init'.")

    comparison = ModelComparison()
    for seed in seeds:
        prep = _preprocessing_with_seed(preprocessing, seed if "split" in vary else None)
        m = copy.copy(model)
        if "init" in vary and hasattr(m, "random_seed"):
            m.random_seed = seed
        logger.info("Training '%s' with seed %s", name, seed)
        td = prep.pipeline(file_path, model=m)  # inputs from the model, if the preprocessing has none
        m.pipeline(td, plot=False, save_model=False)
        run_name = f"seed_{seed}"
        if run is not None:
            from physXAI.io.run import Run

            sub = Run(run.path / name, run_name, overwrite=run.overwrite)
            sub.save(name, prep, m, td, plot=plot)
        comparison.add(name, td, run=run_name, model=m, preprocessing=prep)
    return comparison


def _preprocessing_with_seed(preprocessing, seed: int | None):
    from physXAI.preprocessing.preprocessing import PreprocessingData

    config = preprocessing.get_config()
    config.pop("transform", None)
    if seed is not None:
        config["random_state"] = seed
    prep = PreprocessingData.from_config(config)
    prep.transform = preprocessing.transform
    return prep


# --- comparison ----------------------------------------------------------------------------------
class ModelComparison:
    """
    Compares the metrics of several models with several runs each.

    Create it with `ModelComparison.load(...)` from saved runs, with `repeat_training`, or add evaluated
    training data with `add`.
    """

    def __init__(self, records: Iterable[RunRecord] = ()):
        self.records: list[RunRecord] = list(records)

    def __add__(self, other: ModelComparison) -> ModelComparison:
        return ModelComparison([*self.records, *other.records])

    def __len__(self) -> int:
        return len(self.records)

    def __repr__(self) -> str:
        counts = {m: sum(r.model == m for r in self.records) for m in self.models}
        return f"ModelComparison({counts})"

    @property
    def models(self) -> list[str]:
        """Labels of the models, in the order they were added."""
        return list(dict.fromkeys(r.model for r in self.records))

    # --- creation ------------------------------------------------------------------------------
    def add(self, label: str, training_data, run: str | None = None, model=None, preprocessing=None) -> ModelComparison:
        """
        Adds an evaluated training data object (after `model.pipeline(td)`).

        Args:
            label: Name of the model.
            training_data: The evaluated training data, or its config dict.
            run: Name of the run. Defaults to its number.
            model: Optional model object (its config is shown in `runs()`).
            preprocessing: Optional preprocessing object.
        """
        config = training_data if isinstance(training_data, dict) else training_data.get_config()
        prep_config = preprocessing.get_config() if preprocessing is not None else None
        config = _normalize(config, prep_config)
        if run is None:
            run = str(sum(r.model == label for r in self.records))
        self.records.append(
            RunRecord(
                model=label,
                run=run,
                training_data=config,
                model_config=model.get_config() if model is not None else None,
                preprocessing=prep_config,
            )
        )
        return self

    @classmethod
    def load(
        cls,
        paths: str | Path | Iterable[str | Path] | Mapping[str, str | Path],
        name: str | None = None,
    ) -> ModelComparison:
        """
        Loads saved runs: every file `*training_data.json` is one run, with the configs next to it.

        Supported are runs written by `Run.save` / `save_for_agentlib` (`{name}_training_data.json`) and
        by the Logger of physXAI < 1.0 (`training_data.json`, `model_config.json`, `preprocessing_config.json`).

        A model is a directory holding its runs, e.g. `results/cmnn/<run>/...` with any number of run
        directories named arbitrarily (`results/cmnn/a1b2c3/`, `results/cmnn/x9y8z7/`, ...):

        - `load('results')`: every sub directory of `results` is a model (`cmnn`, `ann`, ...),
        - `load(['results/cmnn', 'other/ann'])`: every given directory is a model, named after it,
        - `load({'CMNN': 'results/cmnn'})`: every given directory is a model with the given label.

        Runs are named after their directory below the model directory. If a run directory holds several
        models (e.g. `T_training_data.json` and `P_training_data.json` of AgentLib-MPC), they become separate
        models (`cmnn/T`, `cmnn/P`), unless `name` selects one of them. Files that cannot be read are
        skipped with a warning.

        Args:
            paths: The model directories, see above.
            name: Only load the files `{name}_training_data.json` of every run directory.

        Raises:
            FileNotFoundError: If a model directory holds no runs.
        """
        groups: list[tuple[str, Path, list[Path]]] = list()
        if isinstance(paths, (str, Path)):
            root = Path(paths)
            files = _find_runs(root, name)
            if not files:
                raise FileNotFoundError(f"No runs ('*{TRAINING_DATA_SUFFIX}') found in '{root}'.")
            by_model: dict[tuple[str, Path], list[Path]] = dict()
            for file in files:
                parts = file.parent.relative_to(root).parts
                key = (parts[0], root / parts[0]) if parts else (root.name, root)
                by_model.setdefault(key, []).append(file)
            groups = [(label, base, model_files) for (label, base), model_files in by_model.items()]
            if len(groups) > 1 and all(len(g[2]) == 1 for g in groups):
                logger.warning(
                    "Every sub directory of '%s' is loaded as a model with one run. If they are runs of one "
                    "model, use ModelComparison.load(['%s']).", root, root
                )
        else:
            items = paths.items() if isinstance(paths, Mapping) else ((Path(p).name, p) for p in paths)
            for label, directory in items:
                directory = Path(directory)
                files = _find_runs(directory, name)
                if not files:
                    raise FileNotFoundError(f"No runs ('*{TRAINING_DATA_SUFFIX}') found in '{directory}'.")
                groups.append((label, directory, files))

        comparison = cls()
        for label, base, files in groups:
            several = len({_file_prefix(f) for f in files}) > 1
            for file in files:
                prefix = _file_prefix(file)
                model = f"{label}/{prefix}" if several and prefix else label
                record = _read_run(file, model, file.parent.relative_to(base).parts)
                if record is not None:
                    comparison.records.append(record)
        logger.info("Loaded %r", comparison)
        return comparison

    # --- tables --------------------------------------------------------------------------------
    def runs(self) -> pd.DataFrame:
        """One row per run: model class, seeds, number of inputs, training time and file."""
        rows = list()
        for r in self.records:
            prep, model_config = r.preprocessing or {}, r.model_config or {}
            rows.append(
                {
                    "model": r.model,
                    "run": r.run,
                    "model_class": model_config.get("__class_name__"),
                    "split_seed": prep.get("random_state"),
                    "init_seed": model_config.get("random_seed"),
                    "n_inputs": len(prep["inputs"]) if prep.get("inputs") else None,
                    "training_time": r.training_data.get("training_time"),
                    "physXAI_version": r.training_data.get("physXAI_version") or prep.get("physXAI_version"),
                    "path": r.path,
                }
            )
        return pd.DataFrame(rows)

    def metrics(self) -> pd.DataFrame:
        """Single step metrics: one row per model, run, output and split."""
        rows = list()
        for r in self.records:
            for output, splits in _metrics_by_output(r.training_data).items():
                for split, kpis in splits.items():
                    rows.append({"model": r.model, "run": r.run, "output": output, "split": split, **kpis})
        return pd.DataFrame(rows, columns=None if rows else ["model", "run", "output", "split", "rmse"])

    def horizon(self) -> pd.DataFrame:
        """Errors over the prediction horizon: one row per model, run, feature, split and step."""
        rows = list()
        for r in self.records:
            hm = r.training_data.get("horizon_metrics")
            if not hm:
                continue
            time_step = hm.get("time_step")
            by_feature = hm.get("by_feature") or {hm.get("feature", ""): {s: hm[s] for s in SPLITS if s in hm}}
            for feature, splits in by_feature.items():
                for split, kpis in splits.items():
                    for i, rmse in enumerate(kpis["rmse"]):
                        rows.append(
                            {
                                "model": r.model,
                                "run": r.run,
                                "feature": feature,
                                "split": split,
                                "step": i + 1,
                                "hours": (i + 1) * time_step / 3600 if time_step else np.nan,
                                "rmse": rmse,
                                "mae": _at(kpis, "mae", i),
                                "bias": _at(kpis, "bias", i),
                                "std": _at(kpis, "std", i),
                            }
                        )
        columns = ["model", "run", "feature", "split", "step", "hours", "rmse", "mae", "bias", "std"]
        return pd.DataFrame(rows, columns=columns)

    def common_horizon(self, split: str = "test") -> tuple[str, float] | None:
        """
        The horizon all runs with horizon metrics reach: ('hours', value) if all time steps are known,
        else ('step', value). None without horizon metrics.
        """
        horizon = self.horizon()
        horizon = horizon[horizon["split"] == split]
        if horizon.empty:
            return None
        axis = "hours" if horizon["hours"].notna().all() else "step"
        return axis, float(horizon.groupby(["model", "run", "feature"])[axis].max().min())

    def scores(self, split: str = "test", horizon: float | None = None) -> pd.DataFrame:
        """
        One row per run with scalar scores, the basis of `summary` and of the box plots:

        - `RMSE <output>`, `Bias <output>`, `R2 <output>`: single step metrics of the split,
        - `Horizon RMSE <feature>`: mean RMSE over the horizon (up to `horizon`),
        - `RMSE <feature> @ <horizon>` and `Bias <feature> @ <horizon>`: at the end of the horizon.

        Args:
            split: 'train', 'val' or 'test'.
            horizon: End of the horizon, in hours (or steps if the time step is unknown). Defaults to the
                     horizon all runs reach, so models with different evaluation horizons are comparable.
        """
        rows = {(r.model, r.run): {"model": r.model, "run": r.run} for r in self.records}
        metrics = self.metrics()
        if not metrics.empty:
            for row in metrics[metrics["split"] == split].itertuples():
                entry = rows[(row.model, row.run)]
                for kpi in ("rmse", "bias", "r2"):
                    if kpi in metrics.columns and not pd.isna(getattr(row, kpi)):
                        entry[f"{kpi.upper() if kpi != 'bias' else 'Bias'} {row.output}"] = getattr(row, kpi)

        common = self.common_horizon(split)
        if common is not None:
            axis, end = common
            end = end if horizon is None else horizon
            data = self.horizon()
            data = data[(data["split"] == split) & (data[axis] <= end + 1e-9)]
            label = _horizon_label(axis, end)
            for (model, run, feature), group in data.groupby(["model", "run", "feature"], sort=False):
                entry = rows[(model, run)]
                last = group.sort_values(axis).iloc[-1]
                entry[f"Horizon RMSE {feature}"] = float(group["rmse"].mean())
                entry[f"RMSE {feature} @ {label}"] = float(last["rmse"])
                entry[f"Bias {feature} @ {label}"] = float(last["bias"])
        return pd.DataFrame(list(rows.values()))

    def summary(self, split: str = "test", horizon: float | None = None) -> pd.DataFrame:
        """
        Distribution of the scores (see `scores`) per model: number of runs, mean, standard deviation,
        minimum, median and maximum. Rows: (score, model).
        """
        scores = self.scores(split, horizon)
        columns = [c for c in scores.columns if c not in ("model", "run")]
        long = scores.melt(id_vars=["model", "run"], value_vars=columns, var_name="score").dropna(subset=["value"])
        summary = long.groupby(["score", "model"], sort=False)["value"].agg(["count", "mean", "std", "min", "median", "max"])
        return summary.rename(columns={"count": "runs"})

    # --- plots ---------------------------------------------------------------------------------
    def plot(self, split: str = "test", horizon: float | None = None, band: str | None = "iqr", show: bool = False):
        """
        Plots the distributions of the errors of all models:

        - box plots of the single step RMSE per output (every point is a run),
        - RMSE and bias over the horizon: median over the runs and a band ('iqr': 25 % - 75 % (default),
          'range': min - max, None: no band),
        - box plots of the mean RMSE over the horizon and of the RMSE at its end,
        - a table with the summary.

        Args:
            split: 'train', 'val' or 'test'.
            horizon: End of the compared horizon (see `scores`).
            band: 'iqr', 'range' or None.
            show: Whether to open the figure.

        Returns:
            plotly.graph_objects.Figure
        """
        from physXAI.plotting.plotting import subplots

        if band not in ("range", "iqr", None):
            raise ValueError("band has to be 'range', 'iqr' or None.")
        colors = {m: MODEL_COLORS[i % len(MODEL_COLORS)] for i, m in enumerate(self.models)}
        scores = self.scores(split, horizon)
        figures = list()

        # one panel per output: outputs differ in their units (e.g. Change(T) and the trajectory T of an RNN)
        for column in [c for c in scores.columns if c.startswith("RMSE ") and " @ " not in c]:
            figures.append(
                {
                    "title": f"{column} ({split}), every point is one run",
                    "type": "scatter",
                    "figure": self._box_figure(scores, [column], colors, "RMSE"),
                }
            )

        common = self.common_horizon(split)
        if common is not None:
            axis, end = common
            end = end if horizon is None else horizon
            data = self.horizon()
            data = data[(data["split"] == split) & (data[axis] <= end + 1e-9)]
            x_title = "Prediction horizon [h]" if axis == "hours" else "Prediction step"
            for feature in data["feature"].unique():
                feature_data = data[data["feature"] == feature]
                for kpi, title in (("rmse", "RMSE"), ("bias", "Bias")):
                    figures.append(
                        {
                            "title": f"{title} of {feature} over the horizon ({split}), median and {band or 'no'} band",
                            "type": "scatter",
                            "figure": self._horizon_figure(feature_data, axis, kpi, band, colors, x_title, title),
                        }
                    )
            horizon_scores = [c for c in scores.columns if c.startswith("Horizon RMSE ") or (c.startswith("RMSE ") and " @ " in c)]
            figures.append(
                {
                    "title": f"Mean RMSE over the horizon and RMSE at {_horizon_label(axis, end)} ({split})",
                    "type": "scatter",
                    "figure": self._box_figure(scores, horizon_scores, colors, "RMSE"),
                }
            )

        figures.append({"title": f"Summary ({split})", "type": "table", "figure": _table_figure(self.summary(split, horizon))})
        fig = subplots(f"Model comparison: {', '.join(self.models)}", *figures, show=False)
        _deduplicate_legend(fig)
        fig.update_layout(boxmode="group")
        if show:
            fig.show()
        return fig

    @staticmethod
    def _box_figure(scores: pd.DataFrame, columns: list[str], colors: dict, y_title: str) -> go.Figure:
        fig = go.Figure()
        for model, color in colors.items():
            rows = scores[scores["model"] == model]
            x, y, text = [], [], []
            for column in columns:
                values = rows[column] if column in rows else pd.Series(dtype=float)
                for run, value in zip(rows["run"], values, strict=False):
                    if not pd.isna(value):
                        x.append(column)
                        y.append(value)
                        text.append(run)
            if not y:
                continue
            fig.add_trace(
                go.Box(
                    x=x,
                    y=y,
                    text=text,
                    name=model,
                    legendgroup=model,
                    marker={"color": color, "size": 8},
                    line={"width": 2},
                    boxpoints="all",
                    jitter=0.4,
                    pointpos=0,
                    hovertemplate="%{text}: %{y:.4g}<extra>" + model + "</extra>",
                )
            )
        fig.update_layout(xaxis_title="", yaxis_title=y_title)
        return fig

    @staticmethod
    def _horizon_figure(data: pd.DataFrame, axis: str, kpi: str, band: str | None, colors: dict, x_title: str,
                        y_title: str) -> go.Figure:
        fig = go.Figure()
        for model, color in colors.items():
            rows = data[data["model"] == model]
            if rows.empty:
                continue
            grouped = rows.groupby(axis)[kpi]
            x = np.asarray(grouped.median().index, dtype=float)
            median = grouped.median().to_numpy()
            if band is not None and rows["run"].nunique() > 1:
                low = grouped.min() if band == "range" else grouped.quantile(0.25)
                high = grouped.max() if band == "range" else grouped.quantile(0.75)
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([x, x[::-1]]),
                        y=np.concatenate([high.to_numpy(), low.to_numpy()[::-1]]),
                        fill="toself",
                        fillcolor=_transparent(color, 0.18),
                        line={"width": 0},
                        name=model,
                        legendgroup=model,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=median,
                    mode="lines",
                    name=model,
                    legendgroup=model,
                    line={"color": color, "width": 2},
                    hovertemplate="%{x:.3g}: %{y:.4g}<extra>" + model + " (median)</extra>",
                )
            )
        if kpi == "bias":
            fig.add_hline(y=0, line={"color": "gray", "width": 1, "dash": "dot"})
        fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)
        return fig

    # --- saving --------------------------------------------------------------------------------
    def save(self, run, name: str = "comparison", split: str = "test", horizon: float | None = None,
             band: str | None = "iqr") -> dict:
        """
        Saves the comparison into a run directory: `{name}.html` (plot), `{name}_summary.csv`,
        `{name}_scores.csv` (one row per run), `{name}_metrics.csv` and `{name}_horizon.csv`.

        Returns:
            dict: The paths of the written files.
        """
        from physXAI.plotting.plotting import save_figure

        files = {"plot": save_figure(self.plot(split, horizon, band=band), run.file(f"{name}.html"))}
        tables = {
            "summary": self.summary(split, horizon),
            "scores": self.scores(split, horizon),
            "metrics": self.metrics(),
            "horizon": self.horizon(),
        }
        for kind, table in tables.items():
            path = run.file(f"{name}_{kind}.csv")
            table.to_csv(path, index=kind == "summary")
            files[kind] = path
        logger.info("Saved the comparison of %s in %s", ", ".join(self.models), run.path)
        return files


# --- helpers -------------------------------------------------------------------------------------
def _find_runs(directory: Path, name: str | None = None) -> list[Path]:
    files = sorted(p for p in directory.rglob(f"*{TRAINING_DATA_SUFFIX}") if p.is_file())
    if name is not None:
        files = [f for f in files if _file_prefix(f) == name]
    return files


def _file_prefix(file: Path) -> str:
    """'T' for 'T_training_data.json', '' for 'training_data.json'."""
    return file.name[: -len(TRAINING_DATA_SUFFIX)].rstrip("_")


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_run(file: Path, label: str, parts: tuple[str, ...]) -> RunRecord | None:
    """Reads a training data json and its sibling configs ('{prefix}_model.json' or legacy 'model_config.json')."""
    prefix = file.name[: -len(TRAINING_DATA_SUFFIX)]
    name = prefix.rstrip("_")
    # runs in sub directories are named after them, runs side by side after the prefix of their files
    run_parts = list(parts) if parts else ([name] if name else [])
    try:
        training_data = _read_json(file)
        model_config = _read_json(file.with_name(f"{prefix}model.json"))
        preprocessing = _read_json(file.with_name(f"{prefix}preprocessing.json"))
        if not prefix:  # layout of physXAI < 1.0
            model_config = model_config or _read_json(file.with_name("model_config.json"))
            preprocessing = preprocessing or _read_json(file.with_name("preprocessing_config.json"))
    except (OSError, ValueError) as e:
        logger.warning("Skipped run '%s': %s", file, e)
        return None
    if not isinstance(training_data, dict):
        logger.warning("Skipped run '%s': no training data config.", file)
        return None
    return RunRecord(
        model=label,
        run="/".join(run_parts) or "run",
        training_data=_normalize(training_data, preprocessing),
        model_config=model_config,
        preprocessing=preprocessing,
        path=str(file),
    )


def _output_name(preprocessing: dict | None) -> str:
    output = (preprocessing or {}).get("output")
    if isinstance(output, list):
        return " + ".join(map(str, output)) if output else "output"
    return str(output) if output else "output"


def _normalize(config: dict, preprocessing: dict | None = None) -> dict:
    """
    Brings a training data config of physXAI < 1.0 into the current form (without changing `config`):

    - flat metrics ('RMSE Test', ...) become `metrics.by_output`, named after the output of the preprocessing,
    - the RMSE per step of multi step models ('rmse_test_l', ...) becomes `horizon_metrics`.
    """
    metrics = config.get("metrics") or {}
    if metrics.get("by_output") and config.get("horizon_metrics") is not None:
        return config
    config = dict(config)
    output = _output_name(preprocessing)
    if not metrics.get("by_output"):
        by_output = _metrics_by_output(config)
        if by_output:
            config["metrics"] = {**metrics, "by_output": {output: by_output["output"]}}
    if config.get("horizon_metrics") is None:
        steps = {split: metrics.get(f"rmse_{split}_l") for split in SPLITS}
        steps = {split: {"rmse": values} for split, values in steps.items() if values}
        if steps:
            config["horizon_metrics"] = {
                "feature": output,
                "time_step": (preprocessing or {}).get("time_step"),
                **steps,
            }
    return config


def _at(kpis: dict, key: str, i: int) -> float:
    values = kpis.get(key)
    return values[i] if values is not None and i < len(values) else np.nan


def _metrics_by_output(config: dict) -> dict[str, dict[str, dict]]:
    """{output: {split: kpis}}, also from the flat metrics of physXAI < 1.0."""
    metrics = config.get("metrics") or {}
    if metrics.get("by_output"):
        return metrics["by_output"]
    flat = dict()
    for split in SPLITS:
        kpis = metrics.get(f"{split}_kpis")
        if not kpis:
            continue
        values = {key.rsplit(" ", 1)[0].lower(): value for key, value in kpis.items()}
        flat[split] = {k: v for k, v in values.items() if k in ("mse", "rmse", "r2", "mae", "bias")}
    return {"output": flat} if flat else {}


def _horizon_label(axis: str, end: float) -> str:
    return f"{end:g} h" if axis == "hours" else f"step {end:g}"


def _transparent(color: str, alpha: float) -> str:
    color = color.lstrip("#")
    r, g, b = (int(color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _table_figure(summary: pd.DataFrame) -> go.Figure:
    table = summary.reset_index()
    values = list()
    for column in table.columns:
        if column != "runs" and pd.api.types.is_float_dtype(table[column]):
            values.append(table[column].map(lambda v: f"{v:.4g}" if not pd.isna(v) else ""))
        else:
            values.append(table[column].astype(str))
    return go.Figure(go.Table(header={"values": list(table.columns)}, cells={"values": values}))


def _deduplicate_legend(fig: go.Figure) -> None:
    """Shows every model once in the legend."""
    seen = set()
    for trace in fig.data:
        if trace.type == "table" or trace.showlegend is False:
            continue
        if trace.name in seen:
            trace.showlegend = False
        else:
            seen.add(trace.name)


# --- command line --------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> ModelComparison:
    """
    Compares saved runs from the command line::

        python -m physXAI.evaluation.comparison results                       # sub directories are models
        python -m physXAI.evaluation.comparison results/cmnn results/ann      # every directory is a model
        python -m physXAI.evaluation.comparison results/cmnn --models         # one directory as one model

    The summary is printed, plot and tables are saved into `--out` (default: `<first path>/comparison`).
    """
    import argparse

    from physXAI.io.run import Run

    parser = argparse.ArgumentParser(prog="python -m physXAI.evaluation.comparison", description=main.__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="directory of model directories, or model directories")
    parser.add_argument("--models", action="store_true", help="every given path is a model (also a single one)")
    parser.add_argument("--name", help="only files '{name}_training_data.json'")
    parser.add_argument("--split", default="test", choices=SPLITS)
    parser.add_argument("--horizon", type=float, help="end of the compared horizon in hours")
    parser.add_argument("--band", default="iqr", choices=["iqr", "range", "none"])
    parser.add_argument("--out", help="output directory")
    parser.add_argument("--show", action="store_true", help="open the plot")
    args = parser.parse_args(argv)

    paths = args.paths if (args.models or len(args.paths) > 1) else args.paths[0]
    comparison = ModelComparison.load(paths, name=args.name)
    band = None if args.band == "none" else args.band
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(comparison.summary(args.split, args.horizon).to_string(float_format="{:.4g}".format))
    out = Path(args.out) if args.out else Path(args.paths[0]) / "comparison"
    run = Run(out.parent, out.name, overwrite=True)
    comparison.save(run, split=args.split, horizon=args.horizon, band=band)
    if args.show:
        comparison.plot(args.split, args.horizon, band=band, show=True)
    print(f"Saved to {run.path}")
    return comparison


if __name__ == "__main__":  # pragma: no cover
    main()
