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
    >>> run = Run("results", "TAir_compare", overwrite=True)
    >>> repeat_training(file_path, prep, LinearRegressionModel(), seeds=5, run=run, name="linreg")
    >>> repeat_training(file_path, prep, ClassicalANNModel(epochs=50), seeds=5, run=run, name="ann")
    >>> comparison = ModelComparison.load(run.path)  # one model per sub directory
    >>> comparison.summary()
    >>> comparison.paired("linreg")  # run by run on the same splits: differences, corrected confidence interval, p-value
    >>> comparison.save(run, baseline="linreg")  # comparison.html and csv files

With the same seeds and the same split settings, all models use the same splits of the data, so their runs can be
compared in pairs (`paired`), which removes the spread caused by the split.
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
        model: The physXAI model. It is not changed, every run uses a new model created from its config.
        seeds: Number of runs (seeds 0, 1, ...) or the seeds. Use the same seeds for all compared models,
               then they are trained and tested on the same splits.
        run: Optional `Run`. Every run is saved into `<run>/<name>/seed_<seed>/` with `Run.save`.
        name: Name of the model (label in the comparison and name of the saved files).
        vary: What the seed changes: 'split' (the random split of the data) and / or 'init'
              (initialization of the weights, order of the batches and seeds of blocks such as `px.RBF`).
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
        m = _model_with_seed(model, seed if "init" in vary else None)
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


def _model_with_seed(model, seed: int | None):
    """
    A new, untrained model from the config of `model`, so runs share no state (e.g. the weights of a trainable
    `px.Pretrained` block). With a seed, it replaces the seed of the weights and the batch order and the seeds of the
    blocks of expressions (k-means of `px.RBF`).
    """
    from physXAI.models.models import AbstractModel

    try:
        config = copy.deepcopy(model.get_config())
    except ValueError as e:  # e.g. a Pretrained block holding a model object instead of a path
        logger.warning("The model is copied instead of created from its config (%s): runs may share state.", e)
        m = copy.copy(model)
        if seed is not None and hasattr(m, "random_seed"):
            m.random_seed = seed
        return m
    if seed is not None:
        if "random_seed" in config:
            config["random_seed"] = seed
        _seed_expressions(config, seed)
    return AbstractModel.model_from_config(config)


def _seed_expressions(config, seed: int) -> None:
    """Sets `random_state` of all nodes of serialized expressions (`physXAI.expr.to_config`) inside a config."""
    if isinstance(config, dict):
        for node in config.get("nodes", []) if isinstance(config.get("nodes"), list) else []:
            if isinstance(node, dict) and "random_state" in node.get("params", {}):
                node["params"]["random_state"] = seed
        for value in config.values():
            _seed_expressions(value, seed)
    elif isinstance(config, list):
        for value in config:
            _seed_expressions(value, seed)


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

    def paired(
        self,
        baseline: str,
        models: str | Iterable[str] | None = None,
        split: str = "test",
        horizon: float | None = None,
        by: str = "auto",
        confidence: float = 0.95,
        corrected: bool = True,
    ) -> pd.DataFrame:
        """
        Paired comparison of models with a baseline: the scores (see `scores`) are compared run by run on the same
        split of the data, so the spread caused by the split cancels out. Two models whose distributions overlap in
        `summary` can still differ clearly, if one of them is better on (almost) every split.

        Runs are paired by the seed of their split (`by='split_seed'`, the `random_state` of the preprocessing) or by
        their name (`by='run'`, e.g. 'seed_0' of `repeat_training`). 'auto' takes the split seed if it identifies the
        runs of both models, else the name. The same seed gives the same split only with the same split settings:
        the block length depends on the window length (horizon, warmup) and on the size of the data. Every pair is
        therefore checked with the fingerprint of the split (`same_data`, saved with physXAI >= 1.0). Single step
        and recurrent models on the same split can still be evaluated on different windows (warmup inside the split).

        Differences are model - baseline. RMSE and the absolute bias are better when lower, R2 when higher.

        The runs are no independent samples: repeated splits of the same data share most of their training data, so a
        plain paired t-test (or a Wilcoxon test) is too optimistic. Confidence interval and p-value are therefore
        those of the corrected resampled t-test (Nadeau & Bengio 2003, Machine Learning 52, 239-281): the variance of
        the mean difference is `var(d) * (1/n + n_test/n_train)` instead of `var(d) / n`. The ratio `n_test/n_train`
        (`resampling`) is taken from the split summary of the runs (else from the split sizes of the preprocessing);
        it is 0 if all runs of the baseline used the same split (`vary=('init',)`), then only the initialization
        varies. Several scores and models are tested at once: correct the p-values for multiple testing (e.g. Holm)
        before drawing conclusions from several of them.

        Args:
            baseline: Label of the reference model.
            models: Models to compare with the baseline. Defaults to all others.
            split: 'train', 'val' or 'test'.
            horizon: End of the compared horizon (see `scores`).
            by: 'auto', 'split_seed' or 'run'.
            confidence: Level of the confidence interval of the mean difference.
            corrected: Corrected resampled t-test (default). False gives the plain paired t-test, which assumes
                       independent runs.

        Returns:
            pd.DataFrame: One row per (score, model) with the number of `pairs`, the medians of the `baseline` and the
            `model`, the `mean_difference` with its confidence interval (`ci_low`, `ci_high`), the
            `median_difference`, the share of pairs in which the model is `better`, the two-sided `p_value` of the
            (corrected) t-test, the `resampling` ratio n_test/n_train of the correction and the share of pairs on the
            `same_data` (NaN if unknown).
        """
        from scipy import stats

        if by not in ("auto", "split_seed", "run"):
            raise ValueError("by has to be 'auto', 'split_seed' or 'run'.")
        if baseline not in self.models:
            raise ValueError(f"Unknown baseline '{baseline}', the models are {self.models}.")
        others = [m for m in self.models if m != baseline] if models is None else (
            [models] if isinstance(models, str) else list(models))
        unknown = [m for m in others if m not in self.models]
        if unknown:
            raise ValueError(f"Unknown models {unknown}, the models are {self.models}.")

        scores = self.scores(split, horizon).set_index(["model", "run"])
        records = {(r.model, r.run): r for r in self.records}
        rows = list()
        for model in others:
            pairs = self._pairs(baseline, model, by)
            resampling = _resampling_ratio([records[(baseline, a)] for a, _ in pairs], split) if corrected else 0.0
            same = [_same_split(records[(baseline, a)], records[(model, b)], split) for a, b in pairs]
            known = [s for s in same if s is not None]
            same_data = float(np.mean(known)) if known else np.nan
            if known and not all(known):
                logger.warning(
                    "Paired comparison of '%s' with '%s': %d of %d pairs were evaluated on different %s data (other "
                    "split settings, e.g. block length or horizon). Use the same split settings for paired comparisons.",
                    model, baseline, len(known) - sum(known), len(known), split,
                )
            for column in scores.columns:
                if not column.startswith(("RMSE ", "Bias ", "R2 ", "Horizon RMSE ")):
                    continue
                a = np.array([scores.at[(baseline, p), column] for p, _ in pairs], dtype=float)
                b = np.array([scores.at[(model, q), column] for _, q in pairs], dtype=float)
                finite = np.isfinite(a) & np.isfinite(b)
                if not finite.any():
                    continue
                a, b = a[finite], b[finite]
                name = column
                if column.startswith("Bias "):
                    a, b, name = np.abs(a), np.abs(b), f"|{column}|"
                difference = b - a
                better = difference > 0 if column.startswith("R2 ") else difference < 0
                n = len(difference)
                mean = float(np.mean(difference))
                ci, p_value = (np.nan, np.nan), np.nan
                if n > 1:
                    se = float(np.std(difference, ddof=1)) * np.sqrt(1.0 / n + resampling)
                    half = stats.t.ppf((1 + confidence) / 2, n - 1) * se
                    ci = (mean - half, mean + half)
                    if se > 0:
                        p_value = float(2 * stats.t.sf(abs(mean) / se, n - 1))
                    elif mean != 0:
                        p_value = 0.0  # the same difference on every split
                rows.append({
                    "score": name, "model": model, "pairs": n, "baseline": float(np.median(a)),
                    "model_median": float(np.median(b)), "mean_difference": mean, "ci_low": ci[0], "ci_high": ci[1],
                    "median_difference": float(np.median(difference)), "better": float(np.mean(better)),
                    "p_value": p_value, "resampling": resampling, "same_data": same_data,
                })
        columns = ["score", "model", "pairs", "baseline", "model_median", "mean_difference", "ci_low", "ci_high",
                   "median_difference", "better", "p_value", "resampling", "same_data"]
        return pd.DataFrame(rows, columns=columns).set_index(["score", "model"])

    def _pairs(self, baseline: str, model: str, by: str) -> list[tuple[str, str]]:
        """Pairs of run names (baseline, model) on the same split, see `paired`."""
        keys = dict()
        for mode in (["split_seed", "run"] if by == "auto" else [by]):
            for label in (baseline, model):
                runs = [r for r in self.records if r.model == label]
                values = [_split_seed(r) if mode == "split_seed" else r.run for r in runs]
                if any(v is None for v in values) or len(set(values)) != len(values):
                    break  # the runs of this model are not identified by this key
                keys[label] = dict(zip(values, (r.run for r in runs), strict=True))
            else:
                common = [k for k in keys[baseline] if k in keys[model]]
                if common:
                    unpaired = len(keys[baseline]) + len(keys[model]) - 2 * len(common)
                    if unpaired:
                        logger.info("Paired comparison of '%s' with '%s' by %s: %d run(s) without a partner are left out.",
                                    model, baseline, mode, unpaired)
                    return [(keys[baseline][k], keys[model][k]) for k in common]
            keys = dict()
        raise ValueError(
            f"The runs of '{baseline}' and '{model}' cannot be paired (by='{by}'): they need the same split seeds "
            "(random_state of the preprocessing) or the same run names, e.g. repeat_training with the same seeds."
        )

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
             band: str | None = "iqr", baseline: str | None = None) -> dict:
        """
        Saves the comparison into a run directory: `{name}.html` (plot), `{name}_summary.csv`,
        `{name}_scores.csv` (one row per run), `{name}_metrics.csv`, `{name}_horizon.csv` and, with a `baseline`,
        `{name}_paired.csv` (see `paired`).

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
        if baseline is not None:
            tables["paired"] = self.paired(baseline, split=split, horizon=horizon)
        for kind, table in tables.items():
            path = run.file(f"{name}_{kind}.csv")
            table.to_csv(path, index=kind in ("summary", "paired"))
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


def _split_seed(record: RunRecord):
    return (record.preprocessing or {}).get("random_state")


def _resampling_ratio(records: list[RunRecord], split: str) -> float:
    """
    Ratio n_test / n_train of the corrected resampled t-test (`paired`): 0 if all runs used the same split, else from
    the split summary of the runs or the split sizes of their preprocessing.
    """
    prints = [((r.training_data.get("summary") or {}).get("fingerprint") or {}).get(split) for r in records]
    if None not in prints and len(set(prints)) == 1:
        return 0.0
    ratios = list()
    for r in records:
        summary, prep = r.training_data.get("summary") or {}, r.preprocessing or {}
        if summary.get("train") and summary.get(split) is not None:
            ratios.append(summary[split] / summary["train"])
        elif prep.get("test_size") is not None and prep.get("val_size") is not None:
            size = prep["val_size"] if split == "val" else prep["test_size"] if split == "test" else None
            train = 1.0 - prep["test_size"] - prep["val_size"]
            if size is not None and train > 0:
                ratios.append(size / train)
    if not ratios:
        logger.warning("Paired comparison: the split sizes of the runs are unknown, the t-test is not corrected for "
                       "the overlap of the splits (too optimistic).")
        return 0.0
    return float(np.mean(ratios))


def _same_split(a: RunRecord, b: RunRecord, split: str) -> bool | None:
    """Whether two runs were evaluated on the same rows of a split (fingerprint of the split), None if unknown."""
    prints = [((r.training_data.get("summary") or {}).get("fingerprint") or {}).get(split) for r in (a, b)]
    return None if None in prints else prints[0] == prints[1]


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
        python -m physXAI.evaluation.comparison results --baseline linreg     # paired with the runs of linreg

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
    parser.add_argument("--baseline", help="model for a paired comparison with all other models")
    parser.add_argument("--out", help="output directory")
    parser.add_argument("--show", action="store_true", help="open the plot")
    args = parser.parse_args(argv)

    paths = args.paths if (args.models or len(args.paths) > 1) else args.paths[0]
    comparison = ModelComparison.load(paths, name=args.name)
    band = None if args.band == "none" else args.band
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(comparison.summary(args.split, args.horizon).to_string(float_format="{:.4g}".format))
        if args.baseline:
            print(f"\nPaired with '{args.baseline}' (difference = model - baseline):")
            print(comparison.paired(args.baseline, split=args.split, horizon=args.horizon)
                  .to_string(float_format="{:.4g}".format))
    out = Path(args.out) if args.out else Path(args.paths[0]) / "comparison"
    run = Run(out.parent, out.name, overwrite=True)
    comparison.save(run, split=args.split, horizon=args.horizon, band=band, baseline=args.baseline)
    if args.show:
        comparison.plot(args.split, args.horizon, band=band, show=True)
    print(f"Saved to {run.path}")
    return comparison


if __name__ == "__main__":  # pragma: no cover
    main()
