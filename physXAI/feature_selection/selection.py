"""
Feature selection by backward elimination (experimental).

A small single step error does not imply a good MPC performance. The selection therefore
uses the error of the recursive prediction over a horizon (`criterion='horizon'`, requires
`PreprocessingSingleStep(eval_horizon=...)`) and repeats every evaluation on several block splits
(`n_repeats`), so the spread of the criterion can be compared to the differences between candidates.
Optionally, the sign of the sensitivity of the prediction to the control inputs is checked
(`controls`): a model whose prediction reacts to a control with the wrong sign is of little use in
an MPC, even if its error is small.

The criterion is computed on the validation data, which also stops the training early. The score of the selected
inputs is therefore optimistic: evaluate the final model on the test data (`td.metrics`, `td.horizon_metrics`).

    >>> fs = FeatureSelection(prep, LinearRegressionModel(), controls={'oveHeaPumY_u': 1}, n_repeats=3)
    >>> result = fs.run(file_path)
    >>> result.selected.inputs
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from physXAI.preprocessing.preprocessing import (
    LAG_PATTERN,
    PreprocessingData,
    PreprocessingSingleStep,
    output_feature,
)
from physXAI.training.recursive import step_weights

logger = logging.getLogger("physXAI")


@dataclass
class Candidate:
    """
    Evaluation of one set of inputs (mean over the repeats).

    Attributes:
        inputs: The inputs.
        score: Criterion (lower is better).
        score_std: Standard deviation of the criterion over the repeats.
        single_step_rmse: Single step RMSE on the validation data.
        horizon_rmse: Step weighted RMSE of the recursive prediction on the validation data.
        horizon_bias: Absolute bias of the recursive prediction at `mpc_horizon` on the validation data.
        sensitivity: Per control, the share of validation samples with the expected sign of the sensitivity.
        plausible: Whether all sensitivities reach `min_sign_share`.
    """

    inputs: list[str]
    score: float
    score_std: float
    single_step_rmse: float
    horizon_rmse: float | None = None
    horizon_bias: float | None = None
    sensitivity: dict[str, float] = field(default_factory=dict)
    plausible: bool = True


@dataclass
class FeatureSelectionResult:
    """
    Result of a feature selection.

    Attributes:
        criterion: The criterion used.
        steps: {number of inputs: all evaluated candidates of that size}.
        selected: The selected candidate.
    """

    criterion: str
    steps: dict[int, list[Candidate]]
    selected: Candidate

    @property
    def best(self) -> dict[int, Candidate]:
        """Best (plausible, if any) candidate per number of inputs."""
        return {n: _best(candidates) for n, candidates in self.steps.items()}

    def table(self) -> pd.DataFrame:
        """All candidates as table."""
        rows = list()
        for n, candidates in self.steps.items():
            for c in candidates:
                rows.append({"n_inputs": n, **asdict(c), "inputs": ", ".join(c.inputs)})
        return pd.DataFrame(rows)

    def plot(self, show: bool = False) -> go.Figure:
        """Criterion (with its spread) and single step RMSE of the best candidate per number of inputs."""
        best = dict(sorted(self.best.items()))
        n = list(best)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=n,
                y=[c.score for c in best.values()],
                mode="lines+markers",
                name=f"Criterion ({self.criterion})",
                error_y=dict(type="data", array=[c.score_std for c in best.values()], visible=True),
                marker=dict(color="blue"),
            )
        )
        if self.criterion != "single_step":
            fig.add_trace(
                go.Scatter(
                    x=n,
                    y=[c.single_step_rmse for c in best.values()],
                    mode="lines+markers",
                    name="Single step RMSE",
                    marker=dict(color="green"),
                )
            )
        implausible = [k for k, c in best.items() if not c.plausible]
        if implausible:
            fig.add_trace(
                go.Scatter(
                    x=implausible,
                    y=[best[k].score for k in implausible],
                    mode="markers",
                    name="Implausible sensitivity",
                    marker=dict(color="red", symbol="x", size=12),
                )
            )
        fig.add_trace(
            go.Scatter(
                x=[len(self.selected.inputs)],
                y=[self.selected.score],
                mode="markers",
                name="Selected",
                marker=dict(color="black", symbol="star", size=14),
            )
        )
        fig.update_layout(xaxis_title="Number of inputs", yaxis_title="Validation error", title="Feature selection")
        if show:
            fig.show()
        return fig

    def save(self, run, name: str = "feature_selection") -> dict:
        """
        Saves the result into a run directory: `{name}.json` (all candidates and their criteria) and
        `{name}.html` (plot).

        Args:
            run: The run directory (`physXAI.io.Run`).
            name: Name of the files.

        Returns:
            dict: The paths of the written files.
        """
        from physXAI.plotting.plotting import save_figure

        return {
            "json": run.save_json(f"{name}.json", self.get_config()),
            "plot": save_figure(self.plot(), run.file(f"{name}.html")),
        }

    def get_config(self) -> dict:
        return {
            "criterion": self.criterion,
            "selected": asdict(self.selected),
            "steps": {str(n): [asdict(c) for c in candidates] for n, candidates in self.steps.items()},
        }


def _best(candidates: list[Candidate]) -> Candidate:
    plausible = [c for c in candidates if c.plausible]
    return min(plausible or candidates, key=lambda c: c.score)


class FeatureSelection:
    """
    Backward elimination of inputs.

    Starting from all inputs of the preprocessing, every step removes the input whose removal gives the
    best criterion. The preprocessing passed in is not changed.

    Args:
        preprocessing: Preprocessing holding all candidate inputs (`PreprocessingSingleStep`).
        model: The physXAI model, trained for every candidate (without plots and saving).
        criterion: 'horizon' (default if the preprocessing has an `eval_horizon`): step weighted RMSE of the
                   recursive prediction on the validation data, plus `bias_weight` times its absolute bias at
                   `mpc_horizon`. 'single_step': single step RMSE on the validation data.
        mpc_horizon: Step (or time delta string) of the horizon at which the bias is evaluated. Defaults to
                     the evaluation horizon.
        first_step_weight: Optional share of the first step in the horizon RMSE (None: all steps alike).
        bias_weight: Weight of the absolute bias in the criterion.
        n_repeats: Number of block splits (seeds) every candidate is evaluated on.
        fixed_inputs: Inputs which are never removed. Controls and the recursive feature are always kept.
        ascending_lag_order: Remove the highest lag of a feature first and the feature itself last.
        controls: {control input: expected sign (+1 / -1)} of the sensitivity of the prediction.
        min_sign_share: Share of validation samples that needs the expected sign for a plausible model.
        tolerance: Relative tolerance: the smallest set of inputs with a criterion within
                   (1 + tolerance) * best criterion is selected.
    """

    def __init__(
        self,
        preprocessing: PreprocessingData,
        model,
        criterion: str | None = None,
        mpc_horizon: int | str | None = None,
        first_step_weight: float | None = None,
        bias_weight: float = 0.0,
        n_repeats: int = 1,
        fixed_inputs: list[str] | None = None,
        ascending_lag_order: bool = True,
        controls: dict[str, int] | None = None,
        min_sign_share: float = 0.95,
        tolerance: float = 0.0,
    ):
        if not isinstance(preprocessing, PreprocessingSingleStep):
            raise NotImplementedError(
                "Feature selection supports single step models only. To select on the multi step error, use "
                "PreprocessingSingleStep(eval_horizon=...)."
            )
        if preprocessing.inputs is None:
            raise ValueError("The preprocessing needs the candidate inputs.")
        if preprocessing.val_size <= 0:
            raise ValueError("Feature selection needs validation data (val_size > 0).")
        has_horizon = preprocessing.eval_horizon is not None
        criterion = criterion or ("horizon" if has_horizon else "single_step")
        if criterion not in ("horizon", "single_step"):
            raise ValueError(f"Unknown criterion '{criterion}', use 'horizon' or 'single_step'.")
        if criterion == "horizon" and not has_horizon:
            raise ValueError("criterion='horizon' requires PreprocessingSingleStep(eval_horizon=...).")
        if n_repeats < 1:
            raise ValueError("n_repeats has to be at least 1.")
        self.preprocessing = preprocessing
        self.model = model
        self.criterion = criterion
        self.mpc_horizon = mpc_horizon
        self.first_step_weight = first_step_weight
        self.bias_weight = bias_weight
        self.n_repeats = n_repeats
        self.controls = dict(controls or {})
        self.min_sign_share = min_sign_share
        self.ascending_lag_order = ascending_lag_order
        self.tolerance = tolerance
        fixed = list(fixed_inputs or []) + list(self.controls)
        if criterion == "horizon":
            fixed.append(output_feature(preprocessing.output[0]))
        self.fixed_inputs = list(dict.fromkeys(fixed))
        missing = [c for c in self.controls if c not in preprocessing.inputs]
        if missing:
            raise ValueError(f"The controls {missing} are not part of the inputs.")

    # --- candidates ---------------------------------------------------------------------------
    def removable(self, inputs: list[str]) -> list[str]:
        """Inputs which may be removed from `inputs` in the next step."""
        result = list()
        for name in inputs:
            if name in self.fixed_inputs:
                continue
            if self.ascending_lag_order:
                match = LAG_PATTERN.match(name)
                base, lag = (match.group(1), int(match.group(2))) if match else (name, 0)
                if f"{base}_lag{lag + 1}" in inputs:
                    continue
            result.append(name)
        return result

    def _preprocessing(self, inputs: list[str], seed: int) -> PreprocessingData:
        config = self.preprocessing.get_config()
        config.pop("transform", None)
        config.update(inputs=inputs, random_state=seed)
        prep = PreprocessingData.from_config(config)
        prep.transform = self.preprocessing.transform
        return prep

    # --- evaluation ---------------------------------------------------------------------------
    def evaluate(self, file_path: str, inputs: list[str]) -> Candidate:
        """Trains and evaluates the model with `inputs` on every repeat."""
        scores, single, horizon, bias, signs = [], [], [], [], {c: [] for c in self.controls}
        for repeat in range(self.n_repeats):
            prep = self._preprocessing(inputs, self.preprocessing.random_state + repeat)
            td = prep.pipeline(file_path)
            trained = self.model.pipeline(td, plot=False, save_model=False)
            rmse = next(iter(td.metrics.by_output.values()))["val"]["rmse"]
            single.append(rmse)
            score = rmse
            if self.criterion == "horizon":
                h_rmse, h_bias = self._horizon_kpis(td)
                horizon.append(h_rmse)
                bias.append(h_bias)
                score = h_rmse + self.bias_weight * h_bias
            scores.append(score)
            for control in self.controls:
                signs[control].append(self._sign_share(trained, td, control))
        sensitivity = {c: float(np.mean(v)) for c, v in signs.items()}
        candidate = Candidate(
            inputs=list(inputs),
            score=float(np.mean(scores)),
            score_std=float(np.std(scores)),
            single_step_rmse=float(np.mean(single)),
            horizon_rmse=float(np.mean(horizon)) if horizon else None,
            horizon_bias=float(np.mean(bias)) if bias else None,
            sensitivity=sensitivity,
            plausible=all(v >= self.min_sign_share for v in sensitivity.values()),
        )
        logger.info(
            "%d inputs, criterion %.4g ± %.2g%s: %s",
            len(inputs),
            candidate.score,
            candidate.score_std,
            "" if candidate.plausible else " (implausible sensitivity)",
            inputs,
        )
        return candidate

    def _horizon_kpis(self, td) -> tuple[float, float]:
        metrics = td.horizon_metrics
        if metrics is None or "val" not in metrics.kpis:
            raise ValueError(
                "No recursive evaluation on the validation data (the recursive feature is needed as input)."
            )
        kpis = metrics.kpis["val"]
        weights = step_weights(len(kpis["rmse"]), self.first_step_weight)
        step = self.mpc_horizon
        if isinstance(step, str):
            step = int(round(pd.Timedelta(step).total_seconds() / td.time_step))
        step = min(step or len(kpis["bias"]), len(kpis["bias"]))
        return float(np.sum(weights * np.asarray(kpis["rmse"]))), float(abs(kpis["bias"][step - 1]))

    def _sign_share(self, trained, td, control: str) -> float:
        """Share of validation samples whose prediction changes with the expected sign of the control."""
        X = td.X("val").astype(np.float64)
        j = td.inputs.index(control)
        delta = 1e-3 * (np.std(X[:, j]) or 1.0)
        upper, lower = X.copy(), X.copy()
        upper[:, j] += delta
        lower[:, j] -= delta
        sensitivity = (_predict(trained, upper) - _predict(trained, lower)).reshape(len(X), -1)[:, 0]
        return float(np.mean(np.sign(sensitivity) == np.sign(self.controls[control])))

    # --- selection ----------------------------------------------------------------------------
    def run(self, file_path: str) -> FeatureSelectionResult:
        """Runs the backward elimination on the data in `file_path`."""
        inputs = list(self.preprocessing.inputs)
        steps = {len(inputs): [self.evaluate(file_path, inputs)]}
        while True:
            options = self.removable(inputs)
            if not options or len(inputs) <= 1:
                break
            candidates = [self.evaluate(file_path, [i for i in inputs if i != name]) for name in options]
            steps[len(inputs) - 1] = candidates
            inputs = _best(candidates).inputs
        best = {n: _best(c) for n, c in steps.items()}
        pool = [c for c in best.values() if c.plausible] or list(best.values())
        threshold = min(c.score for c in pool) * (1 + self.tolerance)
        selected = min((c for c in pool if c.score <= threshold), key=lambda c: (len(c.inputs), c.score))
        logger.info("Selected %d inputs: %s", len(selected.inputs), selected.inputs)
        return FeatureSelectionResult(self.criterion, steps, selected)


def _predict(model, X: np.ndarray) -> np.ndarray:
    import keras

    if isinstance(model, keras.Model):
        return np.asarray(model.predict(X.astype(np.float32), verbose=0, batch_size=4096))
    return np.asarray(model.predict(X))
