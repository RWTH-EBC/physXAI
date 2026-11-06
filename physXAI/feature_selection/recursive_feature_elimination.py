"""
Legacy interface of the feature selection, see `physXAI.feature_selection.selection.FeatureSelection`.
"""

from __future__ import annotations

import logging

from physXAI.feature_selection.selection import FeatureSelection, FeatureSelectionResult
from physXAI.io.run import Run
from physXAI.preprocessing.preprocessing import PreprocessingData

logger = logging.getLogger("physXAI")


def recursive_feature_elimination_pipeline(
    file_path: str,
    preprocessing: PreprocessingData,
    model,
    ascending_lag_order: bool = True,
    use_multi_step_error: bool = True,
    save_models: bool = False,
    fixed_inputs: list[str] | None = None,
    automatic: bool = True,
    run: Run | None = None,
    **kwargs,
) -> tuple[dict, list[str]]:
    """
    Backward elimination of inputs (see `FeatureSelection` for the options in `kwargs`).

    Args:
        file_path: Path to the data.
        preprocessing: Preprocessing holding all candidate inputs. It is not changed.
        model: The physXAI model.
        ascending_lag_order: Remove the highest lag of a feature first.
        use_multi_step_error: Use the error of the recursive prediction, if the preprocessing has an
                              `eval_horizon`, else the single step error.
        save_models: Not supported anymore (models of the candidates are not saved).
        fixed_inputs: Inputs which are never removed.
        automatic: Without effect, the selection never asks for input.
        run: If given, the result is saved into its directory (`feature_selection.json` and `.html`).

    Returns:
        tuple: {number of inputs: [{'inputs', 'kpi', 'kpi_single_step'}, ...]} and the selected inputs.
    """
    if save_models:
        logger.warning("save_models is not supported anymore, the models of the candidates are not saved.")
    horizon = use_multi_step_error and getattr(preprocessing, "eval_horizon", None) is not None
    selection = FeatureSelection(
        preprocessing,
        model,
        criterion="horizon" if horizon else "single_step",
        fixed_inputs=fixed_inputs,
        ascending_lag_order=ascending_lag_order,
        **kwargs,
    )
    result = selection.run(file_path)
    if run is not None:
        save_result(result, run)
    runs = {
        n: [{"inputs": c.inputs, "kpi": c.score, "kpi_single_step": c.single_step_rmse} for c in candidates]
        for n, candidates in result.steps.items()
    }
    return runs, result.selected.inputs


def save_result(result: FeatureSelectionResult, run: Run, name: str = "feature_selection") -> dict:
    """
    Saves the result of a feature selection into a run directory.

    Args:
        result: The result of `FeatureSelection.run`.
        run: The run directory.
        name: Name of the files: `{name}.json` (all candidates and their criteria) and `{name}.html` (plot).

    Returns:
        dict: The paths of the written files.
    """
    from physXAI.plotting.plotting import save_figure

    return {
        "json": run.save_json(f"{name}.json", result.get_config()),
        "plot": save_figure(result.plot(), run.file(f"{name}.html")),
    }
