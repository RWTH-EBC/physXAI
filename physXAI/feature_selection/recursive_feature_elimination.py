import os
import re
import numpy as np
from physXAI.plotting.plotting import plot_recFeatureSelection
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep, PreprocessingData, output_feature
from physXAI.utils.logging import Logger
from physXAI.models.models import SingleStepModel


def search_best_features(runs: dict, multi_step: bool, use_multi_step_error: bool, automatic: bool = False):
    sorted_kpis = dict()
    min_value = np.inf
    min_index = None
    for k, v in runs.items():
        values = list()
        for f in v:
            if multi_step and not use_multi_step_error:
                values.append(f['kpi_single_step'])
            else:
                values.append(f['kpi'])
        index = values.index(min(values))
        sorted_kpis[k] = {
            'inputs': v[index]['inputs'],
            'kpi': values[index],
        }
        if values[index] < min_value:
            min_value = values[index]
            min_index = k

    if not automatic:
        try:
            max_features = int(input("Enter number of features. Otherwise features are selected based on RMSE."))
        except ValueError:
            max_features = np.inf
    else:
        max_features = np.inf

    Logger.print('Selected features:', 'info')
    if max_features == np.inf:
        inputs = sorted_kpis[min_index]['inputs']
    else:
        inputs = sorted_kpis[max_features]['inputs']
    Logger.print(inputs, 'info')
    return inputs


def _uses_horizon(preprocessing: PreprocessingData) -> bool:
    """Whether the models are evaluated over a horizon (recursively, as in an MPC)."""
    if not isinstance(preprocessing, PreprocessingSingleStep):
        raise NotImplementedError(
            "Feature selection supports single step models only. To select features on the multi step error, "
            "use PreprocessingSingleStep(eval_horizon=...)."
        )
    return preprocessing.eval_horizon is not None


def _kpis(td, multi_step: bool) -> dict:
    """Validation RMSE (mean over the horizon for multi step evaluations) and single step validation RMSE."""
    single = td.metrics.val_kpis['RMSE Val']
    if not multi_step:
        return {'kpi': single}
    if td.horizon_metrics is None or 'val' not in td.horizon_metrics.kpis:
        raise ValueError('The multi step error requires the recursive feature as input and validation data.')
    return {'kpi': float(np.mean(td.horizon_metrics.kpis['val']['rmse'])), 'kpi_single_step': single}


def recursive_feature_elimination(file_path: str, preprocessing: PreprocessingData,
                                  model: SingleStepModel, ascending_lag_order: bool = True,
                                  use_multi_step_error: bool = True, save_models: bool = False,
                                  fixed_inputs: list[str] = None):
    assert preprocessing.val_size > 0, 'Value Error: For Feature Selection, preprocessing.val_size must be > 0.'
    multi_step = _uses_horizon(preprocessing)
    recursive_feature = output_feature(preprocessing.output[0])

    if fixed_inputs is None:
        fixed_inputs = list()

    if Logger._logger is None:
        Logger.setup_logger()

    Logger.print('Feature Selection', 'info')

    org_inputs = preprocessing.inputs
    inputs = preprocessing.inputs
    input_length = len(inputs)

    runs = dict()

    # Train original model
    td = preprocessing.pipeline(file_path)
    path = f'model_{input_length}'
    p = os.path.join(Logger._logger, path)
    model.pipeline(td, save_path=p, plot=False, save_model=save_models)

    # Evaluate model
    runs[input_length] = [{'inputs': inputs, **_kpis(td, multi_step)}]

    # Recursive feature elimination
    for j in range(input_length - 1, 0, -1):
        Logger.print(f'Features {j + 1}', 'info')
        Logger.print(inputs, 'info')

        # Reduced input features
        new_inputs = list()
        for i, v in enumerate(inputs):
            if multi_step and v == recursive_feature:  # needed for the recursive evaluation
                continue
            if ascending_lag_order:
                if '_lag' not in v:
                    if v + '_lag1' in inputs:
                        continue
                else:
                    match = int(re.search(r"_lag(\d+)", v).group(1))
                    if v.replace(f'_lag{match}', f'_lag{match + 1}') in inputs:  # pragma: no cover
                        continue  # pragma: no cover
            if v in fixed_inputs:
                continue
            new_inputs.append([item for item in inputs if item != v])

        if len(new_inputs) == 0:
            break

        # Evaluate Kpis for new inputs
        kpis = dict()
        kpis_add = dict()

        for i, v in enumerate(new_inputs):

            preprocessing.inputs = v
            td = preprocessing.pipeline(file_path)
            path = f'model_{j}_{i}'
            p = os.path.join(Logger._logger, path)
            model.pipeline(td, save_path=p, plot=False, save_model=save_models)

            result = _kpis(td, multi_step)
            kpis[i] = result['kpi']
            if multi_step:
                kpis_add[i] = result['kpi_single_step']

        if not multi_step:
            run = [{'inputs': new_inputs[i], 'kpi': kpis[i]} for i in range(len(new_inputs))]
        else:
            run = [{'inputs': new_inputs[i], 'kpi': kpis[i], 'kpi_single_step': kpis_add[i]}
                   for i in range(len(new_inputs))]

        # Choose best models
        if multi_step and not use_multi_step_error:
            key_filter = int(min(kpis_add, key=kpis_add.get))
        else:
            key_filter = int(min(kpis, key=kpis.get))
        inputs = new_inputs[key_filter]
        runs[j] = run
    Logger.print(f'Features {1}', 'info')
    Logger.print(inputs, 'info')

    preprocessing.inputs = org_inputs

    return runs


def recursive_feature_elimination_pipeline(file_path: str,
                                           preprocessing: PreprocessingData,
                                           model: SingleStepModel, ascending_lag_order: bool = True,
                                           use_multi_step_error: bool = True, save_models: bool = False,
                                           fixed_inputs: list[str] = None,
                                           automatic: bool = False):

    runs = recursive_feature_elimination(file_path, preprocessing, model, ascending_lag_order, use_multi_step_error,
                                         save_models, fixed_inputs)

    multi_step = _uses_horizon(preprocessing)
    plot_recFeatureSelection(runs, multi_step, use_multi_step_error)

    inputs = search_best_features(runs, multi_step, use_multi_step_error, automatic=automatic)

    return runs, inputs
