import os
import re
import numpy as np
from physXAI.plotting.plotting import plot_recFeatureSelection
from physXAI.preprocessing.preprossesing import PreprocessingSingleStep, PreprocessingMultiStep, \
    PreprocessingData
from physXAI.preprocessing.training_data import TrainingDataMultiStep
from physXAI.utils.logging import Logger
from physXAI.evaluation.metrics import Metrics
from physXAI.models.models import SingleStepModel


def search_best_features(runs: dict, multi_step: bool, use_multi_step_error: bool):
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

    try:
        max_features = int(input("Enter number of features. Otherwise features are selected based on RMSE."))
    except ValueError:
        max_features = np.inf

    print('Selected features:')
    if max_features == np.inf:
        inputs = sorted_kpis[min_index]['inputs']
    else:
        inputs = sorted_kpis[max_features]['inputs']
    print(inputs)
    return inputs


def recursive_feature_elimination(file_path: str, preprocessing: PreprocessingData,
                                  model: SingleStepModel, ascending_lag_order: bool = True,
                                  use_multi_step_error: bool = True, save_models: bool = False,
                                  fixed_inputs: list[str] = None):
    assert preprocessing.val_size > 0, 'Value Error: For Feature Selection, preprocessing.val_size must be > 0.'

    if fixed_inputs is None:
        fixed_inputs = list()

    print('Feature Selection')
    Metrics.print_evaluate = False

    if Logger._logger is None:
        Logger.setup_logger()

    org_inputs = preprocessing.inputs
    inputs = preprocessing.inputs
    input_length = len(inputs)

    runs = dict()

    # Train original model
    td = preprocessing.pipeline(file_path)
    path = f'model_{input_length}'
    p = os.path.join(Logger._logger, path)
    model.pipeline(td, save_path=p, plot=False, save_model=save_models)
    val_kpi = td.metrics.val_kpis['RMSE Val']

    # Evaluate model
    if isinstance(preprocessing, PreprocessingSingleStep):
        runs[input_length] = [{'inputs': inputs, 'kpi': val_kpi}]
    elif isinstance(td, TrainingDataMultiStep):
        val_kpi_single = td.single_step_metrics.val_kpis['RMSE Val']
        runs[input_length] = [{'inputs': inputs, 'kpi': val_kpi, 'kpi_single_step': val_kpi_single}]
    else:
        raise NotImplementedError

    # Recursive feature elimination
    for j in range(input_length - 1, 0, -1):
        print(f'Features {j + 1}')
        print(inputs)

        # Reduced input features
        new_inputs = list()
        for i, v in enumerate(inputs):
            if isinstance(preprocessing, PreprocessingMultiStep) and preprocessing.init_features[0] == v:
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

            val_kpi = td.metrics.val_kpis['RMSE Val']
            kpis[i] = val_kpi
            if isinstance(preprocessing, PreprocessingMultiStep):
                val_kpi = td.single_step_metrics.val_kpis['RMSE Val']
                kpis_add[i] = val_kpi

        if isinstance(preprocessing, PreprocessingSingleStep):
            run = [{'inputs': new_inputs[i], 'kpi': kpis[i]} for i in range(len(new_inputs))]
        else:
            run = [{'inputs': new_inputs[i], 'kpi': kpis[i], 'kpi_single_step': kpis_add[i]}
                   for i in range(len(new_inputs))]

        # Choose best models
        if isinstance(preprocessing, PreprocessingMultiStep) and not use_multi_step_error:
            key_filter = int(min(kpis_add, key=kpis_add.get))
        else:
            key_filter = int(min(kpis, key=kpis.get))
        inputs = new_inputs[key_filter]
        runs[j] = run
    print(f'Features {1}')
    print(inputs)

    preprocessing.inputs = org_inputs

    return runs


def recursive_feature_elimination_pipeline(file_path: str,
                                           preprocessing: PreprocessingData,
                                           model: SingleStepModel, ascending_lag_order: bool = True,
                                           use_multi_step_error: bool = True, save_models: bool = False,
                                           fixed_inputs: list[str] = None):

    runs = recursive_feature_elimination(file_path, preprocessing, model, ascending_lag_order, use_multi_step_error,
                                         save_models, fixed_inputs)

    plot_recFeatureSelection(runs, isinstance(preprocessing, PreprocessingMultiStep), use_multi_step_error)

    inputs = search_best_features(runs, isinstance(preprocessing, PreprocessingMultiStep), use_multi_step_error)

    return runs, inputs
