"""
Blocked k-fold cross validation of the number of training epochs.

With little data the validation split of a single training holds few independent samples (e.g. two sequences of
24 h in 14 days of data). The epoch picked by the early stopping on it is then largely random: a training may stop
after a few epochs (underfitted) or run long (overfitted). The cross validation uses every block outside the test
data once for validation instead:

1. `folds` trainings: the blocks outside the test data are divided into `folds` contiguous groups, group j is the
   validation data of training j (`BlockSplit(folds=k, fold=j)`). Each training uses early stopping (patience,
   minimum number of epochs) and records its validation curve.
2. The mean validation curve over the folds, on the epochs all folds reached, gives the number of epochs E*
   (its minimum, not before `min_epochs`).
3. The final model is trained on all data outside the test blocks (`BlockSplit(folds=k, fold=None)`, no validation
   data, hence no early stopping) for E* epochs.

The test blocks are drawn as without cross validation and are the same in all trainings. A model trained in two
phases (`RecursiveTraining(mode='single_then_multi')`) gets the epochs of both phases from their own curves; a
closed form fit (`ModularANN(solver='qp' / 'ols')`) has no epochs, only its multi step phase is cross validated.
The cost is (folds + 1) trainings.

Example:
    >>> prep = PreprocessingMultiStep(inputs=[...], output=px.Change('T'), warmup_width='6h', label_width='24h')
    >>> model = RNNModel(rnn_layer='LSTM', rnn_units=16, epochs=500, early_stopping_epochs=50)
    >>> result = fit_cross_validated(prep, model, 'data.csv', CrossValidation(folds=5, min_epochs=20))
    >>> result.td.cross_validation['epochs']     # e.g. {'training': 87}
    >>> run.save_for_agentlib('T', result.preprocessing, result.model, result.td)
"""

from __future__ import annotations

import copy
import logging
from dataclasses import asdict, dataclass

import numpy as np

logger = logging.getLogger("physXAI")

TRAINING, RECURSIVE = "training", "recursive"


@dataclass
class CrossValidation:
    """
    Settings of the cross validation of the number of epochs.

    Args:
        folds: Number of folds (contiguous groups of the blocks outside the test data).
        patience: Patience of the early stopping in the folds. None: the one of the model.
        min_epochs: Minimum number of epochs: the early stopping in the folds starts watching from this epoch
                    on, and E* is not chosen below it.
    """

    folds: int = 5
    patience: int | None = None
    min_epochs: int = 0

    def get_config(self) -> dict:
        return asdict(self)


@dataclass
class CrossValidated:
    """Result of `fit_cross_validated`: the preprocessing of the final training, the trained model and its data."""

    preprocessing: object
    model: object
    td: object


def phases(model) -> list[str]:
    """Training phases of `model` whose number of epochs is cross validated."""
    from physXAI.models.modular.modular_ann import CLOSED_FORM

    recursive = getattr(model, "recursive_training", None)
    closed_form = getattr(model, "solver", None) in CLOSED_FORM
    out = list()
    if not closed_form and (recursive is None or recursive.mode == "single_then_multi"):
        out.append(TRAINING)
    if recursive is not None:
        out.append(RECURSIVE)
    return out


def _curve(td, model, phase: str) -> np.ndarray:
    """Validation curve of a phase from the training records of `td`."""
    record = td.training_record if phase == TRAINING else td.recursive_training_record
    history = getattr(record, "history", record) or dict()
    key = "val_data_loss" if phase == TRAINING and getattr(model, "physics", None) is not None else "val_loss"
    if key not in history:
        raise RuntimeError(f"The {phase} phase recorded no '{key}': the fold had no validation data.")
    return np.asarray(history[key], dtype=float)


def _fold_model(model, cv: CrossValidation):
    """Untrained copy of `model` with the early stopping of the folds."""
    m = copy.deepcopy(model)
    if cv.patience is not None:
        m.early_stopping_epochs = cv.patience
    m.early_stopping_start = cv.min_epochs
    recursive = getattr(m, "recursive_training", None)
    if recursive is not None:
        recursive.early_stopping_start = cv.min_epochs
        if cv.patience is not None:
            recursive.early_stopping_epochs = cv.patience
        if (m.early_stopping_epochs or recursive.early_stopping_epochs) is None:
            raise ValueError("The cross validation needs an early stopping patience (model or CrossValidation).")
    elif m.early_stopping_epochs is None:
        raise ValueError("The cross validation needs an early stopping patience (model or CrossValidation).")
    return m


def _has_recursive_validation(td, model) -> bool:
    """Whether the validation data of `td` holds a window of the multi step training of `model`. With little data
    (short blocks, long lags) a fold may hold none: its validation blocks are shorter than horizon + lags."""
    from physXAI.evaluation.rollout import recursion_spec, rollout_data

    config = model.recursive_training
    spec = recursion_spec(td)
    if spec is None:  # fit_recursive raises the error
        return True
    return rollout_data(td, spec, "val", config.resolve_horizon(td), config.stride) is not None


def _train(model, td):
    """Fits a model without evaluating it (the folds only need the validation curves)."""
    model._check_training_data(td)
    model.prepare()
    keras_model = model.generate_model(td=td)
    model.compile_model(keras_model)
    model.fit_model(keras_model, td)


def cross_validate(preprocessing, model, file_path, cv: CrossValidation | None = None) -> dict:
    """
    Trains `cv.folds` copies of the untrained `model`, each validated on another fold, and picks the number of
    epochs per training phase from the mean validation curve.

    Args:
        preprocessing: A preprocessing with a `BlockSplit` of strategy 'blocks' (its test_size is kept).
        model: The untrained physXAI model (not changed).
        file_path: The csv file of the data.
        cv: Settings, defaults to `CrossValidation()`.

    Returns:
        dict: 'epochs' ({phase: E*}), 'mean_curves' and 'curves' ({phase: validation curves}), 'fold_best_epochs'
        ({phase: best epoch of every fold}), 'folds_used' ({phase: folds behind the curves}) and the settings.
        A fold whose validation data holds no window of the multi step horizon does not count for the multi step
        phase (its multi step training is skipped); an error only if no fold holds one.
    """
    cv = cv or CrossValidation()
    todo = phases(model)
    if not todo:
        raise ValueError(f"{type(model).__name__} has no trained epochs to cross validate (closed form fit).")
    curves: dict[str, list[np.ndarray]] = {phase: list() for phase in todo}
    used: dict[str, list[int]] = {phase: list() for phase in todo}
    for fold in range(cv.folds):
        prep = copy.deepcopy(preprocessing)
        prep.split = preprocessing.split.with_fold(cv.folds, fold)
        m = _fold_model(model, cv)
        td = prep.pipeline(file_path, model=m)
        fold_phases = list(todo)
        if RECURSIVE in todo and not _has_recursive_validation(td, m):
            # without validation windows the multi step phase would run all its epochs without early stopping and
            # record no curve: leave it out in this fold (the single step phase is still validated)
            logger.warning(
                "Cross validation fold %d/%d: no validation window of the multi step training, the fold does not "
                "count for its epochs.",
                fold + 1,
                cv.folds,
            )
            fold_phases.remove(RECURSIVE)
            if not fold_phases:
                continue
            m.recursive_training = None
        _train(m, td)
        for phase in fold_phases:
            curves[phase].append(_curve(td, m, phase))
            used[phase].append(fold)
        logger.info(
            "Cross validation fold %d/%d: best epoch %s",
            fold + 1,
            cv.folds,
            {p: int(np.argmin(curves[p][-1])) + 1 for p in fold_phases},
        )

    epochs, mean_curves = dict(), dict()
    for phase in todo:
        if not curves[phase]:
            raise RuntimeError(
                f"No fold holds validation data for the {phase} phase: reduce the folds, the horizon or the lags, "
                "or use longer blocks."
            )
        length = min(len(c) for c in curves[phase])
        mean = np.mean([c[:length] for c in curves[phase]], axis=0)
        first = min(max(cv.min_epochs, 1), length) - 1
        epochs[phase] = int(first + np.argmin(mean[first:])) + 1
        mean_curves[phase] = mean.tolist()
    result = {
        "epochs": epochs,
        "mean_curves": mean_curves,
        "curves": {p: [c.tolist() for c in cs] for p, cs in curves.items()},
        "fold_best_epochs": {p: [int(np.argmin(c)) + 1 for c in cs] for p, cs in curves.items()},
        # folds (0-based) behind the curves of each phase: a fold without validation windows of the multi step
        # training is missing in its phase
        "folds_used": used,
        "settings": cv.get_config(),
    }
    logger.info("Cross validation: epochs %s (best epochs of the folds %s)", epochs, result["fold_best_epochs"])
    return result


def fit_cross_validated(
    preprocessing, model, file_path, cv: CrossValidation | None = None, result: dict | None = None
) -> CrossValidated:
    """
    Cross validates the number of epochs (`cross_validate`) and trains the final model on all data outside the test
    blocks for that number of epochs. The result of the cross validation is stored in `td.cross_validation`.

    Args:
        preprocessing: A preprocessing with a `BlockSplit` of strategy 'blocks' (not changed).
        model: The untrained physXAI model (not changed, the trained copy is returned).
        file_path: The csv file of the data.
        cv: Settings, defaults to `CrossValidation()`.
        result: The return value of an earlier `cross_validate` of the same model on the same data (e.g. with another
                seed, i.e. other test blocks and initialization): its epochs are used without a new cross
                validation ('reused': True in `td.cross_validation`). Costs one training instead of folds + 1.

    Returns:
        CrossValidated: preprocessing of the final training (split without validation data), the trained model
        (evaluated, `model.trained_model` set) and its training data.
    """
    cv = cv or CrossValidation()
    if result is None:
        result = cross_validate(preprocessing, model, file_path, cv)
    else:
        if set(result["epochs"]) != set(phases(model)):
            raise ValueError(
                f"The given cross validation has the phases {sorted(result['epochs'])}, the model {phases(model)}."
            )
        result = dict(result, reused=True)
        logger.info("Cross validation reused: epochs %s", result["epochs"])
    final = copy.deepcopy(model)
    if TRAINING in result["epochs"]:
        final.epochs = result["epochs"][TRAINING]
    if RECURSIVE in result["epochs"]:
        final.recursive_training.epochs = result["epochs"][RECURSIVE]
    prep = copy.deepcopy(preprocessing)
    prep.split = preprocessing.split.with_fold(cv.folds, None)
    td = prep.pipeline(file_path, model=final)
    final.pipeline(td, plot=False, save_model=False)
    td.cross_validation = result
    return CrossValidated(prep, final, td)
