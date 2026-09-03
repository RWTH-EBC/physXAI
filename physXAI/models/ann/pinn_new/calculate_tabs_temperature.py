import numpy as np

from typing import Optional, Sequence

t_tabs_column = "T_tabs_core_mea"
t_tabs_k1_column = "T_tabs_core_mea_k1"

def calculate_tabs_temperature_scale(
    x: np.ndarray,
    columns: Sequence[str],
    rc_kwargs: Optional[dict] = None,
    use_case: str = "UC2"
) -> float:
    """
    """
    x = np.asarray(x)

    if x.shape[1] != len(columns):
        raise ValueError("the number of columns must match the number of features in x!")

    if len(x) < 2:
        raise ValueError("at least two measurement rows are required to calculate the T_tabs scale!")

    if rc_kwargs is None:
        raise ValueError("rc_kwargs must be provided!")

    if use_case != "UC2":
        raise ValueError("TABS physics loss can only be used for usecase='UC2'!")

    required_features = {
        "t_tabs_index": t_tabs_column,
        "t_tabs_k1_index": t_tabs_k1_column,
    }

    for key, column in required_features.items():
        if rc_kwargs.get(key) is None and column in columns:
            rc_kwargs[key] = columns.index(column)

        if rc_kwargs.get(key) is None:
            raise ValueError(f"the input column '{column}' or rc_kwargs['{key}'] is required to calculate the TABS physics loss scale!")

    t_tabs_k1_index = rc_kwargs["t_tabs_k1_index"]
    t_tabs_index = rc_kwargs["t_tabs_index"]

    if isinstance(t_tabs_k1_index, (int, np.integer)):
        t_tabs_indices = [t_tabs_index]
    else:
        t_tabs_indices = list(t_tabs_index)

    if isinstance(t_tabs_k1_index, (int, np.integer)):
        t_tabs_k1_indices = [t_tabs_k1_index]
    else:
        t_tabs_k1_indices = list(t_tabs_k1_index)

    measured_t_tabs_k = np.take(x, t_tabs_indices, axis=1).sum(axis=1)
    measured_t_tabs_k1 = np.take(x, t_tabs_k1_indices, axis=1).sum(axis=1)

    measured_delta_t_tabs = measured_t_tabs_k1 - measured_t_tabs_k

    if not np.isfinite(measured_delta_t_tabs).all():
        raise ValueError("the measured T_tabs chnages contain NaN or Inf!")

    scale = float(np.std(measured_delta_t_tabs, ddof=1))

    return scale

    