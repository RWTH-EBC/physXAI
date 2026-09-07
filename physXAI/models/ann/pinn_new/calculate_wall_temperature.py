import keras
import pathlib
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Optional, Sequence

from physXAI.models.ann.pinn_new.rc_layers import RC2R2CPhysNetLayerUC1, RC2R2CPhysNetLayerUC2


time_column = "time"
t_air_column = "TAir"
t_amb_column = "TDryBul"
v_flow_ahu_column = "V_flow_AHU"
t_ahu_sub_column = "T_AHU_sup"
t_sup_w_h_column = "T_sup_w_h"
y_valve_h_column = "y_valve_h"
h_dir_nor_column = "HDirNor"

input_column = [
    t_air_column,
    t_amb_column,
    v_flow_ahu_column,
    t_ahu_sub_column,
    t_sup_w_h_column,
    y_valve_h_column,
    h_dir_nor_column,
]


parameters_RC = {
    "r_win": 0.15444015444015444 + 1.0 / (7.0 * 2.7),
    "r_ext": 0.001440291630693675 + 1.0 / ((19.0969 + 9.5226 + 9.5226) * 2.1993222711911513),
    "c_air": 68860.74019524,
    "theta_solar_init": 1.75,
    "alpha_init": 0.1,
    "beta_init": 0.7,
}


def calculate_wall_temperature_for_scaling(
    x: np.ndarray,
    measured_change_t_air: np.ndarray,
    columns: Sequence[str],
    rc_kwargs: Optional[dict] = None,
    use_case: str = "UC1",
) -> np.ndarray:

    x = np.asarray(x)

    if x.ndim != 2:
        raise ValueError("x must be two-dimensional!")

    measured_change_t_air = np.asarray(measured_change_t_air, dtype=np.float32).reshape(-1, 1)

    if len(x) != len(measured_change_t_air):
        raise ValueError("x and measured_change_t_air must contain the same number of rows!")

    if rc_kwargs is None:
        raise ValueError("rc_kwargs must be provided!")

    rc_kwargs = dict(rc_kwargs)

    physics_layer_classes = {
        "UC1": RC2R2CPhysNetLayerUC1,
        "UC2": RC2R2CPhysNetLayerUC2,
    }

    physics_layer_class = physics_layer_classes[use_case]

    physics_layer = physics_layer_class(
        trainable_rc=False,
        use_internal_gains=False,
        **rc_kwargs,
    )

    physics_layer.trainable = False

    t_wall = physics_layer(
        [
            keras.ops.convert_to_tensor(x, dtype="float32"),
            keras.ops.convert_to_tensor(measured_change_t_air, dtype="float32"),
        ],
        training=False,
    )

    t_wall = keras.ops.convert_to_numpy(t_wall).reshape(-1)

    if not np.isfinite(t_wall).all():
        raise ValueError("The calculated wall temperature contains NaN or Inf!")

    return t_wall


def calculate_wall_temperature_timeseries(
    data: pd.DataFrame,
    time_step: float = 300.0,
    rc_kwargs: Optional[dict] = None,
) -> pd.DataFrame:

    if time_column not in data.columns and data.index.name == time_column:
        data = data.reset_index()

    required_columns = [
        time_column,
        *input_column,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError("Missing columns: " + ", ".join(missing_columns))

    if len(data) < 2:
        raise ValueError("At least two measurement rows are required!")

    if time_step <= 0:
        raise ValueError("time_step must be positive!")

    data = data.loc[:, required_columns].copy()

    data[required_columns] = data[required_columns].apply(pd.to_numeric, errors="raise")

    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("The required measurement columns contain NaN or Inf!")

    time = data[time_column].to_numpy(dtype=float)
    time_differences = np.diff(time)

    if not np.allclose(
        time_differences,
        float(time_step),
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(f"The time grid must be {float(time_step):g} seconds throughout. Found: {np.unique(time_differences)}!")

    current = data.iloc[:-1].copy()

    t_air = data[t_air_column].to_numpy(dtype=float)
    measured_change_t_air = t_air[1:] - t_air[:-1]

    calculation_kwargs = {} if rc_kwargs is None else dict(rc_kwargs)
    calculation_kwargs["time_step"] = float(time_step)

    t_wall = calculate_wall_temperature_for_scaling(
        x=current[input_column].to_numpy(),
        measured_change_t_air=measured_change_t_air,
        columns=input_column,
        rc_kwargs=calculation_kwargs,
    )

    current_time = current[time_column].to_numpy(dtype=float)

    return pd.DataFrame(
        {
            time_column: current_time,
            "time_next": time[1:],
            "time_hours": (current_time - current_time[0]) / 3600.0,
            t_air_column: current[t_air_column].to_numpy(dtype=float),
            "TAir_next_measured": t_air[1:],
            "Change(TAir)": measured_change_t_air,
            "TWall": t_wall,
        }
    )


def main(
    data_path: Path,
    output_path: Optional[Path] = None,
    csv_delimiter: str = ",",
    time_step: float = 300.0,
) -> Path:

    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Measurement file not found: {data_path}!")

    data = pd.read_csv(data_path, delimiter=csv_delimiter)

    result = calculate_wall_temperature_timeseries(data=data, time_step=time_step)

    if output_path is None:
        output_path = data_path.with_name("wall_temperature_2r2c.csv")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(output_path, index=False)

    return output_path


if __name__ == "__main__":
    data_path = str(pathlib.Path(__file__).parents[5]) + "\\0_agentlib_configs\\training_data\\sim\\W28T.csv"
    output = str(pathlib.Path(__file__).parents[5]) + "\\0_agentlib_configs\\results\\walltemperature\\walltemperature.csv"

    time_step = 300

    main(
        data_path=data_path,
        output_path=output,
        csv_delimiter=",",
        time_step=time_step,
    )