import os

from physXAI.utils.logging import Logger
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep
from physXAI.preprocessing.constructed import Feature, FeatureConstruction

from physXAI.models.ann.ann_design import PINNModel
from physXAI.models.ann.pinn_new.rc_layers import RC1R1CLayer


# -------------------------------------------------------------------------
# 1. Logger
# -------------------------------------------------------------------------
Logger.setup_logger(
    folder_name="test_pinn_1r1c_w28t",
    override=True,
    print_level="info"
)


# -------------------------------------------------------------------------
# 2. Datenpfad
# -------------------------------------------------------------------------
file_path = "C:\\Users\\phe-dwe\\Downloads\\W28T.csv"


# -------------------------------------------------------------------------
# 3. Zielgröße erstellen: Change(TAir) = TAir_t - TAir_{t-1}
#    PreprocessingSingleStep verschiebt y danach um -1.
#    Dadurch lernt das Modell: TAir_{t+1} - TAir_t
# -------------------------------------------------------------------------
FeatureConstruction.reset()

t_air = Feature("TAir")
t_air_lag1 = t_air.lag(1, previous=False)

delta_t_air = t_air - t_air_lag1
delta_t_air.rename("Change(TAir)")


# -------------------------------------------------------------------------
# 4. Inputs und Output
# -------------------------------------------------------------------------
inputs = [
    "TAir",
    "TDryBul",
    "V_flow_AHU",
    "T_AHU_sup",
    "HDirNor",
]

output = "Change(TAir)"


# -------------------------------------------------------------------------
# 5. Preprocessing
# -------------------------------------------------------------------------
prep = PreprocessingSingleStep(
    inputs=inputs,
    output=output,
    csv_delimiter=",",
    time_index_col="time",
    time_step=300,
    test_size=0.1,
    val_size=0.1,
    random_state=42,
)

td = prep.pipeline(file_path)

print("Input-Spalten:", td.columns)
print("X_train:", td.X_train_single.shape)
print("y_train:", td.y_train_single.shape)


# -------------------------------------------------------------------------
# 6. Indizes für RC-Layer bestimmen
# -------------------------------------------------------------------------
t_ambient_index = td.columns.index("TDryBul")
t_room_index = td.columns.index("TAir")
v_flow_ahu_index = td.columns.index("V_flow_AHU")
t_ahu_sup_index = td.columns.index("T_AHU_sup")
h_dir_nor_index = td.columns.index("HDirNor")

# -------------------------------------------------------------------------
# 7. Modell erstellen
# -------------------------------------------------------------------------
m = PINNModel(
    rc_layer=RC1R1CLayer,

    t_room_column="TAir",

    rc_kwargs={
        "time_step": 300.0,
        "resistance": 0.05,
        "capacitance": 2_000_000,
        "t_ambient_index": t_ambient_index,
        "theta_solar_init": 0.5,
        "t_room_index": t_room_index,
        "v_flow_ahu_index": v_flow_ahu_index,
        "t_ahu_sup_index": t_ahu_sup_index,
        "h_dir_nor_index": h_dir_nor_index,
    },

    predict_delta=True,

    n_layers=2,
    n_neurons=32,
    activation_function="softplus",
    rescale_output=True,

    trainable_rc=True,
    physics_loss_weight=1.0,
    physics_loss_reduction="mean",

    epochs=100,
    batch_size=64,
    learning_rate=0.001,
    early_stopping_epochs=None,
    random_seed=42,
)


# -------------------------------------------------------------------------
# 8. Training + Evaluation
# -------------------------------------------------------------------------
model = m.pipeline(
    td,
    plot=False,
    save_model=False,
)

print("\n--- Gefundene physikalische Parameter ---")
# Geht alle Variablen des trainierten Keras-Modells durch
for var in model.trainable_variables:
    # Filtert nach unseren RC-Variablen
    if 'raw_resistance' in var.name or 'raw_capacitance' in var.name:
        # Achtung: Wir haben die Variablen im Layer mit Softplus initialisiert, 
        # um sie positiv zu halten. Wir müssen sie hier also auch wieder durch 
        # die Softplus-Funktion jagen, um den echten physikalischen Wert zu sehen!
        import tensorflow as tf
        echter_wert = tf.nn.softplus(var).numpy()
        print(f"{var.name}: {echter_wert}")

print(td.metrics.get_config())
print(td.training_record.history.keys())
print("Letzter Trainings-Loss:", td.training_record.history["loss"][-1])
