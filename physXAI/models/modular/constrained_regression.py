from abc import abstractmethod
import os
import time
from typing import Union
import numpy as np
from physXAI.models.models import SingleStepModel, register_model
from physXAI.models.modular.modular_expression import ModularAdd, ModularExpression, ModularFeature, ModularMul, ModularPow, ModularSub, ModularTrueDiv
from physXAI.plotting.plotting import plot_metrics_table, plot_multi_rmse, plot_prediction_correlation, plot_predictions, subplots
from physXAI.preprocessing.constructed import FeatureBase
from physXAI.preprocessing.training_data import TrainingData, TrainingDataGeneric, TrainingDataMultiStep
import casadi as ca
from physXAI.utils.logging import Logger, create_full_path, get_full_path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


@register_model
class ConstrainedRegression(SingleStepModel):
    """
    A constrained (montonone, convex, concave) regression model solved using Casadi / IPOPT.
    """
    allowed_input_types = [
        ModularFeature,
        ModularAdd,
        ModularSub,
        ModularMul,
        ModularTrueDiv,
        ModularPow,
    ]

    def __init__(self, inputs: list[Union[ModularExpression, FeatureBase]], monotonies: dict[str, int] = None, convexities: dict[str, int] = None):
        super().__init__()
        self.inputs = [inp if isinstance(inp, ModularExpression) else inp.input() for inp in inputs]

        for inp in self.inputs:
            if type(inp) not in self.allowed_input_types:
                raise ValueError(f"Input type {type(inp)} is not allowed for ConstrainedRegression. Allowed types: {self.allowed_input_types}")
            
        if monotonies is None:
            monotonies = {}
        self.monotonies = monotonies
            
        if convexities is None:
            convexities = {}
        self.convexities = convexities
            
        self.opti = ca.Opti()
        self.w_vec = None


    def generate_model(self, **kwargs):
        """
        Generates Casadi optimization problem based on the specified inputs and constraints.
        """
        td = kwargs['td']
        n_features = td.X_train_single.shape[1]
        input_layer = keras.layers.Input(shape=(n_features,))

        inps = list()
        for x in self.inputs:
            y = x.construct(input_layer, td)
            inps.append(y)
        l = keras.layers.Dense(units=1, activation='linear', name='ConstrainedRegression')(keras.layers.Concatenate()(inps))
        model = keras.models.Model(inputs=input_layer, outputs=l)

        X = list()
        sym_raw = {name: ca.MX.sym(name) for name in td.columns}
        x_sym_list = list()
        for inp in self.inputs:
            try:
                X_0, sym_0 = inp.get_value(td, input_layer, sym_raw)
                X.append(X_0)
                x_sym_list.append(sym_0)
            except NotImplementedError:
                raise ValueError(f"Input type {type(inp)} does not implement get_value method, but is specified as allowed type for ConstrainedRegression. Please implement get_value method for this input type or remove it from allowed_input_types.")
        X = np.column_stack(X)
        x_sym = ca.vertcat(*x_sym_list)
        y_sym = 0

        # Map constraint names to composite or raw feature indices
        monotonies_composite_idx = {}  # Maps composite feature index to monotonicity value
        monotonies_raw = {}  # Maps raw feature name to monotonicity value
        
        for constraint_name, mono_value in self.monotonies.items():
            found = False
            
            # Check if it's a composite feature (by input name)
            for i, inp in enumerate(self.inputs):
                if constraint_name == inp.name:
                    monotonies_composite_idx[i] = mono_value
                    found = True
                    break
            
            # If not found, check if it's a raw feature
            if not found and constraint_name in sym_raw:
                monotonies_raw[constraint_name] = mono_value
                found = True
            
            if not found:
                raise ValueError(f"Monotonicity specified for unknown input '{constraint_name}'. "
                                f"Available composite features: {[inp.name for inp in self.inputs]}. "
                                f"Available raw features: {list(sym_raw.keys())}")
        
        # Same for convexities
        convexities_composite_idx = {}
        convexities_raw = {}
        
        for constraint_name, convex_value in self.convexities.items():
            found = False
            
            # Check if it's a composite feature (by input name)
            for i, inp in enumerate(self.inputs):
                if constraint_name == inp.name:
                    convexities_composite_idx[i] = convex_value
                    found = True
                    break
            
            # If not found, check if it's a raw feature
            if not found and constraint_name in sym_raw:
                convexities_raw[constraint_name] = convex_value
                found = True
            
            if not found:
                raise ValueError(f"Convexity specified for unknown input '{constraint_name}'. "
                                f"Available composite features: {[inp.name for inp in self.inputs]}. "
                                f"Available raw features: {list(sym_raw.keys())}")

        w_sym_list = []

        w_0 = self.opti.variable()
        term = 1
        y_sym += w_0 * term
        w_sym_list.append(w_0)

        for i, inp in enumerate(self.inputs):
            w_i = self.opti.variable()
            term = x_sym[i]
            y_sym += w_i * term
            w_sym_list.append(w_i)

        w_vec = ca.vertcat(*w_sym_list)
        self.w_vec = w_vec

        # Gradients w.r.t. composite features
        grad_sym = ca.gradient(y_sym, x_sym)     
        hess_sym, _ = ca.hessian(y_sym, x_sym)   

        # Gradients w.r.t. raw features (for raw feature constraints)
        grad_raw_sym = {}
        for raw_name in sym_raw:
            if raw_name in monotonies_raw or raw_name in convexities_raw:
                grad_raw_sym[raw_name] = ca.gradient(y_sym, sym_raw[raw_name])

        f_pred = ca.Function('f_pred', [x_sym, w_vec], [y_sym])
        f_grad = ca.Function('f_grad', [x_sym, w_vec], [grad_sym])
        f_hess = ca.Function('f_hess', [x_sym, w_vec], [ca.diag(hess_sym)])
        
        # Functions for raw feature gradients
        f_grad_raw = {}
        for raw_name, grad_raw in grad_raw_sym.items():
            f_grad_raw[raw_name] = ca.Function(f'f_grad_{raw_name}', [x_sym, w_vec], [grad_raw])

        N, _ = td.y_train_single.shape
        y = td.y_train_single

        y_pred_all = f_pred.map(N)(X.T, ca.repmat(w_vec, 1, N))
        error = y_pred_all - y.reshape(1, N)
        obj = ca.mtimes(error, error.T)
        self.opti.minimize(obj)

        # TODO: Check if constraints should be applied only to training data or generally
        
        # Constraints on composite features (via gradients w.r.t. x_sym)
        grad_all = f_grad.map(N)(X.T, ca.repmat(w_vec, 1, N))
        for feat_idx, mono in monotonies_composite_idx.items():
            if mono > 0:
                self.opti.subject_to(grad_all[feat_idx, :].T >= 0)
            elif mono < 0:
                self.opti.subject_to(grad_all[feat_idx, :].T <= 0)
        
        hess_all = f_hess.map(N)(X.T, ca.repmat(w_vec, 1, N))
        for feat_idx, convex in convexities_composite_idx.items():
            if convex > 0:
                self.opti.subject_to(hess_all[feat_idx, :].T >= 0)
            elif convex < 0:
                self.opti.subject_to(hess_all[feat_idx, :].T <= 0)
        
        # Constraints on raw features (via gradients w.r.t. sym_raw)
        for raw_name, mono in monotonies_raw.items():
            grad_raw_all = f_grad_raw[raw_name].map(N)(X.T, ca.repmat(w_vec, 1, N))
            if mono > 0:
                self.opti.subject_to(grad_raw_all.T >= 0)
            elif mono < 0:
                self.opti.subject_to(grad_raw_all.T <= 0)
        
        for raw_name, convex in convexities_raw.items():
            # For raw features, we need the Hessian w.r.t. that feature
            # But this is scalar anyway (1 variable), so we need the second derivative
            hess_raw = ca.gradient(grad_raw_sym[raw_name], sym_raw[raw_name])
            f_hess_raw = ca.Function(f'f_hess_{raw_name}', [x_sym, w_vec], [hess_raw])
            hess_raw_all = f_hess_raw.map(N)(X.T, ca.repmat(w_vec, 1, N))
            if convex > 0:
                self.opti.subject_to(hess_raw_all.T >= 0)
            elif convex < 0:
                self.opti.subject_to(hess_raw_all.T <= 0)

        return model 

    def compile_model(self, model):
        if Logger.check_print_level('WARNING'):
            print_level = 0
        else:
            print_level = 5

        opts = {'ipopt.print_level': print_level, 'expand': True}
        self.opti.solver('ipopt', opts)

    def fit_model(self, model, td: TrainingDataGeneric):
        start_time = time.perf_counter()
        sol = self.opti.solve()
        stop_time = time.perf_counter()
        td.add_training_time(stop_time - start_time)

        weights_val = sol.value(self.w_vec)

        if Logger.check_print_level('INFO'):
            print("Optimized weights:")
            print("Bias:", weights_val[0])
            for i, inp in enumerate(self.inputs):
                print(f"Weight for {inp.name}:", weights_val[i+1])

        for l in model.layers:
            if l.name == 'ConstrainedRegression':
                l.set_weights([weights_val[1:].reshape(-1, 1), np.array([weights_val[0]])])
                l.trainable = False

    def plot(self, td: TrainingDataGeneric):
        """
        Generates and displays various plots related to model performance.

        Args:
            td (TrainingDataGeneric): The TrainingData object
        """

        fig1 = plot_prediction_correlation(td)
        fig2 = plot_predictions(td)
        fig3 = plot_metrics_table(td)

        if isinstance(td, TrainingData):
            subplots(
                "Linear Regression",
                {"title": "Prediction Correlation", "type": "scatter", "figure": fig1},
                {"title": "Predictions Sorted", "type": "scatter", "figure": fig2},
                {"title": "Performance Metrics", "type": "table", "figure": fig3}
            )
        elif isinstance(td, TrainingDataMultiStep):
            fig4 = plot_multi_rmse(td)
            subplots(
                "Linear Regression",
                # {"title": "Prediction Correlation", "type": "scatter", "figure": fig1},
                {"title": "Predictions Sorted", "type": "scatter", "figure": fig2},
                {"title": "Prediction Step RMSE", "type": "scatter", "figure": fig4},
                {"title": "Performance Metrics", "type": "table", "figure": fig3}
            )
        else:
            raise NotImplementedError

    def save_model(self, model, save_path: str):
        """
        Saves the Keras model to the specified path.

        Args:
            model (keras.Model): The Keras model to save.
            save_path (str): The directory or full path where the model should be saved.
        """

        if save_path is None:
            save_path = Logger.get_model_savepath()

        if not save_path.endswith('.keras'):
            save_path += '.keras'

        save_path = create_full_path(save_path)
        model.save(save_path)

    def load_model(self, load_path: str):
        """
        Loads a Keras model from the specified path.

        Args:
            load_path (str): The path from which to load the model.

        Returns:
            keras.Model: The loaded Keras model.
        """

        load_path = get_full_path(load_path)
        model = keras.saving.load_model(load_path)
        return model
    
    def get_config(self) -> dict:
        c = super().get_config()
        c.update({
            'inputs': [inp.name for inp in self.inputs],
            'monotonies': self.monotonies,
            'convexities': self.convexities
        })
        return c