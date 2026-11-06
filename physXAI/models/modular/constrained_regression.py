import logging
import os
import time

import casadi as ca
import numpy as np

from physXAI.expr import Expr, as_expr
from physXAI.expr import from_config as expr_from_config
from physXAI.expr import to_config as expr_to_config
from physXAI.expr.core import BuildContext
from physXAI.io.run import input_path, output_path
from physXAI.models.models import SingleStepModel, register_model
from physXAI.preprocessing.training_data import TrainingDataGeneric

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

logger = logging.getLogger("physXAI")

#: Name of the dense layer holding the weights. It has to contain 'dense' for the CasADi export.
LAYER_NAME = 'dense_constrained_regression'


@register_model
class ConstrainedRegression(SingleStepModel):
    """
    A constrained (monotone, convex, concave) linear regression on expressions, solved with CasADi / IPOPT.

    Args:
        inputs: Data only expressions (`physXAI.expr`) or column names, combined linearly.
        monotonies: {name: 1 | -1}. The name is either the name of an input expression or of a raw
                    data column, in which case the derivative with respect to the raw column is constrained.
        convexities: {name: 1 (convex) | -1 (concave)}, names as for `monotonies`.
    """

    plot_title = "Constrained Regression"

    def __init__(self, inputs: list[Expr | str] | dict, monotonies: dict[str, int] = None,
                 convexities: dict[str, int] = None, **kwargs):
        super().__init__()
        if isinstance(inputs, dict):
            inputs = expr_from_config(inputs)
        self.inputs = [as_expr(inp) for inp in inputs]
        trainable = [inp.name for inp in self.inputs if inp.trainable]
        if trainable:
            raise ValueError(f"ConstrainedRegression requires data only inputs, but {trainable} are trainable.")
        self.monotonies = dict(monotonies or {})
        self.convexities = dict(convexities or {})
        self.opti = None
        self.w_vec = None

    @property
    def required_inputs(self) -> list[str]:
        return list(dict.fromkeys(v for inp in self.inputs for v in inp.free_vars()))

    def generate_model(self, **kwargs):
        """
        Generates the keras model and the CasADi optimization problem of the regression.
        """
        td = kwargs['td']
        columns = list(td.inputs)
        X_data = td.X_train_single
        train_values = {name: X_data[:, i] for i, name in enumerate(columns)}
        ctx = BuildContext(keras.layers.Input(shape=(len(columns),)), columns, train_values)
        tensors = [ctx.tensor(inp) for inp in self.inputs]
        x = tensors[0] if len(tensors) == 1 else keras.layers.Concatenate()(tensors)
        output = keras.layers.Dense(units=1, activation='linear', name=LAYER_NAME)(x)
        model = keras.models.Model(inputs=ctx.input, outputs=output)

        # A new problem for every fit, so constraints are not added twice
        self.opti = ca.Opti()
        n_samples = X_data.shape[0]
        X = np.column_stack([np.broadcast_to(inp.evaluate(train_values), (n_samples,)) for inp in self.inputs])
        raw_names = self.required_inputs
        x_sym_raw = {name: ca.MX.sym(name) for name in raw_names}
        X_raw = np.column_stack([train_values[name] for name in raw_names])
        x_sym = ca.vertcat(*[inp.to_casadi(x_sym_raw) for inp in self.inputs])

        # Map constraint names to composite or raw feature indices
        monotonies = {}  # Maps composite feature index to monotonicity value
        monotonies_raw = {}  # Maps raw feature name to monotonicity value

        for constraint_name, mono_value in self.monotonies.items():
            found = False

            # Check if it's a raw feature
            for i, inp in enumerate(x_sym_raw.keys()):
                if constraint_name == inp:
                    if mono_value != 0:
                        monotonies_raw[i] = mono_value
                    found = True
                    break

            # Check if it's a composite feature (by input name)
            if not found:
                for i, inp in enumerate(self.inputs):
                    if constraint_name == inp.name:
                        if mono_value != 0:
                            monotonies[i] = mono_value
                        found = True
                        break

            if not found:
                raise ValueError(f"Monotonicity specified for unknown input '{constraint_name}'. "
                                f"Available raw features: {list(x_sym_raw.keys())}"
                                f"Available composite features: {[inp.name for inp in self.inputs]}. "
                                )

        # Same for convexities
        convexities = {}
        convexities_raw = {}

        for constraint_name, convex_value in self.convexities.items():
            found = False

            # Check if it's a raw feature
            for i, inp in enumerate(x_sym_raw.keys()):
                if constraint_name == inp:
                    if convex_value != 0:
                        convexities_raw[i] = convex_value
                    found = True
                    break

            # Check if it's a composite feature (by input name)
            if not found:
                for i, inp in enumerate(self.inputs):
                    if constraint_name == inp.name:
                        if convex_value != 0:
                            convexities[i] = convex_value
                        found = True
                        break

            if not found:
                raise ValueError(f"Convexity specified for unknown input '{constraint_name}'. "
                                f"Available raw features: {list(x_sym_raw.keys())}"
                                f"Available composite features: {[inp.name for inp in self.inputs]}. "
                               )


        x_sym_raw = ca.vertcat(*x_sym_raw.values())
        x_sym_regression = ca.MX.sym('regression_input', len(self.inputs))
        y_sym = 0
        w_sym_list = []

        w_0 = self.opti.variable()
        w_sym_list.append(w_0)
        y_sym += w_0

        for i in range(len(self.inputs)):
            w_i = self.opti.variable()
            w_sym_list.append(w_i)
            y_sym += w_i * x_sym_regression[i]

        w_vec = ca.vertcat(*w_sym_list)
        self.w_vec = w_vec

        f_pred = ca.Function('f_pred', [x_sym_regression, w_vec], [y_sym])


        # Symbolic gradients/hessians w.r.t. composite features
        if monotonies:
            grad_sym = ca.gradient(y_sym, x_sym_regression)
        if convexities:
            hess_sym, _ = ca.hessian(y_sym, x_sym_regression)

        # Symbolic gradients/hessians w.r.t. raw features
        f_pred_raw = f_pred(x_sym, w_vec)
        if monotonies_raw:
            grad_sym_raw = ca.gradient(f_pred_raw, x_sym_raw)
        if convexities_raw:
            hess_sym_raw, _ = ca.hessian(f_pred_raw, x_sym_raw)


        N, _ = td.y_train_single.shape
        y = td.y_train_single

        y_pred_all = f_pred.map(N)(X.T, ca.repmat(w_vec, 1, N))
        error = y_pred_all - y.reshape(1, N)
        obj = ca.mtimes(error, error.T)
        self.opti.minimize(obj)


        # TODO: Check if constraints should be applied only to training data or generally
        # Constraints on composite features (via gradients w.r.t. x_sym_regression)
        if monotonies:
            for feat_idx, mono in monotonies.items():
                grad_i = grad_sym[feat_idx]
                if ca.depends_on(grad_i, x_sym_regression):
                    # Gradient depends on inputs, need per-point constraints
                    f_grad_i = ca.Function('f_grad_i', [x_sym_regression, w_vec], [grad_i])
                    grad_vals = f_grad_i.map(N)(X.T, ca.repmat(w_vec, 1, N)).T
                    if mono > 0:
                        self.opti.subject_to(grad_vals >= 0)
                    elif mono < 0:
                        self.opti.subject_to(grad_vals <= 0)
                else:
                    # Gradient is constant w.r.t. inputs, single constraint suffices
                    # Wrap in Function to eliminate raw MX.sym references
                    f_grad_i = ca.Function('f_grad_i', [x_sym_regression, w_vec], [grad_i])
                    grad_val = f_grad_i(ca.DM.zeros(x_sym_regression.shape), w_vec)
                    if mono > 0:
                        self.opti.subject_to(grad_val >= 0)
                    elif mono < 0:
                        self.opti.subject_to(grad_val <= 0)

        if convexities:
            for feat_idx, convex in convexities.items():
                hess_i = ca.diag(hess_sym)[feat_idx]
                if hess_i.is_constant():
                    # Hessian is constant (e.g. zero for linear model), skip
                    continue
                if ca.depends_on(hess_i, x_sym_regression):
                    f_hess_i = ca.Function('f_hess_i', [x_sym_regression, w_vec], [hess_i])
                    hess_vals = f_hess_i.map(N)(X.T, ca.repmat(w_vec, 1, N)).T
                    if convex > 0:
                        self.opti.subject_to(hess_vals >= 0)
                    elif convex < 0:
                        self.opti.subject_to(hess_vals <= 0)
                else:
                    # Wrap in Function to eliminate raw MX.sym references
                    f_hess_i = ca.Function('f_hess_i', [x_sym_regression, w_vec], [hess_i])
                    hess_val = f_hess_i(ca.DM.zeros(x_sym_regression.shape), w_vec)
                    if convex > 0:
                        self.opti.subject_to(hess_val >= 0)
                    elif convex < 0:
                        self.opti.subject_to(hess_val <= 0)

        # Constraints on raw features (via gradients w.r.t. x_sym_raw)
        if monotonies_raw:
            for feat_idx, mono in monotonies_raw.items():
                grad_raw_i = grad_sym_raw[feat_idx]
                if ca.depends_on(grad_raw_i, x_sym_raw):
                    f_grad_raw_i = ca.Function('f_grad_raw_i', [x_sym_raw, w_vec], [grad_raw_i])
                    grad_vals = f_grad_raw_i.map(N)(X_raw.T, ca.repmat(w_vec, 1, N)).T
                    if mono > 0:
                        self.opti.subject_to(grad_vals >= 0)
                    elif mono < 0:
                        self.opti.subject_to(grad_vals <= 0)
                else:
                    # Wrap in Function to eliminate raw MX.sym references
                    f_grad_raw_i = ca.Function('f_grad_raw_i', [x_sym_raw, w_vec], [grad_raw_i])
                    grad_val = f_grad_raw_i(ca.DM.zeros(x_sym_raw.shape), w_vec)
                    if mono > 0:
                        self.opti.subject_to(grad_val >= 0)
                    elif mono < 0:
                        self.opti.subject_to(grad_val <= 0)

        if convexities_raw:
            for feat_idx, convex in convexities_raw.items():
                hess_raw_i = ca.diag(hess_sym_raw)[feat_idx]
                if hess_raw_i.is_constant():
                    continue
                if ca.depends_on(hess_raw_i, x_sym_raw):
                    f_hess_raw_i = ca.Function('f_hess_raw_i', [x_sym_raw, w_vec], [hess_raw_i])
                    hess_vals = f_hess_raw_i.map(N)(X_raw.T, ca.repmat(w_vec, 1, N)).T
                    if convex > 0:
                        self.opti.subject_to(hess_vals >= 0)
                    elif convex < 0:
                        self.opti.subject_to(hess_vals <= 0)
                else:
                    # Wrap in Function to eliminate raw MX.sym references
                    f_hess_raw_i = ca.Function('f_hess_raw_i', [x_sym_raw, w_vec], [hess_raw_i])
                    hess_val = f_hess_raw_i(ca.DM.zeros(x_sym_raw.shape), w_vec)
                    if convex > 0:
                        self.opti.subject_to(hess_val >= 0)
                    elif convex < 0:
                        self.opti.subject_to(hess_val <= 0)

        return model

    def compile_model(self, model):
        print_level = 5 if logger.isEnabledFor(logging.INFO) else 0

        opts = {'ipopt.print_level': print_level, 'expand': True}
        self.opti.solver('ipopt', opts)

    def fit_model(self, model, td: TrainingDataGeneric):
        start_time = time.perf_counter()
        sol = self.opti.solve()
        stop_time = time.perf_counter()
        td.add_training_time(stop_time - start_time)

        weights_val = sol.value(self.w_vec)

        if logger.isEnabledFor(logging.INFO):
            weights = "\n".join(f"  {inp.name}: {weights_val[i + 1]:.6g}" for i, inp in enumerate(self.inputs))
            logger.info("Optimized bias: %.6g, weights:\n%s", weights_val[0], weights)

        for layer in model.layers:
            if layer.name == LAYER_NAME:
                layer.set_weights([weights_val[1:].reshape(-1, 1), np.array([weights_val[0]])])
                layer.trainable = False

    def save_model(self, model, save_path: str):
        """
        Saves the Keras model to the specified path.

        Args:
            model (keras.Model): The Keras model to save.
            save_path (str): The directory or full path where the model should be saved.
        """
        save_path = output_path(save_path, '.keras')
        model.save(save_path)

    def load_model(self, load_path: str):
        """
        Loads a Keras model from the specified path.

        Args:
            load_path (str): The path from which to load the model.

        Returns:
            keras.Model: The loaded Keras model.
        """

        load_path = input_path(load_path)
        model = keras.saving.load_model(load_path)
        return model

    def get_config(self) -> dict:
        c = super().get_config()
        c.update({
            'inputs': expr_to_config(self.inputs),
            'monotonies': self.monotonies,
            'convexities': self.convexities
        })
        return c
