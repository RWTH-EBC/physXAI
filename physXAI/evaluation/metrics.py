import math
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from physXAI.preprocessing.training_data import TrainingData, TrainingDataMultiStep, TrainingDataGeneric


class Metrics:
    """
    A class to calculate and store regression metrics (MSE, RMSE, R2)
    for training, validation, and test datasets.
    """

    print_evaluate = True

    def __init__(self, td: TrainingDataGeneric):
        """
                Initializes the Metrics object by calculating metrics for train, validation (if available),
                and test sets.

                Args:
                    td (TrainingDataGeneric): An object containing the training data.
        """

        self.train_kpis = self.evaluate(td.y_train_single, td.y_train_pred, label='Train')
        if td.y_val_single is not None:
            self.val_kpis = self.evaluate(td.y_val_single, td.y_val_pred, label='Val')
        else:
            self.val_kpis = None
        self.test_kpis = self.evaluate(td.y_test_single, td.y_test_pred, label='Test')

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str = '') -> dict[str, float]:
        """
        Calculates Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2) score.

        Args:
            y_true (np.ndarray): The true target values.
            y_pred (np.ndarray): The predicted target values.
            label (str, optional): A label for the dataset being evaluated (e.g., 'Train', 'Test').
                                   Defaults to ''.

        Returns:
            dict: A dict containing MSE, RMSE, and R2 scores.
        """

        mse = mean_squared_error(y_true, y_pred)
        rmse = math.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        kpis = dict()
        kpis['MSE' + ' ' + label] = mse
        kpis['RMSE' + ' ' + label] = rmse
        kpis['R2' + ' ' + label] = r2

        if Metrics.print_evaluate:
            # print(f"{label} MSE: {mse:.2f}")
            print(f"{label} RMSE: {rmse:.2f}")
            print(f"{label} R2: {r2:.2f}")

        return kpis

    def get_metrics(self, nround: int = 2) -> (list[str], list[float]):
        """
          Returns a list of metric labels and their corresponding rounded values.

          Args:
              nround (int, optional): The number of decimal places to round the metric values to.
                                      Defaults to 2.

          Returns:
              tuple[list[str], list[float]]: A tuple containing a list of metric labels and a list of
                                             their corresponding rounded values.
        """

        kpis = self.train_kpis
        if self.val_kpis is not None:
            kpis = kpis | self.val_kpis
        kpis = kpis | self.test_kpis
        return list(kpis.keys()), [round(v, nround) for v in kpis.values()]

    def get_config(self) -> dict:
        return {
            'train_kpis': self.train_kpis,
            'val_kpis': self.val_kpis,
            'test_kpis': self.test_kpis,
        }


class MetricsPINN(Metrics):
    """
    A class to calculate and store metrics for Physics-Informed Neural Networks (PINNs).
    It evaluates performance using a list of provided loss functions.
    """

    def __init__(self, td: TrainingDataGeneric, pinn_losses: list):
        """
        Initializes the MetricsPINN object by calculating metrics for train,
        validation (if available), and test sets using specified PINN loss functions.

        Args:
            td (TrainingData): An object containing the true and predicted values.
            pinn_losses (list): A list of loss functions to be used for evaluation.
        """

        self.train_kpis = self.evaluate(td.y_train_single, td.y_train_pred_single, label='Train',
                                        pinn_losses=pinn_losses)
        if td.y_val is not None:
            self.val_kpis = self.evaluate(td.y_val_single, td.y_val_pred_single, label='Val', pinn_losses=pinn_losses)
        else:
            self.val_kpis = None
        self.test_kpis = self.evaluate(td.y_test_single, td.y_test_pred_single, label='Test', pinn_losses=pinn_losses)

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str = '', **kwargs) -> dict[str, float]:
        """
        Calculates metrics based on the provided list of PINN loss functions.

        Args:
            y_true (np.ndarray): The true target values.
            y_pred (np.ndarray): The predicted target values.
            label (str, optional): A label for the dataset being evaluated (e.g., 'Train', 'Test').
                                   Defaults to ''.

        Returns:
            dict: A dictionary where keys are loss function names appended with the label,
                  and values are the calculated loss values.
        """

        kpis = dict()
        for loss in kwargs['pinn_losses']:
            val = float(loss(y_true, y_pred))
            kpis[loss.__name__ + ' ' + label] = val
            print(f"{loss.__name__ + ' ' + label}: {val:.2f}")

        return kpis


class MetricsMultiStep(Metrics):
    """
    A class to calculate and store regression metrics (MSE, RMSE, R2) for multi-step
    time series forecasting models. It evaluates overall performance and performance at each step.
    """

    def __init__(self, td: TrainingDataMultiStep):
        """
        Initializes the MetricsMultiStep object. Calculates overall metrics for train,
        validation (if available), and test sets, as well as step-wise RMSE for each set.

        Args:
            td (TrainingDataMultiStep): An object containing the true and predicted values
                                        for multi-step forecasts.
        """

        self.train_kpis = self.evaluate(td.y_train.reshape(-1, 1).astype(float), td.y_train_pred.reshape(-1, 1).astype(float), label='Train')
        if td.y_val is not None:
            self.val_kpis = self.evaluate(td.y_val.reshape(-1, 1).astype(float), td.y_val_pred.reshape(-1, 1).astype(float), label='Val')
        else:
            self.val_kpis = None
        self.test_kpis = self.evaluate(td.y_test.reshape(-1, 1).astype(float), td.y_test_pred.reshape(-1, 1).astype(float), label='Test')

        # Stepwise RMSE
        rmse_train_l = list[float]()
        rmse_val_l = list[float]()
        rmse_test_l = list[float]()
        for i in range(td.y_train.shape[1]):
            _, rmse_train, _ = self.evaluate_step(td.y_train, td.y_train_pred, i)
            _, rmse_test, _ = self.evaluate_step(td.y_test, td.y_test_pred, i)
            rmse_train_l.append(rmse_train)
            rmse_test_l.append(rmse_test)
            if td.y_val is not None:
                _, rmse_val, _ = self.evaluate_step(td.y_val, td.y_val_pred, i)
                rmse_val_l.append(rmse_val)
        self.rmse_train_l = rmse_train_l
        if td.y_val is not None:
            self.rmse_val_l = rmse_val_l
        else:
            self.rmse_val_l = None
        self.rmse_test_l = rmse_test_l

    @staticmethod
    def evaluate_step(y_true: np.ndarray, y_pred: np.ndarray, step: int) -> (float, float, float):
        """
        Calculates Mean Squared Error (MSE), Root Mean Squared Error (RMSE),
        and R-squared (R2) score for a specific step in multi-step predictions.

        Args:
            y_true (np.ndarray): The true target values (samples, steps, features).
            y_pred (np.ndarray): The predicted target values (samples, steps, features).
            step (int): The specific forecast step to evaluate (0-indexed).

        Returns:
            tuple[float, float, float]: A tuple containing MSE, RMSE, and R2 score for the specified step.
        """

        mse = mean_squared_error(y_true[:, step, :].reshape(-1, 1), y_pred[:, step, :].reshape(-1, 1))
        rmse = math.sqrt(mse)
        r2 = r2_score(y_true[:, step, :].reshape(-1, 1), y_pred[:, step, :].reshape(-1, 1))

        return mse, rmse, r2

    def get_config(self) -> dict:
        c = super().get_config()
        c.update({
            'rmse_train_l': self.rmse_train_l,
            'rmse_val_l': self.rmse_val_l,
            'rmse_test_l': self.rmse_test_l,
        })
        return c
