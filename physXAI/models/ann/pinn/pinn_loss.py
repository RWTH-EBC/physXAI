def multi_y_loss(loss, additional_losses: list[float], name: str = None):
    """
    Creates a custom loss function that computes a weighted average of a base loss
    applied to a primary target and several additional targets.

    This is useful for PINNs. The `y` tensor is expected to have its first column (`y[:, 0]`) as the primary target,
    and subsequent columns (`y[:, 1]`, `y[:, 2]`, etc.) as additional targets.

    Args:
        loss (callable): The base loss function (e.g., keras.losses.MeanSquaredError())
                         that takes `y_true` and `y_pred` and returns a scalar loss.
        additional_losses (list of float): A list of weights.
                                                    Each weight `additional_losses[i]` corresponds
                                                    to the loss calculated between `y_pred` and
                                                    the (i+1)-th additional target `y[:, i+1]`.
        name (str, optional): An optional name for the returned loss function.
                              Defaults to None, in which case the inner function's
                              default name (`loss_function`) is used.

    Returns:
        callable: A custom loss function that takes `y` (true multi-component targets)
                  and `y_pred` (model predictions) and returns a single scalar loss value.
    """

    # Generate custom loss based on additional_losses
    def loss_function(y, y_pred):
        ls = loss(y_pred, y[:, 0])
        # Add additional losses
        for i in range(0, len(additional_losses)):
            ls += + additional_losses[i] * loss(y_pred, y[:, i + 1])
        # Rescale loss to match dimension of single loss
        ls = ls / (len(additional_losses) + 1)
        return ls

    # Add loss name if provided
    if name is not None:  # pragma: no cover
        loss_function.__name__ = name  # pragma: no cover
    return loss_function
