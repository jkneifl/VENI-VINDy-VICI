from .base_callback import Callback


class SaveCoefficientsCallback(Callback):

    def __init__(self, freq=1, **kwargs):
        """
        Callback for logging SINDy coefficients during training.

        Parameters
        ----------
        freq : int, default=1
            Frequency of saving the coefficients (every freq-th epoch).
        **kwargs
            Additional keyword arguments.
        """
        self.freq = freq
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        # only save coefficients every freq epochs
        if (epoch + 1) % self.freq == 0:
            # add the current epoch to the logs
            logs = logs or {}
            # get the current weights of the sindy layer
            sindy_layer = self.model.sindy_layer
            coeffs = sindy_layer._coeffs
            # save coeffs to training history
            # .copy() is needed so each snapshot is independent; without it,
            # numpy() returns a view of the tensor storage and all history
            # entries end up pointing to the same (latest) values.
            if isinstance(coeffs, list) or isinstance(coeffs, tuple):
                logs.update({"coeffs_mean": coeffs[1].detach().cpu().numpy().copy()})
                logs.update({"coeffs_scale": coeffs[2].detach().cpu().numpy().copy()})
            else:
                logs.update({"coeffs": coeffs.detach().cpu().numpy().copy()})
