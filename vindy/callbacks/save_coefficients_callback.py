import tensorflow as tf


class SaveCoefficientsCallback(tf.keras.callbacks.Callback):

    def __init__(self, freq=1, **kwargs):
        """
        Callback for logging SINDy coefficients during training.

        Parameters
        ----------
        freq : int, default=1
            Frequency of saving the coefficients (every freq-th epoch).
        **kwargs
            Additional keyword arguments passed to ``tf.keras.callbacks.Callback``.
        """
        self.freq = freq
        super().__init__(**kwargs)

    def on_epoch_end(self, epoch, logs=None):
        # only save coefficients every freq epochs
        if (epoch + 1) % self.freq == 0:
            # add the current epoch to the logs
            logs = logs or {}
            # get the current weights of the sindy layer
            sindy_layer = self.model.sindy_layer
            coeffs = sindy_layer._coeffs
            # save coeffs to training history
            if isinstance(coeffs, list) or isinstance(coeffs, tuple):
                logs.update({"coeffs_mean": coeffs[1].numpy()})
                logs.update({"coeffs_scale": coeffs[2].numpy()})
            else:
                logs.update({"coeffs": coeffs})
