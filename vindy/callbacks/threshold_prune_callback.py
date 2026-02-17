import tensorflow as tf
from vindy.layers import SindyLayer

class ThresholdPruneCallback(tf.keras.callbacks.Callback):

    def __init__(self, freq=1, threshold=0.01, on_train_end=False, start_epoch=0):
        """
        Callback for thresholding SINDy coefficients during training.

        This callback sets all coefficients of the SINDy layer to zero if their
        value is below a certain threshold.

        Parameters
        ----------
        freq : int, default=1
            Frequency of coefficient cancellation (every freq-th epochs).
        threshold : float, default=0.01
            Threshold for cancellation (coefficients with |value| < threshold are zeroed).
        on_train_end : bool, default=False
            Perform thresholding at end of training.
        start_epoch : int, default=0
            First epoch for which the thresholding is applied.
        """
        self.freq = freq
        self.threshold = threshold
        self.on_train_end_ = on_train_end
        self.start_epoch = start_epoch
        # super init
        super(ThresholdPruneCallback, self).__init__()


    def on_epoch_end(self, epoch, logs=None):
        # only cancel coefficients every freq epochs
        if (epoch - self.start_epoch + 1) % self.freq == 0 and epoch >= self.start_epoch:
            self.prune_weights()

    def on_train_end(self, logs=None):
        if self.on_train_end_:
            self.prune_weights()

    def prune_weights(self):
        sindy_layer = self.model.sindy_layer
        if isinstance(sindy_layer, SindyLayer):
            tf.print(
                f"Thresholding coefficients below {self.threshold}"
            )
            sindy_layer.prune_weights(self.threshold)
        else:
            tf.print(
                "Thresholding coefficients is only implemented for SINDy layer use pdf thresholding for VINDy"
            )
