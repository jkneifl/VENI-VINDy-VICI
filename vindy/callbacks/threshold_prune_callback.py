import tensorflow as tf
from vindy.layers import SindyLayer

class ThresholdPruneCallback(tf.keras.callbacks.Callback):

    def __init__(self, freq=1, threshold=0.01, on_train_end=False, start_epoch=0):
        """
        Callback for the SINDy layer. This callback is used to set all coefficients of the SINDy layer to zero if their
        value is below a certain threshold
        :param freq: frequency of the cancellation of the coefficients (every freq-th epochs)
        :param threshold: for the cancellation of the coefficients (get canceled if value < threshold)
        :param on_train_end: perform thresholding at end of training
        :param start_epoch: first epoch for which the thresholding is applied
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
