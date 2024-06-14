import tensorflow as tf
from vindy.layers import SindyLayer

class ThresholdPruneCallback(tf.keras.callbacks.Callback):

    def __init__(self, threshold=0.01, start_epoch=0, epochs=1):
        """
        Callback for the SINDy layer. This callback is used to set all coefficients of the VINDy layer to zeroif their
        value is below a certain threshold
        :param threshold:
        :param start_epoch:
        :param epochs:
        """
        self.threshold_ = threshold
        self.start_epoch = start_epoch
        self.prune_weights = False
        self.epochs = epochs
        # super init
        super(ThresholdPruneCallback, self).__init__()

    def on_epoch_begin(self, epoch, logs=None):
        if epoch >= self.start_epoch:
            self.threshold = self.threshold_ * tf.reduce_min([(epoch - self.start_epoch) / self.epochs, 1])
            self.prune_weights = True

    def on_batch_begin(self, batch, logs=None):
        if self.prune_weights:

            for layer in self.model.sindy.layers:
                if isinstance(layer, SindyLayer):
                    layer.prune_weights(self.threshold)