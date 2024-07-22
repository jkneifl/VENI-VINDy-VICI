import tensorflow as tf
from vindy.layers.vindy_layer import VindyLayer


class PDFThresholdCallback(tf.keras.callbacks.Callback):

    def __init__(self, freq=1, threshold=1, on_train_end=False, **kwargs):
        """
        Callback for the VINDy layer. This callback is used to set all coefficients of the VINDy layer to zero if their
        corresponding probability density function at zero is above the threshold
        :param freq: frequency of the cancelation of the coefficients
        :param threshold: threshold for the cancelation of the coefficients (get canceled if pdf(0) > threshold)
        :param kwargs:
        """
        self.freq = freq
        self.threshold = threshold
        self.on_train_end_ = on_train_end
        super().__init__(**kwargs)

    def on_epoch_end(self, epoch, logs=None):
        # only save coefficients every freq epochs
        if (epoch + 1) % self.freq == 0:
            self.cancel_coefficients()

    def on_train_end(self, logs=None):
        if self.on_train_end_:
            self.cancel_coefficients()

    def cancel_coefficients(self):
        """
        Cancel the coefficients of the SINDy layer if their corresponding probability density function at zero is above
        the threshhold, i.e. if pdf(0) > self.threshhold
        :return:
        """

        sindy_layer = self.model.sindy_layer
        if isinstance(sindy_layer, VindyLayer):
            # get current
            sindy_layer.pdf_thresholding(threshold=self.threshold)
        else:
            tf.print(
                "Canceling coefficients is only implemented for variational SINDy"
            )
