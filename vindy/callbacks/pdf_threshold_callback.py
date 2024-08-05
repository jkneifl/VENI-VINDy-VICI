import tensorflow as tf
from vindy.layers.vindy_layer import VindyLayer


class PDFThresholdCallback(tf.keras.callbacks.Callback):

    def __init__(self, freq=1, threshold=1, on_train_end=False, **kwargs):
        """
        Callback for the VINDy layer. This callback is used to set all coefficients of the VINDy layer to zero if their
        corresponding probability density function at zero is above the threshold.

        Parameters
        ----------
        freq : int
            Frequency of the cancelation of the coefficients.
        threshold : int
            Threshold for the cancelation of the coefficients (get canceled if pdf(0) > threshold).
        on_train_end : bool
            Whether to cancel coefficients at the end of training.
        kwargs : dict
            Additional keyword arguments.
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
        the threshold, i.e. if pdf(0) > self.threshold

        Returns
        -------
        None
        """

        sindy_layer = self.model.sindy_layer
        if isinstance(sindy_layer, VindyLayer):
            # get current
            sindy_layer.pdf_thresholding(threshold=self.threshold)
        else:
            tf.print(
                "Canceling coefficients is only implemented for variational SINDy"
            )
