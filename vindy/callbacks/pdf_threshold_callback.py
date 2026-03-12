from vindy.layers.vindy_layer import VindyLayer
from .base_callback import Callback


class PDFThresholdCallback(Callback):

    def __init__(self, freq=1, threshold=1, on_train_end=False, **kwargs):
        """
        Callback for the VINDy layer to cancel coefficients based on PDF thresholding.

        This callback sets all coefficients of the VINDy layer to zero if their
        corresponding probability density function at zero is above the threshold.

        Parameters
        ----------
        freq : int, default=1
            Frequency of coefficient cancellation (every freq-th epochs).
        threshold : float, default=1
            Threshold for cancellation (coefficients with pdf(0) > threshold are zeroed).
        on_train_end : bool, default=False
            Perform thresholding at end of training.
        **kwargs
            Additional keyword arguments.
        """
        self.freq = freq
        self.threshold = threshold
        self.on_train_end_ = on_train_end
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        # only save coefficients every freq epochs
        if (epoch + 1) % self.freq == 0:
            self.cancel_coefficients()

    def on_train_end(self, logs=None):
        if self.on_train_end_:
            self.cancel_coefficients()

    def cancel_coefficients(self):
        """
        Cancel coefficients based on their probability density at zero.

        Cancels the coefficients of the SINDy layer if their corresponding
        probability density function at zero is above the threshold, i.e.,
        if pdf(0) > self.threshold.
        """

        sindy_layer = self.model.sindy_layer
        if isinstance(sindy_layer, VindyLayer):
            # get current
            sindy_layer.pdf_thresholding(threshold=self.threshold)
        else:
            print(
                "Canceling coefficients is only implemented for variational SINDy"
            )
