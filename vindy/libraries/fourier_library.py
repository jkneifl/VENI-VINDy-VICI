import tensorflow as tf
from .base_library import BaseLibrary


class FourierLibrary(BaseLibrary):
    """
    Library for Fourier features.
    """

    def __init__(self, freqs=[1]):
        """
        Initialize the Fourier library.

        Parameters
        ----------
        freqs : list of int, optional
            List of frequencies (default: [1])
        """
        self.freqs = freqs
        self.fcn = [tf.sin, tf.cos, tf.math.sigmoid]
        self.fcn_names = ["sin", "cos", "sigmoid"]

    @tf.function
    def __call__(self, x):
        """
        Transform input x to trigonometric features of

        Parameters
        ----------
        x : array-like of shape (n_samples, 2*reduce_order)
            Latent variable and its time derivative.

        Returns
        -------
        array-like
            Trigonometric features.
        """
        x_fourier = []
        for f in self.freqs:
            for fcn in self.fcn:
                x_fourier += [fcn(f * x)]
        x_fourier = tf.concat(x_fourier, axis=1)
        return x_fourier

    def get_names(self, x):
        """
        Construct the names of the features for the input x.

        Parameters
        ----------
        x : array-like
            Input data.

        Returns
        -------
        list of str
            Names of the trigonometric features.
        """
        # ensure that x is a list
        if not isinstance(x, list):
            x = [x]
        x_fourier = []
        for f in self.freqs:
            for fcn in self.fcn_names:
                for x_ in x:
                    x_fourier += [f'{fcn}({f} * {x_})']
        return x_fourier

