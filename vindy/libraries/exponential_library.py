import tensorflow as tf
from .base_library import BaseLibrary


class ExponentialLibrary(BaseLibrary):
    """
    Library for exponential features.
    """

    def __init__(self, coeff=[1]):
        self.coeff = coeff

    @tf.function
    def __call__(self, x):
        """
        Construct exponential features for the input x.

        Parameters
        ----------
        x : any
            Input data (n_samples, 2*reduce_order).

        Returns
        -------
        any
            Exponential features.
        """
        x_exp = []
        for c in self.coeff:
            x_exp += [tf.exp(c * x)]
        x_exp = tf.concat(x_exp, axis=1)
        return x_exp

    def get_names(self, x):
        """
        Construct the names of the exponential features for the input x.

        Parameters
        ----------
        x : any
            Input data.

        Returns
        -------
        list of str
            Names of the exponential features.
        """
        # ensure that x is a list
        if not isinstance(x, list):
            x = [x]
        x_exp = []
        for x_ in x:
            for c in self.coeff:
                x_exp += [f'exp({c} * {x_})']
        return x_exp
