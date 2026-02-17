import tensorflow as tf
from .base_library import BaseLibrary


class ExponentialLibrary(BaseLibrary):

    def __init__(self, coeff=[1]):
        self.coeff = coeff

    @tf.function
    def __call__(self, x):
        """
        Transform input x to exponential features.

        Parameters
        ----------
        x : array-like of shape (n_samples, 2*reduce_order)
            Latent variable and its time derivative.

        Returns
        -------
        x_exp : tf.Tensor
            Exponential features.
        """
        x_exp = []
        for c in self.coeff:
            x_exp += [tf.exp(c * x)]
        x_exp = tf.concat(x_exp, axis=1)
        return x_exp

    def get_names(self, x):
        """
        Construct features for the input x.

        Parameters
        ----------
        x : array-like
            Input data.

        Returns
        -------
        list of str
            Feature names in exponential form.
        """
        # ensure that x is a list
        if not isinstance(x, list):
            x = [x]
        x_exp = []
        for x_ in x:
            for c in self.coeff:
                x_exp += [f'exp({c} * {x_})']
        return x_exp
