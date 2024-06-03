import tensorflow as tf
from .base_library import BaseLibrary


class ExponentialLibrary(BaseLibrary):

    def __init__(self, coeff=[1]):
        self.coeff = coeff

    @tf.function
    def __call__(self, x):
        """
        transform input x to exponential features
        :param x: array-like of shape (n_samples, 2*reduce_order), latent variable and its time derivative
        :return: polynomial features
        """
        x_exp = []
        for c in self.coeff:
            x_exp += [tf.exp(c * x)]
        x_exp = tf.concat(x_exp, axis=1)
        return x_exp

    def get_names(self, x):
        """
        construct features for the input x
        :param x: input
        :return: feature
        """
        # ensure that x is a list
        if not isinstance(x, list):
            x = [x]
        x_exp = []
        for x_ in x:
            for c in self.coeff:
                x_exp += [f'exp({c} * {x_})']
        return x_exp
