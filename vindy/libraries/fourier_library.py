import tensorflow as tf
from .base_library import BaseLibrary


class FourierLibrary(BaseLibrary):

    def __init__(self, freqs=[1]):
        self.freqs = freqs
        self.fcn = [tf.sin, tf.cos, tf.math.sigmoid]
        self.fcn_names = ["sin", "cos", "sigmoid"]

    @tf.function
    def __call__(self, x):
        """
        transform input x to trigonometric features of order self.poly_order
        :param x: array-like of shape (n_samples, 2*reduce_order), latent variable and its time derivative
        :return: polynomial features
        """
        x_fourier = []
        for f in self.freqs:
            for fcn in self.fcn:
                x_fourier += [fcn(f * x)]
        x_fourier = tf.concat(x_fourier, axis=1)
        return x_fourier

    def get_names(self, x):
        """
        construct features for the input x
        :param x: input
        :return: feature
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

