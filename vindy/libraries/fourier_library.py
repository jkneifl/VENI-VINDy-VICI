import tensorflow as tf
from .base_library import BaseLibrary


class FourierLibrary(BaseLibrary):

    def __init__(self, freqs=[1]):
        self.freqs = freqs

    @tf.function
    def __call__(self, x):
        """
        transform input x to trigonometric features of order self.poly_order
        :param x: array-like of shape (n_samples, 2*reduce_order), latent variable and its time derivative
        :return: polynomial features
        """
        x_fourier = []
        for f in self.freqs:
            x_fourier += [tf.sin(f * x), tf.cos(f * x), tf.math.sigmoid(f * x)]
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
        for x_ in x:
            for f in self.freqs:
                x_fourier += [f'sin({f} * {x_})', f'cos({f} * {x_})', f'sigmoid({f} * {x_})']
        return x_fourier

