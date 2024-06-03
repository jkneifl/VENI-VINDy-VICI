import tensorflow as tf
from .base_library import BaseLibrary


class ForceLibrary(BaseLibrary):

    def __init__(self, functions=[tf.sin, tf.cos]):
        self.functions = functions

    @tf.function
    def __call__(self, x):
        """
        transform input x to force features following force = amplitude * sin(omega * t)
        :param x: array-like of shape (n_samples, 2*reduce_order), latent variable and its time derivative
        :return: polynomial features
        """
        x_force = []
        for func in self.functions:
            x_force += [x[:, 2:] * func(x[:, 1:2] * x[:, 0:1])]

        x_force = tf.concat(x_force, axis=1)
        return x_force

    def get_names(self, x):
        """
        construct features for the input x
        :param x: input
        :return: feature
        """
        x_force = []
        for func in self.functions:
            x_force += [f'{x[2]}*{func.__name__}({x[1]}*{x[0]})']
        return x_force
