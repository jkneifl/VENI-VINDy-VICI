import tensorflow as tf
from .base_library import BaseLibrary


class ForceLibrary(BaseLibrary):
    """
    Library for force features.
    """

    def __init__(self, functions=[tf.sin, tf.cos]):
        """
        Initialize the force library.

        Parameters
        ----------
        functions : list of callable, optional
            List of functions to apply (default: [tf.sin, tf.cos])
        """
        self.functions = functions

    @tf.function
    def __call__(self, x):
        """
        Transform input x to force features following force = amplitude * sin(omega * t).

        Parameters
        ----------
        x : array-like of shape (n_samples, 2*reduce_order)
            Latent variable and its time derivative.

        Returns
        -------
        array-like
            Force features.
        """
        x_force = []
        for func in self.functions:
            x_force += [x[:, 2:] * func(x[:, 1:2] * x[:, 0:1])]

        x_force = tf.concat(x_force, axis=1)
        return x_force

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
            Names of the force features.
        """
        x_force = []
        for func in self.functions:
            x_force += [f'{x[2]}*{func.__name__}({x[1]}*{x[0]})']
        return x_force
