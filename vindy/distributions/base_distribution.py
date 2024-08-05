import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod

class BaseDistribution(tf.keras.layers.Layer, ABC):
    """
    Base class for distribution layers implementing a call function to sample from the distribution and a
    KL divergence function to compute the KL divergence between two distributions.

    Methods
    -------
    call(inputs)
        Sample from the distribution.
    KL_divergence()
        Compute the KL divergence between two distributions.
    prob_density_fcn(x, mean, scale)
        Compute the probability density function.
    variance(scale)
        Compute the variance of the distribution.
    reverse_log(log_scale)
        Convert the log scale to scale.
    plot(mean, scale, ax=None)
        Plot the probability density function of the distribution.
    """

    @abstractmethod
    def call(self, inputs):
        """
        Sample from the distribution.

        Parameters
        ----------
        inputs : tuple
            Inputs to the distribution.

        Returns
        -------
        tf.Tensor
            Sampled values from the distribution.
        """
        pass

    @abstractmethod
    def KL_divergence(self):
        """
        Compute the KL divergence between two distributions.

        Returns
        -------
        tf.Tensor
            KL divergence.
        """
        pass

    @abstractmethod
    def prob_density_fcn(self, x, mean, scale):
        """
        Compute the probability density function.

        Parameters
        ----------
        x : float
            Input value.
        mean : float
            Mean of the distribution.
        scale : float
            Scale of the distribution.

        Returns
        -------
        float
            Probability density function value.
        """
        pass

    @abstractmethod
    def variance(self, scale):
        """
        Compute the variance of the distribution.

        Parameters
        ----------
        scale : float
            Scale of the distribution.

        Returns
        -------
        float
            Variance.
        """
        pass

    def reverse_log(self, log_scale):
        """
        Convert the log scale to scale.

        Parameters
        ----------
        log_scale : float
            Log scale.

        Returns
        -------
        tf.Tensor
            Scale.
        """
        return tf.exp(log_scale)

    def plot(self, mean, scale, ax=None):
        """
        Plot the probability density function of the distribution.

        Parameters
        ----------
        mean : float
            Mean of the distribution.
        scale : float
            Scale of the distribution.
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes object to plot on. If None, a new axis is created.

        Returns
        -------
        None
        """
        if ax is None:
            ax = plt.gca()
        variance = self.variance(scale)
        x = (np.linspace(-1*variance, 1*variance, 3000) + mean)
        # find first positive value
        try:
            idx = np.where(x > 0)[0][0]
            x = np.insert(x, idx, 0)
        except IndexError:
            pass

        if isinstance(x, tf.Tensor):
            x = x.numpy().squeeze()
        x = x.squeeze()
        # plt.figure()
        # get current axis
        ax.plot(x, self.prob_density_fcn(x, mean, scale))
        # fill area under curve
        ax.fill_between(x, self.prob_density_fcn(x, mean, scale), alpha=0.3)
        # plt.show()