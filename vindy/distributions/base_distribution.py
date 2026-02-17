import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from abc import abstractmethod

class BaseDistribution(tf.keras.layers.Layer):
    """
    Base class for probabilistic distributions used in variational encoders.

    Subclasses should implement sampling and log-probability computations.

    Methods
    -------
    call(inputs)
        Return samples and any auxiliary outputs (e.g. mean/logvar).
    """

    @abstractmethod
    def call(self, inputs):
        """
        Sample from the distribution.

        Parameters
        ----------
        inputs : array-like
            Inputs used to parameterize the distribution (for instance mean/logvar).

        Returns
        -------
        tuple or tf.Tensor
            Samples (and optionally auxiliary statistics).
        """
        pass

    @abstractmethod
    def KL_divergence(self):
        """
        Compute the KL divergence between two distributions.

        Returns
        -------
        tf.Tensor
            Scalar KL divergence.
        """
        pass

    @abstractmethod
    def prob_density_fcn(self, x, mean, scale):
        """
        Probability density function.

        Parameters
        ----------
        x : array-like
            Points at which to evaluate the density.
        mean : float or array-like
            Distribution mean/loc parameter.
        scale : float or array-like
            Scale parameter (std, scale, etc.).

        Returns
        -------
        array-like
            Density values at x.
        """
        pass

    @abstractmethod
    def variance(self, scale):
        """
        Variance as a function of the distribution scale parameter.

        Parameters
        ----------
        scale : float or array-like
            Scale parameter of the distribution.

        Returns
        -------
        float or array-like
            Variance corresponding to the provided scale.
        """
        pass

    def reverse_log(self, log_scale):
        """
        Converts the log scale to scale following
            s = exp(log(s)) = exp(log_scale)

        Parameters
        ----------
        log_scale : array-like
            Logarithm of the scale parameter.

        Returns
        -------
        array-like
            Scale (exp(log_scale)).
        """
        return tf.exp(log_scale)

    def plot(self, mean, scale, ax=None):
        """
        Plots the probability density function of the distribution.

        Parameters
        ----------
        mean : float or array-like
            Mean/loc of the distribution.
        scale : float or array-like
            Scale parameter.
        ax : matplotlib.axes.Axes, optional
            Axis to draw on. If None, uses current axis.
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