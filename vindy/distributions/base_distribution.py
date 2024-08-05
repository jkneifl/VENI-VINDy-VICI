import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod

class BaseDistribution(tf.keras.layers.Layer, ABC):
    """
    Base class for distribution layers implementing a call function to sample from the distribution and a
    KL divergence function to compute the KL divergence between two distributions.
    """

    @abstractmethod
    def call(self, inputs):
        """
        Sample from the distribution
        :param inputs:
        :return:
        """
        pass

    @abstractmethod
    def KL_divergence(self):
        """
        Compute the KL divergence between two distributions
        :return:
        """
        pass

    @abstractmethod
    def prob_density_fcn(self, x, mean, scale):
        """
        Probability density function
        :param x: input
        :param loc: mean
        :param scale: scale
        :return:
        """
        pass

    @abstractmethod
    def variance(self, scale):
        """
        Probability density function
        :param x: input
        :param loc: mean
        :param scale: scale
        :return:
        """
        pass

    def reverse_log(self, log_scale):
        """
        Converts the log scale to scale following
            s = exp(log(s)) = exp(log_scale)
        :param log_scale:
        :return:
        """
        return tf.exp(log_scale)

    def plot(self, mean, scale, ax=None):
        """
        Plots the probability density function of the Laplace distribution
        :return:
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