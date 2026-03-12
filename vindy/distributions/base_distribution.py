import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from abc import abstractmethod, ABC

class BaseDistribution(nn.Module, ABC):
    """
    Base class for probabilistic distributions used in variational encoders.

    Subclasses should implement sampling and log-probability computations.

    Methods
    -------
    forward(inputs)
        Return samples and any auxiliary outputs (e.g. mean/logvar).
    """

    @abstractmethod
    def forward(self, *inputs):
        """
        Sample from the distribution.

        Parameters
        ----------
        inputs : array-like
            Inputs used to parameterize the distribution (for instance mean/logvar).

        Returns
        -------
        tuple or torch.Tensor
            Samples (and optionally auxiliary statistics).
        """
        pass

    @abstractmethod
    def KL_divergence(self):
        """
        Compute the KL divergence between two distributions.

        Returns
        -------
        torch.Tensor
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
        if isinstance(log_scale, torch.Tensor):
            return torch.exp(log_scale)
        return np.exp(log_scale)

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
        # convert tensors to numpy for plotting
        if isinstance(mean, torch.Tensor):
            mean = mean.detach().cpu().numpy()
        if isinstance(variance, torch.Tensor):
            variance = variance.detach().cpu().numpy()
        x = (np.linspace(-1*variance, 1*variance, 3000) + mean)
        # find first positive value
        try:
            idx = np.where(x > 0)[0][0]
            x = np.insert(x, idx, 0)
        except IndexError:
            pass

        x = np.asarray(x).squeeze()
        ax.plot(x, self.prob_density_fcn(x, mean, scale))
        # fill area under curve
        ax.fill_between(x, self.prob_density_fcn(x, mean, scale), alpha=0.3)
