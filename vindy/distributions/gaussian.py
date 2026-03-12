import numpy as np
import torch
from .base_distribution import BaseDistribution


class Gaussian(BaseDistribution):
    """
    Layer for a Gaussian distribution that can be used to perform the reparameterization trick.

    This layer samples from a Gaussian distribution and computes the KL divergence between
    two Gaussian distributions. Uses (z_mean, z_log_var) to sample arguments from a normal
    distribution with mean z_mean and log variance z_log_var (the log variance is used to
    ensure that the variance is positive).
    """

    def __init__(self, prior_mean=0.0, prior_variance=1.0, **kwargs):
        """
        Initialize the Gaussian distribution layer.

        Parameters
        ----------
        prior_mean : float, optional
            Mean of the prior distribution (default is 0.0).
        prior_variance : float, optional
            Variance of the prior distribution (default is 1.0).
        **kwargs
            Additional arguments passed to nn.Module.
        """
        super(Gaussian, self).__init__(**kwargs)
        assert isinstance(prior_mean, float), "prior mean must be a float"
        assert (
            isinstance(prior_variance, float) and prior_variance > 0
        ), "prior variance must be a float > 0"
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.prior_deviation = float(np.sqrt(self.prior_variance))

    def forward(self, z_mean, z_log_var):
        """
        Draw a sample from a normal distribution using the reparameterization trick.

        Sample y ~ N(z_mean, exp(z_log_var)) from a normal distribution with mean z_mean and
        log variance z_log_var using the reparameterization trick.

        Parameters
        ----------
        z_mean : torch.Tensor
            Mean of the distribution.
        z_log_var : torch.Tensor
            Log variance of the distribution.

        Returns
        -------
        torch.Tensor
            Sampled values from the normal distribution.
        """
        epsilon = torch.randn_like(z_mean)
        return z_mean + self.log_var_to_deviation(z_log_var) * epsilon

    def KL_divergence(self, mean, log_var):
        """
        Compute the KL divergence between two univariate normal distributions.

        Computes the KL divergence between two univariate normal distributions p(x) ~ N(mu1, sigma1)
        and q(x) ~ N(mu2, sigma2) following:
            KL(p,q) = log(sigma2/sigma1) + (sigma1^2 + (mu1-mu2)^2) / (2*sigma2^2) - 1/2

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the first normal distribution.
        log_var : torch.Tensor
            Log variance of the first normal distribution.

        Returns
        -------
        torch.Tensor
            KL divergence value.
        """
        sigma1 = self.log_var_to_deviation(log_var)
        sigma2 = self.prior_deviation

        kl = (
            torch.log(torch.tensor(sigma2, dtype=mean.dtype, device=mean.device) / sigma1)
            + (sigma1**2 + (mean - self.prior_mean) ** 2) / (2 * sigma2**2)
            - 1 / 2
        )
        return kl

    def log_var_to_deviation(self, log_var):
        """
        Convert log variance to standard deviation.

        Parameters
        ----------
        log_var : torch.Tensor
            Log variance.

        Returns
        -------
        torch.Tensor
            Standard deviation.
        """
        return torch.exp(0.5 * log_var)

    def variance_to_log_scale(self, variance):
        """
        Convert variance to log scale.

        Parameters
        ----------
        variance : torch.Tensor
            Variance.

        Returns
        -------
        torch.Tensor
            Log variance.
        """
        return torch.log(variance)

    def prob_density_fcn(self, x, mean, variance):
        """
        Probability density function of the Gaussian distribution.

        Parameters
        ----------
        x : array-like
            Points at which to evaluate the density.
        mean : float or array-like
            Mean of the distribution.
        variance : float or array-like
            Variance of the distribution.

        Returns
        -------
        array-like
            Probability density at x.
        """
        return np.exp(-0.5 * (x - mean) ** 2 / variance) / np.sqrt(2 * np.pi * variance)

    def variance(self, log_var):
        """
        Convert log variance to variance.

        Parameters
        ----------
        log_var : array-like
            Log variance.

        Returns
        -------
        array-like
            Variance (exp(log_var)).
        """
        return np.exp(log_var)
