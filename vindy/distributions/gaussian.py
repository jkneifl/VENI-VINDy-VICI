import numpy as np
import tensorflow as tf
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
            Additional arguments passed to tensorflow.keras.layers.Layer.
        """
        super(Gaussian, self).__init__(**kwargs)
        assert isinstance(prior_mean, float), "prior mean must be a float"
        assert (
            isinstance(prior_variance, float) and prior_variance > 0
        ), "prior variance must be a float > 0"
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.prior_deviation = tf.math.sqrt(self.prior_variance)

    def call(self, inputs):
        """
        Draw a sample from a normal distribution using the reparameterization trick.

        Sample y ~ N(z_mean, exp(z_log_var)) from a normal distribution with mean z_mean and
        log variance z_log_var using the reparameterization trick. Log variance is used to
        ensure numerical stability.

        The variance relationship: variance = measurement_noise_factor^2

        Sampling formula:
            x = mu + measurement_noise_factor * epsilon, where epsilon ~ N(0, 1)

        Rewritten with log variance:
            x = mu + exp(0.5 * log_var) * epsilon = mu + (measurement_noise_factor^2)^0.5 * epsilon

        Parameters
        ----------
        inputs : tuple of tf.Tensor
            Tuple containing (z_mean, z_log_var) where z_mean is the mean and
            z_log_var is the log variance.

        Returns
        -------
        tf.Tensor
            Sampled values from the normal distribution.
        """
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        # create random normal distributed coefficients with mean 0 and std 1
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + self.log_var_to_deviation(z_log_var) * epsilon

    def KL_divergence(self, mean, log_var):
        """
        Compute the KL divergence between two univariate normal distributions.

        Computes the KL divergence between two univariate normal distributions p(x) ~ N(mu1, sigma1)
        and q(x) ~ N(mu2, sigma2) following:
            KL(p,q) = log(sigma2/sigma1) + (sigma1^2 + (mu1-mu2)^2) / (2*sigma2^2) - 1/2

        In case of a unitary Gaussian q(x) = N(0,1) the KL divergence simplifies to:
            KL(p,q) = log(1/sigma1) + (sigma1^2 + mu1^2 -1) / 2

        Which can be rewritten using the log variance log_var1 = log(sigma1**2) as:
            KL(p,q) = -0.5 * (1 + log_var1 - mu1^2 - exp(log_var1))

        Parameters
        ----------
        mean : tf.Tensor
            Mean of the first normal distribution.
        log_var : tf.Tensor
            Log variance of the first normal distribution.

        Returns
        -------
        tf.Tensor
            KL divergence value.
        """
        sigma1 = self.log_var_to_deviation(log_var)
        sigma2 = self.prior_deviation

        kl = (
            tf.math.log(sigma2 / sigma1)
            + (sigma1**2 + (mean - self.prior_mean) ** 2) / (2 * sigma2**2)
            - 1 / 2
        )
        return kl

    def log_var_to_deviation(self, log_var):
        """
        Convert log variance to standard deviation.

        Converts the log variance to standard deviation (variance = measurement_noise_factor^2) following:
            measurement_noise_factor = exp(0.5 * log(measurement_noise_factor^2)) = (measurement_noise_factor^2)^0.5

        Parameters
        ----------
        log_var : tf.Tensor
            Log variance.

        Returns
        -------
        tf.Tensor
            Standard deviation.
        """
        return tf.exp(0.5 * log_var)

    def variance_to_log_scale(self, variance):
        """
        Convert variance to log scale.

        Parameters
        ----------
        variance : tf.Tensor
            Variance.

        Returns
        -------
        tf.Tensor
            Log variance.
        """
        return tf.math.log(variance)

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
