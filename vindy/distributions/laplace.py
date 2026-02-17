import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from .base_distribution import BaseDistribution

class Laplace(BaseDistribution):
    """
    Laplace distribution layer for the reparameterization trick.

    This layer samples from a Laplace distribution using the reparameterization
    trick and computes KL divergence between two Laplace distributions.
    """

    def __init__(self, prior_mean=0., prior_scale=1., **kwargs):
        """
        Initialize Laplace distribution layer.

        Parameters
        ----------
        prior_mean : float, default=0.0
            Mean (location) of the prior distribution.
        prior_scale : float, default=1.0
            Scale factor of the prior distribution.
        **kwargs
            Additional keyword arguments passed to ``tf.keras.layers.Layer``.
        """
        super(Laplace, self).__init__(**kwargs)
        assert isinstance(prior_mean, float), "prior mean must be a float"
        assert isinstance(prior_scale, float) and prior_scale > 0, "prior scale must be a float > 0"
        self.prior_mean = prior_mean
        self.prior_scale = prior_scale


    def call(self, inputs):
        """
        Draw a sample from a Laplace distribution using the reparameterization trick.

        Sample y ~ L(loc, exp(log_scale)) using the reparameterization trick:
        x = mu + exp(log_scale) * epsilon, where epsilon ~ L(0, 1)

        Parameters
        ----------
        inputs : list of tf.Tensor
            [loc, log_scale] where loc is the location and log_scale is the
            log scale of the distribution.

        Returns
        -------
        tf.Tensor
            Samples from the Laplace distribution.
        """

        loc, log_scale = inputs
        dim = tf.shape(loc)[1]
        batch = tf.shape(loc)[0]

        # create random Laplacian distributed coefficients with mean 0 and scale 1
        laplace_dist = tfp.distributions.Laplace(0, 1)
        epsilon = laplace_dist.sample(sample_shape=(batch, dim))
        return loc + self.reverse_log(log_scale) * epsilon

    def KL_divergence(self, mean, log_scale):
        """
        Compute KL divergence between two univariate Laplace distributions.

        For p(x) ~ L(mu1, s1) and q(x) ~ L(mu2, s2), the KL divergence is:
        KL(p,q) = log(s2/s1) + (s1*exp(-|mu1-mu2|/s1) + |mu1-mu2|)/s2 - 1

        See supplemental material of Meyer, G. P. (2021). An alternative
        probabilistic interpretation of the huber loss. CVPR 2021.

        Parameters
        ----------
        mean : tf.Tensor
            Mean (location) of the first Laplace distribution.
        log_scale : tf.Tensor
            Log scale of the first Laplace distribution.

        Returns
        -------
        tf.Tensor
            KL divergence.
        """
        mu1 = mean
        mu2 = self.prior_mean
        s1 = self.reverse_log(log_scale)
        s2 = self.prior_scale
        mu_diff = tf.math.abs(mu1 - mu2)
        kl = tf.math.log(s2/s1) + (s1*tf.math.exp(-mu_diff/s1) + mu_diff)/s2 - 1
        return kl

    def prob_density_fcn(self, x, loc, scale):
        """
        Probability density function of the Laplace distribution.

        Parameters
        ----------
        x : array-like
            Points at which to evaluate the density.
        loc : float or array-like
            Location (mean) of the distribution.
        scale : float or array-like
            Scale parameter of the distribution.

        Returns
        -------
        array-like
            Probability density at x.
        """
        return np.exp(-np.abs(x-loc)/scale) / (2*scale)

    def variance_to_log_scale(self, variance):
        """
        Convert variance to log scale.

        Parameters
        ----------
        variance : tf.Tensor
            Variance of the distribution.

        Returns
        -------
        tf.Tensor
            Log scale.
        """
        return tf.math.log(tf.math.sqrt(0.5 * variance))

    def variance(self, log_scale):
        """
        Compute the variance of the Laplace distribution.

        Parameters
        ----------
        log_scale : array-like
            Log scale factor.

        Returns
        -------
        array-like
            Variance (2*scale^2).
        """
        scale = self.reverse_log(log_scale)
        return 2*scale**2