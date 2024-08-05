import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from .base_distribution import BaseDistribution

class Laplace(BaseDistribution):
    """
    Layer for a Laplace distribution that can be used to perform the reparameterization trick by sampling from an
    unit Laplace and to compute the KL divergence between two Laplace distributions.

    Methods
    -------
    call(inputs)
        Draw a sample y ~ L(z_mean, exp(z_log_var)) from a Laplace distribution with location loc and log scale.
    KL_divergence(mean, log_scale)
        Compute the KL divergence between two univariate Laplace distributions.
    log_scale_to_deviation(log_scale)
        Convert the log scale to standard deviation.
    variance_to_log_scale(variance)
        Convert the variance to log scale.
    prob_density_fcn(x, loc, scale)
        Compute the probability density function.
    variance(log_scale)
        Compute the variance of the Laplace distribution.
    """

    def __init__(self, prior_mean=0., prior_scale=1., **kwargs):
        """
        Initialize the Laplace distribution layer.

        Parameters
        ----------
        prior_mean : float
            Mean (location) of the prior distribution.
        prior_scale : float
            Scale factor of the prior distribution.
        kwargs : dict
            Additional keyword arguments.
        """
        super(Laplace, self).__init__(**kwargs)
        assert isinstance(prior_mean, float), "prior mean must be a float"
        assert isinstance(prior_scale, float) and prior_scale > 0, "prior scale must be a float > 0"
        self.prior_mean = prior_mean
        self.prior_scale = prior_scale


    def call(self, inputs):
        """
        Draw a sample y ~ L(z_mean, exp(z_log_var)) from a Laplace distribution with location loc and
        log scale using the reparameterization trick. (log scale is used to ensure numerical stability)
        y = loc + scale * epsilon    with    epsilon ~ L(0, 1)

        Parameters
        ----------
        inputs : tuple
            A tuple containing loc and log_scale.

        Returns
        -------
        tf.Tensor
            Sampled values from the Laplace distribution.
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
        Computes the KL divergence between two univariate Laplace distributions p(x) ~ L(mu1, s1) and
        q(x) ~ L(mu2, s2) following KL(p,q) = log(s2/s1) + (s1*exp(-|mu1-mu2|/s1) + |mu1-mu2|)/s2 - 1
        See supplemental material of
        Meyer, G. P. (2021). An alternative probabilistic interpretation of the huber loss. In Proceedings of the
        ieee/cvf conference on computer vision and pattern recognition (pp. 5261-5269).
        https://openaccess.thecvf.com/content/CVPR2021/supplemental/Meyer_An_Alternative_Probabilistic_CVPR_2021_supplemental.pdf

        Parameters
        ----------
        mean : float
            Mean (location) of the first Laplace distribution.
        log_scale : float
            Log scale of the first Laplace distribution.

        Returns
        -------
        tf.Tensor
            KL divergence.
        """

        """


        :param mean: mean (location) of the first Laplace distribution
        :param log_scale: log scale of the first Laplace distribution
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
        Compute the probability density function of the Laplace distribution.

        Parameters
        ----------
        x : float
            Input value.
        loc : float
            Mean (location) of the distribution.
        scale : float
            Scale of the distribution.

        Returns
        -------
        float
            Probability density function value.
        """
        return np.exp(-np.abs(x-loc)/scale) / (2*scale)

    def variance_to_log_scale(self, variance):
        """
        Convert the variance to log scale.

        Parameters
        ----------
        variance : float
            Variance.

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
        log_scale : float
            Log scale factor.

        Returns
        -------
        float
            Variance.
        """
        scale = self.reverse_log(log_scale)
        return 2*scale**2