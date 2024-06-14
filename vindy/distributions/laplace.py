import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from .base_distribution import BaseDistribution

class Laplace(BaseDistribution):
    """
    Layer for a Laplace distribution that can be used to perform the reparameterization trick by sampling from an
    unit Laplace and to compute the KL divergence between two Laplace distributions.
    """

    def __init__(self, prior_mean=0., prior_scale=1., **kwargs):
        """
        :param prior_mean: mean (location) of the prior distribution
        :param prior_scale: scale factor of the prior distribution
        :param kwargs: passed to tensorflow.keras.layers.Layer
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

        y = loc + scale * epsilon
            epsilon ~ L(0, 1)

        rewritten with log scale:
            x = mu + exp(log_scale) * epsilon = mu + exp(log(scale)) * epsilon

        :param inputs:
        :return:
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
        q(x) ~ L(mu2, s2) following
            KL(p,q) = log(s2/s1) + (s1*exp(-|mu1-mu2|/s1) + |mu1-mu2|)/s2 - 1
        See supplemental material of
            Meyer, G. P. (2021). An alternative probabilistic interpretation of the huber loss.
            In Proceedings of the ieee/cvf conference on computer vision and pattern recognition (pp. 5261-5269).
        https://openaccess.thecvf.com/content/CVPR2021/supplemental/Meyer_An_Alternative_Probabilistic_CVPR_2021_supplemental.pdf

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
        Probability density function of the Laplace distribution
        :param x: input
        :param loc: mean
        :param scale: scale
        :return:
        """
        return np.exp(-np.abs(x-loc)/scale) / (2*scale)

    def variance_to_log_scale(self, variance):
        """
        Converts the variance to log scale
        :param variance:
        :return:
        """
        return tf.math.log(tf.math.sqrt(0.5 * variance))

    def variance(self, log_scale):
        """
        Computes the variance of the Laplace distribution
        :param log_scale: log scale factor
        :return:
        """
        scale = self.reverse_log(log_scale)
        return 2*scale**2