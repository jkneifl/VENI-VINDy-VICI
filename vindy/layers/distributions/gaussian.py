import numpy as np
import tensorflow as tf
from vindy.layers.distributions.base_distribution import BaseDistribution


class Gaussian(BaseDistribution):
    """
    Layer for a Gaussian distribution that can be used to perform the reparameterization trick by sampling from an
    unit Gaussian and to compute the KL divergence between two Gaussian distributions.
    Uses (z_mean, z_log_var) to sample arguments from a normal distribution with mean z_mean and log variance z_log_var
    (the log variance is used to ensure that the variance is positive)
    """

    def __init__(self, prior_mean=0.0, prior_variance=1.0, **kwargs):
        """
        :param prior_mean: mean of the prior distribution
        :param prior_variance: variance of the prior distribution
        :param kwargs: arguments passed to tensorflow.keras.layers.Layer
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
        Draw a sample y ~ N(z_mean, exp(z_log_var)) from a normal distribution with mean z_mean and
        log variance z_log_var using the reparemeterization trick. (log variance is used to ensure numerical stability)

        variance = measurement_noise_factor^2

        x = mu + measurement_noise_factor * epsilon
            epsilon ~ N(0, 1)

        rewritten with log variance:
            x = mu + exp(0.5 * log_var) * epsilon = mu + (measurement_noise_factor^2)^0.5 * epsilon

        :param inputs:
        :return:
        """
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        # create random normal distributed coefficients with mean 0 and std 1
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + self.log_var_to_deviation(z_log_var) * epsilon

    def KL_divergence(self, mean, log_var):
        """
        Computes the KL divergence between two univariate normal distributions p(x) ~ N(mu1, sigma1) and
        q(x) ~ N(mu2, sigma2) following
            KL(p,q) = log(sigma2/sigma1) + (sigma1^2 + (mu1-mu2)^2) / (2*sigma2^2) - 1/2
        in case of a unitary Gaussian q(x) = N(0,1) the KL divergence simplifies to
            KL(p,q) = log(1/sigma1) + (sigma1^2 + mu1^2 -1) / 2
        which can be rewritten using the log variance log_var1 = log(sigma1**2) as
            KL(p,q) = -0.5 * (1 + log_var1 - mu1^2 - exp(log_var1))
        :param mean: mean of the first normal distribution
        :param log_var: log variance of a given normal distribution
        :return:
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
        Converts the log variance to standard deviation (variance = measurement_noise_factor^2) following
            measurement_noise_factor = exp(0.5 * log(measurement_noise_factor^2)) = (measurement_noise_factor^2)^0.5
        :param log_var:
        :return:
        """
        return tf.exp(0.5 * log_var)

    def variance_to_log_scale(self, variance):
        """
        Converts the variance to log scale (log variance = log(measurement_noise_factor^2)) following
            log(measurement_noise_factor^2) = log(measurement_noise_factor^2)
        :param variance:
        :return:
        """
        return tf.math.log(variance)

    def prob_density_fcn(self, x, mean, variance):
        """

        :param mean:
        :param variance:
        :return:
        """
        return np.exp(-0.5 * (x - mean) ** 2 / variance) / np.sqrt(2 * np.pi * variance)

    def variance(self, log_var):
        """
        Converts the log variance to variance (variance = measurement_noise_factor^2)
        :return:
        """
        return np.exp(log_var)
