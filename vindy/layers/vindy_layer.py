import numpy as np
import logging
import tensorflow as tf
import matplotlib.pyplot as plt
from .sindy_layer import SindyLayer
from vindy.layers.distributions import Gaussian, BaseDistribution

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


class VindyLayer(SindyLayer):

    def __init__(self, beta=1, priors=Gaussian(0.0, 1.0), **kwargs):
        """
        Layer for variational identification of nonlinear dynamics (VINDy) approximation of the
        time derivative of the latent variable. Feature libraries are applied to the latent variables and a
        (sparse) variational inference is performed to obtain the coefficients
        :param beta: scaling factor for the KL divergence
        :param priors: prior distribution for the coefficients
        :param kwargs: see SindyLayer
        """
        super(VindyLayer, self).__init__(**kwargs)
        self.assert_additional_args(beta, priors)
        self.priors = priors
        self.beta = beta
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_sindy")

    def assert_additional_args(self, beta, priors):
        """
        Asserts that the additional arguments are valid
        :param beta:
        :param priors:
        :return:
        """
        # assert that input arguments are valid that are not checked in the super class
        assert isinstance(beta, float) or isinstance(beta, int), "beta must be a float"
        # priors
        if isinstance(priors, list):
            assert (
                len(priors) == self.n_dofs
            ), f"Number of priors must match the number of dofs ({self.n_dofs})"
            for prior in priors:
                assert isinstance(prior, BaseDistribution), (
                    "All priors must be an instance inheriting from " "BaseDistribution"
                )
        else:
            assert isinstance(
                priors, BaseDistribution
            ), "priors must be a class inheriting from BaseDistribution"

    def init_weigths(self, kernel_regularizer):
        super(VindyLayer, self).init_weigths(kernel_regularizer)

        # initialize the log variance of the coefficients
        init = tf.random_uniform_initializer(minval=-1, maxval=1)
        l1, l2 = kernel_regularizer.l1, kernel_regularizer.l2
        scale_regularizer = LogVarL1L2Regularizer(l1=l1, l2=l2)
        self.kernel_scale = self.add_weight(
            name="SINDy_log_scale",
            initializer=init,
            shape=self.kernel_shape,
            dtype=self.dtype_,
            regularizer=scale_regularizer,
        )

    @property
    def loss_trackers(self):
        return dict(kl_sindy=self.kl_loss_tracker)

    @property
    def _coeffs(self):
        """
        Returns the coefficients of the SINDy layer which are sampled from a normal distribution parametrized by the
        layer's kernel (weights)
        :return:
        """
        # split the kernel into mean and log variance
        coeffs_mean, coeffs_log_scale = self.kernel, self.kernel_scale
        # draw samples from the normal distribution
        if isinstance(self.priors, list):
            trainable_coeffs = []
            for i, prior in enumerate(self.priors):
                trainable_coeffs.append(
                    prior([coeffs_mean[i : i + 1], coeffs_log_scale[i : i + 1]])
                )
            trainable_coeffs = tf.concat(trainable_coeffs, axis=0)
        else:
            trainable_coeffs = self.priors([coeffs_mean, coeffs_log_scale])

        # fill the coefficient matrix with the trainable coefficients
        coeffs = self.fill_coefficient_matrix(trainable_coeffs)

        return coeffs, coeffs_mean, coeffs_log_scale

    def kl_loss(self, mean, scale):
        """
        Computes the KL divergence between the priors and the coefficient distributions of the VINDy layer
        :param mean:
        :param scale:
        :return:
        """
        if isinstance(self.priors, list):
            kl_loss = 0
            for prior in self.priors:
                kl_loss += prior.KL_divergence(mean, scale)
        else:
            kl_loss = self.priors.KL_divergence(mean, scale)

        return self.beta * tf.reduce_sum(kl_loss)

    def get_sindy_coeffs(self):
        _, coeffs_mean, _ = self._coeffs
        coeffs = self.fill_coefficient_matrix(coeffs_mean)
        return coeffs.numpy()

    # @tf.function
    def call(self, inputs, training=False):
        """
        Applies the VINDy layer to the arguments, i.e. applies the feature libraries to the arguments,
        samples the coefficients from a normal distribution parametrized by the layer's kernel (weights) and
        computes the dot product of the features and the coefficients
        :param inputs:
        :param training:
        :return:
        """
        # todo: think about whether we want to have deterministic coefficients during inference (after training) or not
        z_features = self.tfFeat(inputs)
        coeffs, coeffs_mean, coeffs_log_var = self._coeffs
        if training:
            z_dot = z_features @ tf.transpose(coeffs)
            return [z_dot, coeffs_mean, coeffs_log_var]
        else:
            # in case of evaluation, we use the mean of the coefficients
            z_dot = z_features @ tf.transpose(self.fill_coefficient_matrix(coeffs_mean))
            return z_dot

    def visualize_coefficients(self, x_range=None, z=None, mu=None):
        """
        Visualizes the coefficients of the SINDy layer as distributions
        :return:
        """
        # get coefficient parameterization
        _, mean, log_scale = self._coeffs
        _ = self._visualize_coefficients(
            mean.numpy(), log_scale.numpy(), x_range, z, mu
        )

    def _visualize_coefficients(self, mean, log_scale, x_range=None, z=None, mu=None):

        # get name of the corresponding features
        feature_names = self.get_feature_names(z, mu)
        n_variables = self.state_dim
        n_plots = int(self.n_dofs / n_variables)
        mean = mean.reshape(n_variables, n_plots).T
        log_scale = log_scale.reshape(n_variables, n_plots).T
        # create a plot with one subplot for each (trainablie) coefficient
        # for j in range(n_figures):
        fig, axs = plt.subplots(
            n_plots, n_variables, figsize=(n_variables * 10, 10), sharex=True
        )
        # in case of a one-dimensional system, we append a dimension to axs
        if n_variables == 1:
            axs = axs[:, np.newaxis]
        for j in range(n_variables):
            for i in range(n_plots):
                # draw a vertical line at 0
                axs[i][j].axvline(x=0, color="black", linestyle="--")
                # plot the distribution of the coefficients
                if isinstance(self.priors, list):
                    distribution = self.priors[i]
                else:
                    distribution = self.priors
                scale = distribution.reverse_log(log_scale[i, j])
                distribution.plot(mean[i, j], scale, ax=axs[i][j])
                # put feature name as ylabel
                axs[i][j].set_ylabel(f"${feature_names[i]}$", rotation=90, labelpad=10)
                # set x range
                if x_range is not None:
                    axs[i][j].set_xlim(x_range)
        plt.tight_layout()
        # ensure that ylabel don't overlap with axis ticks
        plt.subplots_adjust(left=0.1)
        return fig

    def pdf_thresholding(self, threshold: float = 1.0):
        """
        Cancel the coefficients of the SINDy layer if their corresponding probability density function at zero is above
        the threshold, i.e. if pdf(0) > threshold
        :param threshold:
        :return:
        """
        # get current
        _, loc, log_scale = self._coeffs
        feature_names = np.array([self.get_feature_names()] * self.state_dim).flatten()
        # cancel coefficients
        for i, (loc_, log_scale_) in enumerate(zip(loc[:-1], log_scale[:-1])):
            # plot the distribution of the coefficients
            if isinstance(self.priors, list):
                distribution = self.priors[i]
            else:
                distribution = self.priors
            scale = distribution.reverse_log(log_scale_)
            zero_density = distribution.prob_density_fcn(x=0, loc=loc_, scale=scale)
            if zero_density > threshold:
                # cancel the coefficient
                loc[i].assign(0)
                log_scale[i].assign(-10)
                logging.info(f"Canceling coefficient {feature_names[i]}")
        self.print()


class LogVarL1L2Regularizer(tf.keras.regularizers.Regularizer):
    """
    Regularizer for the log variance of the coefficients in the VINDy layer
    """

    def __init__(self, l1=0.0, l2=0.0):
        # The default value for l1 and l2 are different from the value in l1_l2
        # for backward compatibility reason. Eg, L1L2(l2=0.1) will only have l2
        # and no l1 penalty.
        l1 = 0.0 if l1 is None else l1
        l2 = 0.0 if l2 is None else l2

        self.l1 = l1
        self.l2 = l2

    def __call__(self, x):
        regularization = 0
        if self.l1:
            regularization += self.l1 * tf.reduce_sum(tf.abs(tf.exp(0.5 * x)))
        if self.l2:
            # equivalent to "self.l2 * tf.reduce_sum(tf.square(x))"
            regularization += 2.0 * self.l2 * tf.nn.l2_loss(tf.exp(0.5 * x))
        return regularization
