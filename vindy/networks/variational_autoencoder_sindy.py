import logging
import tensorflow as tf
import numpy as np
from vindy.distributions import Gaussian
from .autoencoder_sindy import AutoencoderSindy

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


class VAESindy(AutoencoderSindy):
    """
    Model to discover low-dimensional dynamics of a high-dimensional system using variational autoencoders and SINDy/VINDy.
    """

    def __init__(self, beta, **kwargs):
        """
        Initialize the VAE SINDy model.

        Parameters
        ----------
        beta : float
            Weight of the KL divergence term in the loss function.
        kwargs : dict
            Additional arguments for AutoencoderSindy.
        """
        # assert that input arguments are valid
        if not hasattr(self, "config"):
            self._init_to_config(locals())
        assert isinstance(beta, float) or isinstance(beta, int), "beta must be a float"
        self.beta = beta
        super(VAESindy, self).__init__(**kwargs)

    def create_loss_trackers(self):
        """
        Create loss trackers for the model.
        """
        super(VAESindy, self).create_loss_trackers()
        self.loss_trackers["kl"] = tf.keras.metrics.Mean(name="kl_loss")

    def build_encoder(self, x):
        """
        Build the variational encoder part of the model.

        Parameters
        ----------
        x : array-like
            Input data.

        Returns
        -------
        tuple
            Input tensor and latent variable tensor.
        """
        x_input = tf.keras.Input(shape=(x.shape[1],), dtype=self.dtype_)
        z_ = x_input
        for n_neurons in self.layer_sizes:
            z_ = tf.keras.layers.Dense(
                n_neurons,
                activation=self.activation,
                kernel_regularizer=self.kernel_regularizer,
            )(z_)

        zero_initializer = tf.keras.initializers.Zeros()
        z_mean = tf.keras.layers.Dense(
            self.reduced_order,
            name="z_mean",
            kernel_regularizer=self.kernel_regularizer,
        )(z_)

        z_log_var = tf.keras.layers.Dense(
            self.reduced_order,
            name="z_log_var",
            kernel_initializer=zero_initializer,
            kernel_regularizer=self.kernel_regularizer,
        )(z_)

        z = Gaussian()([z_mean, z_log_var])
        self.variational_encoder = tf.keras.Model(
            x_input, [z_mean, z_log_var, z], name="encoder"
        )
        return x_input, z

    def kl_loss(self, mean, log_var):
        """
        Calculate the KL divergence loss for Gaussian distributions.

        Parameters
        ----------
        mean : tf.Tensor
            Mean of the latent variable distribution.
        log_var : tf.Tensor
            Log variance of the latent variable distribution.

        Returns
        -------
        tf.Tensor
            KL divergence loss.
        """
        kl_loss = -0.5 * (1 + log_var - tf.square(mean) - tf.exp(log_var))
        # sum over the latent dimension is correct as it reflects the kl divergence for a multivariate isotropic Gaussian
        kl_loss = self.beta * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

        return kl_loss

    def _training_encoding(self, x, losses):
        """
        For compatibility with the parent class we need a method that only returns the latent variable
        but not the mean and log variance. The mean and log variance are used to calculate the KL divergence

        Parameters
        ----------
        x : array-like
            Full state.
        losses : dict
            Dictionary of losses.

        Returns
        -------
        tuple
            Latent variable and updated losses.
        """
        z_mean, z_log_var, z = self.variational_encoder(x)
        kl_loss = self.kl_loss(z_mean, z_log_var)
        losses["kl"] = kl_loss
        losses["loss"] += kl_loss
        return z, losses

    def encode(self, x, training=False):
        """
        Encode full state to latent distribution and return its mean.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features, n_dof_per_feature)
            Full state.
        training : bool, optional
            Whether the model is in training mode (default is False).

        Returns
        -------
        array-like of shape (n_samples, reduced_order)
            Latent variable.
        """
        x = self.flatten(x)
        z_mean, _, _ = self.variational_encoder(x)
        return z_mean

    def call(self, inputs, _=None):
        """
        Perform a forward pass through the model.

        Parameters
        ----------
        inputs : array-like
            Input data.

        Returns
        -------
        array-like
            Reconstructed data.
        """

        z_mean, z_log_var, z = self.encode(inputs)
        reconstruction = self.decode(z)
        return reconstruction

    @staticmethod
    def reconstruction_loss(x, x_reconstruction):
        """
        Compute the reconstruction loss of the autoencoder as log(mse) as stated in https://arxiv.org/pdf/2006.10273.pdf

        Parameters
        ----------
        x : array-like
            Input data.
        x_reconstruction : array-like
            Reconstructed data.

        Returns
        -------
        tf.Tensor
            Reconstruction loss.
        """
        return tf.math.log(
            2 * np.pi * tf.reduce_mean(tf.keras.losses.mse(x, x_reconstruction)) + 1
        )
