import logging
import tensorflow as tf
import numpy as np
from vindy.distributions import Gaussian
from .autoencoder_sindy import AutoencoderSindy

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


class VENI(AutoencoderSindy):
    """Variational Encoder Network for system identification.

    The VENI model combines a variational autoencoder with a SINDy
    layer to discover low-dimensional dynamics from high-dimensional
    observations.

    Parameters
    ----------
    beta : float
        Weight of the KL divergence term in the loss function.
    **kwargs
        Additional keyword arguments forwarded to ``AutoencoderSindy``.
    """

    def __init__(self, beta, **kwargs):
        # assert that input arguments are valid
        if not hasattr(self, "config"):
            self._init_to_config(locals())
        assert isinstance(beta, float) or isinstance(beta, int), "beta must be a float"
        self.beta = beta
        super(VENI, self).__init__(**kwargs)

    def create_loss_trackers(self):
        """Create loss trackers used during training.

        Extends the base trackers by adding a tracker for the KL loss.
        """
        super(VENI, self).create_loss_trackers()
        self.loss_trackers["kl"] = tf.keras.metrics.Mean(name="kl_loss")

    def build_encoder(self, x):
        """Build the variational encoder network.

        Parameters
        ----------
        x : array-like
            Example input array used to infer input shapes.

        Returns
        -------
        x_input : tf.keras.Input
            The encoder input tensor.
        z : tf.Tensor
            Sampled latent variable from the learned Gaussian.
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
        """Compute the KL divergence between the learned Gaussian and the unit Gaussian.

        Parameters
        ----------
        mean : tf.Tensor
            Mean of the approximate posterior.
        log_var : tf.Tensor
            Log-variance of the approximate posterior.

        Returns
        -------
        tf.Tensor
            Scalar KL divergence loss scaled by ``self.beta``.
        """
        kl_loss = -0.5 * (1 + log_var - tf.square(mean) - tf.exp(log_var))
        # sum over the latent dimension is correct as it reflects the kl divergence for a multivariate isotropic Gaussian
        kl_loss = self.beta * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

        return kl_loss

    def _training_encoding(self, x, losses):
        """Get latent encoding used during training and accumulate KL loss.

        Parameters
        ----------
        x : array-like
            Input observations.
        losses : dict
            Mutable dict where computed losses are stored/accumulated.

        Returns
        -------
        tuple
            (z, losses) where ``z`` is the sampled latent variable and
            ``losses`` includes the KL contribution.
        """
        z_mean, z_log_var, z = self.variational_encoder(x)
        kl_loss = self.kl_loss(z_mean, z_log_var)
        losses["kl"] = kl_loss
        losses["loss"] += kl_loss
        return z, losses

    def encode(self, x, training=False, mean_or_sample="mean"):
        """Encode input to latent space and return mean or sample.

        Parameters
        ----------
        x : array-like
            Full state observations with shape ``(n_samples, n_features, ...)``.
        training : bool, optional
            If True, run in training mode (unused here).
        mean_or_sample : {'mean', 'sample'}, optional
            Return the mean of the posterior or a sample from it.

        Returns
        -------
        tf.Tensor
            Latent representation (mean or sample) of shape ``(n_samples, reduced_order)``.
        """
        x = self.flatten(x)
        z_mean, _, z = self.variational_encoder(x)
        if mean_or_sample == "mean":
            return z_mean
        elif mean_or_sample == "sample":
            return z
        else:
            raise ValueError("mean_or_sample must be either 'mean' or 'sample'")

    def call(self, inputs, _=None):
        z_mean, z_log_var, z = self.encode(inputs)
        reconstruction = self.decode(z)
        return reconstruction

    @staticmethod
    def reconstruction_loss(x, x_reconstruction):
        """Reconstruction loss used for the variational autoencoder.

        The implementation follows the log-MSE variant referenced in the
        VINDy paper.

        Parameters
        ----------
        x : array-like
            Original inputs.
        x_reconstruction : array-like
            Reconstructed inputs from the decoder.

        Returns
        -------
        tf.Tensor
            Scalar reconstruction loss.
        """
        return tf.math.log(
            2 * np.pi * tf.reduce_mean(tf.keras.losses.mse(x, x_reconstruction)) + 1
        )
