import logging
import torch
import torch.nn as nn
import numpy as np
from vindy.distributions import Gaussian
from .autoencoder_sindy import AutoencoderSindy, _get_activation

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
        assert isinstance(beta, float) or isinstance(beta, int), "beta must be a float"
        self.beta = beta
        super(VENI, self).__init__(**kwargs)

    def build_encoder(self, input_dim):
        """Build the variational encoder network.

        Parameters
        ----------
        input_dim : int
            Input dimension.
        """
        layers = []
        prev_dim = input_dim
        for n_neurons in self.layer_sizes:
            layers.append(nn.Linear(prev_dim, n_neurons))
            layers.append(_get_activation(self.activation_name))
            prev_dim = n_neurons

        self.encoder_backbone = nn.Sequential(*layers)
        self.z_mean_layer = nn.Linear(prev_dim, self.reduced_order)
        self.z_log_var_layer = nn.Linear(prev_dim, self.reduced_order)
        # initialize z_log_var_layer weights to zeros
        nn.init.zeros_(self.z_log_var_layer.weight)
        nn.init.zeros_(self.z_log_var_layer.bias)
        self.gaussian_sampling = Gaussian()

        # Create a wrapper encoder that outputs the sampled z
        self.encoder = _VariationalEncoder(
            self.encoder_backbone, self.z_mean_layer, self.z_log_var_layer, self.gaussian_sampling
        )

    def variational_encode(self, x):
        """
        Variational encoding: returns (z_mean, z_log_var, z).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        tuple
            (z_mean, z_log_var, z)
        """
        h = self.encoder_backbone(x)
        z_mean = self.z_mean_layer(h)
        z_log_var = self.z_log_var_layer(h)
        z = self.gaussian_sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z

    def kl_loss(self, mean, log_var):
        """Compute the KL divergence between the learned Gaussian and the unit Gaussian.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the approximate posterior.
        log_var : torch.Tensor
            Log-variance of the approximate posterior.

        Returns
        -------
        torch.Tensor
            Scalar KL divergence loss scaled by ``self.beta``.
        """
        kl_loss = -0.5 * (1 + log_var - torch.square(mean) - torch.exp(log_var))
        # sum over the latent dimension
        kl_loss = self.beta * torch.mean(torch.sum(kl_loss, dim=1))
        return kl_loss

    def _training_encoding(self, x, losses):
        """Get latent encoding used during training and accumulate KL loss.

        Parameters
        ----------
        x : torch.Tensor
            Input observations.
        losses : dict
            Mutable dict where computed losses are stored/accumulated.

        Returns
        -------
        tuple
            (z, losses)
        """
        z_mean, z_log_var, z = self.variational_encode(x)
        kl_loss = self.kl_loss(z_mean, z_log_var)
        losses["kl"] = kl_loss
        losses["loss"] = losses["loss"] + kl_loss
        return z, losses

    def encode(self, x, training=False, mean_or_sample="mean"):
        """Encode input to latent space and return mean or sample.

        Parameters
        ----------
        x : array-like or torch.Tensor
            Full state observations.
        training : bool, optional
            Unused.
        mean_or_sample : {'mean', 'sample'}, optional
            Return the mean of the posterior or a sample from it.

        Returns
        -------
        torch.Tensor
            Latent representation.
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=self.torch_dtype)
        x = self.flatten(x)
        z_mean, _, z = self.variational_encode(x)
        if mean_or_sample == "mean":
            return z_mean
        elif mean_or_sample == "sample":
            return z
        else:
            raise ValueError("mean_or_sample must be either 'mean' or 'sample'")

    @staticmethod
    def reconstruction_loss(x, x_reconstruction):
        """Reconstruction loss used for the variational autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            Original inputs.
        x_reconstruction : torch.Tensor
            Reconstructed inputs from the decoder.

        Returns
        -------
        torch.Tensor
            Scalar reconstruction loss.
        """
        return torch.log(
            2 * np.pi * torch.mean((x - x_reconstruction) ** 2) + 1
        )


class _VariationalEncoder(nn.Module):
    """Wrapper module that outputs sampled z from the variational encoder."""

    def __init__(self, backbone, z_mean_layer, z_log_var_layer, gaussian_sampling):
        super().__init__()
        self.backbone = backbone
        self.z_mean_layer = z_mean_layer
        self.z_log_var_layer = z_log_var_layer
        self.gaussian_sampling = gaussian_sampling

    def forward(self, x):
        h = self.backbone(x)
        z_mean = self.z_mean_layer(h)
        z_log_var = self.z_log_var_layer(h)
        z = self.gaussian_sampling(z_mean, z_log_var)
        return z
