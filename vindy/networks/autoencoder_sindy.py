import numpy as np
import logging
import torch
import torch.nn as nn
from .base_model import BaseModel
from vindy.utils.jacobian import batch_jacobian, batch_hessian

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


# Map activation name strings to PyTorch activation modules
_ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "selu": nn.SELU,
    "elu": nn.ELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
    "linear": nn.Identity,
    "gelu": nn.GELU,
    "swish": nn.SiLU,
    "silu": nn.SiLU,
}


def _get_activation(activation):
    """Return an nn.Module activation from a string or callable."""
    if isinstance(activation, str):
        act_cls = _ACTIVATION_MAP.get(activation.lower())
        if act_cls is None:
            raise ValueError(f"Unknown activation: {activation}. "
                             f"Available: {list(_ACTIVATION_MAP.keys())}")
        return act_cls()
    if isinstance(activation, nn.Module):
        return activation
    raise TypeError(f"activation must be a string or nn.Module, got {type(activation)}")


class AutoencoderSindy(BaseModel):

    def __init__(
        self,
        sindy_layer,
        reduced_order,
        x,
        mu=None,
        scaling="individual",
        layer_sizes=None,
        activation="selu",
        second_order=True,
        l1: float = 0,
        l2: float = 0,
        l_rec: float = 1,
        l_dz: float = 1,
        l_dx: float = 1,
        l_int: float = 0,
        dt=0,
        dtype="float32",
        **kwargs,
    ):
        """
        Autoencoder with SINDy dynamics in the latent space.

        Parameters
        ----------
        sindy_layer : SindyLayer
            Instance of a SINDy-compatible layer.
        reduced_order : int
            Dimensionality of the latent space.
        x : array-like
            Example input data used to infer shapes and build the model.
        mu : array-like, optional
            Optional parameter/control inputs.
        scaling : str, optional
            Method used to scale inputs before encoding.
        layer_sizes : list of int, optional
            Hidden layer sizes for the encoder/decoder networks.
        activation : str or callable, optional
            Activation function for encoder/decoder hidden layers.
        second_order : bool, optional
            If True, the model treats dynamics as second-order.
        l1, l2 : float, optional
            Kernel regularization coefficients for encoder/decoder.
        l_rec, l_dz, l_dx, l_int : float, optional
            Weights for different loss components.
        dt : float, optional
            Time-step used for integration loss.
        dtype : str, optional
            Floating point precision.
        **kwargs
            Additional keyword arguments.
        """

        # set default layer sizes to avoid mutable default argument
        if layer_sizes is None:
            layer_sizes = [10, 10, 10]

        # assert that input arguments are valid
        self.assert_arguments(locals())

        self.dtype_ = dtype
        self.torch_dtype = torch.float32 if dtype == "float32" else torch.float64
        super(AutoencoderSindy, self).__init__(**kwargs)

        self._init_to_config(locals())

        self.sindy_layer = sindy_layer
        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.reduced_order = reduced_order
        self.second_order = second_order
        self.scaling = scaling
        # weighting of the different losses
        self.l_rec, self.l_dz, self.l_dx, self.l_int = l_rec, l_dz, l_dx, l_int
        self.dt = dt

        # kernel regularization weights for encoder/decoder
        self.l1, self.l2 = l1, l2

        # create the model
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=self.torch_dtype)
        self.x_shape = x.shape[1:]
        if len(self.x_shape) == 1:
            self.flatten, self.unflatten = self.flatten_dummy, self.flatten_dummy
        elif len(self.x_shape) == 2:
            self.flatten, self.unflatten = self.flatten3d, self.unflatten3d

        x = self.flatten(x)
        self.build_model(x, mu)

    def assert_arguments(self, arguments):
        """
        Validate initialization arguments.

        Parameters
        ----------
        arguments : dict
            Mapping of argument names to values.
        """
        # base class asserts
        super(AutoencoderSindy, self).assert_arguments(arguments)
        # additional asserts for the autoencoder
        assert isinstance(
            arguments["reduced_order"], int
        ), "reduced_order must be an integer"
        assert arguments["scaling"] in self._scaling_methods, (
            f"scaling must be one of " f"{self._scaling_methods}"
        )
        # network architecture
        assert isinstance(
            arguments["layer_sizes"], list
        ), "layer_sizes must be a list of integers"
        for layer_size in arguments["layer_sizes"]:
            assert isinstance(layer_size, int), "layer_sizes must be a list of integers"
        # loss weights
        for scale_factor in ["l1", "l2", "l_rec", "l_dz", "l_dx", "l_int"]:
            assert type(arguments[scale_factor]) in (
                float,
                int,
            ), f"{scale_factor} must be of type int/float"

    def compile(
        self,
        optimizer=None,
        loss=None,
        sindy_optimizer=None,
        **kwargs,
    ):
        """
        Configure optimizers for training.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer class or instance, optional
            Optimizer for the autoencoder parameters. Defaults to Adam(lr=1e-3).
        loss : callable, optional
            Loss function (unused, kept for API compatibility).
        sindy_optimizer : torch.optim.Optimizer class or instance, optional
            Optimizer for the SINDy parameters. If None, uses same config as main optimizer.
        """
        if optimizer is None:
            ae_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
            self.ae_optimizer = torch.optim.Adam(ae_params, lr=1e-3)
        elif isinstance(optimizer, torch.optim.Optimizer):
            self.ae_optimizer = optimizer
        else:
            # Assume it's a partial or factory
            ae_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
            self.ae_optimizer = optimizer(ae_params)

        if sindy_optimizer is None:
            sindy_params = list(self.sindy_layer.parameters())
            self.sindy_optimizer = torch.optim.Adam(sindy_params, lr=1e-3)
        elif isinstance(sindy_optimizer, torch.optim.Optimizer):
            self.sindy_optimizer = sindy_optimizer
        else:
            sindy_params = list(self.sindy_layer.parameters())
            self.sindy_optimizer = sindy_optimizer(sindy_params)

    @staticmethod
    def reconstruction_loss(x, x_pred):
        """
        Calculate the reconstruction loss as mean squared error.

        Parameters
        ----------
        x : torch.Tensor
            Ground-truth inputs.
        x_pred : torch.Tensor
            Reconstructed inputs.

        Returns
        -------
        torch.Tensor
            Mean squared error between x and x_pred.
        """
        return torch.mean((x - x_pred) ** 2)

    def build_model(self, x, mu):
        """
        Assemble the encoder, decoder and SINDy sub-models.

        Parameters
        ----------
        x : torch.Tensor
            Example input used to determine shapes.
        mu : array-like, optional
            Parameter inputs (used for shape inference only).
        """
        input_dim = x.shape[1]
        self.build_encoder(input_dim)
        self.build_decoder(input_dim)

    def build_encoder(self, input_dim):
        """
        Build a fully connected encoder.

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
        layers.append(nn.Linear(prev_dim, self.reduced_order))
        self.encoder = nn.Sequential(*layers)

    def build_decoder(self, output_dim):
        """
        Build a fully connected decoder with reversed layer sizes.

        Parameters
        ----------
        output_dim : int
            Output dimension (matches input dimension).
        """
        layers = []
        prev_dim = self.reduced_order
        for n_neurons in reversed(self.layer_sizes):
            layers.append(nn.Linear(prev_dim, n_neurons))
            layers.append(_get_activation(self.activation_name))
            prev_dim = n_neurons
        layers.append(nn.Linear(prev_dim, output_dim))
        self.decoder = nn.Sequential(*layers)

    def encoder_regularization_loss(self):
        """
        Compute L1/L2 regularization loss for encoder and decoder parameters.

        Returns
        -------
        torch.Tensor
            Regularization loss.
        """
        reg = torch.tensor(0.0, dtype=self.torch_dtype, device=next(self.parameters()).device)
        if self.l1 == 0 and self.l2 == 0:
            return reg
        for module in [self.encoder, self.decoder]:
            for param in module.parameters():
                if self.l1 > 0:
                    reg = reg + self.l1 * torch.sum(torch.abs(param))
                if self.l2 > 0:
                    reg = reg + self.l2 * torch.sum(param ** 2)
        return reg

    def _train_step(self, inputs):
        """
        Perform one training step.

        Parameters
        ----------
        inputs : list
            Input data for the training step.

        Returns
        -------
        dict
            Dictionary of loss values.
        """
        losses = self.build_loss(inputs)
        return losses

    def _eval_step(self, inputs):
        """
        Perform one evaluation step.

        Parameters
        ----------
        inputs : list
            Input data for the validation step.

        Returns
        -------
        dict
            Dictionary of loss values.
        """
        # second order systems dx_ddt = f(x, dx_dt, mu)
        x, dx_dt, dx_ddt, x_int, dx_int, mu, mu_int = self.split_inputs(inputs)

        # only perform reconstruction if no identification loss is used
        if self.l_dx == 0 and self.l_dz == 0:
            losses = self._get_loss_rec(x)
        elif self.second_order:
            losses = self._get_loss_2nd_eval(x, dx_dt, dx_ddt, mu)
        else:
            losses = self._get_loss_eval(x, dx_dt, mu)

        return losses

    def build_loss(self, inputs):
        """
        Build and compute the loss, then update weights.

        Parameters
        ----------
        inputs : list of array-like
            List of input arrays.

        Returns
        -------
        dict
            Dictionary of computed losses.
        """
        # second order systems dx_ddt = f(x, dx_dt, mu)
        x, dx_dt, dx_ddt, x_int, dx_int, mu, mu_int = self.split_inputs(inputs)

        self.ae_optimizer.zero_grad()
        self.sindy_optimizer.zero_grad()

        # only perform reconstruction if no identification loss is used
        if self.l_dx == 0 and self.l_dz == 0:
            losses = self._get_loss_rec(x)
        # calculate loss for second order systems (includes two time derivatives)
        elif self.second_order:
            losses = self.get_loss_2nd(x, dx_dt, dx_ddt, mu, x_int, dx_int, mu_int)
        # calculate loss for first order systems
        else:
            losses = self.get_loss(x, dx_dt, mu, x_int, mu_int)

        losses["loss"].backward()
        self.ae_optimizer.step()
        if self.l_dx > 0 or self.l_dz > 0:
            self.sindy_optimizer.step()

        return {k: v.detach() for k, v in losses.items()}

    def calc_latent_time_derivatives(
        self, x, dx_dt, dx_ddt=None, mean_or_sample="mean"
    ):
        """
        Calculate time derivatives of latent variables given time derivatives of the inputs.

        Parameters
        ----------
        x : array-like
            Full state.
        dx_dt : array-like
            First time derivative of the full state.
        dx_ddt : array-like, optional
            Second time derivative of the full state.
        mean_or_sample : {'mean', 'sample'}, optional
            Whether to use the mean or a sample from the encoder distribution.

        Returns
        -------
        tuple
            ``(z, dz_dt[, dz_ddt])`` as numpy arrays.
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=self.torch_dtype)
        if isinstance(dx_dt, np.ndarray):
            dx_dt = torch.tensor(dx_dt, dtype=self.torch_dtype)
        if dx_ddt is not None and isinstance(dx_ddt, np.ndarray):
            dx_ddt = torch.tensor(dx_ddt, dtype=self.torch_dtype)

        # in case the variables are not vectorized flatten them
        if len(x.shape) > 2:
            if dx_ddt is not None:
                x, dx_dt, dx_ddt = [self.flatten(v) for v in [x, dx_dt, dx_ddt]]
                dx_ddt = dx_ddt.to(dtype=self.torch_dtype).unsqueeze(-1)
            else:
                x, dx_dt = [self.flatten(v) for v in [x, dx_dt]]

        x = x.to(dtype=self.torch_dtype)
        dx_dt = dx_dt.to(dtype=self.torch_dtype).unsqueeze(-1)
        if dx_ddt is not None:
            dx_ddt = dx_ddt.to(dtype=self.torch_dtype)
            if dx_ddt.dim() == 2:
                dx_ddt = dx_ddt.unsqueeze(-1)

        x.requires_grad_(True)

        # forward pass of encoder and time derivative of latent variable
        z = self.encode(x, mean_or_sample=mean_or_sample)
        dz_dx = batch_jacobian(z, x, create_graph=dx_ddt is not None)

        # calculate first time derivative
        dz_dt = dz_dx @ dx_dt
        dz_dt = dz_dt.squeeze(2)

        # calculate second time derivative if needed
        if dx_ddt is not None:
            # We need the Hessian for second-order derivatives
            # dz_ddt = dz_ddx @ dx_dt @ dx_dt + dz_dx @ dx_ddt
            dz_ddx = batch_hessian(dz_dx, x, create_graph=False)

            dz_ddt = (
                torch.squeeze(dz_ddx @ dx_dt.unsqueeze(1), dim=3) @ dx_dt
                + dz_dx @ dx_ddt
            )
            dz_ddt = dz_ddt.squeeze(2)
            return z.detach().cpu().numpy(), dz_dt.detach().cpu().numpy(), dz_ddt.detach().cpu().numpy()
        else:
            return z.detach().cpu().numpy(), dz_dt.detach().cpu().numpy()

    def _training_encoding(self, x, losses):
        """
        Encode input to latent representation during training.

        Parameters
        ----------
        x : torch.Tensor
            Input data.
        losses : dict
            Dictionary to store losses.

        Returns
        -------
        tuple
            (z, losses)
        """
        z = self.encoder(x)
        return z, losses

    def _get_loss_rec(self, x):
        """
        Calculate reconstruction loss of autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            Full state.

        Returns
        -------
        dict
            Dictionary of losses.
        """
        losses = dict(loss=torch.tensor(0.0, dtype=self.torch_dtype, device=x.device))
        z, losses = self._training_encoding(x, losses)
        x_pred = self.decoder(z)

        # calculate losses
        rec_loss = self.l_rec * self.reconstruction_loss(x, x_pred)
        losses["rec"] = rec_loss
        reg_loss = self.encoder_regularization_loss() + self.sindy_layer.regularization_loss()
        losses["reg"] = reg_loss
        losses["loss"] = losses["loss"] + rec_loss + reg_loss

        return losses

    def get_loss(self, x, dx_dt, mu, x_int=None, mu_int=None):
        """
        Calculate loss for first order system.

        Parameters
        ----------
        x : torch.Tensor
            Full state.
        dx_dt : torch.Tensor
            Time derivative of state.
        mu : torch.Tensor, optional
            Control input.
        x_int : torch.Tensor, optional
            Integration state trajectory.
        mu_int : torch.Tensor, optional
            Integration control trajectory.

        Returns
        -------
        dict
            Dictionary of individual losses.
        """
        losses = dict(loss=torch.tensor(0.0, dtype=self.torch_dtype, device=x.device))

        x = x.to(dtype=self.torch_dtype)
        dx_dt = dx_dt.to(dtype=self.torch_dtype).unsqueeze(-1)

        # forward pass of encoder and time derivative of latent variable
        x = x.requires_grad_(True)
        z, losses = self._training_encoding(x, losses)
        dz_dx = batch_jacobian(z, x, create_graph=True)

        # calculate first time derivative of the latent variable by application of the chain rule
        dz_dt = dz_dx @ dx_dt

        # sindy approximation of the time derivative of the latent variable
        sindy_pred, sindy_mean, sindy_log_var = self.evaluate_sindy_layer(z, None, mu)
        dz_dt_sindy = sindy_pred.unsqueeze(-1)

        # forward pass of decoder and time derivative of reconstructed variable
        x_ = self.decoder(z)
        if self.l_dx > 0:
            dx_dz = batch_jacobian(x_, z, create_graph=True)
            # calculate first time derivative of the reconstructed state
            dxf_dt = dx_dz @ dz_dt_sindy
            dx_loss = self.l_dx * torch.mean(
                (torch.cat([dxf_dt], dim=1) - torch.cat([dx_dt], dim=1)) ** 2
            )
            losses["dx"] = dx_loss
            losses["loss"] = losses["loss"] + dx_loss

        # SINDy consistency loss
        if self.l_int and x_int is not None:
            int_loss = self.get_int_loss([x_int, mu_int])
            losses["int"] = int_loss
            losses["loss"] = losses["loss"] + int_loss

        # calculate losses
        reg_loss = self.encoder_regularization_loss() + self.sindy_layer.regularization_loss()
        rec_loss = self.l_rec * self.reconstruction_loss(x, x_)
        dz_loss = torch.log(
            2 * np.pi * torch.mean((dz_dt - dz_dt_sindy) ** 2) + 1
        )

        losses["loss"] = losses["loss"] + rec_loss + dz_loss + reg_loss

        # calculate kl divergence for variational sindy
        if sindy_mean is not None:
            kl_loss_sindy = self.sindy_layer.kl_loss(sindy_mean, sindy_log_var)
            losses["kl_sindy"] = kl_loss_sindy
            losses["loss"] = losses["loss"] + kl_loss_sindy

        losses["reg"] = reg_loss
        losses["rec"] = rec_loss
        losses["dz"] = dz_loss

        return losses

    def _get_loss_eval(self, x, dx_dt, mu):
        """Evaluation-only loss for first order (no gradients through encoder)."""
        with torch.no_grad():
            z = self.encoder(x)
            x_ = self.decoder(z)
        rec_loss = self.l_rec * self.reconstruction_loss(x, x_)
        losses = dict(loss=rec_loss, rec=rec_loss)
        return losses

    def get_loss_2nd(
        self, x, dx_dt, dx_ddt, mu, x_int=None, dx_dt_int=None, mu_int=None
    ):
        """
        Calculate loss for second order system.

        Parameters
        ----------
        x : torch.Tensor
            Full state.
        dx_dt : torch.Tensor
            Time derivative of state.
        dx_ddt : torch.Tensor
            Second time derivative of state.
        mu : torch.Tensor, optional
            Control input.
        x_int, dx_dt_int, mu_int : torch.Tensor, optional
            Integration data.

        Returns
        -------
        dict
            Dictionary of individual losses.
        """
        losses = dict(loss=torch.tensor(0.0, dtype=self.torch_dtype, device=x.device))

        x = x.to(dtype=self.torch_dtype)
        dx_dt = dx_dt.to(dtype=self.torch_dtype).unsqueeze(-1)
        dx_ddt = dx_ddt.to(dtype=self.torch_dtype).unsqueeze(-1)

        # forward pass of encoder and time derivative of latent variable
        x = x.requires_grad_(True)
        z, losses = self._training_encoding(x, losses)
        dz_dx = batch_jacobian(z, x, create_graph=True)

        # Compute Hessian: dz_ddx (batch, m, n, n)
        dz_ddx = batch_hessian(dz_dx, x, create_graph=True)

        # calculate first time derivative
        dz_dt = dz_dx @ dx_dt

        # calculate second time derivative
        dz_ddt = (
            torch.squeeze(dz_ddx @ dx_dt.unsqueeze(1), dim=3) @ dx_dt
            + dz_dx @ dx_ddt
        )

        # sindy approximation
        sindy_pred, sindy_mean, sindy_log_var = self.evaluate_sindy_layer(z, dz_dt, mu)
        dz_dt_sindy = sindy_pred[:, : self.reduced_order].unsqueeze(-1)
        dz_ddt_sindy = sindy_pred[:, self.reduced_order :].unsqueeze(-1)

        # forward pass of decoder and time derivative of reconstructed variable
        x_ = self.decoder(z)
        if self.l_dx > 0:
            dx_dz = batch_jacobian(x_, z, create_graph=True)
            dx_ddz = batch_hessian(dx_dz, z, create_graph=True)

            dxf_dt = dx_dz @ dz_dt_sindy
            dxf_ddt = (
                torch.squeeze(dx_ddz @ dz_dt_sindy.unsqueeze(1), dim=3) @ dz_dt_sindy
            ) + dx_dz @ dz_ddt_sindy

            dx_loss = self.l_dx * torch.mean(
                (torch.cat([dxf_dt, dxf_ddt], dim=1) - torch.cat([dx_dt, dx_ddt], dim=1)) ** 2
            )
            losses["dx"] = dx_loss
            losses["loss"] = losses["loss"] + dx_loss

        # SINDy consistency loss
        if self.l_int and x_int is not None:
            int_loss = self.get_int_loss([x_int, dx_dt_int, mu_int])
            losses["int"] = int_loss
            losses["loss"] = losses["loss"] + int_loss

        # calculate kl divergence for variational sindy
        if sindy_mean is not None:
            kl_loss_sindy = self.sindy_layer.kl_loss(sindy_mean, sindy_log_var)
            losses["kl_sindy"] = kl_loss_sindy
            losses["loss"] = losses["loss"] + kl_loss_sindy

        # calculate losses
        reg_loss = self.encoder_regularization_loss() + self.sindy_layer.regularization_loss()
        rec_loss = self.l_rec * self.reconstruction_loss(x, x_)
        dz_loss = self.l_dz * torch.mean(
            (torch.cat([dz_dt, dz_ddt], dim=1) - torch.cat([dz_dt_sindy, dz_ddt_sindy], dim=1)) ** 2
        )

        losses["loss"] = losses["loss"] + rec_loss + dz_loss + reg_loss
        losses["reg"] = reg_loss
        losses["rec"] = rec_loss
        losses["dz"] = dz_loss

        return losses

    def _get_loss_2nd_eval(self, x, dx_dt, dx_ddt, mu):
        """Evaluation-only loss for second order (no gradients)."""
        with torch.no_grad():
            z = self.encoder(x)
            x_ = self.decoder(z)
        rec_loss = self.l_rec * self.reconstruction_loss(x, x_)
        losses = dict(loss=rec_loss, rec=rec_loss)
        return losses

    def encode(self, x, training=False, mean_or_sample="mean"):
        """
        Encode full state to latent variables.

        Parameters
        ----------
        x : array-like or torch.Tensor
            Full state input.
        training : bool, optional
            Unused, for API compatibility.
        mean_or_sample : {'mean', 'sample'}, optional
            Unused for deterministic encoder.

        Returns
        -------
        torch.Tensor
            Latent representation.
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=self.torch_dtype)
        x = self.flatten(x)
        z = self.encoder(x)
        return z

    def decode(self, z):
        """
        Decode latent variable to full state.

        Parameters
        ----------
        z : torch.Tensor
            Latent variable.

        Returns
        -------
        torch.Tensor
            Reconstructed full state.
        """
        x_rec = self.decoder(z)
        return self.unflatten(x_rec)

    def reconstruct(self, x, _=None):
        """
        Reconstruct full state from inputs.

        Parameters
        ----------
        x : array-like
            Full state input.

        Returns
        -------
        torch.Tensor
            Reconstructed full state.
        """
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec
