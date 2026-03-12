import logging
import numpy as np
import torch
from .base_model import BaseModel

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


class IdentificationNetwork(BaseModel):

    def __init__(
        self,
        sindy_layer,
        x,
        mu=None,
        scaling="individual",
        second_order=True,
        l_dz: float = 1,
        l_int: float = 0,
        dt=0,
        dtype="float32",
        **kwargs,
    ):
        """
        Identification network using a SINDy layer.

        Parameters
        ----------
        sindy_layer : SindyLayer
            SINDy-compatible layer used to model system dynamics.
        x : array-like
            Example input data used to infer shapes.
        mu : array-like, optional
            Optional control/parameter inputs.
        scaling : str, optional
            Scaling strategy for inputs.
        second_order : bool, optional
            Whether the underlying system is second-order.
        l_dz : float, optional
            Weight for latent derivative loss.
        l_int : float, optional
            Weight for integration consistency loss.
        dt : float, optional
            Time-step for finite differences.
        dtype : str, optional
            Float dtype.
        **kwargs
            Forwarded to the base model.
        """

        # assert that input arguments are valid
        self.assert_arguments(locals())

        self.dtype_ = dtype
        self.torch_dtype = torch.float32 if dtype == "float32" else torch.float64
        super(IdentificationNetwork, self).__init__(**kwargs)

        self._init_to_config(locals())

        self.sindy_layer = sindy_layer
        self.second_order = second_order
        self.reduced_order = sindy_layer.state_dim
        # weighting of the different losses
        self.l_dz, self.l_int = l_dz, l_int
        self.dt = dt
        self.scaling = scaling

        # create the model
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=self.torch_dtype)
        self.x_shape = x.shape[1:]
        if len(self.x_shape) == 1:
            self.flatten, self.unflatten = self.flatten_dummy, self.flatten_dummy
        elif len(self.x_shape) == 2:
            self.flatten, self.unflatten = self.flatten3d, self.unflatten3d

    def compile(
        self,
        optimizer=None,
        loss=None,
        sindy_optimizer=None,
        **kwargs,
    ):
        """
        Configure optimizer for training.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer, optional
            Optimizer for the SINDy parameters. Defaults to Adam(lr=1e-3).
        loss : callable, optional
            Unused, for API compatibility.
        sindy_optimizer : optional
            Unused for this model (single optimizer suffices).
        """
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.sindy_layer.parameters(), lr=1e-3)
        elif isinstance(optimizer, torch.optim.Optimizer):
            self.optimizer = optimizer
        else:
            self.optimizer = optimizer(self.sindy_layer.parameters())

    def _train_step(self, inputs):
        """
        Perform one training step.

        Parameters
        ----------
        inputs : list
            Input data.

        Returns
        -------
        dict
            Dictionary of loss values.
        """
        return self.build_loss(inputs)

    def _eval_step(self, inputs):
        """
        Perform one evaluation step.

        Parameters
        ----------
        inputs : list
            Input data.

        Returns
        -------
        dict
            Dictionary of loss values.
        """
        x, dx_dt, dx_ddt, x_int, dx_int, mu, mu_int = self.split_inputs(inputs)
        if self.second_order:
            losses = self._get_loss_2nd_eval(x, dx_dt, dx_ddt, mu)
        else:
            losses = self._get_loss_eval(x, dx_dt, mu)
        return losses

    def build_loss(self, inputs):
        """
        Compute training loss and apply optimizer steps.

        Parameters
        ----------
        inputs : list
            Input data.

        Returns
        -------
        dict
            Dictionary with loss components.
        """
        x, dx_dt, dx_ddt, x_int, dx_int, mu, mu_int = self.split_inputs(inputs)

        self.optimizer.zero_grad()

        if self.second_order:
            losses = self.get_loss_2nd(x, dx_dt, dx_ddt, mu, x_int, dx_int, mu_int)
        else:
            losses = self.get_loss(x, dx_dt, mu, x_int, mu_int)

        losses["loss"].backward()
        self.optimizer.step()

        return {k: v.detach() for k, v in losses.items()}

    def get_loss(self, z, dz_dt, mu, z_int=None, mu_int=None):
        """
        Calculate loss for first order system.

        Parameters
        ----------
        z : torch.Tensor
            State.
        dz_dt : torch.Tensor
            Time derivative of state.
        mu : torch.Tensor, optional
            Control input.
        z_int, mu_int : torch.Tensor, optional
            Integration data.

        Returns
        -------
        dict
            Dictionary of losses.
        """
        losses = dict(loss=torch.tensor(0.0, dtype=self.torch_dtype, device=z.device))

        z = z.to(dtype=self.torch_dtype)
        dz_dt = dz_dt.to(dtype=self.torch_dtype).unsqueeze(-1)

        # sindy approximation of the time derivative of the latent variable
        sindy_pred, sindy_mean, sindy_log_var = self.evaluate_sindy_layer(z, None, mu)
        dz_dt_sindy = sindy_pred.unsqueeze(-1)

        # SINDy consistency loss
        if self.l_int and z_int is not None:
            int_loss = self.get_int_loss([z_int, mu_int])
            losses["int"] = int_loss
            losses["loss"] = losses["loss"] + int_loss

        # calculate losses
        reg_loss = self.sindy_layer.regularization_loss()
        dz_loss = self.l_dz * torch.mean(
            (torch.cat([dz_dt], dim=1) - torch.cat([dz_dt_sindy], dim=1)) ** 2
        )
        losses["reg"] = reg_loss
        losses["dz"] = dz_loss
        losses["loss"] = losses["loss"] + dz_loss + reg_loss

        # calculate kl divergence for variational sindy
        if sindy_mean is not None:
            kl_loss_sindy = self.sindy_layer.kl_loss(sindy_mean, sindy_log_var)
            losses["kl_sindy"] = kl_loss_sindy
            losses["loss"] = losses["loss"] + kl_loss_sindy

        return losses

    def _get_loss_eval(self, z, dz_dt, mu):
        """Evaluation loss for first order."""
        z = z.to(dtype=self.torch_dtype)
        dz_dt = dz_dt.to(dtype=self.torch_dtype).unsqueeze(-1)
        with torch.no_grad():
            sindy_pred = self.sindy_layer(
                torch.cat([z, mu], dim=1) if mu is not None else z
            )
        if isinstance(sindy_pred, list):
            sindy_pred = sindy_pred[0]
        dz_dt_sindy = sindy_pred.unsqueeze(-1)
        dz_loss = self.l_dz * torch.mean((dz_dt - dz_dt_sindy) ** 2)
        return dict(loss=dz_loss, dz=dz_loss)

    def get_loss_2nd(
        self, z, dz_dt, dz_ddt, mu, z_int=None, dz_dt_int=None, mu_int=None
    ):
        """
        Calculate loss for second order system.

        Parameters
        ----------
        z, dz_dt, dz_ddt : torch.Tensor
            State and derivatives.
        mu : torch.Tensor, optional
            Control input.
        z_int, dz_dt_int, mu_int : torch.Tensor, optional
            Integration data.

        Returns
        -------
        dict
            Dictionary of losses.
        """
        losses = dict(loss=torch.tensor(0.0, dtype=self.torch_dtype, device=z.device))

        z = z.to(dtype=self.torch_dtype)
        dz_dt = dz_dt.to(dtype=self.torch_dtype).unsqueeze(-1)
        dz_ddt = dz_ddt.to(dtype=self.torch_dtype).unsqueeze(-1)

        # sindy approximation of the time derivative of the latent variable
        sindy_pred, sindy_mean, sindy_log_var = self.evaluate_sindy_layer(z, dz_dt, mu)
        dz_dt_sindy = sindy_pred[:, : self.reduced_order].unsqueeze(-1)
        dz_ddt_sindy = sindy_pred[:, self.reduced_order :].unsqueeze(-1)

        # SINDy consistency loss
        if self.l_int and z_int is not None:
            int_loss = self.get_int_loss([z_int, dz_dt_int, dz_dt_int, mu_int])
            losses["int"] = int_loss
            losses["loss"] = losses["loss"] + int_loss

        # calculate kl divergence for variational sindy
        if sindy_mean is not None:
            kl_loss_sindy = self.sindy_layer.kl_loss(sindy_mean, sindy_log_var)
            losses["kl_sindy"] = kl_loss_sindy
            losses["loss"] = losses["loss"] + kl_loss_sindy

        # calculate losses
        reg_loss = self.sindy_layer.regularization_loss()
        dz_loss = self.l_dz * torch.mean(
            (torch.cat([dz_dt, dz_ddt], dim=1) - torch.cat([dz_dt_sindy, dz_ddt_sindy], dim=1)) ** 2
        )

        losses["loss"] = losses["loss"] + dz_loss + reg_loss
        losses["reg"] = reg_loss
        losses["dz"] = dz_loss

        return losses

    def _get_loss_2nd_eval(self, z, dz_dt, dz_ddt, mu):
        """Evaluation loss for second order."""
        z = z.to(dtype=self.torch_dtype)
        dz_dt = dz_dt.to(dtype=self.torch_dtype).unsqueeze(-1)
        dz_ddt = dz_ddt.to(dtype=self.torch_dtype).unsqueeze(-1)
        with torch.no_grad():
            sindy_input = torch.cat([z, dz_dt.reshape(-1, dz_dt.shape[1])], dim=1)
            if mu is not None:
                sindy_input = torch.cat([sindy_input, mu], dim=1)
            sindy_pred = self.sindy_layer(sindy_input)
        if isinstance(sindy_pred, list):
            sindy_pred = sindy_pred[0]
        dz_dt_sindy = sindy_pred[:, : self.reduced_order].unsqueeze(-1)
        dz_ddt_sindy = sindy_pred[:, self.reduced_order :].unsqueeze(-1)
        dz_loss = self.l_dz * torch.mean(
            (torch.cat([dz_dt, dz_ddt], dim=1) - torch.cat([dz_dt_sindy, dz_ddt_sindy], dim=1)) ** 2
        )
        return dict(loss=dz_loss, dz=dz_loss)
