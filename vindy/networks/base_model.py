import numpy as np
from abc import ABC, abstractmethod
import os
import logging
import datetime
import pickle
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from vindy.layers import SindyLayer
from vindy.callbacks import CallbackList
import matplotlib.pyplot as plt

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


# abstract base class for autoencoder SINDy models
class BaseModel(nn.Module, ABC):

    def _init_to_config(self, init_locals):
        """
        Save model initialization parameters to config.

        Parameters
        ----------
        init_locals : dict
            Local variables from the __init__ function.
        """
        pass

    def assert_arguments(self, arguments):
        """
        Validate that the arguments passed to the model are valid.

        Parameters
        ----------
        arguments : dict
            All arguments passed to the model.
        """
        # assert that sindy_layer is of correct class
        assert type(arguments["x"]) in (
            np.ndarray,
            torch.Tensor,
        ), "x must be of type np.ndarray or torch.Tensor"
        assert len(arguments["x"].shape) in (2, 3), (
            "x must be of shape (n_samples, n_features) or "
            "(n_samples, n_nodes, n_dofs)"
        )
        if arguments["mu"] is not None:
            assert type(arguments["mu"]) in (
                np.ndarray,
                torch.Tensor,
            ), "mu must be of type np.ndarray or torch.Tensor"
            assert (
                arguments["x"].shape[0] == arguments["mu"].shape[0]
            ), "x and mu must have the same number of samples"

        assert isinstance(arguments["sindy_layer"], SindyLayer), (
            "sindy_layer must be an object of a subclass of " "SindyLayer"
        )
        assert isinstance(
            arguments["second_order"], bool
        ), "second_order must be a boolean"
        # loss weights
        for scale_factor in ["l_dz", "l_int"]:
            assert type(arguments[scale_factor]) in (
                float,
                int,
            ), f"{scale_factor} must be of type int/float"
        assert arguments["dtype"] in [
            "float32",
            "float64",
        ], "dtype must be either float32 or float64"

    def save(self, path: str = None):
        """
        Save the model weights to a given path.

        Parameters
        ----------
        path : str, optional
            Path to the folder where the model should be saved. If None, a default
            path with timestamp is created.
        """
        if path is None:
            path = (
                f"results/saved_models/{self.__class__.__name__}/"
                f'{datetime.datetime.now().strftime("%Y_%m_%d-%H:%M:%S")}/'
            )
        os.makedirs(path, exist_ok=True)
        weights_path = os.path.join(path, "model.pt")
        torch.save(self.state_dict(), weights_path)

    def load(self, path: str):
        """
        Load model weights from the given path.

        Parameters
        ----------
        path : str
            Path to the saved model directory.
        """
        weights_path = os.path.join(path, "model.pt")
        self.load_state_dict(torch.load(weights_path, weights_only=True))

    @staticmethod
    def flatten_dummy(x):
        return x

    def flatten3d(self, x):
        if isinstance(x, torch.Tensor):
            return x.reshape(-1, self.x_shape[0] * self.x_shape[1])
        return x.reshape(-1, self.x_shape[0] * self.x_shape[1])

    def unflatten3d(self, x):
        if isinstance(x, torch.Tensor):
            return x.reshape(-1, self.x_shape[0], self.x_shape[1])
        return x.reshape(-1, self.x_shape[0], self.x_shape[1])

    def print(self, z=None, mu=None, precision=3):
        self.sindy_layer.print(z, mu, precision)

    def sindy_coeffs(self):
        """
        Return the coefficients of the SINDy model.

        Returns
        -------
        array-like
            SINDy coefficient matrix.
        """
        return self.sindy_layer.get_sindy_coeffs()

    @abstractmethod
    def compile(self, optimizer, loss, sindy_optimizer=None, **kwargs):
        """Configure optimizers and loss function for training."""
        pass

    @abstractmethod
    def _train_step(self, batch):
        """Perform one training step. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _eval_step(self, batch):
        """Perform one evaluation step. Must be implemented by subclasses."""
        pass

    def fit(self, x, y=None, validation_data=None, epochs=1, batch_size=32, callbacks=None, verbose=1):
        """
        Train the model.

        Parameters
        ----------
        x : list of array-like
            Training data as a list of tensors/arrays.
        y : array-like, optional
            Target data (unused, for API compatibility).
        validation_data : tuple or list, optional
            Validation data as (x_val, y_val) or list of tensors.
        epochs : int, default=1
            Number of training epochs.
        batch_size : int, default=32
            Batch size for training.
        callbacks : list of Callback, optional
            List of callbacks to run during training.
        verbose : int, default=1
            Verbosity mode (0=silent, 1=progress).

        Returns
        -------
        dict
            Training history mapping metric names to lists of values per epoch.
        """
        # setup callbacks
        cb_list = CallbackList(callbacks or [])
        cb_list.set_model(self)

        # flatten and cast the input
        train_tensors = []
        for x_ in x:
            if isinstance(x_, np.ndarray):
                x_ = torch.tensor(x_, dtype=self.torch_dtype)
            else:
                x_ = x_.to(dtype=self.torch_dtype)
            try:
                train_tensors.append(self.flatten(x_))
            except Exception:
                train_tensors.append(x_)

        # create dataset and dataloader
        dataset = TensorDataset(*train_tensors)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # validation data
        val_dataloader = None
        if validation_data is not None:
            if isinstance(validation_data, tuple):
                val_x, val_y = validation_data
            else:
                val_x = validation_data
            val_tensors = []
            for x_ in val_x:
                if isinstance(x_, np.ndarray):
                    x_ = torch.tensor(x_, dtype=self.torch_dtype)
                else:
                    x_ = x_.to(dtype=self.torch_dtype)
                try:
                    val_tensors.append(self.flatten(x_))
                except Exception:
                    val_tensors.append(x_)
            val_dataset = TensorDataset(*val_tensors)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        history = {}
        cb_list.on_train_begin()

        for epoch in range(epochs):
            cb_list.on_epoch_begin(epoch)

            # Training
            self.train()
            epoch_losses = {}
            n_batches = 0
            for batch in dataloader:
                losses = self._train_step([batch])
                for key, val in losses.items():
                    if isinstance(val, torch.Tensor):
                        val = val.item()
                    if key not in epoch_losses:
                        epoch_losses[key] = 0.0
                    epoch_losses[key] += val
                n_batches += 1

            # Average losses
            for key in epoch_losses:
                epoch_losses[key] /= n_batches

            # Validation
            if val_dataloader is not None:
                self.eval()
                val_losses = {}
                n_val_batches = 0
                with torch.no_grad():
                    for batch in val_dataloader:
                        losses = self._eval_step([batch])
                        for key, val in losses.items():
                            if isinstance(val, torch.Tensor):
                                val = val.item()
                            val_key = f"val_{key}"
                            if val_key not in val_losses:
                                val_losses[val_key] = 0.0
                            val_losses[val_key] += val
                        n_val_batches += 1
                for key in val_losses:
                    val_losses[key] /= n_val_batches
                for key, val in val_losses.items():
                    if key not in history:
                        history[key] = []
                    history[key].append(val)
                epoch_losses.update(val_losses)

            # Logging
            if verbose:
                loss_str = " - ".join(
                    f"{k}: {v:.3E}" for k, v in epoch_losses.items()
                )
                print(f"Epoch {epoch + 1}/{epochs} - {loss_str}")

            cb_list.on_epoch_end(epoch, epoch_losses)

            # Store in history
            for key, val in epoch_losses.items():
                if key not in history:
                    history[key] = []
                history[key].append(val)

        cb_list.on_train_end()

        return history

    def concatenate_sindy_input(self, z, dzdt=None, mu=None):
        """
        Concatenate state, derivative, and parameters for SINDy layer input.

        Parameters
        ----------
        z : torch.Tensor
            Latent state.
        dzdt : torch.Tensor, optional
            Time derivative of latent state.
        mu : torch.Tensor, optional
            Parameters.

        Returns
        -------
        torch.Tensor
            Concatenated input tensor for SINDy layer.
        """
        quantities_to_concatenate = [z]
        if dzdt is not None:
            quantities_to_concatenate.append(dzdt)
        if mu is not None:
            quantities_to_concatenate.append(mu)
        z_sindy = torch.cat(quantities_to_concatenate, dim=-1)
        return z_sindy

    def split_inputs(self, inputs):
        """
        Split the inputs into state, derivative, and parameters.

        Parameters
        ----------
        inputs : list
            Input data containing state and optional derivatives/parameters.

        Returns
        -------
        tuple
            (x, dx_dt, dx_ddt, x_int, dx_int, mu, mu_int) with unpacked components.
        """
        # initialize variables as None
        x, dx_dt, dx_ddt, x_int, dx_int, mu, mu_int = [None] * 7

        # second order systems dx_ddt = f(x, dx_dt, mu)
        if self.second_order:
            if len(inputs[0]) == 7:
                [x, dx_dt, dx_ddt, x_int, dx_int, mu, mu_int] = inputs[0]
            elif len(inputs[0]) == 4:
                [x, dx_dt, dx_ddt, mu] = inputs[0]
            elif len(inputs[0]) == 3:
                # second order system without parameter / arguments
                if inputs[0][0].shape == inputs[0][0].shape:
                    [x, dx_dt, dx_ddt] = inputs[0]

        # first order systems dx_dt = f(x, mu)
        else:
            if len(inputs[0]) == 5:
                [x, dx_dt, x_int, mu, mu_int] = inputs[0]
            # first order system with parameter / arguments
            if len(inputs[0]) == 3:
                [x, dx_dt, mu] = inputs[0]
            # first order system without parameter / arguments
            elif len(inputs[0]) == 2:
                [x, dx_dt] = inputs[0]

        return x, dx_dt, dx_ddt, x_int, dx_int, mu, mu_int

    def get_int_loss(self, inputs):
        """
        Integrate the identified dynamical system and compare to true dynamics.

        Parameters
        ----------
        inputs : list
            Input data containing state trajectories and parameters.

        Returns
        -------
        torch.Tensor
            Integration consistency loss.
        """
        # only evaluate if there is an integration loss
        if len(inputs) == 3:
            x_int, dx_dt_int, mu_int = inputs

            # reshape so all timesteps are in the batch dimension
            dx_dt_int = dx_dt_int.reshape(-1, dx_dt_int.shape[-1])
            dx_dt_int = dx_dt_int.to(dtype=self.torch_dtype).unsqueeze(-1)

            # forward pass of encoder and time derivative of latent variable
            x_int_flat = x_int.reshape(-1, x_int.shape[-1])
            x_int_flat = x_int_flat.requires_grad_(True)

            z_int = self.encoder(x_int_flat)
            from vindy.utils.jacobian import batch_jacobian
            dz_dx_int = batch_jacobian(z_int, x_int_flat, create_graph=True)
            dz_dt_int = dz_dx_int @ dx_dt_int

            # reshape to sequences again
            z_int = z_int.reshape(-1, mu_int.shape[1], self.reduced_order)
            dz_dt_int = dz_dt_int.reshape(-1, mu_int.shape[1], self.reduced_order)

            s = torch.cat([z_int, dz_dt_int], dim=2)

        elif len(inputs) == 2:
            x_int, mu_int = inputs
            # reshape for encoding
            x_int_flat = x_int.reshape(-1, x_int.shape[-1])
            z_int = self.encode(x_int_flat)
            # reshape to sequences again
            z_int = z_int.reshape(-1, mu_int.shape[1], self.reduced_order)
            s = z_int
        else:
            x_int = inputs[0]
            # reshape for encoding
            x_int_flat = x_int.reshape(-1, x_int.shape[-1])
            z_int = self.encode(x_int_flat)
            # reshape to sequences again
            z_int = z_int.reshape(-1, x_int.shape[1], self.reduced_order)
            s = z_int

        s_max = torch.amax(torch.abs(s), dim=1)
        sol = s[:, 0, :]
        int_loss = torch.tensor(0.0, dtype=self.torch_dtype, device=s.device)
        total_steps = z_int.shape[1]
        # Runge Kutta 4 integration scheme
        for i in range(1, total_steps):
            sindy_input_i = torch.cat([sol, mu_int[:, i]], dim=1)
            k1 = self.sindy_layer(sindy_input_i)
            k2 = self.sindy_layer(torch.cat([sol + self.dt / 2 * k1, mu_int[:, i]], dim=1))
            k3 = self.sindy_layer(torch.cat([sol + self.dt / 2 * k2, mu_int[:, i]], dim=1))
            k4 = self.sindy_layer(torch.cat([sol + self.dt * k3, mu_int[:, i]], dim=1))
            sol = sol + 1 / 6 * self.dt * (k1 + 2 * k2 + 2 * k3 + k4)
            sol = torch.where(torch.abs(sol) > s_max, s_max, sol)
            int_loss = int_loss + torch.mean(
                (sol[:, : self.reduced_order] - s[:, i, : self.reduced_order]) ** 2
            )

        if torch.isnan(int_loss) or int_loss > 1e0:
            print("Integration loss is NaN. Setting to 0.0.")
            int_loss = torch.tensor(0.0, dtype=self.torch_dtype, device=s.device)
        else:
            int_loss = self.l_int * int_loss / total_steps
        return int_loss

    def evaluate_sindy_layer(self, z, dz_dt, mu):
        """
        Evaluate the SINDy layer.

        Parameters
        ----------
        z : torch.Tensor
            Latent variable.
        dz_dt : torch.Tensor, optional
            Time derivative of the latent variable (only for second order models).
        mu : torch.Tensor, optional
            Parameters.

        Returns
        -------
        tuple
            (sindy_pred, sindy_mean, sindy_log_var) - prediction and optional
            variational parameters.
        """
        # sindy approximation of the time derivative of the latent variable
        if mu is None:
            if self.second_order:
                sindy_input = torch.cat([z, dz_dt.reshape(-1, dz_dt.shape[1])], dim=1)
            else:
                sindy_input = z
        else:
            if self.second_order:
                sindy_input = torch.cat([z, dz_dt.reshape(-1, dz_dt.shape[1]), mu], dim=1)
            else:
                sindy_input = torch.cat([z, mu], dim=1)

        sindy_pred_ = self.sindy_layer(sindy_input)

        if isinstance(sindy_pred_, list):
            sindy_pred = sindy_pred_[0]
            sindy_mean = sindy_pred_[1]
            sindy_log_var = sindy_pred_[2]
        else:
            sindy_pred = sindy_pred_
            sindy_mean = None
            sindy_log_var = None

        return sindy_pred, sindy_mean, sindy_log_var

    def vis_modes(self, x, n_modes=3):
        """
        Visualize the reconstruction of the reduced coefficients.

        Parameters
        ----------
        x : array-like
            Input data.
        n_modes : int, default=3
            Number of modes to visualize.
        """
        n_modes = min(n_modes, x.shape[1])
        if isinstance(x, np.ndarray):
            x_t = torch.tensor(x, dtype=self.torch_dtype)
        else:
            x_t = x.to(dtype=self.torch_dtype)
        with torch.no_grad():
            z = self.encoder(self.flatten(x_t))
            x_rec = self.decoder(z)
        z = z.cpu().numpy()
        x_rec = x_rec.cpu().numpy()
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        # visualize modes in subplots
        fig, axs = plt.subplots(n_modes + self.reduced_order, 1, figsize=(10, 10))
        # plot latent variables
        for i in range(self.reduced_order):
            axs[i].plot(z[:, i], color="k")
            axs[i].set_title(f"z_{i}")
        for i in range(n_modes):
            axs[i + self.reduced_order].plot(x[:, i])
            axs[i + self.reduced_order].plot(x_rec[:, i])
            axs[i + self.reduced_order].set_title("Mode {}".format(i))
        # add legend
        axs[i].legend(["Original", "Reconstructed"])
        plt.show()

    def integrate(self, z0, t, mu=None, method="RK45", sindy_fcn=None):
        """
        Integrate the model using scipy.integrate.solve_ivp.

        Parameters
        ----------
        z0 : array-like
            Initial state.
        t : array-like
            Time points to evaluate the solution at.
        mu : array-like or callable, optional
            Parameters to use in the model.
        method : str, default='RK45'
            Integration method to use.
        sindy_fcn : callable, optional
            Custom SINDy function.

        Returns
        -------
        OdeResult
            Solution from scipy.integrate.solve_ivp.
        """
        return self.sindy_layer.integrate(z0, t, mu, method, sindy_fcn)

    @property
    def _scaling_methods(self):
        return ["individual", "global", "individual_sqrt", "none"]

    def define_scaling(self, x):
        """
        Define the scaling factor for given training data.

        Parameters
        ----------
        x : torch.Tensor
            Training data.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=self.torch_dtype)
        # scale the data if requested
        if self.scaling == "individual":
            scale_factor = 1 / torch.amax(torch.abs(x), dim=0)
            # replace inf with ones to avoid division by zero
            scale_factor = torch.where(
                torch.isinf(scale_factor),
                torch.ones_like(scale_factor),
                scale_factor,
            )
        elif self.scaling == "individual_sqrt":
            scale_factor = 1 / torch.sqrt(torch.amax(torch.abs(x), dim=0))
            scale_factor = torch.where(
                torch.isinf(scale_factor),
                torch.ones_like(scale_factor),
                scale_factor,
            )
        elif self.scaling == "global":
            scale_factor = torch.tensor(1.0 / torch.amax(torch.abs(x)).item(),
                                        dtype=self.torch_dtype)
        else:
            scale_factor = torch.tensor(1.0, dtype=self.torch_dtype)

        self.register_buffer('scale_factor', scale_factor)

    def scale(self, x):
        # scale the data
        x = x * self.scale_factor
        return x

    def rescale(self, x):
        # rescale the data
        x = x / self.scale_factor
        return x
