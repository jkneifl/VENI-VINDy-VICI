import numpy as np
import scipy
from abc import ABC
import os
import logging
import datetime
import pickle
import tensorflow as tf
import keras
from vindy.layers import SindyLayer
import matplotlib.pyplot as plt

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


# abstract base class for autoencoder SINDy models
class BaseModel(tf.keras.Model, ABC):
    """
    Abstract base class for autoencoder SINDy models.
    """

    def _init_to_config(self, init_locals):
        """
        Save the parameters with which the model was initialized except for the data itself.

        Parameters
        ----------
        init_locals : dict
            Local variables from the `__init__` function.
        """

        # ! broken due to tf (can't copy layer https://github.com/keras-team/keras/issues/19383)
        # todo: check this again
        # ! commented out for now

        # sig = inspect.signature(self.__init__)
        # keys = [param.name for param in sig.parameters.values()]
        # values = [init_locals[name] for name in keys]
        # init_dict = dict(zip(keys, values))

        # deep copy kwargs so we can manipulate them without changing the original
        # if "kwargs" in init_dict.keys():
        #     init_dict["kwargs"] = copy.deepcopy(init_dict["kwargs"])
        # # we don't want to save the data itself
        # init_dict["x"] = None
        # init_dict["mu"] = None
        # if "x" in init_dict["kwargs"].keys():
        #     init_dict["kwargs"]["x"] = None
        # if "mu" in init_dict["kwargs"].keys():
        #     init_dict["kwargs"]["mu"] = None
        # self.config = init_dict

    def assert_arguments(self, arguments):
        """
        Assert that the arguments passed to the model are valid.

        Parameters
        ----------
        arguments : dict
            All arguments passed to the model.

        Raises
        ------
        AssertionError
            If any of the arguments are invalid.
        """
        # assert that sindy_layer is of correct class
        assert type(arguments["x"]) in (
            np.ndarray,
            tf.Tensor,
        ), "x must be of type np.ndarray or tf.Tensor"
        assert len(arguments["x"].shape) in (2, 3), (
            "x must be of shape (n_samples, n_features) or "
            "(n_samples, n_nodes, n_dofs)"
        )
        if arguments["mu"] is not None:
            assert type(arguments["mu"]) in (
                np.ndarray,
                tf.Tensor,
            ), "mu must be of type np.ndarray or tf.Tensor"
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
            tf.float32,
            tf.float64,
        ], "dtype must be either float32 or float64"

    def save(self, path: str = None):
        """
        Save the model weights and configuration to a given path.

        Parameters
        ----------
        path : str, optional
            Path to the folder where the model should be saved (default is None).
        """
        if path is None:
            path = (
                f"results/saved_models/{self.__class__.__name__}/"
                f'{datetime.datetime.now().strftime("%Y_%m_%d-%H:%M:%S")}/'
            )
        weights_path = os.path.join(path, "weights/")
        model_path = os.path.join(path, f"model_config.pkl")
        self.save_weights(weights_path)
        # self.config['class_name'] = self.__class__.__name__
        with open(model_path, "wb") as outp:  # Overwrites any existing file.
            # pickle.dump(self.config, outp)
            pickle.dump(self.config["kwargs"]["sindy_layer"], outp)

    @staticmethod
    def load(
        aesindy,
        x=None,
        mu=None,
        mask=None,
        fixed_coeffs=None,
        path: str = None,
        kwargs_overwrite: dict = {},
    ):
        """
        Load the model from the given path.

        Parameters
        ----------
        aesindy : class
            The class of the model to be loaded.
        x : array-like, optional
            Data needed to initialize the model (default is None).
        mu : array-like, optional
            Parameters used to create the model the first time (default is None).
        mask : array-like, optional
            Mask for the model (default is None).
        fixed_coeffs : array-like, optional
            Fixed coefficients for the model (default is None).
        path : str, optional
            Path to the model (default is None).
        kwargs_overwrite : dict, optional
            Additional kwargs to overwrite the config (default is {}).

        Returns
        -------
        BaseModel
            Loaded model.
        """

        weights_path = os.path.join(path, "weights/")
        model_path = os.path.join(path, f"model_config.pkl")
        with open(model_path, "rb") as file:  # Overwrites any existing file.
            init_dict = pickle.load(file)
        init_dict["x"] = x
        init_dict["mu"] = mu
        init_dict["mask"] = mask
        init_dict["fixed_coeffs"] = fixed_coeffs
        kwargs = init_dict.pop("kwargs")
        # overwrite the kwargs with values from kwargs_overwrite
        kwargs.update(kwargs_overwrite)
        if "x" in kwargs:
            kwargs.pop("x")
        if "mu" in kwargs:
            kwargs.pop("mu")
        if "fixed_coeffs" in kwargs:
            kwargs.pop("fixed_coeffs")
        if "mask" in kwargs:
            kwargs.pop("mask")
        loaded_model = aesindy(**init_dict, **kwargs)
        loaded_model.load_weights(weights_path)

        return loaded_model

    @staticmethod
    def flatten_dummy(x):
        return x

    def flatten3d(self, x):
        return tf.reshape(x, [-1, self.x_shape[0] * self.x_shape[1]])

    def unflatten3d(self, x):
        return tf.reshape(x, [-1, self.x_shape[0], self.x_shape[1]])

    def rhs_(self, t, z):
        if len(z.shape) == 1:
            z = tf.expand_dims(z, 0)
        return self.sindy(z)  # .numpy()

    def print(self, z=None, mu=None, precision=3):
        for layer in self.sindy.layers:
            if isinstance(layer, SindyLayer):
                layer.print(z, mu, precision)

    def integrate(self, z0, t, mu=None, method="RK45"):
        """
        Integrate the model using `scipy.integrate.solve_ivp`.

        Parameters
        ----------
        z0 : array-like
            Initial state.
        t : array-like
            Time points to evaluate the solution at.
        mu : array-like or callable, optional
            Parameters to use in the model (default is None).
        method : str, optional
            Integration method to use (default is "RK45").

        Returns
        -------
        scipy.integrate.OdeResult
            Solution of the integration.
        """
        if mu is not None:
            if not callable(mu):
                mu_fun = scipy.interpolate.interp1d(
                    t, mu, axis=0, kind="cubic", fill_value="extrapolate"
                )
                t = t[:-1]
                logging.warning(
                    "Last time point dropped in simulation because "
                    "interpolation of control input was used. To avoid "
                    "this, pass in a callable for u."
                )
            else:
                mu_fun = mu

            def rhs(t, x):
                return self.rhs_(t, np.concatenate([x, mu_fun(t)], axis=0))[0]

        else:

            def rhs(t, x):
                return self.rhs_(t, x)[0]

        # z = tf.Tensor(z0[np.newaxis], dtype=self.dtype_)
        # tensorflow tensor form numpy
        z0 = tf.cast(z0, dtype=self.dtype_)
        # %timeit self.sindy(z, dtype=self.dtype_)
        sol = scipy.integrate.solve_ivp(
            rhs,
            t_span=[t[0], t[-1]],
            t_eval=t,
            y0=z0,
            method=method,
            # rtol=1e-6
        )
        return sol

    def sindy_coeffs(self):
        """
        Return the coefficients of the SINDy model.

        Returns
        -------
        array-like
            Coefficients of the SINDy model.
        """
        return self.sindy_layer.get_sindy_coeffs()

    def fit(self, x, y=None, validation_data=None, **kwargs):
        """
        Wrapper for the fit function of the keras model to flatten the data if necessary.

        Parameters
        ----------
        x : array-like
            Training data.
        y : array-like, optional
            Target data (default is None).
        validation_data : tuple or array-like, optional
            Data on which to evaluate the loss and any model metrics at the end of each epoch (default is None).
        kwargs : dict
            Additional arguments for the fit function.

        Returns
        -------
        History
            A `History` object. Its `History.history` attribute is a record of training loss values and metrics values at successive epochs.
        """
        # flatten and cast the input
        for i, x_ in enumerate(x):
            x_ = tf.cast(x_, self.dtype_)
            try:
                x[i] = self.flatten(x_)
            except tf.errors.InvalidArgumentError:
                x[i] = x_
        if validation_data is not None:
            if isinstance(validation_data, tuple):
                validation_x, validation_y = validation_data
            else:
                validation_x = validation_data
                validation_y = None
            for i, x_ in enumerate(validation_x):
                try:
                    validation_x[i] = x_
                except tf.errors.InvalidArgumentError:
                    validation_x[i] = x_
            validation_data = (validation_x, validation_y)
        return super(BaseModel, self).fit(
            x, y, validation_data=validation_data, **kwargs
        )

    def concatenate_sindy_input(self, z, dzdt=None, mu=None):
        """
        Concatenate the state, its derivative, and the parameters to the input of the SINDy layer.

        Parameters
        ----------
        z : array-like
            State.
        dzdt : array-like, optional
            Derivative of the state (default is None).
        mu : array-like, optional
            Parameters (default is None).

        Returns
        -------
        array-like
            Concatenated input for the SINDy layer.
        """
        quantities_to_concatenate = [z]
        if dzdt is not None:
            quantities_to_concatenate.append(dzdt)
        if mu is not None:
            quantities_to_concatenate.append(mu)
        # concatenate the state and the parameters to the input of the SINDy layer
        z_sindy = keras.layers.Concatenate(axis=1)(quantities_to_concatenate)
        return z_sindy

    def build_sindy(self, z, mu):
        """
        Build the model for the forward pass of the SINDy layer.

        Parameters
        ----------
        z : array-like
            Latent state.
        mu : array-like
            Parameters.

        Returns
        -------
        tuple
            Concatenated input for the SINDy layer and the output of the SINDy layer.
        """
        # sindy
        dzdt = None
        if self.second_order:
            dzdt = tf.keras.Input(shape=(self.reduced_order,))
        # in case we have parameters we add them to the input of the SINDy layer
        if mu is not None:
            mu = tf.keras.Input(shape=(mu.shape[1],))
        # concatenate the state and the parameters to the input of the SINDy layer
        z_sindy = self.concatenate_sindy_input(z, dzdt, mu)
        z_dot = self.sindy_layer(z_sindy)
        return z_sindy, z_dot

    def split_inputs(self, inputs):
        """
        Split the inputs into the state, its derivative, and the parameters (if present).

        Parameters
        ----------
        inputs : list
            List of inputs.

        Returns
        -------
        tuple
            Split inputs.
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

    @tf.function
    def train_step(self, inputs):
        """
        Perform one training step.

        Parameters
        ----------
        inputs : list
            List of inputs.

        Returns
        -------
        dict
            Dictionary of losses.
        """

        # perform forwad pass, calculate loss and update weights
        losses = self.build_loss(inputs)

        # update loss tracker
        for key, loss_tracker in self.loss_trackers.items():
            loss_tracker.update_state(losses[key])

        # # sort losses for logging
        # losses = {key: losses[key] for key in self.loss_trackers.keys()}

        return losses

    @tf.function
    def test_step(self, inputs):
        """
        Perform one test step.

        Parameters
        ----------
        inputs : list
            List of inputs.

        Returns
        -------
        dict
            Dictionary of losses.
        """
        # perform forwad pass, calculate loss for validation data
        losses = self.build_loss(inputs)

        # update loss tracker
        for key, loss_tracker in self.loss_trackers.items():
            loss_tracker.update_state(losses[key])

        return losses

    @tf.function
    def get_int_loss(self, inputs):
        """
        Integrate the identified dynamical system and compare the result to the true dynamics.

        Parameters
        ----------
        inputs : list
            List of inputs.

        Returns
        -------
        float
            Integration loss.
        """
        # todo: use tensorflow's built in ode solver

        # only evaluate if there is an integration loss
        if len(inputs) == 3:
            x_int, dx_dt_int, mu_int = inputs

            # reshape so all timesteps are in the batch dimension
            dx_dt_int = tf.reshape(dx_dt_int, shape=(-1, dx_dt_int.shape[-1]))
            dx_dt_int = tf.expand_dims(tf.cast(dx_dt_int, dtype=self.dtype_), axis=-1)

            # forward pass of encoder and time derivative of latent variable
            x_int = tf.reshape(x_int, shape=(-1, x_int.shape[-1]))

            with tf.GradientTape() as t12:
                t12.watch(x_int)
                z_int = self.encoder(x_int)
            dz_dx_int = t12.batch_jacobian(z_int, x_int)
            dz_dt_int = dz_dx_int @ dx_dt_int

            # reshape to sequences again
            z_int = tf.reshape(z_int, shape=[-1, mu_int.shape[1], self.reduced_order])
            dz_dt_int = tf.reshape(
                dz_dt_int, shape=[-1, mu_int.shape[1], self.reduced_order]
            )

            s = tf.concat([z_int, dz_dt_int], axis=2)

        elif len(inputs) == 2:
            x_int, mu_int = inputs
            # reshape for encoding
            x_int = tf.reshape(x_int, shape=(-1, x_int.shape[-1]))
            z_int = self.encode(x_int)
            # reshape to sequences again
            z_int = tf.reshape(z_int, shape=[-1, mu_int.shape[1], self.reduced_order])
            s = z_int
        else:
            x_int = inputs[0]

            # reshape for encoding
            x_int = tf.reshape(x_int, shape=(-1, x_int.shape[-1]))
            z_int = self.encode(x_int)
            # reshape to sequences again
            z_int = tf.reshape(
                z_int, shape=[-1, inputs.shape[1], self.reduced_order]
            )  # x_int?
            s = z_int

        s_max = tf.reduce_max(tf.abs(s), axis=1)
        sol = s[:, 0, :]
        int_loss = 0
        total_steps = z_int.shape[1]
        # # # Runge Kutta 4 integration scheme
        for i in range(1, total_steps):
            k1 = self.sindy(tf.concat([sol, mu_int[:, i]], axis=1))
            k2 = self.sindy(tf.concat([sol + self.dt / 2 * k1, mu_int[:, i]], axis=1))
            k3 = self.sindy(tf.concat([sol + self.dt / 2 * k2, mu_int[:, i]], axis=1))
            k4 = self.sindy(tf.concat([sol + self.dt * k3, mu_int[:, i]], axis=1))
            sol = sol + 1 / 6 * self.dt * (k1 + 2 * k2 + 2 * k3 + k4)
            sol = tf.where(tf.abs(sol) > s_max, s_max, sol)
            int_loss += self.compiled_loss(
                sol[:, : self.reduced_order], s[:, i, : self.reduced_order]
            )

        # todo: this is only working in eager execution due to scipy interpolation of the input parameters
        # t = tf.linspace(0.0, self.dt * (total_steps - 1), total_steps)[:, 0]
        #
        # if not callable(mu_int):
        #     mu_fun = scipy.interpolate.interp1d(
        #         t, mu_int, axis=1, kind="cubic", fill_value="extrapolate"
        #     )
        #
        # # tensorflow ide solver
        # def ode_fn(t, y):
        #     # return y * mu_fun(t)[:, :2]
        #     return self.sindy(tf.concat([y, mu_fun(t)], axis=1))
        #
        # results = tfp.math.ode.DormandPrince().solve(
        #     ode_fn,
        #     0.0,
        #     s[:, 0, :],
        #     solution_times=tf.linspace(0.0, self.dt * (total_steps - 1), total_steps)[
        #         :, 0
        #     ],
        # )
        # sol = tf.transpose(results.states, [1, 0, 2])
        # int_loss = tf.reduce_mean(
        #     self.compiled_loss(
        #         tf.reshape(
        #             sol[:, :, self.reduced_order], shape=(-1, self.reduced_order)
        #         ),
        #         tf.reshape(s[:, :, self.reduced_order], shape=(-1, self.reduced_order)),
        #     )
        # )

        if tf.math.is_nan(int_loss) or int_loss > 1e0:
            # tf logging warning
            tf.print("Integration loss is NaN. Setting to 0.0.")
            int_loss = 0.0
        else:
            int_loss = self.l_int * int_loss / total_steps
        return int_loss

    def evaluate_sindy_layer(self, z, dz_dt, mu):
        """
        Evaluate the SINDy layer.

        Parameters
        ----------
        z : array-like
            Latent variable.
        dz_dt : array-like
            Time derivative of the latent variable (only required for second order models).
        mu : array-like
            Parameters.

        Returns
        -------
        tuple
            SINDy prediction, mean, and log variance.
        """
        # sindy approximation of the time derivative of the latent variable
        if mu is None:
            if self.second_order:
                sindy_pred_ = self.sindy_layer(
                    tf.concat([z, tf.reshape(dz_dt, [-1, dz_dt.shape[1]])], axis=1),
                    training=True,
                )
            else:
                sindy_pred_ = self.sindy_layer(z, training=True)
        else:
            if self.second_order:
                sindy_pred_ = self.sindy_layer(
                    tf.concat([z, tf.reshape(dz_dt, [-1, dz_dt.shape[1]]), mu], axis=1),
                    training=True,
                )
            else:
                sindy_pred_ = self.sindy_layer(
                    tf.concat([z, mu], axis=1), training=True
                )

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
        Visualize the reconstruction of the reduced coefficients of the PCA modes.

        Parameters
        ----------
        x : array-like
            Input data.
        n_modes : int, optional
            Number of modes to visualize (default is 3).
        """
        n_modes = min(n_modes, x.shape[1])
        z = self.encoder(self.flatten(x))
        x_rec = self.decoder(z)
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

    # @tf.function
