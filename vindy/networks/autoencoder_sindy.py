import numpy as np
import logging
import tensorflow as tf
from .base_model import BaseModel

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


class AutoencoderSindy(BaseModel):

    def __init__(
        self,
        sindy_layer,
        reduced_order,
        x,
        mu=None,
        scaling="individual",
        layer_sizes=[10, 10, 10],
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
        Initialize the AutoencoderSindy model.

        Parameters
        ----------
        sindy_layer : SindyLayer
            Layer to identify the governing equations of the latent dynamics.
        reduced_order : int
            Order of the reduced model.
        x : array-like
            Input data.
        mu : array-like, optional
            Parameter data (default is None).
        scaling : str, optional
            Scaling method (default is "individual").
        layer_sizes : list of int, optional
            Sizes of the layers in the encoder and decoder (default is [10, 10, 10]).
        activation : str, optional
            Activation function (default is "selu").
        second_order : bool, optional
            Whether the system is second order (default is True).
        l1 : float, optional
            L1 regularization factor for the autoencoder layers (default is 0).
        l2 : float, optional
            L2 regularization factor for the autoencoder layers (default is 0).
        l_rec : float, optional
            Weight of the reconstruction loss (default is 1).
        l_dz : float, optional
            Weight of the derivative loss (default is 1).
        l_dx : float, optional
            Weight of the derivative loss (default is 1).
        l_int : float, optional
            Weight of the integration loss (default is 0).
        dt : float, optional
            Time step (default is 0).
        dtype : str, optional
            Data type (default is "float32").
        kwargs : dict
            Additional arguments.
        """

        # assert that input arguments are valid
        self.assert_arguments(locals())

        tf.keras.backend.set_floatx(dtype)
        self.dtype_ = dtype
        super(AutoencoderSindy, self).__init__(**kwargs)

        if not hasattr(self, "config"):
            self._init_to_config(locals())

        self.sindy_layer = sindy_layer
        self.layer_sizes = layer_sizes
        self.activation = tf.keras.activations.get(activation)
        self.reduced_order = reduced_order
        self.second_order = second_order
        self.scaling = scaling
        # weighting of the different losses
        self.l_rec, self.l_dz, self.l_dx, self.l_int = l_rec, l_dz, l_dx, l_int
        self.dt = dt

        # kernel regularization weights
        self.l1, self.l2 = l1, l2
        self.kernel_regularizer = tf.keras.regularizers.l1_l2(l1=l1, l2=l2)

        # create the model
        self.x_shape = x.shape[1:]
        if len(self.x_shape) == 1:
            self.flatten, self.unflatten = self.flatten_dummy, self.flatten_dummy
        elif len(self.x_shape) == 2:
            self.flatten, self.unflatten = self.flatten3d, self.unflatten3d

        x = self.flatten(x)
        # some subclasses initialize weights before building the model
        if hasattr(self, "init_weights"):
            self.init_weights()
        self.build_model(x, mu)

        # create loss tracker
        self.create_loss_trackers()

    @property
    def _scaling_methods(self):
        return ["individual", "global", "individual_sqrt", "none"]

    def assert_arguments(self, arguments):
        """
        Asserts that the arguments passed to the model are valid
        :param arguments: all arguments passed to the model
        :return:
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

    def create_loss_trackers(self):
        """
        Create loss trackers for the model.
        """
        self.loss_trackers = dict()
        self.loss_trackers["loss"] = tf.keras.metrics.Mean(name="loss")
        self.loss_trackers["rec"] = tf.keras.metrics.Mean(name="rec")
        if self.l_dz > 0:
            self.loss_trackers["dz"] = tf.keras.metrics.Mean(name="dz")
        if self.l_dx > 0:
            self.loss_trackers["dx"] = tf.keras.metrics.Mean(name="dx")
        if self.l_int > 0:
            self.loss_trackers["int"] = tf.keras.metrics.Mean(name="int")
        self.loss_trackers["reg"] = tf.keras.metrics.Mean(name="reg")
        # update dict with sindy layer loss trackers
        if self.l_dx > 0 or self.l_dz > 0:
            self.loss_trackers.update(self.sindy_layer.loss_trackers)

    def compile(
        self,
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        sindy_optimizer=None,
        **kwargs,
    ):
        """
        Wrapper for the compile function of the keras model to enable the use of different optimizers for different parts of the model.

        Parameters
        ----------
        optimizer : tf.keras.optimizers.Optimizer, optional
            Optimizer for the autoencoder (default is Adam with learning rate 1e-3).
        loss : tf.keras.losses.Loss, optional
            Loss function for the autoencoder (default is BinaryCrossentropy).
        sindy_optimizer : tf.keras.optimizers.Optimizer, optional
            Optimizer for the SINDy part of the model (default is None).
        kwargs : dict
            Additional arguments for the compile function.
        """
        super(AutoencoderSindy, self).compile(optimizer=optimizer, loss=loss, **kwargs)
        if sindy_optimizer is None:
            self.sindy_optimizer = tf.keras.optimizers.get(optimizer)
            # in case we call the optimizer to update different parts of the model separately we need to build it first
            trainable_weights = self.get_trainable_weights()
            self.sindy_optimizer.build(trainable_weights)
        else:
            self.sindy_optimizer = tf.keras.optimizers.get(sindy_optimizer)
        # self.reconstrution_loss = self.compiled_loss

    @staticmethod
    def reconstruction_loss(x, x_pred):
        """
        Calculate the reconstruction loss.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            Original input data.
        x_pred : array-like of shape (n_samples, n_features)
            Reconstructed input data.

        Returns
        -------
        tf.Tensor
            Reconstruction loss.
        """
        return tf.reduce_mean(tf.square(x - x_pred))

    def get_trainable_weights(self):
        """
        Return the trainable weights of the model.

        Returns
        -------
        list
            List of trainable weights.
        """
        return (
            self.encoder.trainable_weights
            + self.decoder.trainable_weights
            + self.sindy.trainable_weights
        )

    def build_model(self, x, mu):
        """
        Build the model.

        Parameters
        ----------
        x : array-like
            Full state.
        mu : array-like
            Parameters.
        """
        x = tf.cast(x, dtype=self.dtype_)

        # encoder
        x_input, z = self.build_encoder(x)

        # sindy
        z_sindy, z_dot = self.build_sindy(z, mu)

        # build the decoder
        x = self.build_decoder(z)

        # build the models
        self.encoder = tf.keras.Model(inputs=x_input, outputs=z, name="encoder")
        self.decoder = tf.keras.Model(inputs=z, outputs=x, name="decoder")
        self.sindy = tf.keras.Model(inputs=z_sindy, outputs=z_dot, name="sindy")

    def define_scaling(self, x):
        """
        Define the scaling factor for given training data.

        Parameters
        ----------
        x : array-like
            Training data.
        """
        # scale the data if requested
        if self.scaling == "individual":
            # scale by squareroot of individual max value
            self.scale_factor = 1 / (tf.reduce_max(tf.abs(x), axis=0))
            # self.model_noise_factor = 1 / (tf.reduce_max(tf.abs(x), axis=0))
            # replace inf with ones to avoid division by zero
            self.scale_factor = tf.where(
                tf.math.is_inf(self.scale_factor),
                tf.ones_like(self.scale_factor),
                self.scale_factor,
            )
        elif self.scaling == "individual_sqrt":
            # scale by squareroot of individual max value
            self.scale_factor = 1 / tf.sqrt(tf.reduce_max(tf.abs(x), axis=0))
            # self.model_noise_factor = 1 / (tf.reduce_max(tf.abs(x), axis=0))
            # replace inf with ones to avoid division by zero
            self.scale_factor = tf.where(
                tf.math.is_inf(self.scale_factor),
                tf.ones_like(self.scale_factor),
                self.scale_factor,
            )
        elif self.scaling == "global":
            # scale all features by global  max value
            self.scale_factor = 1.0 / tf.reduce_max(tf.abs(x))
        else:
            self.scale_factor = 1.0

    def scale(self, x):
        """
        Scale the data.

        Parameters
        ----------
        x : array-like
            Input data.

        Returns
        -------
        array-like
            Scaled data.
        """

        # scale the data
        x = x * self.scale_factor
        return x

    def rescale(self, x):
        """
        Rescale the data.

        Parameters
        ----------
        x : array-like
            Scaled data.

        Returns
        -------
        array-like
            Rescaled data.
        """
        # rescale the data
        x = x / self.scale_factor
        return x

    def build_encoder(self, x):
        """
        Build a fully connected encoder with layers of size layer_sizes.

        Parameters
        ----------
        x : array-like
            Input to the autoencoder.

        Returns
        -------
        x_input : tf.keras.Input
            Input tensor.
        z : tf.Tensor
            Latent variable.
        """
        x_input = tf.keras.Input(shape=(x.shape[1],), dtype=self.dtype_)
        z = x_input
        for n_neurons in self.layer_sizes:
            z = tf.keras.layers.Dense(
                n_neurons,
                activation=self.activation,
                kernel_regularizer=self.kernel_regularizer,
            )(z)
        z = tf.keras.layers.Dense(
            self.reduced_order,
            activation="linear",
            kernel_regularizer=self.kernel_regularizer,
        )(z)
        return x_input, z

    def build_decoder(self, z):
        """
        Build a fully connected decoder with layers of reversed sizes in layer_sizes.

        Parameters
        ----------
        z : array-like
            Latent variable.

        Returns
        -------
        tf.Tensor
            Reconstructed full state.
        """
        # new decoder
        x_ = z
        for n_neurons in reversed(self.layer_sizes):
            x_ = tf.keras.layers.Dense(
                n_neurons,
                activation=self.activation,
                kernel_regularizer=self.kernel_regularizer,
            )(x_)
        x = tf.keras.layers.Dense(
            self.x_shape[0],
            activation="linear",
            kernel_regularizer=self.kernel_regularizer,
        )(x_)
        return x

    @tf.function
    def build_loss(self, inputs):
        """
        Split input into state, its derivative, and the parameters, perform the forward pass, calculate the loss,
        and update the weights.

        Parameters
        ----------
        inputs : list
            List of array-like objects.

        Returns
        -------
        dict
            Dictionary of losses.
        """
        # second order systems dx_ddt = f(x, dx_dt, mu)
        x, dx_dt, dx_ddt, x_int, dx_int, mu, mu_int = self.split_inputs(inputs)

        # forward pass
        with tf.GradientTape() as tape:
            # only perform reconstruction if no identification loss is used
            if self.l_dx == 0 and self.l_dz == 0:
                losses = self.get_loss_rec(x)
            # calculate loss for second order systems (includes two time derivatives)
            elif self.second_order:
                losses = self.get_loss_2nd(x, dx_dt, dx_ddt, mu, x_int, dx_int, mu_int)
            # calculate loss for first order systems
            else:
                losses = self.get_loss(x, dx_dt, mu, x_int, mu_int)

            # split trainable variables for autoencoder and dynamics so that you can use separate optimizers
            trainable_weights = self.get_trainable_weights()
            # grads = tape.gradient(losses["loss"], trainable_weights)
            # self.optimizer.apply_gradients(zip(grads, trainable_weights))

            n_ae_weights = len(self.encoder.trainable_weights) + len(
                self.decoder.trainable_weights
            )
            grads = tape.gradient(losses["loss"], trainable_weights)
            grads_autoencoder = grads[:n_ae_weights]
            grads_sindy = grads[n_ae_weights:]

            # adjust weights for autoencoder
            self.optimizer.apply_gradients(
                zip(grads_autoencoder, trainable_weights[:n_ae_weights])
            )
            # in case of only reconstructing the data without dynamics there won't be gradients for the dynamics
            if self.l_dx > 0 or self.l_dz > 0:
                # adjust sindy weights with separate optimizer
                self.sindy_optimizer.apply_gradients(
                    zip(grads_sindy, trainable_weights[n_ae_weights:])
                )

        return losses

    def calc_latent_time_derivatives(self, x, dx_dt, dx_ddt=None):
        """
        Calculate time derivatives of latent variables given the time derivatives of the input variables
        (used for comparison with SINDy)

        Parameters
        ----------
        x : array-like
            Full state.
        dx_dt : array-like
            Time derivative of state.
        dx_ddt : array-like, optional
            Second time derivative of state (default is None).

        Returns
        -------
        tuple
            Latent variables and their time derivatives.
        """
        # in case the variables are not vectorized but in their physical geometrical description flatten them
        if len(x.shape) > 2:
            if dx_ddt is not None:
                x, dx_dt, dx_ddt = [self.flatten(x) for x in [x, dx_dt, dx_ddt]]
                dx_ddt = tf.expand_dims(tf.cast(dx_ddt, dtype=self.dtype_), axis=-1)
            else:
                x, dx_dt = [self.flatten(x) for x in [x, dx_dt]]

        x = tf.cast(x, self.dtype_)
        dx_dt = tf.expand_dims(tf.cast(dx_dt, dtype=self.dtype_), axis=-1)
        if dx_ddt is not None:
            dx_ddt = tf.expand_dims(tf.cast(dx_ddt, dtype=self.dtype_), axis=-1)

        # forward pass of encoder and time derivative of latent variable
        if dx_ddt is not None:
            with tf.GradientTape() as t11:
                with tf.GradientTape() as t12:
                    t12.watch(x)
                    z = self.encode(x)
                dz_dx = t12.batch_jacobian(z, x)
            dz_ddx = t11.batch_jacobian(dz_dx, x)
        else:
            with tf.GradientTape() as t12:
                t12.watch(x)
                z = self.encode(x)
            dz_dx = t12.batch_jacobian(z, x)

        # calculate first time derivative of the latent variable by application of the chain rule
        #   dz_dt  = dz_dx @ dx_dt
        #           = dz_dxr @ (V^T @ dx_dt)
        dz_dt = dz_dx @ dx_dt
        dz_dt = tf.squeeze(dz_dt, axis=2)

        # calculate second time derivative of the latent variable
        #   dz_ddt  = dz_ddz @ (V^T @ dx_dt) + dz_dx @ dx_ddt
        #           = dz_ddxr @ (V^T @ dx_dt) @ (V^T @ dx_dt) + dz_dxr @ (V^T @ dx_ddt)
        if dx_ddt is not None:
            dz_ddt = tf.squeeze(
                dz_ddx @ tf.expand_dims(dx_dt, axis=1), axis=3
            ) @ dx_dt + dz_dx @ tf.expand_dims(tf.squeeze(dx_ddt, axis=-1), axis=-1)
            dz_ddt = tf.squeeze(dz_ddt, axis=2)
            return z.numpy(), dz_dt.numpy(), dz_ddt.numpy()
        else:
            return z.numpy(), dz_dt.numpy()

    def _training_encoding(self, x, losses):
        """
        For compatibility with the class we need a method that only returns the latent variable but not the mean and
        log variance. The mean and log variance are stored in the class attributes so that they can be accessed by the
        get_loss method.

        Parameters
        ----------
        x : array-like
            Full state.
        losses : dict
            Dictionary of losses.

        Returns
        -------
        tf.Tensor
            Latent variable.
        """
        z = self.encoder(x)
        return z, losses

    @tf.function
    def get_loss_rec(self, x):
        """
        Calculate the reconstruction loss.

        Parameters
        ----------
        x : array-like
            Full state.

        Returns
        -------
        tf.Tensor
            Reconstruction loss.
        """
        losses = dict(loss=0)
        z, losses = self._training_encoding(x, losses)
        x_pred = self.decoder(z)

        # calculate losses
        rec_loss = self.l_rec * self.reconstruction_loss(
            x, x_pred
        )  # reconstruction loss
        losses["rec"] = rec_loss
        reg_loss = tf.reduce_sum(self.losses)  # regularization loss
        losses["reg"] = reg_loss
        losses["loss"] += rec_loss + reg_loss  # total loss

        return losses

    @tf.function
    def get_loss(self, x, dx_dt, mu, x_int=None, mu_int=None):
        """
        Calculate loss for first order system.

        Parameters
        ----------
        x : array-like
            Full state.
        dx_dt : array-like
            Time derivative of state.
        mu : array-like
            Control input.
        x_int : array-like, optional
            Full state at future time steps (default is None).
        mu_int : array-like, optional
            Control input at future time steps (default is None).

        Returns
        -------
        dict
            Dictionary of losses.
        """
        losses = dict(loss=0)

        x = tf.cast(x, self.dtype_)
        dx_dt = tf.expand_dims(tf.cast(dx_dt, dtype=self.dtype_), axis=-1)

        # forward pass of encoder and time derivative of latent variable
        with tf.GradientTape() as t12:
            t12.watch(x)
            z, losses = self._training_encoding(x, losses)
            dz_dx = t12.batch_jacobian(z, x)

        # calculate first time derivative of the latent variable by application of the chain rule
        #   dz_ddt  = dz_dx @ dx_dt
        dz_dt = dz_dx @ dx_dt

        # sindy approximation of the time derivative of the latent variable
        sindy_pred, sindy_mean, sindy_log_var = self.evaluate_sindy_layer(z, None, mu)
        dz_dt_sindy = tf.expand_dims(sindy_pred, -1)

        # forward pass of decoder and time derivative of reconstructed variable
        with tf.GradientTape() as t22:
            t22.watch(z)
            x_ = self.decoder(z)
        if self.l_dx > 0:
            dx_dz = t22.batch_jacobian(x_, z)
            # calculate first time derivative of the reconstructed state by application of the chain rule
            dxf_dt = dx_dz @ dz_dt_sindy
            dx_loss = self.l_dx * self.compiled_loss(
                tf.concat([dxf_dt], axis=1), tf.concat([dx_dt], axis=1)
            )
            losses["dx"] = dx_loss
            losses["loss"] += dx_loss

        # SINDy consistency loss
        if self.l_int:
            int_loss = self.get_int_loss([x_int, mu_int])
            losses["int"] = int_loss
            losses["loss"] += int_loss

        # calculate losses
        reg_loss = tf.reduce_sum(self.losses)
        rec_loss = self.l_rec * self.reconstruction_loss(x, x_)
        # dz_loss = self.l_dz * self.compiled_loss(tf.concat([dz_dt], axis=1),
        # tf.concat([dz_dt_sindy], axis=1))
        dz_loss = tf.math.log(
            2 * np.pi * tf.reduce_mean(tf.keras.losses.mse(dz_dt, dz_dt_sindy)) + 1
        )

        losses["loss"] += rec_loss + dz_loss + reg_loss

        # calculate kl divergence for variational sindy
        if sindy_mean is not None:
            kl_loss_sindy = self.sindy_layer.kl_loss(sindy_mean, sindy_log_var)
            losses["kl_sindy"] = kl_loss_sindy
            losses["loss"] += kl_loss_sindy

        losses["reg"] = reg_loss
        losses["rec"] = rec_loss
        losses["dz"] = dz_loss

        return losses

    @tf.function
    def get_loss_2nd(
        self, x, dx_dt, dx_ddt, mu, x_int=None, dx_dt_int=None, mu_int=None
    ):
        """
        Calculate loss for second order system.

        Parameters
        ----------
        x : array-like
            Full state.
        dx_dt : array-like
            Time derivative of state.
        dx_ddt : array-like
            Second time derivative of state.
        mu : array-like
            Control input.
        x_int : array-like, optional
            Full state at future time steps (default is None).
        dx_dt_int : array-like, optional
            Time derivative of state at future time steps (default is None).
        mu_int : array-like, optional
            Control input at future time steps (default is None).

        Returns
        -------
        dict
            Dictionary of losses.
        """
        losses = dict(loss=0)

        x = tf.cast(x, self.dtype_)
        dx_dt = tf.expand_dims(tf.cast(dx_dt, dtype=self.dtype_), axis=-1)
        dx_ddt = tf.expand_dims(tf.cast(dx_ddt, dtype=self.dtype_), axis=-1)

        # forward pass of encoder and time derivative of latent variable
        with tf.GradientTape() as t11:
            with tf.GradientTape() as t12:
                t12.watch(x)
                z, losses = self._training_encoding(x, losses)
            t11.watch(x)
            dz_dx = t12.batch_jacobian(z, x)
        dz_ddx = t11.batch_jacobian(dz_dx, x)

        # calculate first time derivative of the latent variable by application of the chain rule
        #   dz_ddt  = dz_dx @ dx_dt
        dz_dt = dz_dx @ dx_dt

        # calculate second time derivative of the latent variable
        #   dz_ddt  = dz_ddz @ dx_dt + dz_dx @ dx_ddt
        dz_ddt = (
            tf.squeeze(dz_ddx @ tf.expand_dims(dx_dt, axis=1), axis=3) @ dx_dt
            + dz_dx @ dx_ddt
        )

        # sindy approximation of the time derivative of the latent variable
        sindy_pred, sindy_mean, sindy_log_var = self.evaluate_sindy_layer(z, dz_dt, mu)
        dz_dt_sindy = tf.expand_dims(sindy_pred[:, : self.reduced_order], -1)
        dz_ddt_sindy = tf.expand_dims(sindy_pred[:, self.reduced_order :], -1)

        # forward pass of decoder and time derivative of reconstructed variable
        with tf.GradientTape() as t21:
            t21.watch(z)
            with tf.GradientTape() as t22:
                t22.watch(z)
                x_ = self.decoder(z)
                if self.l_dx > 0:
                    dx_dz = t22.batch_jacobian(x_, z)
            if self.l_dx > 0:
                dx_ddz = t21.batch_jacobian(dx_dz, z)

                # calculate first time derivative of the reconstructed state by application of the chain rule
                dxf_dt = dx_dz @ dz_dt_sindy

                # calculate second time derivative of the reconstructed state by application of the chain rule
                dxf_ddt = (
                    tf.squeeze((dx_ddz @ tf.expand_dims(dz_dt_sindy, axis=1)), axis=3)
                    @ dz_dt_sindy
                ) + dx_dz @ dz_ddt_sindy

                dx_loss = self.l_dx * self.compiled_loss(
                    tf.concat([dxf_dt, dxf_ddt], axis=1),
                    tf.concat([dx_dt, dx_ddt], axis=1),
                )
                losses["dx"] = dx_loss
                losses["loss"] += dx_loss

        # SINDy consistency loss
        if self.l_int:
            int_loss = self.get_int_loss([x_int, dx_dt_int, mu_int])
            losses["int"] = int_loss
            losses["loss"] += int_loss

        # calculate kl divergence for variational sindy
        if sindy_mean is not None:
            kl_loss_sindy = self.sindy_layer.kl_loss(sindy_mean, sindy_log_var)
            losses["kl_sindy"] = kl_loss_sindy
            losses["loss"] += kl_loss_sindy

        # calculate losses
        reg_loss = tf.reduce_sum(self.losses)
        rec_loss = self.l_rec * self.reconstruction_loss(x, x_)
        dz_loss = self.l_dz * self.compiled_loss(
            tf.concat([dz_dt, dz_ddt], axis=1),
            tf.concat([dz_dt_sindy, dz_ddt_sindy], axis=1),
        )

        losses["loss"] += rec_loss + dz_loss + reg_loss
        losses["reg"] = reg_loss
        losses["rec"] = rec_loss
        losses["dz"] = dz_loss

        return losses

    # @tf.function
    def encode(self, x):
        """
        Encode full state.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features, n_dof_per_feature)
            Full state.

        Returns
        -------
        array-like of shape (n_samples, reduced_order)
            Latent variable.
        """
        x = self.flatten(x)
        z = self.encoder(x)
        return z

    # @tf.function
    def decode(self, z):
        """
        Decode latent variable.

        Parameters
        ----------
        z : array-like of shape (n_samples, reduced_order)
            Latent variable.

        Returns
        -------
        array-like of shape (n_samples, n_features, n_dof_per_feature)
            Full state.
        """
        x_rec = self.decoder(z)
        return self.unflatten(x_rec)

    @tf.function
    def reconstruct(self, x, _=None):
        """
        Reconstruct full state.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features, n_dof_per_feature)
            Full state.

        Returns
        -------
        array-like of shape (n_samples, n_features, n_dof_per_feature)
            Reconstructed full state.
        """
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec
