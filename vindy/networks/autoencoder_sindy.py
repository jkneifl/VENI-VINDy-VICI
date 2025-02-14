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
        Model to discover low-dimensional dynamics of a high-dimensional system using autoencoders and SINDy
        :param sindy_layer: Layer to identify the governing equatinos of the latent dynamics, must be a class inheriting
            from SindyLayer
        :param reduced_order: Order of the reduced model
        :param x: Input data
        :param y: Output data
        :param l_1: l1 regularization factor for the autoencoder layers
        :param l_2: l1 regularization factor for the autoencoder layers
        :param l_rec: Weight of the reconstruction loss
        :param l_dz: Weight of the derivative loss
        :param l_dx: Weight of the derivative loss
        :param l_int: Weight of the integration loss
        :param kwargs:
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
        wrapper for the compile function of the keras model to enable the use of a different optimizers for different
        parts of the model
        :param optimizer: tf.keras.optimizers object, optimizer for the autoencoder
        :param loss: tf.keras.losses object, loss function for the autoencoder
        :param sindy_optimizer: tf.keras.optimizers object, optimizer for the SINDy part of the model
        :param kwargs: additional kwargs for the compile function
        :return:
        """
        super(AutoencoderSindy, self).compile(optimizer=optimizer, loss=loss, **kwargs)
        if sindy_optimizer is None:
            self.sindy_optimizer = tf.keras.optimizers.get(optimizer)
            # in case we call the optimizer to update different parts of the model separately we need to build it first
            trainable_weights = self.get_trainable_weights()
            self.sindy_optimizer.build(trainable_weights)
        else:
            self.sindy_optimizer = tf.keras.optimizers.get(sindy_optimizer)

    @staticmethod
    def reconstruction_loss(x, x_pred):
        """
        calculate the reconstruction loss
        :param x: array-like of shape (n_samples, n_features)
        :param x_pred: array-like of shape (n_samples, n_features)
        :return:
        """
        return tf.reduce_mean(tf.square(x - x_pred))

    def get_trainable_weights(self):
        """
        Returns the trainable weights of the model
        :return:
        """
        return (
            self.encoder.trainable_weights
            + self.decoder.trainable_weights
            + self.sindy.trainable_weights
        )

    def build_model(self, x, mu):
        """
        build the model
        :param x: array-like of shape (n_samples, n_features), full state
        :param mu: array-like of shape (n_samples, n_params), parameters
        :return:
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
        define the scaling factor for given training data
        :param x:
        :return:
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
        # scale the data
        x = x * self.scale_factor
        return x

    def rescale(self, x):
        # rescale the data
        x = x / self.scale_factor
        return x

    def build_encoder(self, x):
        """
        build a fully connected encoder with layers of size layer_sizes
        :param x: input to the autoencoder
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
        Build a fully connected decoder with layers of reversed sizes in layer_sizes
        :param z:
        :return:
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

    def build_loss(self, inputs):
        """
        split input into state, its derivative and the parameters, perform the forward pass, calculate the loss,
        and update the weights
        :param inputs: list of array-like objects
        :return:
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
        :param x: array-like of shape (n_samples, n_features), full state
        :param dx_dt: array-like of shape (n_samples, n_features), time derivative of state
        :param dx_ddt: array-like of shape (n_samples, n_features), second time derivative of state
        :return: z, dz_dt, dz_ddt: array-like of shape (n_samples, n_latent), latent variables and their time derivatives
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
        For compatibility with the class we need a method that only returns the latent variable
        but not the mean and log variance. The mean and log variance are stored in the class attributes so that they
        can be accessed by the get_loss method.
        :param x:
        :return:
        """
        z = self.encoder(x)
        return z, losses

    def get_loss_rec(self, x):
        """
        calculate reconstruction loss of autoencoder
        :param x: array-like of shape (n_samples, n_features), full state
        :return:
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

    def get_loss(self, x, dx_dt, mu, x_int=None, mu_int=None):
        """
        calculate loss for first order system
        :param x: array-like of shape (n_samples, n_features), full state
        :param dx_dt: array-like of shape (n_samples, n_features), time derivative of state
        :param mu: array-like of shape (n_samples, n_features), control input
        :param x_int: array-like of shape (n_samples, n_features, n_integrationsteps), full state at {t+1,...,t+n_integrationsteps}
        :param mu_int: array-like of shape (n_samples, n_param, n_integrationsteps), control input at {t+1,...,t+n_integrationsteps}
        :return: rec_loss, dz_loss, dx_loss, int_loss, loss: individual losses
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
            dx_loss = self.l_dx * self.compute_loss(
                None, tf.concat([dxf_dt], axis=1), tf.concat([dx_dt], axis=1)
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
        # dz_loss = self.l_dz * self.compute_loss(None, tf.concat([dz_dt], axis=1),
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

    def get_loss_2nd(
        self, x, dx_dt, dx_ddt, mu, x_int=None, dx_dt_int=None, mu_int=None
    ):
        """
        calculate loss for second order system
        :param x: array-like of shape (n_samples, n_features), full state
        :param dx_dt: array-like of shape (n_samples, n_features), time derivative of state
        :param dx_ddt: array-like of shape (n_samples, n_features), second time derivative of state
        :param mu: array-like of shape (n_samples, n_param), control input
        :param x_int: array-like of shape (n_samples, n_features, n_integrationsteps), full state at {t+1,...,t+n_integrationsteps}
        :param dx_dt_int: array-like of shape (n_samples, n_features, n_integrationsteps), time derivative of state at {t+1,...,t+n_integrationsteps}
        :param mu_int: array-like of shape (n_samples, n_param, n_integrationsteps), control input at {t+1,...,t+n_integrationsteps}
        :return: rec_loss, dz_loss, dx_loss, int_loss, loss: individual losses
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

                dx_loss = self.l_dx * self.compute_loss(
                    None,
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
        dz_loss = self.l_dz * self.compute_loss(
            None,
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
        encode full state
        :param x: array-like of shape (n_samples, n_features, n_dof_per_feature), full state
        :return: z: array-like of shape (n_samples, reduced_order), latent variable
        """
        x = self.flatten(x)
        z = self.encoder(x)
        return z

    # @tf.function
    def decode(self, z):
        """
        decode latent variable
        :param z: array-like of shape (n_samples, reduced_order), latent variable
        :return: x: array-like of shape (n_samples, n_features, n_dof_per_feature), full state
        """
        x_rec = self.decoder(z)
        return self.unflatten(x_rec)

    def reconstruct(self, x, _=None):
        """
        reconstruct full state
        :param x: array-like of shape (n_samples, n_features, n_dof_per_feature), full state
        :param x_rec: array-like of shape (n_samples, n_features, n_dof_per_feature), full state
        :return:
        """
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec
