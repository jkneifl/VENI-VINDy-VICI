import logging
import tensorflow as tf
from .base_model import BaseModel

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


class SindyNetwork(BaseModel):

    def __init__(
        self,
        sindy_layer,
        x,
        mu=None,
        second_order=True,
        l_dz: float = 1,
        l_int: float = 0,
        dt=0,
        dtype="float32",
        **kwargs,
    ):
        """
        Model to discover low-dimensional dynamics of a system using SINDy or VINDy
        :param sindy_layer: Layer to identify the governing equatinos of the latent dynamics, must be a class inheriting
            from SindyLayer
        :param x: Input data
        :param mu: parameter data
        :param l_dz: Weight of the derivative loss
        :param l_int: Weight of the integration loss
        :param kwargs:
        """

        # assert that input arguments are valid
        self.assert_arguments(locals())

        tf.keras.backend.set_floatx(dtype)
        self.dtype_ = dtype
        super(SindyNetwork, self).__init__(**kwargs)

        if not hasattr(self, "config"):
            self._init_to_config(locals())

        self.sindy_layer = sindy_layer
        self.second_order = second_order
        # weighting of the different losses
        self.l_dz, self.l_int = l_dz, l_int
        self.dt = dt

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

    def create_loss_trackers(self):
        self.loss_trackers = dict()
        self.loss_trackers["loss"] = tf.keras.metrics.Mean(name="loss")
        self.loss_trackers["dz"] = tf.keras.metrics.Mean(name="dz")
        if self.l_int > 0:
            self.loss_trackers["int"] = tf.keras.metrics.Mean(name="int")
        self.loss_trackers["reg"] = tf.keras.metrics.Mean(name="reg")
        # update dict with sindy layer loss trackers
        self.loss_trackers.update(self.sindy_layer.loss_trackers)

    def get_trainable_weights(self):
        """
        Returns the trainable weights of the model
        :return:
        """
        return self.sindy.trainable_weights

    def build_model(self, z, mu):
        """
        build the model
        :param x: array-like of shape (n_samples, n_features), full state
        :param mu: array-like of shape (n_samples, n_params), parameters
        :return:
        """
        z = tf.keras.Input(shape=(z.shape[1],), dtype=self.dtype_)
        # sindy
        z_sindy, z_dot = self.build_sindy(z, mu)

        # build the models
        self.sindy = tf.keras.Model(inputs=z_sindy, outputs=z_dot, name="sindy")

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
            # calculate loss for second order systems (includes two time derivatives)
            if self.second_order:
                losses = self.get_loss_2nd(x, dx_dt, dx_ddt, mu, x_int, dx_int, mu_int)
            # calculate loss for first order systems
            else:
                losses = self.get_loss(x, dx_dt, mu, x_int, mu_int)

            # split trainable variables for autoencoder and dynamics so that you can use separate optimizers
            trainable_weights = self.get_trainable_weights()
            grads = tape.gradient(losses["loss"], trainable_weights)

            # adjust weights for autoencoder
            self.optimizer.apply_gradients(zip(grads, trainable_weights))

        return losses

    def get_loss(self, z, dz_dt, mu, z_int=None, mu_int=None):
        """
        calculate loss for first order system
        :param z: array-like of shape (n_samples, n_features), full state
        :param dz_dt: array-like of shape (n_samples, n_features), time derivative of state
        :param mu: array-like of shape (n_samples, n_features), control input
        :param z_int: array-like of shape (n_samples, n_features, n_integrationsteps), full state at {t+1,...,t+n_integrationsteps}
        :param mu_int: array-like of shape (n_samples, n_param, n_integrationsteps), control input at {t+1,...,t+n_integrationsteps}
        :return: dz_loss, int_loss, losses: individual losses
        """
        losses = dict(loss=0)

        z = tf.cast(z, self.dtype_)
        dz_dt = tf.expand_dims(tf.cast(dz_dt, dtype=self.dtype_), axis=-1)

        # sindy approximation of the time derivative of the latent variable
        sindy_pred, sindy_mean, sindy_log_var = self.evaluate_sindy_layer(z, None, mu)
        dz_dt_sindy = tf.expand_dims(sindy_pred, -1)

        # SINDy consistency loss
        if self.l_int:
            int_loss = self.get_int_loss([z_int, mu_int])
            losses["int"] = int_loss
            losses["loss"] += int_loss

        # calculate losses
        reg_loss = tf.reduce_sum(self.losses)
        dz_loss = self.l_dz * self.compute_loss(
            None, tf.concat([dz_dt], axis=1), tf.concat([dz_dt_sindy], axis=1)
        )
        losses["reg"] = reg_loss
        losses["dz"] = dz_loss
        losses["loss"] += dz_loss + reg_loss

        # calculate kl divergence for variational sindy
        if sindy_mean is not None:
            kl_loss_sindy = self.sindy_layer.kl_loss(sindy_mean, sindy_log_var)
            losses["kl_sindy"] = kl_loss_sindy
            losses["loss"] += kl_loss_sindy

        return losses

    def get_loss_2nd(
        self, z, dz_dt, dz_ddt, mu, z_int=None, dz_dt_int=None, mu_int=None
    ):
        """
        calculate loss for second order system
        :param x: array-like of shape (n_samples, n_features), full state
        :param dz_dt: array-like of shape (n_samples, n_features), time derivative of state
        :param dz_ddt: array-like of shape (n_samples, n_features), second time derivative of state
        :param mu: array-like of shape (n_samples, n_param), control input
        :param z_int: array-like of shape (n_samples, n_features, n_integrationsteps), full state at {t+1,...,t+n_integrationsteps}
        :param dz_dt_int: array-like of shape (n_samples, n_features, n_integrationsteps), time derivative of state at {t+1,...,t+n_integrationsteps}
        :param mu_int: array-like of shape (n_samples, n_param, n_integrationsteps), control input at {t+1,...,t+n_integrationsteps}
        :return: rec_loss, dz_loss, dx_loss, int_loss, loss: individual losses
        """
        losses = dict(loss=0)

        z = tf.cast(z, self.dtype_)
        dz_dt = tf.expand_dims(tf.cast(dz_dt, dtype=self.dtype_), axis=-1)
        dz_ddt = tf.expand_dims(tf.cast(dz_ddt, dtype=self.dtype_), axis=-1)
        if dz_dt_int is not None:
            dz_dt_int = tf.reshape(dz_dt_int, shape=(-1, dz_dt_int.shape[-1]))
            dz_dt_int = tf.expand_dims(tf.cast(dz_dt_int, dtype=self.dtype_), axis=-1)

        # sindy approximation of the time derivative of the latent variable
        sindy_pred, sindy_mean, sindy_log_var = self.evaluate_sindy_layer(z, dz_dt, mu)
        dz_dt_sindy = tf.expand_dims(sindy_pred[:, : self.reduced_order], -1)
        dz_ddt_sindy = tf.expand_dims(sindy_pred[:, self.reduced_order :], -1)

        # SINDy consistency loss
        if self.l_int:
            int_loss = self.get_int_loss([z_int, dz_dt_int, dz_dt_int, mu_int])
            losses["int"] = int_loss
            losses["loss"] += int_loss

        # calculate kl divergence for variational sindy
        if sindy_mean is not None:
            kl_loss_sindy = self.sindy_layer.kl_loss(sindy_mean, sindy_log_var)
            losses["kl_sindy"] = kl_loss_sindy
            losses["loss"] += kl_loss_sindy

        # calculate losses
        reg_loss = tf.reduce_sum(self.losses)
        dz_loss = self.l_dz * self.compute_loss(
            None,
            tf.concat([dz_dt, dz_ddt], axis=1),
            tf.concat([dz_dt_sindy, dz_ddt_sindy], axis=1),
        )

        losses["loss"] += dz_loss + reg_loss
        losses["reg"] = reg_loss
        losses["dz"] = dz_loss

        return losses
