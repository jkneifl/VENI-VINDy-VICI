import logging
import tensorflow as tf
from .base_model import BaseModel

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


class SindyNetwork(BaseModel):
    """
    Model to discover dynamics of a system using SINDy or VINDy.
    """

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
        Initialize the SINDy network.

        Parameters
        ----------
        sindy_layer : SindyLayer
            Layer to identify the governing equations of the latent dynamics.
        x : array-like
            Input data.
        mu : array-like, optional
            Parameter data (default is None).
        second_order : bool, optional
            Whether the system is second order (default is True).
        l_dz : float, optional
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
        """
        Create loss trackers for the model.
        """
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
        Return the trainable weights of the model.

        Returns
        -------
        list
            List of trainable weights.
        """
        return self.sindy.trainable_weights

    def build_model(self, z, mu):
        """
        Build the model.

        Parameters
        ----------
        z : array-like
            Latent state.
        mu : array-like
            Parameters.
        """
        z = tf.keras.Input(shape=(z.shape[1],), dtype=self.dtype_)
        # sindy
        z_sindy, z_dot = self.build_sindy(z, mu)

        # build the models
        self.sindy = tf.keras.Model(inputs=z_sindy, outputs=z_dot, name="sindy")

    @tf.function
    def build_loss(self, inputs):
        """
        Split input into state, its derivative, and the parameters, perform the forward pass, calculate the loss, and update the weights.

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

    @tf.function
    def get_loss(self, z, dz_dt, mu, z_int=None, mu_int=None):
        """
        Calculate loss for first order system.

        Parameters
        ----------
        z : array-like
            Full state.
        dz_dt : array-like
            Time derivative of state.
        mu : array-like
            Control input.
        z_int : array-like, optional
            Full state at future time steps (default is None).
        mu_int : array-like, optional
            Control input at future time steps (default is None).

        Returns
        -------
        dict
            Dictionary of losses.
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
        dz_loss = self.l_dz * self.compiled_loss(
            tf.concat([dz_dt], axis=1), tf.concat([dz_dt_sindy], axis=1)
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

    @tf.function
    def get_loss_2nd(
        self, z, dz_dt, dz_ddt, mu, z_int=None, dz_dt_int=None, mu_int=None
    ):
        """
        Calculate loss for second order system.

        Parameters
        ----------
        z : array-like
            Full state.
        dz_dt : array-like
            Time derivative of state.
        dz_ddt : array-like
            Second time derivative of state.
        mu : array-like
            Control input.
        z_int : array-like, optional
            Full state at future time steps (default is None).
        dz_dt_int : array-like, optional
            Time derivative of state at future time steps (default is None).
        mu_int : array-like, optional
            Control input at future time steps (default is None).

        Returns
        -------
        dict
            Dictionary of losses.
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
        dz_loss = self.l_dz * self.compiled_loss(
            tf.concat([dz_dt, dz_ddt], axis=1),
            tf.concat([dz_dt_sindy, dz_ddt_sindy], axis=1),
        )

        losses["loss"] += dz_loss + reg_loss
        losses["reg"] = reg_loss
        losses["dz"] = dz_loss

        return losses
