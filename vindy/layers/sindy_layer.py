import tensorflow as tf
import numpy as np
import scipy
import inspect
from vindy.libraries import PolynomialLibrary, BaseLibrary
from sympy import symbols
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


class SindyLayer(tf.keras.layers.Layer):
    """
    Sparse Identification of Nonlinear Dynamics (SINDy) layer.

    This layer evaluates a library of candidate functions on latent inputs and
    performs sparse regression to recover governing coefficients.

    Parameters
    ----------
    state_dim : int
        Number of latent variables (dimension of the latent state z).
    param_dim : int, optional
        Number of parameters (mu). Default is 0.
    feature_libraries : list of BaseLibrary, optional
        Feature libraries applied to the latent variables (e.g. PolynomialLibrary).
    param_feature_libraries : list of BaseLibrary, optional
        Feature libraries applied to parameters (mu).
    second_order : bool, optional
        If True, enforce second-order structure for dynamics (include z_dot features).
    kernel_regularizer : tf.keras.regularizers.Regularizer, optional
        Regularizer applied to the learned coefficient kernel.
    x_mu_interaction : bool, optional
        If True, include interaction features between state and parameters.
    mask : array-like, optional
        Mask to fix or remove certain coefficients from training.
    fixed_coeffs : array-like, optional
        Values for coefficients that are fixed (applied after masking).
    dtype : str, optional
        Data type used by the layer (e.g. 'float32').
    **kwargs
        Additional keyword arguments passed to ``tf.keras.layers.Layer``.
    """

    def __init__(
        self,
        state_dim,
        param_dim=0,
        feature_libraries=None,
        param_feature_libraries=None,
        second_order=True,
        kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-3, l2=0),
        x_mu_interaction=True,
        mask=None,
        fixed_coeffs=None,
        dtype="float32",
        **kwargs,
    ):
        """
        Layer for SINDy approximation of the time derivative of the latent variable.

        Feature libraries are applied to the latent variable and its time derivative,
        and a sparse regression is performed to obtain the governing coefficients.

        Notes
        -----
        This initializer validates arguments, constructs default feature libraries
        when none are provided and prepares internal bookkeeping variables used
        by the layer (masks, coefficient shapes, etc.).
        """
        super(SindyLayer, self).__init__(**kwargs)

        # default libraries
        if feature_libraries is None:
            feature_libraries = [PolynomialLibrary(degree=3)]
        if param_feature_libraries is None:
            param_feature_libraries = []

        # assert that input arguments are valid
        self.assert_arguments(locals())

        self.dtype_ = dtype
        # default library
        if len(feature_libraries) == 0:
            feature_libraries = [PolynomialLibrary(degree=3)]
        self.feature_libraries = feature_libraries
        self.param_feature_libraries = param_feature_libraries

        self.state_dim = state_dim
        self.param_dim = param_dim
        self.x_mu_interaction = x_mu_interaction

        # get feature dimension
        if second_order:
            self.output_dim = 2 * state_dim
        else:
            self.output_dim = state_dim
        self.n_bases_functions = self.features(
            tf.ones((1, self.output_dim + param_dim))
        ).shape[1]

        # set certain values of kernel
        self.mask, self.fixed_coeffs = self.set_mask(mask, fixed_coeffs)

        self.second_order = second_order
        if second_order:
            # enforcing the structure of the 2nd order model
            #   the feature library will look like this [1, z1, ..., zn, z1_dot, ..., zn_dot, ...]
            #   consequently we enforce 2nd order structure [0_(n x n+1) I_(n x n) 0_(n x ...)]
            zero_matrix = tf.zeros(
                shape=[self.state_dim, self.state_dim + 1], dtype=self.dtype_
            )
            eye_matrix = tf.eye(self.state_dim, dtype=self.dtype_)
            zero_matrix2 = tf.zeros(
                shape=[self.state_dim, self.n_bases_functions - int(state_dim * 2) - 1],
                dtype=self.dtype_,
            )
            # apply the structure of the 2nd order model
            fixed_kernel = tf.concat([zero_matrix, eye_matrix, zero_matrix2], axis=1)
            self.mask = tf.concat(
                [tf.zeros(fixed_kernel.shape, dtype=self.dtype_), self.mask], axis=0
            )
            self.fixed_coeffs = tf.concat([fixed_kernel, self.fixed_coeffs], axis=0)

        # initialize sindy coefficients
        self.init_weigths(kernel_regularizer)

    @property
    def loss_trackers(self):
        """
        Return loss trackers used by the layer.

        Returns
        -------
        dict
            Mapping of loss names to Keras Metric objects. By default the SINDy
            layer does not add separate loss trackers and returns an empty dict.
        """
        return dict()

    def _init_to_config(self, init_locals):
        """
        Save initializer arguments into ``self.config`` for serialization.

        Parameters
        ----------
        init_locals : dict
            The locals() mapping from the initializer; used to persist init args.
        """

        sig = inspect.signature(self.__init__)
        keys = [param.name for param in sig.parameters.values()]
        values = [init_locals[name] for name in keys]
        init_dict = dict(zip(keys, values))
        self.config = init_dict

    def assert_arguments(self, arguments):
        """
        Validate initializer arguments and raise informative assertions.

        Parameters
        ----------
        arguments : dict
            Mapping from argument name to value (typically ``locals()`` from __init__).
        """
        assert arguments["dtype"] in [
            "float32",
            "float64",
            tf.float32,
            tf.float64,
        ], "dtype must be either float32 or float64"
        assert isinstance(arguments["state_dim"], int), "state_dim must be an integer"
        assert isinstance(arguments["param_dim"], int), "param_dim must be an integer"
        # assert that mask and fixed_coeffs have the right shape
        assert (
            isinstance(arguments["mask"], np.ndarray)
            or isinstance(arguments["mask"], tf.Tensor)
            or arguments["mask"] is None
        ), "mask must be either None, a numpy array or a tensorflow tensor"
        assert (
            isinstance(arguments["fixed_coeffs"], np.ndarray)
            or isinstance(arguments["fixed_coeffs"], tf.Tensor)
            or arguments["fixed_coeffs"] is None
        ), "fixed_coeffs must be either None, a numpy array or a tensorflow tensor"
        if arguments["mask"] is not None:
            assert (
                arguments["mask"].shape[0] == arguments["state_dim"]
            ), "mask must have shape (state_dim, x)"
        if arguments["fixed_coeffs"] is not None:
            assert (
                arguments["fixed_coeffs"].shape[0] == arguments["state_dim"]
            ), "fixed_coeffs must have shape (state_dim, x)"
        # assert that feature_libraries is a list of veni.libraries objects
        assert isinstance(
            arguments["feature_libraries"], list
        ), "feature_libraries must be a list"
        for lib in arguments["feature_libraries"]:
            assert isinstance(
                lib, BaseLibrary
            ), "feature_libraries must be a list of veni.libraries objects"
        # assert that param_feature_libraries is a list of veni.libraries objects
        assert isinstance(
            arguments["param_feature_libraries"], list
        ), "param_feature_libraries must be a list"
        for lib in arguments["param_feature_libraries"]:
            assert isinstance(
                lib, BaseLibrary
            ), "param_feature_libraries must be a list of veni.libraries objects"
        # assert that second_order and x_mu_interaction are booleans
        assert isinstance(
            arguments["second_order"], bool
        ), "second_order must be a boolean"
        assert isinstance(
            arguments["x_mu_interaction"], bool
        ), "x_mu_interaction must be a boolean"
        assert isinstance(
            arguments["kernel_regularizer"], tf.keras.regularizers.Regularizer
        ), "kernel_regularizer must be a tf.keras.regularizers.Regularizer object"

    @property
    def coefficient_matrix_shape(self):
        return (self.output_dim, self.n_bases_functions)

    @property
    def kernel_shape(self):
        """
        Return the shape of the internal kernel (trainable coefficients).

        Returns
        -------
        tuple
            Kernel shape as (n_dofs, 1).
        """
        return (self.n_dofs, 1)

    def init_weigths(self, kernel_regularizer):

        # get amount of dofs (equals the number of ones in the mask)
        self.n_dofs = int(tf.reduce_sum(self.mask))
        # get ids of dofs
        self.dof_ids = tf.where(tf.equal(self.mask, 1))

        init = tf.random_uniform_initializer(minval=-1, maxval=1)
        self.kernel = self.add_weight(
            name="SINDy_coefficents",
            initializer=init,
            shape=self.kernel_shape,
            dtype=self.dtype_,
            regularizer=kernel_regularizer,
        )

    def set_mask(self, mask, fixed_coeffs=None):
        """
        Normalize and pad mask and fixed coefficient arrays to the expected shape.

        Parameters
        ----------
        mask : array-like or None
            Mask specifying which coefficients are trainable (1) or disabled (0).
        fixed_coeffs : array-like or None
            Fixed coefficient values to be applied for masked entries.

        Returns
        -------
        tuple
            ``(mask, fixed_coeffs)`` both cast to the layer dtype and padded to the
            proper coefficient matrix shape.
        """
        if mask is None:
            mask = tf.ones([self.state_dim, self.n_bases_functions])
        if fixed_coeffs is None:
            fixed_coeffs = tf.zeros([self.state_dim, self.n_bases_functions])

        mask = tf.cast(mask, dtype=self.dtype_)
        if mask.shape != self.coefficient_matrix_shape:
            # bring mask to the right shape by padding ones
            mask = tf.pad(
                mask,
                [[0, 0], [0, self.coefficient_matrix_shape[1] - mask.shape[1]]],
                constant_values=1,
            )
        fixed_coeffs = tf.cast(fixed_coeffs, dtype=self.dtype_)
        if fixed_coeffs.shape != self.coefficient_matrix_shape:
            # bring mask to the right shape by padding zeros
            fixed_coeffs = tf.pad(
                fixed_coeffs,
                [[0, 0], [0, self.coefficient_matrix_shape[1] - fixed_coeffs.shape[1]]],
                constant_values=0,
            )
        return mask, fixed_coeffs

    @property
    def _coeffs(self):
        """
        Get the coefficients of the SINDy layer as a matrix.

        Returns
        -------
        tf.Tensor
            Coefficient matrix with shape (output_dim, n_bases_functions).
        """
        # fill the coefficient matrix with the trainable coefficients
        coeffs = self.fill_coefficient_matrix(self.kernel)

        return coeffs

    def get_sindy_coeffs(self):
        return self._coeffs.numpy()

    def get_prunable_weights(self):
        # Prune bias also, though that usually harms model accuracy too much.
        return [self.kernel]

    def prune_weights(self, threshold=0.01, training=False):
        mask = tf.math.greater(
            tf.math.abs(self.kernel),
            threshold * tf.ones_like(self.kernel, dtype=self.kernel.dtype),
        )
        mask = tf.cast(mask, dtype=self.kernel.dtype)
        self.kernel.assign(tf.multiply(self.kernel, mask))

    def fill_coefficient_matrix(self, trainable_coeffs):
        """
        Insert the trainable coefficients into the full coefficient matrix.

        Parameters
        ----------
        trainable_coeffs : array-like
            Trainable coefficients arranged to match the active DOFs.

        Returns
        -------
        tf.Tensor
            Full coefficient matrix with fixed coefficients applied.
        """
        # create a zero matrix for the coefficients with the correct shape
        coeffs = tf.zeros(self.coefficient_matrix_shape)
        # put the coefficients into the coefficient matrix Xi at the correct positions
        coeffs = tf.tensor_scatter_nd_update(
            coeffs, self.dof_ids, trainable_coeffs[:, 0]
        )

        # apply the mask7
        if self.fixed_coeffs is not None:
            coeffs += self.fixed_coeffs

        return coeffs

    @tf.function
    def call(self, inputs, training=False):
        """
        Forward pass of the SINDy layer: evaluate features and compute prediction.

        Parameters
        ----------
        inputs : tf.Tensor
            Latent variables, shape ``(batch_size, latent_dim)``.
        training : bool, optional
            Whether the call is in training mode.

        Returns
        -------
        tf.Tensor
            Predicted derivatives with shape ``(batch_size, output_dim)``.
        """
        z_features = self.features(inputs)
        z_dot = z_features @ tf.transpose(self._coeffs)
        return z_dot

    def features(self, inputs):
        """
        Compute concatenated features from configured libraries.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor that contains state and (optionally) parameter values.

        Returns
        -------
        tf.Tensor
            Concatenated feature matrix for the SINDy regression.
        """
        # in case we want interaction between parameters and states
        if self.x_mu_interaction:
            z_feat = self.concat_features(inputs, self.feature_libraries)
            return z_feat
        # if we want to apply separate features to parameter and states
        else:
            z_feat = self.concat_features(
                inputs[:, : self.output_dim], self.feature_libraries
            )
            if len(self.param_feature_libraries) > 0:
                param_feat = self.concat_features(
                    inputs[:, self.output_dim :], self.param_feature_libraries
                )
                return tf.concat([z_feat, param_feat], axis=1)
            return z_feat

    def concat_features(self, z, libraries):
        """
        Concatenate outputs of several feature libraries.

        Parameters
        ----------
        z : tf.Tensor
            Input to the feature libraries.
        libraries : list
            Iterable of library objects that are callable on ``z``.

        Returns
        -------
        tf.Tensor
            Concatenated feature outputs along the last axis.
        """
        features = [library(z) for library in libraries]
        z_feat = tf.concat(features, axis=-1)
        return z_feat

    def get_feature_names(self, z=None, mu=None):
        """
        Construct human-readable feature names for states and parameters.

        Parameters
        ----------
        z : list of str, optional
            Names for the state variables. If None, default names are generated.
        mu : list of str, optional
            Names for parameter variables. If None, default names are generated.

        Returns
        -------
        list of sympy.Symbol or str
            Feature names in the order produced by ``features``.
        """

        if z is None:
            z = [f"z_{i}" for i in range(self.output_dim)]
        if mu is None:
            mu = [f"\u03bc_{i}" for i in range(self.param_dim)]

        z = [symbols(z_) for z_ in z]
        mu = [symbols(mu_) for mu_ in mu]

        # in case we want interaction between parameters and states
        if self.x_mu_interaction:
            features = [library.get_names(z + mu) for library in self.feature_libraries]
            # combine lists to one list
            features = [item for sublist in features for item in sublist]
        # if we want to apply separate features to parameter and states
        else:
            z_feat = [library.get_names(z) for library in self.feature_libraries]
            # combine lists to one list
            z_feat = [item for sublist in z_feat for item in sublist]
            param_feat = [
                library.get_names(mu) for library in self.param_feature_libraries
            ]
            # combine lists to one list
            param_feat = [item for sublist in param_feat for item in sublist]
            features = z_feat + param_feat

        return features

    def print(self, z=None, mu=None, precision: int = 3):
        """
        Print the discovered SINDy equations to stdout.

        Parameters
        ----------
        z : list of str, optional
            Variable names for states.
        mu : list of str, optional
            Variable names for parameters.
        precision : int, optional
            Number of decimal places when formatting coefficients.
        """
        print(self.model_equation_to_str(z, mu, precision))

    def model_equation_to_str(self, z=None, mu=None, precision: int = 3):
        """
        Convert coefficients and feature names into a human-readable equation string.

        Parameters
        ----------
        z : list of str, optional
            Names of the state variables.
        mu : list of str, optional
            Names of the parameter variables.
        precision : int, optional
            Decimal precision for printing coefficients.

        Returns
        -------
        str
            Multi-line string with one equation per latent state.
        """
        if z is None:
            z = [f"z{i}" for i in range(self.output_dim)]
        if mu is None:
            mu = [f"\u03bc{i}" for i in range(self.param_dim)]
        if len(z) != self.output_dim:
            raise ValueError(f"arguments should have length {self.output_dim}")
        if len(mu) != self.param_dim:
            raise ValueError(f"mu should have length {self.param_dim}")

        # in case we want interaction between parameters and states
        features = self.get_feature_names(z, mu)

        coeffs = self.get_sindy_coeffs()
        str = ""
        for i, c_ in enumerate(coeffs):
            str += f"d{z[i]} = "
            for j in range(len(c_)):
                if np.round(c_[j], precision) != 0:
                    if c_[j] > 0:
                        str += f"+ {np.abs(c_[j]):.{precision}f}*{features[j]} "
                    else:
                        str += f"- {np.abs(c_[j]):.{precision}f}*{features[j]} "
                    # if j < len(c_) - 1:
                    #     print(' + ', end='')
            str += "\n"
        return str

    def integrate(self, z0, t, mu=None, method="RK45", sindy_fcn=None):
        """
        Integrate the SINDy model forward in time using scipy.integrate.solve_ivp.

        Parameters
        ----------
        z0 : array-like
            Initial state for integration.
        t : array-like
            Time points at which to evaluate the solution.
        mu : array-like or callable, optional
            Parameter trajectory (or callable) to pass to the model.
        method : str, optional
            Integration method for solve_ivp (e.g. 'RK45').
        sindy_fcn : callable, optional
            Callable implementing the right-hand side. If None, uses ``self.rhs_``.

        Returns
        -------
        OdeResult
            The object returned by scipy.integrate.solve_ivp.
        """

        # tensorflow tensor form numpy
        z0 = tf.cast(z0, dtype=self.dtype_)
        t = tf.cast(t, dtype=self.dtype_)

        if sindy_fcn is None:
            sindy_fcn = self.rhs_
        if mu is not None:
            if not callable(mu):
                mu = tf.cast(mu, dtype=self.dtype_)
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
                return sindy_fcn(t, np.concatenate([x, mu_fun(t)], axis=0))[0]

        else:

            def rhs(t, x):
                return sindy_fcn(t, x)[0]

        sol = scipy.integrate.solve_ivp(
            rhs,
            t_span=[t[0], t[-1]],
            t_eval=t,
            y0=z0,
            method=method,
            # rtol=1e-6
        )
        return sol

    def rhs_(self, t, inputs):
        """
        Evaluate the right-hand side z'(t) = f(z, mu) for provided inputs.

        Parameters
        ----------
        t : float
            Current time (unused by default but present for compatibility).
        inputs : array-like
            Flattened inputs (state or state+param) for the RHS function.

        Returns
        -------
        tf.Tensor or ndarray
            Time derivative evaluated at the given inputs.
        """
        if len(inputs.shape) == 1:
            inputs = tf.expand_dims(inputs, 0)
        return self(inputs)


# Provide an explicit signature for Sphinx autodoc so the class constructor
# parameters are rendered even if the tensorflow module is mocked during doc builds.
try:
    SindyLayer.__signature__ = inspect.signature(SindyLayer.__init__)
except Exception:
    # fallback silently if signature cannot be created in constrained environments
    pass

