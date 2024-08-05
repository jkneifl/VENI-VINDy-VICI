import tensorflow as tf
import numpy as np
import inspect
from vindy.libraries import PolynomialLibrary, BaseLibrary
from sympy import symbols


class SindyLayer(tf.keras.layers.Layer):
    """
    Layer for SINDy approximation of the time derivative of the latent variable.
    Feature libraries are applied to the latent variable and its time derivative, and a sparse regression is performed.

    Parameters
    ----------
    state_dim : int
        Number of latent variables.
    param_dim : int, optional
        Number of parameters (default is 0).
    feature_libraries : list, optional
        List of feature libraries for the latent variables (default is [PolynomialLibrary(degree=3)]).
    param_feature_libraries : list, optional
        List of feature libraries for the parameters (default is []).
    second_order : bool, optional
        If True, enforce 2nd order structure (default is True).
    kernel_regularizer : tf.keras.regularizers.Regularizer, optional
        Regularizer for the kernel (default is tf.keras.regularizers.L1L2(l1=1e-3, l2=0)).
    x_mu_interaction : bool, optional
        If True, interaction between latent variables and parameters (default is True).
    mask : array-like, optional
        If required, certain coefficients of the latent governing equations can be fixed and are consequently masked out for training (default is None).
    fixed_coeffs : array-like, optional
        Values for the coefficients that are masked out during training (default is None).
    dtype : str, optional
        Data type of the layer (default is "float32").
    kwargs : dict
        Additional arguments for the TensorFlow layer class.
    """

    def __init__(
        self,
        state_dim,
        param_dim=0,
        feature_libraries: list = [],
        param_feature_libraries: list = [],
        second_order=True,
        kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-3, l2=0),
        x_mu_interaction=True,
        mask=None,
        fixed_coeffs=None,
        dtype="float32",
        **kwargs,
    ):
        """
        Layer for SINDy approximation of the time derivative of the latent variable
        feature libraries are applied to the latent variable and its time derivative and a sparse regression is performed
        :param state_dim: (int) number of latent variables
        :param param_dim: (int) number of parameters
        :param feature_libraries: (list) list of feature libraries for the latent variables
        :param param_feature_libraries: (list) list of feature libraries for the parameters
        :param second_order: (bool) if True, enforce 2nd order structure,
                    i.e. d/dt [z, z_d] = [z_dot, Theta(z, z_dot)@Xi] where Theta(z, z_dot) is a feature library and
                    Xi are the coefficients to be identified
        :param kernel_regularizer: (tf.keras.regularizers.Regularizer) regularizer for the kernel
        :param x_mu_interaction: (bool) if True, interaction between latent variables and parameters
        :param mask: (array-like) If required certain coefficients of the latent governing equations can be fixed
                    and are consequently masked out for training
        :param fixed_coeffs: (array-like) values for the coefficients that are masked out during training
        :param dtype: (str) data type of the layer
        :param kwargs: additional arguments for the tensorflow layer class
        """
        super(SindyLayer, self).__init__(**kwargs)

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
        self.n_bases_functions = self.tfFeat(
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
        Returns the loss trackers of the layer if any. The standard sindy layer has no loss trackers.

        Returns
        -------
        dict
            Loss trackers.
        """
        return dict()

    def _init_to_config(self, init_locals):
        """
        Save the parameters with which the model was initialized except for the data itself.

        Parameters
        ----------
        init_locals : dict
            Local variables from the __init__ function.
        """

        sig = inspect.signature(self.__init__)
        keys = [param.name for param in sig.parameters.values()]
        values = [init_locals[name] for name in keys]
        init_dict = dict(zip(keys, values))
        self.config = init_dict

    def assert_arguments(self, arguments):
        """
        Assert that the arguments passed to the layer are valid.

        Parameters
        ----------
        arguments : dict
            Arguments passed to the layer.
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
        # assert that feature_libraries is a list of aesindy.libraries objects
        assert isinstance(
            arguments["feature_libraries"], list
        ), "feature_libraries must be a list"
        for lib in arguments["feature_libraries"]:
            assert isinstance(
                lib, BaseLibrary
            ), "feature_libraries must be a list of aesindy.libraries objects"
        # assert that param_feature_libraries is a list of aesindy.libraries objects
        assert isinstance(
            arguments["param_feature_libraries"], list
        ), "param_feature_libraries must be a list"
        for lib in arguments["param_feature_libraries"]:
            assert isinstance(
                lib, BaseLibrary
            ), "param_feature_libraries must be a list of aesindy.libraries objects"
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
        """
        Returns the shape of the coefficient matrix.

        Returns
        -------
        tuple
            Shape of the coefficient matrix.
        """
        return (self.output_dim, self.n_bases_functions)

    @property
    def kernel_shape(self):
        """
        Returns the dimension of the kernel (weights) of the SINDy layer.

        Returns
        -------
        tuple
            Shape of the kernel.
        """
        return (self.n_dofs, 1)

    def init_weigths(self, kernel_regularizer):
        """
        Initialize the weights of the SINDy layer.

        Parameters
        ----------
        kernel_regularizer : tf.keras.regularizers.Regularizer
            Regularizer for the kernel.
        """

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
        Set the mask and fixed coefficients for the SINDy layer to mask out certain coefficients and set their values.

        Parameters
        ----------
        mask : array-like or None
            Mask for the coefficients. If None, a mask of ones is used.
        fixed_coeffs : array-like or None, optional
            Fixed coefficients for the masked values. If None, a matrix of zeros is used.

        Returns
        -------
        tuple
            A tuple containing the mask and fixed coefficients.
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
            Coefficient matrix.
        """
        # fill the coefficient matrix with the trainable coefficients
        coeffs = self.fill_coefficient_matrix(self.kernel)

        return coeffs

    def get_sindy_coeffs(self):
        """
        Get the SINDy coefficients as a numpy array.

        Returns
        -------
        np.ndarray
            SINDy coefficients.
        """
        return self._coeffs.numpy()

    def get_prunable_weights(self):
        """
        Get the prunable weights of the SINDy layer.

        Returns
        -------
        list
            List of prunable weights.
        """
        # Prune bias also, though that usually harms model accuracy too much.
        return [self.kernel]

    def prune_weights(self, threshold=0.01, training=False):
        """
        Prune the weights of the SINDy layer by setting values below a threshold to zero.

        Parameters
        ----------
        threshold : float, optional
            Threshold for pruning (default is 0.01).
        training : bool, optional
            Whether the layer is in training mode (default is False).

        Returns
        -------
        None
        """
        mask = tf.math.greater(tf.math.abs(self.kernel), threshold * tf.ones_like(self.kernel, dtype=self.kernel.dtype))
        mask = tf.cast(mask, dtype=self.kernel.dtype)
        self.kernel.assign(tf.multiply(self.kernel, mask))

    def fill_coefficient_matrix(self, trainable_coeffs):
        """
        Fill the coefficient matrix with the trainable coefficients.

        Parameters
        ----------
        trainable_coeffs : array-like
            Trainable coefficients.

        Returns
        -------
        tf.Tensor
            Coefficient matrix filled with the trainable coefficients.
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
        Perform the forward pass of the SINDy layer.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.
        training : bool, optional
            Whether the layer is in training mode (default is False).

        Returns
        -------
        tf.Tensor
            Output tensor after applying the SINDy layer.
        """

        z_features = self.tfFeat(inputs)
        z_dot = z_features @ tf.transpose(self._coeffs)
        return z_dot

    @tf.function
    def tfFeat(self, inputs):
        """
        Combine all features for the SINDy layer.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.

        Returns
        -------
        tf.Tensor
            Combined features.
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

    @tf.function
    def concat_features(self, z, libraries):
        """
        Concatenate features from different libraries.

        Parameters
        ----------
        z : tf.Tensor
            Input tensor.
        libraries : list
            List of feature libraries.

        Returns
        -------
        tf.Tensor
            Concatenated features.
        """
        features = [library(z) for library in libraries]
        z_feat = tf.concat(features, axis=1)
        return z_feat

    def get_feature_names(self, z=None, mu=None):
        """
        Construct feature names for states and parameters.

        Parameters
        ----------
        z : list of str, optional
            Names of the states, e.g., \['z1', 'z2', ...\] (default is None).
        mu : list of str, optional
            Names of the parameters, e.g., \['mu1', 'mu2', ...\] (default is None).

        Returns
        -------
        list of str
            List of feature names.
        """

        if z is None:
            z = [f"z_{i}" for i in range(self.output_dim)]
        if mu is None:
            mu = [f"\u03BC_{i}" for i in range(self.param_dim)]

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
        Print the model equation.

        Parameters
        ----------
        z : list of str, optional
            Names of the states, e.g., \['z1', 'z2', ...\] (default is None).
        mu : list of str, optional
            Names of the parameters, e.g., \['mu1', 'mu2', ...\] (default is None).
        precision : int, optional
            Number of decimal places (default is 3).

        Returns
        -------
        None
        """
        print(self.model_equation_to_str(z, mu, precision))

    def model_equation_to_str(self, z=None, mu=None, precision: int = 3):
        """
        Convert coefficients and feature names into a readable equation.

        Parameters
        ----------
        z : list of str, optional
            Names of the states, e.g., \['z1', 'z2', ...\] (default is None).
        mu : list of str, optional
            Names of the parameters, e.g., \['mu1', 'mu2', ...\] (default is None).
        precision : int, optional
            Number of decimal places (default is 3).

        Returns
        -------
        str
            Model equation as a string.
        """
        if z is None:
            z = [f"z{i}" for i in range(self.output_dim)]
        if mu is None:
            mu = [f"\u03BC{i}" for i in range(self.param_dim)]
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
