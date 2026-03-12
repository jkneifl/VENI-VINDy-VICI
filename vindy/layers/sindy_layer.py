import torch
import torch.nn as nn
import numpy as np
import scipy
import inspect
from vindy.libraries import PolynomialLibrary, BaseLibrary
from sympy import symbols
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


class SindyLayer(nn.Module):
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
    l1 : float, optional
        L1 regularization weight for the kernel.
    l2 : float, optional
        L2 regularization weight for the kernel.
    x_mu_interaction : bool, optional
        If True, include interaction features between state and parameters.
    mask : array-like, optional
        Mask to fix or remove certain coefficients from training.
    fixed_coeffs : array-like, optional
        Values for coefficients that are fixed (applied after masking).
    dtype : str, optional
        Data type used by the layer (e.g. 'float32').
    """

    def __init__(
        self,
        state_dim,
        param_dim=0,
        feature_libraries=None,
        param_feature_libraries=None,
        second_order=True,
        l1=1e-3,
        l2=0.0,
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
        self.torch_dtype = torch.float32 if dtype == "float32" else torch.float64
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
            torch.ones((1, self.output_dim + param_dim), dtype=self.torch_dtype)
        ).shape[1]

        # set certain values of kernel
        self.mask, self.fixed_coeffs = self.set_mask(mask, fixed_coeffs)

        self.second_order = second_order
        if second_order:
            # enforcing the structure of the 2nd order model
            zero_matrix = torch.zeros(
                self.state_dim, self.state_dim + 1, dtype=self.torch_dtype
            )
            eye_matrix = torch.eye(self.state_dim, dtype=self.torch_dtype)
            zero_matrix2 = torch.zeros(
                self.state_dim, self.n_bases_functions - int(state_dim * 2) - 1,
                dtype=self.torch_dtype,
            )
            # apply the structure of the 2nd order model
            fixed_kernel = torch.cat([zero_matrix, eye_matrix, zero_matrix2], dim=1)
            self.mask = torch.cat(
                [torch.zeros_like(fixed_kernel), self.mask], dim=0
            )
            self.fixed_coeffs = torch.cat([fixed_kernel, self.fixed_coeffs], dim=0)

        # register mask and fixed_coeffs as buffers (not trainable, move with model)
        self.register_buffer('_mask', self.mask)
        self.register_buffer('_fixed_coeffs', self.fixed_coeffs)
        # update references to use registered buffers
        self.mask = self._mask
        self.fixed_coeffs = self._fixed_coeffs

        # initialize sindy coefficients
        self.l1, self.l2 = l1, l2
        self.init_weigths()

    @property
    def loss_trackers(self):
        """
        Return loss tracker names used by the layer.

        Returns
        -------
        list
            List of loss tracker name strings. By default empty for SINDy layer.
        """
        return []

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
        ], "dtype must be either float32 or float64"
        assert isinstance(arguments["state_dim"], int), "state_dim must be an integer"
        assert isinstance(arguments["param_dim"], int), "param_dim must be an integer"
        # assert that mask and fixed_coeffs have the right shape
        assert (
            isinstance(arguments["mask"], np.ndarray)
            or isinstance(arguments["mask"], torch.Tensor)
            or arguments["mask"] is None
        ), "mask must be either None, a numpy array or a torch tensor"
        assert (
            isinstance(arguments["fixed_coeffs"], np.ndarray)
            or isinstance(arguments["fixed_coeffs"], torch.Tensor)
            or arguments["fixed_coeffs"] is None
        ), "fixed_coeffs must be either None, a numpy array or a torch tensor"
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
        assert isinstance(arguments["l1"], (float, int)), "l1 must be a float or int"
        assert isinstance(arguments["l2"], (float, int)), "l2 must be a float or int"

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

    def init_weigths(self):

        # get amount of dofs (equals the number of ones in the mask)
        self.n_dofs = int(torch.sum(self.mask).item())
        # get ids of dofs
        self.dof_ids = torch.nonzero(self.mask == 1)
        self.register_buffer('_dof_ids', self.dof_ids)

        self.kernel = nn.Parameter(torch.empty(self.kernel_shape, dtype=self.torch_dtype))
        nn.init.uniform_(self.kernel, -1, 1)

    def regularization_loss(self):
        """
        Compute L1/L2 regularization loss for the kernel.

        Returns
        -------
        torch.Tensor
            Regularization loss.
        """
        reg = torch.tensor(0.0, dtype=self.torch_dtype, device=self.kernel.device)
        if self.l1 > 0:
            reg = reg + self.l1 * torch.sum(torch.abs(self.kernel))
        if self.l2 > 0:
            reg = reg + self.l2 * torch.sum(self.kernel ** 2)
        return reg

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
            mask = torch.ones(self.state_dim, self.n_bases_functions, dtype=self.torch_dtype)
        if fixed_coeffs is None:
            fixed_coeffs = torch.zeros(self.state_dim, self.n_bases_functions, dtype=self.torch_dtype)

        if isinstance(mask, np.ndarray):
            mask = torch.tensor(mask, dtype=self.torch_dtype)
        else:
            mask = mask.to(dtype=self.torch_dtype)
        if mask.shape != (self.state_dim, self.n_bases_functions):
            # bring mask to the right shape by padding ones
            pad_size = self.n_bases_functions - mask.shape[1]
            if pad_size > 0:
                mask = torch.nn.functional.pad(mask, (0, pad_size), value=1.0)

        if isinstance(fixed_coeffs, np.ndarray):
            fixed_coeffs = torch.tensor(fixed_coeffs, dtype=self.torch_dtype)
        else:
            fixed_coeffs = fixed_coeffs.to(dtype=self.torch_dtype)
        if fixed_coeffs.shape != (self.state_dim, self.n_bases_functions):
            # bring fixed_coeffs to the right shape by padding zeros
            pad_size = self.n_bases_functions - fixed_coeffs.shape[1]
            if pad_size > 0:
                fixed_coeffs = torch.nn.functional.pad(fixed_coeffs, (0, pad_size), value=0.0)

        return mask, fixed_coeffs

    @property
    def _coeffs(self):
        """
        Get the coefficients of the SINDy layer as a matrix.

        Returns
        -------
        torch.Tensor
            Coefficient matrix with shape (output_dim, n_bases_functions).
        """
        # fill the coefficient matrix with the trainable coefficients
        coeffs = self.fill_coefficient_matrix(self.kernel)

        return coeffs

    def get_sindy_coeffs(self):
        return self._coeffs.detach().cpu().numpy()

    def get_prunable_weights(self):
        return [self.kernel]

    def prune_weights(self, threshold=0.01, training=False):
        mask = torch.abs(self.kernel) > threshold
        mask = mask.to(dtype=self.kernel.dtype)
        self.kernel.data.copy_(self.kernel.data * mask)

    def fill_coefficient_matrix(self, trainable_coeffs):
        """
        Insert the trainable coefficients into the full coefficient matrix.

        Parameters
        ----------
        trainable_coeffs : array-like
            Trainable coefficients arranged to match the active DOFs.

        Returns
        -------
        torch.Tensor
            Full coefficient matrix with fixed coefficients applied.
        """
        # create a zero matrix for the coefficients with the correct shape
        coeffs = torch.zeros(self.coefficient_matrix_shape, dtype=self.torch_dtype,
                             device=trainable_coeffs.device)
        # put the coefficients into the coefficient matrix Xi at the correct positions
        coeffs[self._dof_ids[:, 0], self._dof_ids[:, 1]] = trainable_coeffs[:, 0]

        # apply the mask
        if self._fixed_coeffs is not None:
            coeffs = coeffs + self._fixed_coeffs

        return coeffs

    def forward(self, inputs):
        """
        Forward pass of the SINDy layer: evaluate features and compute prediction.

        Parameters
        ----------
        inputs : torch.Tensor
            Latent variables, shape ``(batch_size, latent_dim)``.

        Returns
        -------
        torch.Tensor
            Predicted derivatives with shape ``(batch_size, output_dim)``.
        """
        z_features = self.features(inputs)
        z_dot = z_features @ self._coeffs.t()
        return z_dot

    def features(self, inputs):
        """
        Compute concatenated features from configured libraries.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor that contains state and (optionally) parameter values.

        Returns
        -------
        torch.Tensor
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
                return torch.cat([z_feat, param_feat], dim=1)
            return z_feat

    def concat_features(self, z, libraries):
        """
        Concatenate outputs of several feature libraries.

        Parameters
        ----------
        z : torch.Tensor
            Input to the feature libraries.
        libraries : list
            Iterable of library objects that are callable on ``z``.

        Returns
        -------
        torch.Tensor
            Concatenated feature outputs along the last axis.
        """
        features = [library(z) for library in libraries]
        z_feat = torch.cat(features, dim=-1)
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

        # convert to numpy for scipy
        if isinstance(z0, torch.Tensor):
            z0 = z0.detach().cpu().numpy()
        if isinstance(t, torch.Tensor):
            t = t.detach().cpu().numpy()
        z0 = np.asarray(z0, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)

        if sindy_fcn is None:
            sindy_fcn = self.rhs_
        if mu is not None:
            if not callable(mu):
                if isinstance(mu, torch.Tensor):
                    mu = mu.detach().cpu().numpy()
                mu = np.asarray(mu, dtype=np.float64)
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
        )
        return sol

    @torch.no_grad()
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
        ndarray
            Time derivative evaluated at the given inputs.
        """
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=self.torch_dtype)
        if len(inputs.shape) == 1:
            inputs = inputs.unsqueeze(0)
        result = self(inputs)
        return result.detach().cpu().numpy()
