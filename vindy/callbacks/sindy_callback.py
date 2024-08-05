import numpy as np
import tensorflow as tf
import pysindy as ps
from vindy.layers.vindy_layer import VindyLayer


class SindyCallback(tf.keras.callbacks.Callback):
    """
    Callback for the SINDy layer. This callback is used to update the identified coefficients of the SINDy layer.
    """

    def __init__(
        self,
        x,
        dxdt,
        dxddt,
        mu,
        t,
        freq=100,
        train_end=False,
        ensemble=False,
        subset_size=0.5,
        n_subsets=100,
        threshold=0.1,
        thresholder: str = "l0",
        z_names=None,
        mu_names=None,
        print_precision=3,
        **kwargs,
    ):
        """
        Initialize the SINDy callback.

        Parameters
        ----------
        x : array-like
            Training data.
        dxdt : array-like
            Time derivative of training data.
        dxddt : array-like
            Second order time derivative of training data.
        mu : array-like
            Parameter of the system.
        t : array-like
            Time.
        freq : int, optional
            Frequency of the update (default is 100).
        train_end : bool, optional
            Whether to perform update at the end of training (default is False).
        ensemble : bool, optional
            Whether to use ensemble method for SINDy (default is False).
        subset_size : float, optional
            Size of the subset for ensemble method (default is 0.5).
        n_subsets : int, optional
            Number of subsets for ensemble method (default is 100).
        threshold : float, optional
            Threshold for the update (default is 0.1).
        thresholder : str, optional
            Regularization function to use (default is "l0").
        z_names : list of str, optional
            Names of the latent variables (default is None).
        mu_names : list of str, optional
            Names of the parameters (default is None).
        print_precision : int, optional
            Precision of the printed model equation (default is 3).
        kwargs : dict
            Additional arguments.
        """
        assert (
            isinstance(subset_size, float) and subset_size > 0 and subset_size <= 1
        ), "subset_size must be a float between 0 and 1"
        assert (
            isinstance(n_subsets, int) and n_subsets > 0
        ), "n_subsets must be a positive integer"

        self.threshold = threshold
        self.freq = freq
        self.train_end = train_end
        self.ensemble = ensemble
        self.subset_size = subset_size
        self.n_subsets = n_subsets
        self.x = x
        self.dxdt = dxdt
        self.dxddt = dxddt
        self.mu = mu
        self.t = t
        self.thresholder = thresholder
        self.z_names = z_names
        self.mu_names = mu_names
        self.print_precision = print_precision
        super().__init__(**kwargs)

    def prepare_data_for_pysindy(self, t, z_feat, dzdt, dzddt=None):
        """
        Prepare training data for the separate SINDy model.

        Parameters
        ----------
        t : array-like
            Time.
        z_feat : array-like
            Latent variables and their features.
        dzdt : array-like
            Time derivative of latent variables.
        dzddt : array-like, optional
            Second time derivative of latent variables (default is None).

        Returns
        -------
        tuple
            Prepared latent variables, their time derivatives, and time.
        """
        n_samples = z_feat.shape[0]
        n_timesteps = t.shape[0]
        n_sims = int(n_samples / n_timesteps)
        # pysindy requires multiple simulations to be represented as list
        z_feat = list(z_feat.numpy().reshape([n_sims, n_timesteps, -1]))
        if dzddt is None:
            z_dot = list(dzdt.reshape([n_sims, n_timesteps, -1]))
        else:
            z_dot = list(
                np.concatenate([dzdt, dzddt], axis=1).reshape([n_sims, n_timesteps, -1])
            )

        return z_feat, z_dot, t

    def process_data_for_sindy(self):
        """
        Process data for SINDy optimization.

        Returns
        -------
        tuple
            Processed latent variables, their time derivatives, time, fixed coefficients, feature IDs, and mask.
        """
        # concatenate the input for the sindy layer
        mu = self.mu
        # calculate the time derivatives of the latent variable given the training data
        if self.model.second_order:
            z, dzdt, dzddt = self.model.calc_latent_time_derivatives(
                self.x, self.dxdt, self.dxddt
            )
            z_sindy = self.model.concatenate_sindy_input(z, dzdt=dzdt, mu=mu)
        else:
            z, dzdt = self.model.calc_latent_time_derivatives(self.x, self.dxdt, None)
            dzddt = None
            z_sindy = self.model.concatenate_sindy_input(z, dzdt=None, mu=mu)

        # get the mask which specifies which of the features are used in the sindy_model
        mask = self.sindylayer.mask.numpy()
        # the fixed coefficients are the coefficients which are not optimized but fixed to a specific value
        fixed_coeffs = self.sindylayer.fixed_coeffs.numpy()

        # %% for the seperate sindy optimization, we only train on the selected features
        # hence we create a seelction matrix which extracts the selected features

        # ids of the selected features
        feat_ids = np.unique(
            np.argwhere(np.concatenate([mask, fixed_coeffs], axis=0) != 0)[:, 1]
        )
        # create selection matrix
        mask_selection = np.zeros([feat_ids.size, self.sindylayer.n_bases_functions])
        mask_selection[range(feat_ids.size), feat_ids] = 1

        # %% data preparation for the seperate sindy sindy_model

        # apply the basis functions to the latent variable
        z_feat = self.sindylayer.tfFeat(z_sindy)
        # apply the selection matrix to the basis functions
        z_feat = tf.matmul(z_feat, mask_selection.T)

        # bring the data in the right format for pysindy
        z_feat, z_dot, times_train = self.prepare_data_for_pysindy(
            self.t, z_feat, dzdt, dzddt
        )
        return z_feat, z_dot, times_train, fixed_coeffs, feat_ids, mask

    def perform_update(self):
        """
        Perform the SINDy update.
        """
        z_feat, z_dot, times_train, fixed_coeffs, feat_ids, mask = (
            self.process_data_for_sindy()
        )

        if self.ensemble:
            # get partial permutations of the data with replacement
            n_t = z_feat[0].shape[0]
            sindy_models = []
            dt = (times_train[1] - times_train[0])[0]
            for i in range(self.n_subsets):
                perm_ids = np.random.choice(
                    np.arange(n_t), size=int(n_t * self.subset_size), replace=False
                )
                perm_ids.sort()
                z_feat_subset = [z_feat_[perm_ids, :] for z_feat_ in z_feat]
                z_dot_subset = [z_dot_[perm_ids, :] for z_dot_ in z_dot]
                sindy_model = self.call_pySINDy(
                    z_feat_subset, z_dot_subset, dt, fixed_coeffs, feat_ids, mask
                )
                # sindy_model.print()
                sindy_models.append(sindy_model)
            # average the coefficients of the different models
            coeffs = np.array(
                [sindy_model.coefficients() for sindy_model in sindy_models]
            )
            variance = np.var(coeffs, axis=0)
            coeffs = np.mean(coeffs, axis=0)
        else:
            sindy_model = self.call_pySINDy(
                z_feat, z_dot, times_train, fixed_coeffs, feat_ids, mask
            )
            coeffs = sindy_model.coefficients()
            variance = None
        self.update_weights(coeffs, variance)

    def update_weights(self, weights, variance=None):
        """
        Update the weights of the SINDy layer.

        Parameters
        ----------
        weights : array-like
            Weights to update.
        variance : array-like, optional
            Variance of the weights (default is None).
        """
        # %% set the weights of the original sindy_model to the weights of the seperate sindy sindy_model
        # check if weights contain nan or inf values

        # only get trainable weights
        trainable_ids = list(self.sindylayer.dof_ids.numpy())
        weights = np.array([weights[id[0], id[1]] for id in trainable_ids])

        if np.isnan(weights).any() or np.isinf(weights).any():
            tf.print("SINDy optimization failed. NaN values in weights.")
        else:
            # set weights
            for weight_id, w in enumerate(weights):
                self.sindylayer.kernel[weight_id].assign(w)
            # update variance in case of variational sindy layer
            if isinstance(self.sindylayer, VindyLayer):
                # update variance
                if variance is not None:
                    variance = np.array(
                        [variance[id[0], id[1]] for id in trainable_ids]
                    )
                    # calculate log scale from variance
                    priors = self.sindylayer.priors
                    if isinstance(priors, list):
                        log_scale = np.array(
                            [
                                prior.variance_to_log_scale(v).numpy()
                                for prior, v in zip(priors, variance)
                            ]
                        )
                    else:
                        log_scale = priors.variance_to_log_scale(variance).numpy()
                    # set all -inf values to -100
                    log_scale[log_scale == -np.inf] = -100
                    for weight_id, s in enumerate(log_scale):
                        self.sindylayer.kernel_scale[weight_id].assign(s)

    def call_pySINDy(self, z_feat, z_dot, times_train, fixed_coeffs, feat_ids, mask):
        """
        Fit a separate SINDy model using the provided data and constraints.

        Parameters
        ----------
        z_feat : list of array-like
            Latent variables and their features.
        z_dot : list of array-like
            Time derivatives of latent variables.
        times_train : array-like
            Time.
        fixed_coeffs : array-like
            Fixed coefficients for the SINDy model.
        feat_ids : array-like
            IDs of the selected features.
        mask : array-like
            Mask specifying which features are used in the SINDy model.

        Returns
        -------
        pysindy.SINDy
            Fitted SINDy model.
        """
        # %% Define the constraints for the optimization of the seperate sindy sindy_model
        # we want to apply the same mask and fixed coefficients as in the original sindy_model
        # -> rewrite the constraints for pysindy

        # get the fixed coefficients for the selected features
        fixed_coeffs = fixed_coeffs[:, feat_ids]

        # define the lhs constraints (n_constraints x n_features * n_outputs) and the rhs constraints (n_constraints, )
        ids = np.argwhere(mask[:, feat_ids] == 0).squeeze()
        constraint_lhs = np.zeros((ids.shape[0], fixed_coeffs.size))
        constraint_lhs[range(ids.shape[0]), ids[:, 0] * feat_ids.size + ids[:, 1]] = 1
        constraint_rhs = fixed_coeffs[ids[:, 0], ids[:, 1]]
        optimizer = ps.ConstrainedSR3(
            threshold=self.threshold,
            constraint_rhs=constraint_rhs,
            constraint_lhs=constraint_lhs,
            thresholder=self.thresholder,
            max_iter=10000,
        )

        # %% fit the seperate sindy sindy_model
        # feature_names = np.array(self.sindylayer.get_feature_names(z=["z", "dz"]))[feat_ids]
        sindy_model = ps.SINDy(
            # feature_names=feature_names,
            feature_library=ps.PolynomialLibrary(
                degree=1, include_interaction=False, include_bias=False
            ),
            optimizer=optimizer,
        )

        sindy_model.fit(
            z_feat, t=times_train, u=None, multiple_trajectories=True, x_dot=z_dot
        )

        return sindy_model

    def on_train_end(self, logs=None):
        """
        Perform SINDy update at the end of training if specified.

        Parameters
        ----------
        logs : dict, optional
            Dictionary of logs (default is None).
        """
        if self.train_end:
            self.sindylayer = self.model.sindy_layer
            self.perform_update()

    def on_epoch_end(self, epoch, logs=None):
        """
        Perform SINDy update at the end of every specified number of epochs.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        logs : dict, optional
            Dictionary of logs (default is None).
        """
        # only perform sindy update every freq epochs
        if (epoch + 1) % self.freq == 0:
            self.sindylayer = self.model.sindy_layer
            tf.print("Epoch: ", epoch)
            tf.print("Previous SINDy coefficients: ")
            tf.print(
                self.sindylayer.model_equation_to_str(
                    self.z_names, self.mu_names, self.print_precision
                )
            )
            self.perform_update()

            # logging
            tf.print("Updated SINDy coefficients: ")
            tf.print(
                self.sindylayer.model_equation_to_str(
                    self.z_names, self.mu_names, self.print_precision
                )
            )
