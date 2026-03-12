import numpy as np
import logging
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from .sindy_layer import SindyLayer
from vindy.distributions import Gaussian, BaseDistribution

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


class VindyLayer(SindyLayer):

    def __init__(self, beta=1, priors=Gaussian(0.0, 1.0), **kwargs):
        """
        Layer for variational identification of nonlinear dynamics (VINDy).

        Approximates the time derivative of the latent variable. Feature libraries are
        applied to the latent variables and a (sparse) variational inference is performed
        to obtain the coefficients.

        Parameters
        ----------
        beta : float or int, default=1
            Scaling factor for the KL divergence.
        priors : BaseDistribution or list of BaseDistribution, default=Gaussian(0.0, 1.0)
            Prior distribution(s) for the coefficients.
        **kwargs
            Additional keyword arguments, see SindyLayer.
        """
        super(VindyLayer, self).__init__(**kwargs)
        self.assert_additional_args(beta, priors)
        self.priors = priors
        self.beta = beta

    def assert_additional_args(self, beta, priors):
        """
        Validate that the additional arguments are correct.

        Parameters
        ----------
        beta : float or int
            Scaling factor for the KL divergence.
        priors : BaseDistribution or list of BaseDistribution
            Prior distribution(s) for the coefficients.
        """
        # assert that input arguments are valid that are not checked in the super class
        assert isinstance(beta, float) or isinstance(beta, int), "beta must be a float"
        # priors
        if isinstance(priors, list):
            assert (
                len(priors) == self.n_dofs
            ), f"Number of priors must match the number of dofs ({self.n_dofs})"
            for prior in priors:
                assert isinstance(prior, BaseDistribution), (
                    "All priors must be an instance inheriting from " "BaseDistribution"
                )
        else:
            assert isinstance(
                priors, BaseDistribution
            ), "priors must be a class inheriting from BaseDistribution"

    def init_weigths(self):
        super(VindyLayer, self).init_weigths()

        # initialize the log variance of the coefficients
        self.kernel_scale = nn.Parameter(torch.empty(self.kernel_shape, dtype=self.torch_dtype))
        nn.init.uniform_(self.kernel_scale, -1, 1)

    @property
    def loss_trackers(self):
        return ["kl_sindy"]

    def scale_regularization_loss(self):
        """
        Compute regularization loss on the scale parameter.

        Computes l1 * sum(abs(exp(0.5 * kernel_scale))) + l2 * sum(exp(0.5 * kernel_scale)**2)

        Returns
        -------
        torch.Tensor
            Scale regularization loss.
        """
        reg = torch.tensor(0.0, dtype=self.torch_dtype, device=self.kernel_scale.device)
        exp_half = torch.exp(0.5 * self.kernel_scale)
        if self.l1 > 0:
            reg = reg + self.l1 * torch.sum(torch.abs(exp_half))
        if self.l2 > 0:
            reg = reg + self.l2 * torch.sum(exp_half ** 2)
        return reg

    def regularization_loss(self):
        """
        Compute combined regularization loss for kernel and scale.

        Returns
        -------
        torch.Tensor
            Total regularization loss.
        """
        return super().regularization_loss() + self.scale_regularization_loss()

    @property
    def _coeffs(self):
        """
        Get the coefficients of the SINDy layer sampled from the defined distribution.

        Returns the coefficients sampled from the defined distribution parametrized by the
        layer's kernel (weights).

        Returns
        -------
        tuple of torch.Tensor
            Tuple containing (coeffs, coeffs_mean, coeffs_log_scale).
        """
        # split the kernel into mean and log variance
        coeffs_mean, coeffs_log_scale = self.kernel, self.kernel_scale
        # draw samples from the distribution
        if isinstance(self.priors, list):
            trainable_coeffs = []
            for i, prior in enumerate(self.priors):
                trainable_coeffs.append(
                    prior(coeffs_mean[i : i + 1], coeffs_log_scale[i : i + 1])
                )
            trainable_coeffs = torch.cat(trainable_coeffs, dim=0)
        else:
            trainable_coeffs = self.priors(coeffs_mean, coeffs_log_scale)

        # fill the coefficient matrix with the trainable coefficients
        coeffs = self.fill_coefficient_matrix(trainable_coeffs)

        return coeffs, coeffs_mean, coeffs_log_scale

    def kl_loss(self, mean, scale):
        """
        Compute the KL divergence between the priors and coefficient distributions.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the coefficient distributions.
        scale : torch.Tensor
            Scale (log variance) of the coefficient distributions.

        Returns
        -------
        torch.Tensor
            Scaled KL divergence loss.
        """
        if isinstance(self.priors, list):
            kl_loss = torch.tensor(0.0, dtype=self.torch_dtype, device=mean.device)
            for prior in self.priors:
                kl_loss = kl_loss + prior.KL_divergence(mean, scale)
        else:
            kl_loss = self.priors.KL_divergence(mean, scale)

        return self.beta * torch.sum(kl_loss)

    def get_sindy_coeffs(self):
        _, coeffs_mean, _ = self._coeffs
        coeffs = self.fill_coefficient_matrix(coeffs_mean)
        return coeffs.detach().cpu().numpy()

    def forward(self, inputs):
        """
        Apply the VINDy layer to the inputs.

        Applies the feature libraries to the inputs, samples the coefficients from a
        distribution parametrized by the layer's kernel (weights), and computes
        the dot product of the features and the coefficients.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor or list of torch.Tensor
            If training: [z_dot, coeffs_mean, coeffs_log_var]
            If not training: z_dot
        """
        z_features = self.features(inputs)
        coeffs, coeffs_mean, coeffs_log_var = self._coeffs
        if self.training:
            z_dot = z_features @ coeffs.t()
            return [z_dot, coeffs_mean, coeffs_log_var]
        else:
            # in case of evaluation, we use the mean of the coefficients
            z_dot = z_features @ self.fill_coefficient_matrix(coeffs_mean).t()
            return z_dot

    def visualize_coefficients(self, x_range=None, z=None, mu=None):
        """
        Visualize the coefficients of the SINDy layer as distributions.

        Parameters
        ----------
        x_range : tuple, optional
            Range for x-axis.
        z : array-like, optional
            Latent state variable names.
        mu : array-like, optional
            Parameter variable names.
        """
        # get coefficient parameterization
        _, mean, log_scale = self._coeffs
        _ = self._visualize_coefficients(
            mean.detach().cpu().numpy(), log_scale.detach().cpu().numpy(), x_range=x_range, z=z, mu=mu
        )
        plt.show()

    def _visualize_coefficients(
        self,
        mean,
        log_scale,
        x_range=None,
        y_range=None,
        z=None,
        mu=None,
        figsize=None,
        y_ticks=True,
    ):

        # get name of the corresponding features
        feature_names = self.get_feature_names(z, mu)
        n_variables = self.state_dim
        n_plots = int(self.n_dofs / n_variables)
        mean = mean.reshape(n_variables, n_plots).T
        log_scale = log_scale.reshape(n_variables, n_plots).T
        # create a plot with one subplot for each (trainable) coefficient
        if figsize is None:
            figsize = (n_variables * 10, 10)
        fig, axs = plt.subplots(n_plots, n_variables, figsize=figsize, sharex=True)
        # in case of a one-dimensional system, we append a dimension to axs
        if n_variables == 1:
            axs = axs[:, np.newaxis]
        for j in range(n_variables):
            for i in range(n_plots):
                # draw a vertical line at 0
                axs[i][j].axvline(x=0, color="gray", linestyle="-")
                # plot the distribution of the coefficients
                if isinstance(self.priors, list):
                    distribution = self.priors[i]
                else:
                    distribution = self.priors
                scale = distribution.reverse_log(log_scale[i, j])
                distribution.plot(mean[i, j], scale, ax=axs[i][j])
                # put feature name as ylabel
                if j == 0:
                    axs[i][j].set_ylabel(
                        f"${feature_names[i]}$", rotation=90, labelpad=10
                    )
                # set x range
                if x_range is not None:
                    axs[i][j].set_xlim(x_range)
                # set y range
                if y_range is not None:
                    axs[i][j].set_ylim(y_range)
                if not y_ticks:
                    axs[i][j].set_yticks([])
        plt.tight_layout()
        # ensure that ylabel don't overlap with axis ticks
        plt.subplots_adjust(left=0.1)
        return fig

    def pdf_thresholding(self, threshold: float = 1.0):
        """
        Cancel coefficients based on their probability density function at zero.

        Cancels the coefficients of the SINDy layer if their corresponding probability
        density function at zero is above the threshold, i.e., if pdf(0) > threshold.

        Parameters
        ----------
        threshold : float, default=1.0
            Threshold value for cancelling coefficients.
        """
        # get current
        _, loc, log_scale = self._coeffs
        feature_names = np.array([self.get_feature_names()] * self.state_dim).flatten()
        # cancel coefficients
        for i, (loc_, log_scale_) in enumerate(zip(loc[:-1], log_scale[:-1])):
            # plot the distribution of the coefficients
            if isinstance(self.priors, list):
                distribution = self.priors[i]
            else:
                distribution = self.priors
            loc_val = loc_.detach().cpu().numpy()
            log_scale_val = log_scale_.detach().cpu().numpy()
            scale = distribution.reverse_log(log_scale_val)
            zero_density = distribution.prob_density_fcn(x=0, loc=loc_val, scale=scale)
            if zero_density > threshold:
                # cancel the coefficient
                self.kernel.data[i] = 0
                self.kernel_scale.data[i] = -10
                logging.info(
                    f"Canceling coefficient {feature_names[i]} with pdf(0)={zero_density}"
                )
        self.print()

    def integrate_uq(self, z0, t, mu=None, method="RK45"):

        # sample new set of coefficients
        sampled_coeffs, _, _ = self._coeffs

        def sindy_fcn(t_, inputs):
            return self.call_uq(inputs, coeffs=sampled_coeffs)

        return (
            self.integrate(z0, t, mu=mu, method=method, sindy_fcn=sindy_fcn),
            sampled_coeffs,
        )

    @torch.no_grad()
    def call_uq(self, inputs, coeffs):
        """
        Apply the VINDy layer with given coefficients for uncertainty quantification.

        Parameters
        ----------
        inputs : array-like or torch.Tensor
            Input tensor.
        coeffs : torch.Tensor
            Coefficients to use for the computation.

        Returns
        -------
        ndarray
            Time derivative z_dot.
        """
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=self.torch_dtype)
        if len(inputs.shape) == 1:
            inputs = inputs.unsqueeze(0)
        features = self.features(inputs)
        z_dot = features @ coeffs.t()
        return z_dot.detach().cpu().numpy()
