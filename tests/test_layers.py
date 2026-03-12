import torch
import numpy as np
import pytest
from vindy.layers import SindyLayer, VindyLayer
from vindy.libraries import PolynomialLibrary
from vindy.distributions import Laplace


class TestSindyLayer:

    def test_forward_shape_1st(self, sindy_layer):
        x = torch.randn(10, 3)
        out = sindy_layer(x)
        assert out.shape == (10, 3)

    def test_forward_shape_2nd(self):
        layer = SindyLayer(
            state_dim=3,
            feature_libraries=[PolynomialLibrary(degree=2, include_bias=True)],
            second_order=True,
        )
        # second_order expects input dim = 2*state_dim = 6
        x = torch.randn(10, 6)
        out = layer(x)
        assert out.shape == (10, 6)

    def test_features_shape(self, sindy_layer):
        x = torch.randn(10, 3)
        feats = sindy_layer.features(x)
        assert feats.shape == (10, sindy_layer.n_bases_functions)

    def test_regularization(self, sindy_layer):
        reg = sindy_layer.regularization_loss()
        assert reg.shape == ()
        assert reg.item() >= 0

    def test_prune_weights(self, sindy_layer):
        # Set some weights small
        with torch.no_grad():
            sindy_layer.kernel.data[:3] = 0.001
        sindy_layer.prune_weights(threshold=0.01)
        assert torch.all(sindy_layer.kernel.data[:3] == 0)

    def test_coeffs_shape(self, sindy_layer):
        coeffs = sindy_layer.get_sindy_coeffs()
        assert coeffs.shape == sindy_layer.coefficient_matrix_shape

    def test_mask(self):
        mask = torch.ones(3, 10)
        mask[0, 0] = 0  # disable first coefficient of first equation
        layer = SindyLayer(
            state_dim=3,
            feature_libraries=[PolynomialLibrary(degree=2, include_bias=True)],
            second_order=False,
            mask=mask,
        )
        coeffs = layer.get_sindy_coeffs()
        assert coeffs[0, 0] == 0

    def test_fixed_coeffs(self):
        fixed = torch.zeros(3, 10)
        fixed[0, 0] = 5.0
        mask = torch.ones(3, 10)
        mask[0, 0] = 0  # fix this position
        layer = SindyLayer(
            state_dim=3,
            feature_libraries=[PolynomialLibrary(degree=2, include_bias=True)],
            second_order=False,
            mask=mask,
            fixed_coeffs=fixed,
        )
        coeffs = layer.get_sindy_coeffs()
        assert coeffs[0, 0] == pytest.approx(5.0)

    def test_integrate(self, sindy_layer):
        z0 = np.array([1.0, 0.0, 0.0])
        t = np.linspace(0, 1, 50)
        sol = sindy_layer.integrate(z0, t)
        assert hasattr(sol, "y")
        assert sol.y.shape[0] == 3

    def test_model_equation_to_str(self, sindy_layer):
        s = sindy_layer.model_equation_to_str()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_with_mu(self):
        layer = SindyLayer(
            state_dim=2,
            param_dim=1,
            feature_libraries=[PolynomialLibrary(degree=2, include_bias=True)],
            second_order=False,
        )
        x = torch.randn(10, 3)  # 2 state + 1 param
        out = layer(x)
        assert out.shape == (10, 2)


class TestVindyLayer:

    def test_train_mode(self, vindy_layer):
        vindy_layer.train()
        x = torch.randn(10, 3)
        out = vindy_layer(x)
        assert isinstance(out, list)
        assert len(out) == 3
        z_dot, mean, log_var = out
        assert z_dot.shape == (10, 3)

    def test_eval_mode(self, vindy_layer):
        vindy_layer.eval()
        x = torch.randn(10, 3)
        out = vindy_layer(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (10, 3)

    def test_kl_loss(self, vindy_layer):
        mean = torch.randn(vindy_layer.n_dofs, 1)
        scale = torch.randn(vindy_layer.n_dofs, 1)
        kl = vindy_layer.kl_loss(mean, scale)
        assert kl.shape == ()

    def test_pdf_thresholding(self, vindy_layer):
        # With large threshold, some coefficients should survive
        vindy_layer.pdf_thresholding(threshold=0.5)
        # Just verify it runs without error

    def test_integrate_uq(self, vindy_layer):
        z0 = np.array([1.0, 0.0, 0.0])
        t = np.linspace(0, 0.5, 20)
        sol, coeffs = vindy_layer.integrate_uq(z0, t)
        assert hasattr(sol, "y")
        assert isinstance(coeffs, torch.Tensor)
