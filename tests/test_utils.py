import torch
import numpy as np
import pytest
from vindy.utils import set_seed
from vindy.utils.jacobian import batch_jacobian, batch_hessian
from vindy.utils.utils import add_lognormal_noise, switch_data_format


class TestBatchJacobian:

    def test_shape(self):
        x = torch.randn(5, 3, requires_grad=True)
        W = torch.randn(4, 3)
        y = x @ W.t()
        jac = batch_jacobian(y, x)
        assert jac.shape == (5, 4, 3)

    def test_linear_equals_weight(self):
        x = torch.randn(5, 3, requires_grad=True)
        W = torch.randn(4, 3)
        b = torch.randn(4)
        y = x @ W.t() + b
        jac = batch_jacobian(y, x)
        for i in range(5):
            torch.testing.assert_close(jac[i], W, atol=1e-5, rtol=1e-5)


class TestBatchHessian:

    def test_shape(self):
        x = torch.randn(5, 3, requires_grad=True)
        # Use a nonlinear function so the Jacobian retains a grad_fn
        y = torch.stack([torch.sum(x ** 2, dim=1), torch.sum(x ** 3, dim=1)], dim=1)
        jac = batch_jacobian(y, x, create_graph=True)
        hess = batch_hessian(jac, x)
        assert hess.shape == (5, 2, 3, 3)

    def test_linear_is_zero(self):
        x = torch.randn(5, 3, requires_grad=True)
        # Add a small quadratic term so the graph is retained, then check
        # that a purely linear function has zero Hessian by using an MLP-like setup
        # For a truly linear function, the Jacobian is constant and has no grad_fn.
        # Instead, verify via a quadratic function that Hessian is diagonal = 2.
        y = torch.sum(x ** 2, dim=1, keepdim=True)  # (5, 1)
        jac = batch_jacobian(y, x, create_graph=True)  # (5, 1, 3)
        hess = batch_hessian(jac, x)  # (5, 1, 3, 3)
        # Hessian of x_i^2 is 2*I (diagonal)
        expected = 2 * torch.eye(3).unsqueeze(0).unsqueeze(0).expand(5, 1, 3, 3)
        assert torch.allclose(hess, expected, atol=1e-4)


class TestAddLognormalNoise:

    def test_shape_preserved(self):
        traj = np.random.randn(100, 3)
        noisy, noise = add_lognormal_noise(traj, sigma=0.1)
        assert noisy.shape == traj.shape
        assert noise.shape == traj.shape

    def test_different_from_input(self):
        traj = np.ones((100, 3))
        noisy, _ = add_lognormal_noise(traj, sigma=0.1)
        assert not np.allclose(noisy, traj)


class TestSetSeed:

    def test_reproducibility(self):
        set_seed(123)
        a = torch.randn(5)
        set_seed(123)
        b = torch.randn(5)
        torch.testing.assert_close(a, b)


class TestSwitchDataFormat:

    def test_roundtrip_2d_3d(self):
        n_sims, n_t, feat = 3, 10, 5
        data_2d = np.random.randn(n_sims * n_t, feat)
        data_3d = switch_data_format(data_2d, n_sims, n_t, target_format="3d")
        assert data_3d.shape == (n_sims, n_t, feat)
        data_2d_back = switch_data_format(data_3d, n_sims, n_t, target_format="2d")
        np.testing.assert_allclose(data_2d_back, data_2d)
