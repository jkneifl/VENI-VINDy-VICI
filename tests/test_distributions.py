import torch
import pytest
from vindy.distributions import Gaussian, Laplace


class TestGaussian:

    def test_forward_shape(self):
        g = Gaussian()
        mean = torch.randn(5, 3)
        log_var = torch.randn(5, 3)
        out = g(mean, log_var)
        assert out.shape == mean.shape

    def test_kl_standard_normal(self):
        g = Gaussian(prior_mean=0.0, prior_variance=1.0)
        mean = torch.zeros(5, 3)
        log_var = torch.zeros(5, 3)
        kl = g.KL_divergence(mean, log_var)
        assert kl.shape == mean.shape
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)

    def test_kl_positive(self):
        g = Gaussian()
        mean = torch.ones(5, 3)
        log_var = torch.ones(5, 3)
        kl = g.KL_divergence(mean, log_var)
        assert torch.all(kl > 0)


class TestLaplace:

    def test_forward_shape(self):
        lap = Laplace()
        loc = torch.randn(5, 3)
        log_scale = torch.randn(5, 3)
        out = lap(loc, log_scale)
        assert out.shape == loc.shape

    def test_kl_standard(self):
        lap = Laplace(prior_mean=0.0, prior_scale=1.0)
        mean = torch.zeros(5, 3)
        log_scale = torch.zeros(5, 3)  # scale = exp(0) = 1
        kl = lap.KL_divergence(mean, log_scale)
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)

    def test_kl_positive(self):
        lap = Laplace()
        mean = torch.ones(5, 3)
        log_scale = torch.ones(5, 3)
        kl = lap.KL_divergence(mean, log_scale)
        assert torch.all(kl > 0)
