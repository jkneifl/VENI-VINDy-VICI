import torch
import numpy as np
import os
import tempfile
import pytest
from vindy.layers import SindyLayer, VindyLayer
from vindy.libraries import PolynomialLibrary
from vindy.distributions import Laplace
from vindy.networks import IdentificationNetwork


class TestIdentificationFirstOrder:

    def test_training_loss_decreases(self, roessler_data):
        _, x_train, dxdt_train = roessler_data
        layer = SindyLayer(
            state_dim=3,
            feature_libraries=[PolynomialLibrary(degree=2, include_bias=True)],
            second_order=False,
        )
        model = IdentificationNetwork(
            sindy_layer=layer, x=x_train, second_order=False,
        )
        model.compile()
        history = model.fit(
            [x_train, dxdt_train], epochs=5, batch_size=64, verbose=0,
        )
        assert history["loss"][-1] < history["loss"][0]

    def test_with_mu(self, roessler_data):
        _, x_train, dxdt_train = roessler_data
        n = x_train.shape[0]
        mu = np.random.randn(n, 1).astype(np.float32)
        layer = SindyLayer(
            state_dim=3,
            param_dim=1,
            feature_libraries=[PolynomialLibrary(degree=2, include_bias=True)],
            second_order=False,
        )
        model = IdentificationNetwork(
            sindy_layer=layer, x=x_train, mu=mu, second_order=False,
        )
        model.compile()
        history = model.fit(
            [x_train, dxdt_train, mu], epochs=3, batch_size=64, verbose=0,
        )
        assert "loss" in history


class TestIdentificationSecondOrder:

    def test_training(self, second_order_data):
        _, x_train, dxdt_train, dxddt_train = second_order_data
        layer = SindyLayer(
            state_dim=3,
            feature_libraries=[PolynomialLibrary(degree=2, include_bias=True)],
            second_order=True,
        )
        model = IdentificationNetwork(
            sindy_layer=layer, x=x_train, second_order=True,
        )
        model.compile()
        history = model.fit(
            [x_train, dxdt_train, dxddt_train], epochs=5, batch_size=64, verbose=0,
        )
        assert history["loss"][-1] < history["loss"][0]


class TestIdentificationSaveLoad:

    def test_save_load_roundtrip(self, roessler_data):
        _, x_train, dxdt_train = roessler_data
        layer = SindyLayer(
            state_dim=3,
            feature_libraries=[PolynomialLibrary(degree=2, include_bias=True)],
            second_order=False,
        )
        model = IdentificationNetwork(
            sindy_layer=layer, x=x_train, second_order=False,
        )
        model.compile()
        model.fit([x_train, dxdt_train], epochs=2, batch_size=64, verbose=0)
        coeffs_before = model.sindy_coeffs().copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)
            # Create a fresh model and load
            layer2 = SindyLayer(
                state_dim=3,
                feature_libraries=[PolynomialLibrary(degree=2, include_bias=True)],
                second_order=False,
            )
            model2 = IdentificationNetwork(
                sindy_layer=layer2, x=x_train, second_order=False,
            )
            model2.load(tmpdir)
            coeffs_after = model2.sindy_coeffs()

        np.testing.assert_allclose(coeffs_before, coeffs_after, atol=1e-6)


class TestIdentificationIntegrate:

    def test_integrate(self, roessler_data):
        _, x_train, dxdt_train = roessler_data
        layer = SindyLayer(
            state_dim=3,
            feature_libraries=[PolynomialLibrary(degree=2, include_bias=True)],
            second_order=False,
        )
        model = IdentificationNetwork(
            sindy_layer=layer, x=x_train, second_order=False,
        )
        model.compile()
        model.fit([x_train, dxdt_train], epochs=2, batch_size=64, verbose=0)
        z0 = x_train[0]
        t = np.linspace(0, 0.5, 20)
        sol = model.integrate(z0, t)
        assert hasattr(sol, "y")
        assert sol.y.shape[0] == 3


class TestIdentificationVindyVariant:

    def test_vindy_layer(self, roessler_data):
        _, x_train, dxdt_train = roessler_data
        layer = VindyLayer(
            beta=1e-3,
            priors=Laplace(0.0, 1.0),
            state_dim=3,
            feature_libraries=[PolynomialLibrary(degree=2, include_bias=True)],
            second_order=False,
        )
        model = IdentificationNetwork(
            sindy_layer=layer, x=x_train, second_order=False,
        )
        model.compile()
        history = model.fit(
            [x_train, dxdt_train], epochs=3, batch_size=64, verbose=0,
        )
        assert "kl_sindy" in history
