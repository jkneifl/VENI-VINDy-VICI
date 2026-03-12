import torch
import numpy as np
import tempfile
import pytest
from vindy.layers import SindyLayer
from vindy.libraries import PolynomialLibrary
from vindy.networks import AutoencoderSindy


def _make_model(x_train, second_order=False, activation="selu"):
    reduced_order = 2
    layer = SindyLayer(
        state_dim=reduced_order,
        feature_libraries=[PolynomialLibrary(degree=2, include_bias=True)],
        second_order=second_order,
    )
    model = AutoencoderSindy(
        sindy_layer=layer,
        reduced_order=reduced_order,
        x=x_train,
        layer_sizes=[8, 8],
        activation=activation,
        second_order=second_order,
    )
    return model


class TestAutoEncoderShapes:

    def test_encode_decode_shapes(self, roessler_data):
        _, x_train, _ = roessler_data
        model = _make_model(x_train)
        x_t = torch.tensor(x_train[:5], dtype=torch.float32)
        z = model.encode(x_t)
        assert z.shape == (5, 2)
        x_rec = model.decode(z)
        assert x_rec.shape == (5, 3)

    def test_reconstruct(self, roessler_data):
        _, x_train, _ = roessler_data
        model = _make_model(x_train)
        x_rec = model.reconstruct(x_train[:5])
        assert x_rec.shape == (5, 3)


class TestAutoEncoderTraining:

    def test_first_order_training(self, roessler_data):
        _, x_train, dxdt_train = roessler_data
        model = _make_model(x_train, second_order=False)
        model.compile()
        history = model.fit(
            [x_train, dxdt_train], epochs=5, batch_size=64, verbose=0,
        )
        assert history["loss"][-1] < history["loss"][0]

    def test_second_order_training(self, second_order_data):
        _, x_train, dxdt_train, dxddt_train = second_order_data
        model = _make_model(x_train, second_order=True)
        model.compile()
        history = model.fit(
            [x_train, dxdt_train, dxddt_train], epochs=5, batch_size=64, verbose=0,
        )
        assert history["loss"][-1] < history["loss"][0]

    def test_with_mu(self, roessler_data):
        _, x_train, dxdt_train = roessler_data
        n = x_train.shape[0]
        mu = np.random.randn(n, 1).astype(np.float32)
        reduced_order = 2
        layer = SindyLayer(
            state_dim=reduced_order,
            param_dim=1,
            feature_libraries=[PolynomialLibrary(degree=2, include_bias=True)],
            second_order=False,
        )
        model = AutoencoderSindy(
            sindy_layer=layer,
            reduced_order=reduced_order,
            x=x_train,
            mu=mu,
            layer_sizes=[8, 8],
            second_order=False,
        )
        model.compile()
        history = model.fit(
            [x_train, dxdt_train, mu], epochs=3, batch_size=64, verbose=0,
        )
        assert "loss" in history


class TestAutoEncoderSaveLoad:

    def test_save_load_roundtrip(self, roessler_data):
        _, x_train, dxdt_train = roessler_data
        model = _make_model(x_train)
        model.compile()
        model.fit([x_train, dxdt_train], epochs=2, batch_size=64, verbose=0)

        x_t = torch.tensor(x_train[:5], dtype=torch.float32)
        with torch.no_grad():
            z_before = model.encode(x_t).numpy().copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)
            model2 = _make_model(x_train)
            model2.load(tmpdir)
            with torch.no_grad():
                z_after = model2.encode(x_t).numpy()

        np.testing.assert_allclose(z_before, z_after, atol=1e-6)


class TestAutoEncoderActivations:

    @pytest.mark.parametrize("act", ["relu", "selu", "elu", "tanh", "sigmoid",
                                      "leaky_relu", "linear", "gelu", "swish", "silu"])
    def test_various_activations(self, roessler_data, act):
        _, x_train, _ = roessler_data
        model = _make_model(x_train, activation=act)
        x_t = torch.tensor(x_train[:5], dtype=torch.float32)
        z = model.encode(x_t)
        assert z.shape == (5, 2)


class TestCalcLatentDerivatives:

    def test_first_order(self, roessler_data):
        _, x_train, dxdt_train = roessler_data
        model = _make_model(x_train, second_order=False)
        result = model.calc_latent_time_derivatives(x_train[:10], dxdt_train[:10])
        assert len(result) == 2
        z, dz_dt = result
        assert z.shape == (10, 2)
        assert dz_dt.shape == (10, 2)

    def test_second_order(self, second_order_data):
        _, x_train, dxdt_train, dxddt_train = second_order_data
        model = _make_model(x_train, second_order=True)
        result = model.calc_latent_time_derivatives(
            x_train[:10], dxdt_train[:10], dxddt_train[:10],
        )
        assert len(result) == 3
        z, dz_dt, dz_ddt = result
        assert z.shape == (10, 2)
        assert dz_dt.shape == (10, 2)
        assert dz_ddt.shape == (10, 2)
