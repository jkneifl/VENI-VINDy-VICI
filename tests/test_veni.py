import torch
import numpy as np
import tempfile
import pytest
from vindy.layers import SindyLayer
from vindy.libraries import PolynomialLibrary
from vindy.networks import VENI


def _make_veni(x_train):
    reduced_order = 2
    layer = SindyLayer(
        state_dim=reduced_order,
        feature_libraries=[PolynomialLibrary(degree=2, include_bias=True)],
        second_order=False,
    )
    model = VENI(
        beta=1e-3,
        sindy_layer=layer,
        reduced_order=reduced_order,
        x=x_train,
        layer_sizes=[8, 8],
        second_order=False,
    )
    return model


class TestVENIEncode:

    def test_variational_encode(self, roessler_data):
        _, x_train, _ = roessler_data
        model = _make_veni(x_train)
        x_t = torch.tensor(x_train[:5], dtype=torch.float32)
        z_mean, z_log_var, z = model.variational_encode(x_t)
        assert z_mean.shape == (5, 2)
        assert z_log_var.shape == (5, 2)
        assert z.shape == (5, 2)

    def test_encode_mean_vs_sample(self, roessler_data):
        _, x_train, _ = roessler_data
        model = _make_veni(x_train)
        x_t = torch.tensor(x_train[:5], dtype=torch.float32)
        z_mean = model.encode(x_t, mean_or_sample="mean")
        z_sample = model.encode(x_t, mean_or_sample="sample")
        assert z_mean.shape == z_sample.shape
        # They should generally be different (stochastic sampling)
        assert not torch.allclose(z_mean, z_sample)


class TestVENIKLLoss:

    def test_kl_loss(self, roessler_data):
        _, x_train, _ = roessler_data
        model = _make_veni(x_train)
        mean = torch.zeros(5, 2)
        log_var = torch.zeros(5, 2)
        kl = model.kl_loss(mean, log_var)
        assert kl.shape == ()
        assert kl.item() >= 0


class TestVENITraining:

    def test_training(self, roessler_data):
        _, x_train, dxdt_train = roessler_data
        model = _make_veni(x_train)
        model.compile()
        history = model.fit(
            [x_train, dxdt_train], epochs=5, batch_size=64, verbose=0,
        )
        assert history["loss"][-1] < history["loss"][0]
        assert "kl" in history


class TestVENISaveLoad:

    def test_save_load_roundtrip(self, roessler_data):
        _, x_train, dxdt_train = roessler_data
        model = _make_veni(x_train)
        model.compile()
        model.fit([x_train, dxdt_train], epochs=2, batch_size=64, verbose=0)

        x_t = torch.tensor(x_train[:5], dtype=torch.float32)
        with torch.no_grad():
            z_before = model.encode(x_t, mean_or_sample="mean").numpy().copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)
            model2 = _make_veni(x_train)
            model2.load(tmpdir)
            with torch.no_grad():
                z_after = model2.encode(x_t, mean_or_sample="mean").numpy()

        np.testing.assert_allclose(z_before, z_after, atol=1e-6)
