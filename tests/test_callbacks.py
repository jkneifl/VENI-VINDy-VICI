import numpy as np
import pytest
from vindy.layers import SindyLayer, VindyLayer
from vindy.libraries import PolynomialLibrary
from vindy.distributions import Laplace
from vindy.networks import IdentificationNetwork
from vindy.callbacks import (
    CallbackList,
    ThresholdPruneCallback,
    SaveCoefficientsCallback,
)


class TestThresholdPruneCallback:

    def test_coefficients_pruned(self, roessler_data):
        _, x_train, dxdt_train = roessler_data
        layer = SindyLayer(
            state_dim=3,
            feature_libraries=[PolynomialLibrary(degree=2, include_bias=True)],
            second_order=False,
        )
        model = IdentificationNetwork(
            sindy_layer=layer,
            x=x_train,
            second_order=False,
        )
        model.compile()
        cb = ThresholdPruneCallback(freq=1, threshold=0.1)
        model.fit(
            [x_train, dxdt_train],
            epochs=3,
            batch_size=64,
            callbacks=[cb],
            verbose=0,
        )
        coeffs = layer.get_sindy_coeffs()
        # Some coefficients should have been pruned to zero
        assert np.any(coeffs == 0)


class TestSaveCoefficientsCallback:

    def test_history_has_coeffs(self, roessler_data):
        _, x_train, dxdt_train = roessler_data
        layer = VindyLayer(
            beta=1e-3,
            priors=Laplace(0.0, 1.0),
            state_dim=3,
            feature_libraries=[PolynomialLibrary(degree=2, include_bias=True)],
            second_order=False,
        )
        model = IdentificationNetwork(
            sindy_layer=layer,
            x=x_train,
            second_order=False,
        )
        model.compile()
        cb = SaveCoefficientsCallback(freq=1)
        history = model.fit(
            [x_train, dxdt_train],
            epochs=3,
            batch_size=64,
            callbacks=[cb],
            verbose=0,
        )
        assert "coeffs_mean" in history
        # Verify snapshots are independent (not all identical due to shared memory)
        snapshots = history["coeffs_mean"]
        assert len(snapshots) == 3
        assert not np.array_equal(snapshots[0], snapshots[-1])


class TestCallbackList:

    def test_all_callbacks_invoked(self, roessler_data):
        _, x_train, dxdt_train = roessler_data
        layer = SindyLayer(
            state_dim=3,
            feature_libraries=[PolynomialLibrary(degree=2, include_bias=True)],
            second_order=False,
        )
        model = IdentificationNetwork(
            sindy_layer=layer,
            x=x_train,
            second_order=False,
        )
        model.compile()
        cb1 = ThresholdPruneCallback(freq=1, threshold=0.1)
        cb2 = SaveCoefficientsCallback(freq=1)
        cb_list = CallbackList([cb1, cb2])
        cb_list.set_model(model)
        # Just verify all methods can be called
        cb_list.on_train_begin()
        cb_list.on_epoch_begin(0)
        cb_list.on_epoch_end(0, {})
        cb_list.on_train_end()
