import pytest
import numpy as np
import torch
from vindy.utils import set_seed
from vindy.libraries import PolynomialLibrary
from vindy.layers import SindyLayer, VindyLayer
from vindy.distributions import Laplace


@pytest.fixture(autouse=True)
def seed():
    set_seed(42)


@pytest.fixture
def roessler_data():
    """Small synthetic Roessler-like data: 3 trajectories, 100 timesteps, 3 dims."""
    n_traj = 3
    n_t = 100
    dim = 3
    t = np.linspace(0, 5, n_t)
    x_all = []
    dxdt_all = []
    for _ in range(n_traj):
        x = np.column_stack([np.sin(t), np.cos(t), np.sin(2 * t)])
        x += np.random.randn(*x.shape) * 0.01
        dxdt = np.column_stack([np.cos(t), -np.sin(t), 2 * np.cos(2 * t)])
        x_all.append(x)
        dxdt_all.append(dxdt)
    x_train = np.concatenate(x_all, axis=0).astype(np.float32)
    dxdt_train = np.concatenate(dxdt_all, axis=0).astype(np.float32)
    return t, x_train, dxdt_train


@pytest.fixture
def second_order_data(roessler_data):
    """Adds dxddt for second-order tests."""
    t, x_train, dxdt_train = roessler_data
    n = x_train.shape[0]
    dxddt_train = np.random.randn(n, x_train.shape[1]).astype(np.float32) * 0.1
    return t, x_train, dxdt_train, dxddt_train


@pytest.fixture
def poly_library():
    return PolynomialLibrary(degree=2, include_bias=True)


@pytest.fixture
def sindy_layer(poly_library):
    return SindyLayer(
        state_dim=3,
        feature_libraries=[poly_library],
        second_order=False,
    )


@pytest.fixture
def vindy_layer(poly_library):
    return VindyLayer(
        beta=1e-3,
        priors=Laplace(0.0, 1.0),
        state_dim=3,
        feature_libraries=[PolynomialLibrary(degree=2, include_bias=True)],
        second_order=False,
    )
