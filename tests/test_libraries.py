import torch
import pytest
from vindy.libraries import PolynomialLibrary, ExponentialLibrary, FourierLibrary, ForceLibrary


class TestPolynomialLibrary:

    @pytest.mark.parametrize("degree,bias", [(1, True), (2, False), (3, True)])
    def test_output_shape(self, degree, bias):
        lib = PolynomialLibrary(degree=degree, include_bias=bias)
        x = torch.randn(10, 3)
        out = lib(x)
        assert out.shape[0] == 10
        assert out.shape[1] > 0

    def test_values_bias_and_linear(self):
        lib = PolynomialLibrary(degree=1, include_bias=True)
        x = torch.randn(5, 3)
        out = lib(x)
        # first column should be ones (bias)
        torch.testing.assert_close(out[:, 0], torch.ones(5))
        # remaining columns should be the input
        torch.testing.assert_close(out[:, 1:], x)

    def test_get_names(self):
        lib = PolynomialLibrary(degree=2, include_bias=True)
        from sympy import symbols
        x = [symbols("z_0"), symbols("z_1"), symbols("z_2")]
        names = lib.get_names(x)
        assert len(names) > 0
        assert isinstance(names[0], str)


class TestExponentialLibrary:

    def test_output_shape_single_coeff(self):
        lib = ExponentialLibrary(coeff=[1])
        x = torch.randn(10, 3)
        out = lib(x)
        assert out.shape == (10, 3)

    def test_output_shape_multiple_coeffs(self):
        lib = ExponentialLibrary(coeff=[1, 2])
        x = torch.randn(10, 3)
        out = lib(x)
        assert out.shape == (10, 6)

    def test_values(self):
        lib = ExponentialLibrary(coeff=[2])
        x = torch.tensor([[1.0, 2.0]])
        out = lib(x)
        expected = torch.exp(2 * x)
        torch.testing.assert_close(out, expected)


class TestFourierLibrary:

    def test_output_shape(self):
        lib = FourierLibrary(freqs=[1])
        x = torch.randn(10, 2)
        out = lib(x)
        # 3 functions (sin, cos, sigmoid) * 2 dims * 1 freq = 6
        assert out.shape == (10, 6)

    def test_values(self):
        lib = FourierLibrary(freqs=[1])
        x = torch.tensor([[1.0, 2.0]])
        out = lib(x)
        expected = torch.cat([
            torch.sin(x),
            torch.cos(x),
            torch.sigmoid(x),
        ], dim=1)
        torch.testing.assert_close(out, expected)


class TestForceLibrary:

    def test_shape_and_defaults(self):
        lib = ForceLibrary()
        # needs at least 3 columns: t, omega, amplitude(s)
        x = torch.randn(10, 4)
        out = lib(x)
        # 2 functions * (4-2) amplitude columns = 4
        assert out.shape == (10, 4)

    def test_default_functions(self):
        lib = ForceLibrary()
        assert len(lib.functions) == 2
        assert lib.functions[0] is torch.sin
        assert lib.functions[1] is torch.cos


class TestMultipleLibraries:

    def test_concatenated_features(self):
        poly = PolynomialLibrary(degree=1, include_bias=True)
        exp = ExponentialLibrary(coeff=[1])
        x = torch.randn(10, 3)
        out_poly = poly(x)
        out_exp = exp(x)
        combined = torch.cat([out_poly, out_exp], dim=1)
        assert combined.shape[1] == out_poly.shape[1] + out_exp.shape[1]
