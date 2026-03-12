import torch
import scipy
from sympy import sympify, symbols
from .base_library import BaseLibrary


class PolynomialLibrary(BaseLibrary):

    def __init__(self, degree=3, x_dim=2, interaction=True, include_bias=True):
        """
        Polynomial library.

        Parameters
        ----------
        degree : int, default=3
            Polynomial degree.
        x_dim : int, default=2
            Dimension of the input.
        interaction : bool, default=True
            Include interaction terms.
        include_bias : bool, default=True
            Include bias term.
        """
        self.degree = degree
        self.interaction = interaction
        self.include_bias = include_bias

        l = 0
        n = x_dim
        for k in range(self.degree + 1):
            l += int(scipy.special.binom(n + k - 1, k))
        self.n = n
        self.l = l

    def __call__(self, x):
        """
        Transform input x to polynomial features of order self.poly_order.

        Parameters
        ----------
        x : array-like of shape (n_samples, 2*reduce_order)
            Latent variable and its time derivative.

        Returns
        -------
        x_poly : torch.Tensor
            Polynomial features.
        """
        if self.interaction:
            # faster way for one or two dimensional input
            if x.shape[1] <= 2:
                x_poly = x
                x_new = x
                for i in range(1, self.degree):
                    interactions = x[:, 0:1] * x_new
                    sec = x[:, 1:2] ** (i + 1)
                    x_new = torch.cat([interactions, sec], dim=1)
                    x_poly = torch.cat([x_poly, x_new], dim=1)
            # for higher dimensional input
            else:
                x_poly = self.poly_higher_order(x)

        # no interactions
        else:
            x_poly = x
            for i in range(1, self.degree):
                x_new = x ** (i + 1)
                x_poly = torch.cat([x_poly, x_new], dim=1)

        # add ones to the input
        if self.include_bias:
            ones = torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
            x_poly = torch.cat([ones, x_poly], dim=1)

        return x_poly

    def poly_higher_order(self, x):
        """
        Compute polynomial features for higher dimensional input x.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        x_poly : torch.Tensor
            Polynomial features for higher dimensional input.
        """
        x_poly = []
        for d in range(1, self.degree + 1):
            x_poly += self.loop_rec(x, 1, 0, x.shape[1], d)
        x_poly = torch.cat(x_poly, dim=1)
        return x_poly

    def get_names(self, x):
        """
        Construct the names of the features for the input x.

        Parameters
        ----------
        x : array-like of shape (n_samples, 2*reduce_order)
            Latent variable and its time derivative.

        Returns
        -------
        list of str
            List of feature names.
        """
        l = []
        for d in range(1, self.degree + 1):
            l += self.loop_rec_names(x, 1, 0, len(x), d, [])
        #
        if self.include_bias:
            l = [symbols("1")] + l

        # simplify strings by combining powers
        for i, l_ in enumerate(l):
            l[i] = str(sympify(l[i])).replace("**", "^")
        return l

    def loop_rec(self, x, x_i, i, n, d):
        if d > 1:
            feat = []
            for j in range(i, n):
                x_j = x_i * x[:, j : j + 1]
                feat += self.loop_rec(x, x_j, j, n, d - 1)
        else:
            feat = []
            for j in range(i, n):
                x_j = x_i * x[:, j : j + 1]
                feat.append(x_j)

        return feat

    def loop_rec_names(self, x, x_i, i, n, d, l: list):
        if d > 1:
            for j in range(i, n):
                x_j = x_i * x[j]
                l = self.loop_rec_names(x, x_j, j, n, d - 1, l)
        else:
            for j in range(i, n):
                x_j = x_i * x[j]
                l.append(x_j)
        return l
