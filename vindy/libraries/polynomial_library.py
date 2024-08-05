import tensorflow as tf
import scipy
from sympy import sympify, symbols
from .base_library import BaseLibrary


class PolynomialLibrary(BaseLibrary):

    def __init__(self, degree=3, x_dim=2, interaction=True, include_bias=True):
        """
        Polynomial library
        :param degree: polynomial degree (default: 3)
        :param x_dim: dimension of the input (default: 2)
        :param interaction: include interaction terms (default: True)
        :param include_bias: include bias term (default: True)
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

    # @tf.function
    def __call__(self, x):
        """
        transform input x to polynomial features of order self.poly_order
        :param x: array-like of shape (n_samples, 2*reduce_order), latent variable and its time derivative
        :return: polynomial features
        """
        # x_old = x
        if self.interaction:
            # faster way for one or two dimensional input
            if x.shape[1] <= 2:
                x_poly = x
                x_new = x
                for i in range(1, self.degree):
                    interactions = x[:, 0:1] * x_new
                    sec = x[:, 1:2] ** (i + 1)
                    x_new = tf.concat([interactions, sec], axis=1)
                    x_poly = tf.concat([x_poly, x_new], axis=1)
            # for higher dimensional input
            else:
                x_poly = self.poly_higher_order(x)

        # no interactions
        else:
            for i in range(1, self.degree):
                x_new = x ** (i + 1)
                x_poly = tf.concat([x_poly, x_new], axis=1)

        # add ones to the input
        if self.include_bias:
            ones = 0 * x[:, 0:1] + 1
            x_poly = tf.concat([ones, x_poly], axis=1)

        return x_poly

    @tf.function
    def poly_higher_order(self, x):
        """ "
        Compute polynomial features for higher dimensional input x
        """
        x_poly = []
        for d in range(1, self.degree + 1):
            x_poly += self.loop_rec(x, 1, 0, x.shape[1], d)
        x_poly = tf.concat(x_poly, axis=1)
        return x_poly

    def get_names(self, x):
        """
        construct the names of the features for the input x
        :param x:  array-like of shape (n_samples, 2*reduce_order), latent variable and its time derivative
        :return:
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

    @tf.function
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
