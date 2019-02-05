# -*- coding: utf-8 -*-

import numpy as np
import numpy.matlib
from time import time

from .utils import plot_cost

EPS = np.spacing(1)


class BaseNMF(object):
    """Base class for Non-negative Matrix Factorization algorithm

    Parameters:
    -----------
    data: np.ndarray, shape (I, J)
        input data matix
        input matrix is must be Non-negative

     m: int
        number of basis

    beta: float (default: 1)
        set cost function for betaNMF algorithm
        ----------------------------------
        beta = 2 --> EUC distance(EUC)
        beta = 1 --> KL divergence(KL)
        beta = 0 --> IS divergence(IS)
        else     --> beta divergence(beta)
        -----------------------------------

    basis: np.ndarray, shape (I, m)
        initial basis matrix for betaNMF (default: None)
        if basis is None, detmined random values
        basis matrix is also Non-negative

    activation: np.ndarray, shape (m, J)
        initial activation matirix (default: None)
        if activation is None, detmined random values
        activation matrix is also Non-negative

    seed: int or None (default: None)
        seed for initial random basis and activation

    n_iter: int (default: 100)
        number of iteration for algorithm

    interval: int (default: 10)
        interval of print iteration progress


    Attributes:
    -----------
    EPS: array-like, shape ()
        Machine epsilon 2.2204460492503131e-16 for avoid zero divie

    row: int
        row of input matrix (mean: I)

    column: int
        column of input matrix (mean: J)

    nb: int
        number of basis (mean: m)

    _e: np.ndarray, shape (I, J)
        ones matirix whose size is the same with input data matirix

    cost: np.ndarray, shape (n_iter)
        convergence of cost function


    References:
    -----------
    [1] 亀岡弘和. "非負値行列因子分解." 計測と制御 51.9 (2012): 835-844.
    [2] Daich Kitamura - Program Codes

    See also:
    ---------
    http://d-kitamura.net/
    """

    def __init__(self, data, m, beta=1, basis=None, activation=None,
                 seed=None, n_iter=100, interval=10):

        if np.min(data) < 0:
            raise ValueError("Input matrix must be Non-negative.")

        # self.X = data
        self.X = np.maximum(data, EPS)
        self.beta = beta
        self.row, self.column = np.shape(self.X)
        self.nb = m
        self.iter = n_iter
        self.basis = basis
        self.activation = activation
        self.cost = np.zeros(self.iter, dtype="float64")
        self.interval = interval
        self.seed = seed

        self._set_params()

    def __repr__(self):
        return (("I = {0:d}, J = {1:d}, m = {2:d}, β = {3:.f}"
                 ).format(self.row, self.column, self.nb, self.beta))

    def _set_params(self):
        """set basis matrix, activation matrix and phi for NMF

        Attributes:
        -----------
        basis: np.ndarray, shape (I, m)
            initial basis matrix for betaNMF (default: None)
            if basis is None, detmined random values
            basis matrix is also Non-negative

        activation: np.ndarray, shape (m, J)
            initial activation matirix (default: None)
            if activation is None, detmined random values
            activation matrix is also Non-negative

        phi: float
            value for beta divergence NMF

        """
        # random seed
        np.random.seed(self.seed)

        if self.basis is None:
            # random basis
            self.H = np.random.random_sample((self.row, self.nb))
            self.H = np.maximum(self.H, EPS)
        else:
            # initial basis
            self.H = self.basis
            self.H = np.maximum(self.H, EPS)

        if not (np.shape(self.H)[0] == self.row or
                np.shape(self.H)[1] == self.nb):
            raise ValueError("The size of initial basis matrix is incorrect.")

        if self.activation is None:
            # random activation
            self.U = np.random.random_sample((self.nb, self.column))
            self.U = np.maximum(self.U, EPS)
        else:
            # initial activation
            self.U = self.activation
            self.U = np.maximum(self.U, EPS)

        if not (np.shape(self.U)[0] == self.nb or
                np.shape(self.U)[1] == self.column):
            msg = ("The size of initial activation matrix is incorrect.")

            raise ValueError(msg)

        self._e = np.ones((self.row, self.column))

        # set phi
        if self.beta < 1:
            self.phi = 1 / (2 - self.beta)
        elif self.beta < 2:
            self.phi = 1
        else:
            self.phi = 1 / (self.beta - 1)

    def update(self, curve=True):
        """
        culculate convergence of cost function and multiplivate update

        Parameters:
        -----------
        curve: boolen (default: True)
            draw convergence curve or not


        Returns:
        --------
        basis: np.ndarray, shape (I, m)
            decomposed basis matrix

        activation: np.ndarray, shape (m, J)
            decomposed activation matrix
        """

        self._set_iter()

        print("NMF update")
        for i in range(self.iter):
            self.cost[i] = self._divergence()
            self._print_iter_prog(i, self.interval)

        self._print_iter_end()

        if curve is True:

            plot_cost(self.cost, self.iter)

        scale = 1 / np.sum(self.H)
        self.H = self.H * np.matlib.repmat(scale, np.shape(self.H)[0], 1)
        self.U = self.U / np.matlib.repmat(scale.T, 1, np.shape(self.U)[1])

        return self.H, self.U

    def _set_iter(self):
        """set initial values of iteration"""
        self._iter_prev_cost = np.inf
        self._prev_time = time()
        self._iter_prev_time = self._prev_time

    def _print_iter_prog(self, n_iter, interval):
        """print iteration progress"""
        if n_iter % interval == 0:
            cur_time = time()
            cur_cost = self.cost[n_iter]
            print((" Iteration {0:d}\t time {1:.5f}s\t cost change {2:f}"
                   ).format(n_iter, cur_time - self._iter_prev_time,
                            self._iter_prev_cost - cur_cost))
            self._iter_prev_time = cur_time
            self._iter_prev_cost = cur_cost

    def _print_iter_end(self):
        """print messege on the end of iteration"""
        print((" Iteration {0:d}\t time {1:.5f}s\t cost convergence {2:f}"
               ).format(self.iter, time() - self._prev_time, self.cost[-1]))
