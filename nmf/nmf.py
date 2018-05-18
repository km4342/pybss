# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from .base import BaseNMF

EPS = np.spacing(1)


class NMF(BaseNMF):
    """Basic Non-negative Matrix Factorization algorithm

    Parameters:
    -----------
    data: np.ndarray, shape (I, J)
        input data matix
        input matrix must be Non-negative

     m: int
        number of basis

    beta: float (default: 2)
        set cost function for betaNMF algorithm
        -----------------------------------
        beta = 2 --> EUC distance(EUC)
        beta = 1 --> KL divergence(KL)
        beta = 0 --> IS divergence(IS)
        else     --> beta divergence(beta)
        -----------------------------------

    basis: np.ndarray, shape (I, m)
        initial basis matrix (default: None)
        if basis is None, detmined random values
        basis matrix is also Non-negative

    activation: np.ndarray, shape (m, J)
        initial activation matirix (default: None)
        if activation is None, detmined random values
        activation matrix is also Non-negative

    n_iter: int (default: 100)
        number of iteration for algorithm


    References:
    -----------
    [1] 亀岡弘和. "非負値行列因子分解." 計測と制御 51.9 (2012): 835-844.
    [2] Daich Kitamura - Program Codes

    See also:
    ---------
    http://d-kitamura.net/

    """
    def __init__(self, data, m, beta=2, basis=None, activation=None,
                 seed=None, n_iter=100, interval=10):

        super(NMF, self).__init__(data=data, m=m, beta=beta, basis=basis,
                                  activation=activation, seed=None,
                                  n_iter=n_iter, interval=interval)

        self._set_divergence()

    def _set_divergence(self):
        """select divergence for multiplivate update algorithm"""

        if self.beta == 2:
            self._divergence = self._euc_distance

        elif self.beta == 1:
            self._divergence = self._kl_divergence

        elif self.beta == 0:
            self._divergence = self._is_divergence

        else:
            self._divergence = self._beta_divergence

    def _euc_distance(self):
        """cost function: EUC distance"""

        HU = np.dot(self.H, self.U)

        # update basis
        self.H = (self.H *
                  np.dot(self.X, self.U.T) /
                  np.dot(HU, self.U.T))
        self.H = np.maximum(self.H, EPS)

        HU = np.dot(self.H, self.U)

        # update activation
        self.U = (self.U *
                  np.dot(self.H.T, self.X) /
                  np.dot(self.H.T, HU))
        self.U = np.maximum(self.U, EPS)

        # cost function
        HU = np.dot(self.H, self.U)

        return np.sum((self.X - HU)**2)

    def _kl_divergence(self):
        """cost function: KL divergence"""

        HU = np.dot(self.H, self.U)

        # update basis
        self.H = (self.H *
                  np.dot(self.X / HU, self.U.T) /
                  np.dot(self._e, self.U.T))
        self.H = np.maximum(self.H, EPS)

        HU = np.dot(self.H, self.U)

        # update activation
        self.U = (self.U *
                  np.dot(self.H.T, self.X / HU) /
                  np.dot(self.H.T, self._e))
        self.U = np.maximum(self.U, EPS)

        # cost function
        HU = np.dot(self.H, self.U)

        return np.sum(self.X * np.log(self.X / HU) - self.X + HU)

    def _is_divergence(self):
        """cost function: IS divergence"""

        HU = np.dot(self.H, self.U)

        # update basis
        self.H = (self.H *
                  np.sqrt(np.dot(self.X * HU**(-2), self.U.T) /
                          np.dot(HU**(-1), self.U.T)))
        self.H = np.maximum(self.H, EPS)

        HU = np.dot(self.H, self.U)

        # update activation
        self.U = (self.U *
                  np.sqrt(np.dot(self.H.T, self.X * HU**(-2)) /
                          np.dot(self.H.T, HU**(-1))))
        self.U = np.maximum(self.U, EPS)

        # cost function
        HU = np.dot(self.H, self.U)

        return np.sum(self.X / HU - np.log(self.X / HU) - 1)

    def _beta_divergence(self):
        """const function: beta divergence"""

        HU = np.dot(self.H, self.U)

        # update basis
        self.H = (self.H *
                  (np.dot(self.X *
                          np.power(HU, self.beta - 2), self.U.T) /
                   np.dot(np.power(HU, self.beta - 1), self.U.T)) ** self.phi)
        self.H = np.maximum(self.H, EPS)

        HU = np.dot(self.H, self.U)

        # update activation
        self.U = (self.U *
                  (np.dot(self.H.T, self.X *
                          np.power(HU, self.beta - 2)) /
                   np.dot(self.H.T, np.power(HU, self.beta - 1))) ** self.phi)
        self.U = np.maximum(self.U, EPS)

        # cost function
        HU = np.dot(self.H, self.U)

        return (np.sum(abs(self.X**(self.beta) / self.beta * (self.beta - 1) +
                           HU**(self.beta) / self.beta - self.X *
                           HU**(self.beta - 1) / (self.beta - 1))))


###############################################################################
class ConvolutiveNMF(BaseNMF):
    """convolutive NMF algorithm

    Parameters:
    -----------
    data: np.ndarray, shape (I, J)
        input data matix
        input matrix must be Non-negative

    References:
    -----------
    [1] P. Smaragdis, "Convolutive Speech Bases and their Application to Speech
        Separation," IEEE Trans. Speech Audio Process., vol. 15, no. 1,
        pp. 1-12, Jan. 2007.
    """



###############################################################################
class SupervisedNMF(NMF, BaseNMF):
    """supervised NMF algorithm for KL divergence

    Parameters:
    -----------
    data: np.ndarray, shape (I, J)
        input data matix
        input matrix must be Non-negative

     m: int
        number of basis

    basis: np.ndarray, shape (I, m)
        supervised basis matrix
        basis matrix must be Non-negative

    n_iter: int (default: 100)
        update iteration for NMF algorithm and set_cost

    index: int (default: 0)
        index of mixture sound basis matrix


    References:
    -----------
    [1] P.Smaragdis. "Supervised and semi-supervised separation of sounds from
        single-channel mixtures." Independent Component Analysis and Signal
        Separation (2007): 414-421.

    """
    def __init__(self, data, m, basis, n_iter=100, interval=10, index=0):

        super(SupervisedNMF, self).__init__(data=data, m=m, beta=1,
                                            n_iter=n_iter, interval=interval)

        super(SupervisedNMF, self)._set_divergence()

        self._set_supervised(basis=basis, index=index)

    def _set_supervised(self, basis, index):
        """Set supervised basis matirix"""

        self.last_index = index + np.shape(basis)[1]

        if (self.last_index) > np.shape(self.H)[1]:
            ValueError("The index is incorrect.")

        self.H[:, index: self.last_index] = basis
        self.H = np.maximum(self.H, EPS)

    def _euc_distance(self):
        """cost function: EUC distance"""

        HU = np.dot(self.H, self.U)

        # update activation
        self.U = (self.U *
                  np.dot(self.H.T, self.X) /
                  np.dot(self.H.T, HU))
        self.U = np.maximum(self.U, EPS)

        # cost function
        HU = np.dot(self.H, self.U)

        return np.sum((self.X - HU)**2)

    def _kl_divergence(self):
        """cost function: KL divergence"""

        HU = np.dot(self.H, self.U)

        # update activation
        self.U = (self.U *
                  np.dot(self.H.T, self.X / HU) /
                  np.dot(self.H.T, self._e))
        self.U = np.maximum(self.U, EPS)

        # cost function
        HU = np.dot(self.H, self.U)
        return np.sum(self.X * np.log(self.X / HU) - self.X + HU)

    def _is_divergence(self):
        """cost function: IS divergence"""

        HU = np.dot(self.H, self.U)

        # update activation
        self.U = (self.U *
                  np.sqrt(np.dot(self.H.T, self.X * np.power(HU, -2)) /
                          np.dot(self.H.T, np.power(HU, -1))))
        self.U = np.maximum(self.U, EPS)

        # cost function
        HU = np.dot(self.H, self.U)

        return np.sum(self.X / HU - np.log(self.X / HU) - 1)

    def _beta_divergence(self):
        """cost function: beta divergence"""

        HU = np.dot(self.H, self.U)

        # update activation
        self.U = (self.U *
                  (np.dot(self.H.T, self.X *
                          np.power(HU, self.beta - 2)) /
                   np.dot(self.H.T, np.power(HU, self.beta - 1))) ** phi)
        self.U = np.maximum(self.U, EPS)

        # cost function
        HU = np.dot(self.H, self.U)

        return (np.sum(abs(self.X**(self.beta) / self.beta * (self.beta - 1) +
                           HU**(self.beta) / self.beta - self.X *
                           HU**(self.beta - 1) / (self.beta - 1))))


###############################################################################
class SemiNMF(NMF, BaseNMF):
    """semi-supervized NMF algorithm

    Parameters:
    -----------
    data: np.ndarray, shape (I, J)
        input data matix
        input matrix must be Non-negative

     m: int
        number of basis

    basis: np.ndarray, shape (I, m)
        supervised basis matrix
        basis matrix must be Non-negative

    n_iter: int (default: 100)
        update iteration for NMF algorithm and set_cost


    Notes:
    ------
    basis matrix and activation matrix is defined as follows:
    -------------------
    >>> Y = HU
    >>> H = [T , F]
    >>> U = [V , G]
    >>> Y = TV + FG
    -------------------
    basis matrix T is supervised basis. basis matrix F and activation
    matrix V, G are defined as random values.


    References:
    -----------
    [1] P.Smaragdis. "Supervised and semi-supervised separation of sounds from
        single-channel mixtures." Independent Component Analysis and Signal
        Separation (2007): 414-421.

    """
    def __init__(self, data, m, basis, n_iter=100, interval=10):

        super(SemiNMF, self).__init__(data=data, m=m, beta=1,
                                      n_iter=n_iter, interval=interval)

        super(SemiNMF, self)._set_divergence()

        self._set_supervised(basis=basis)

    def _set_supervised(self, basis):
        """Set supervised basis matirix"""

        nb = np.shape(basis)[1]

        # supervised basis
        self.T = basis
        self.T = np.maximum(self.T, EPS)

        # random basis
        self.F = np.random.random_sample((self.row, self.nb - nb))
        self.F = np.maximum(self.F, EPS)

        # activation
        self.V = np.random.random_sample((nb, self.column))
        self.V = np.maximum(self.V, EPS)

        self.G = np.random.random_sample((self.nb - nb, self.column))
        self.G = np.maximum(self.G, EPS)

    def _euc_distance(self):
        """cost function: EUC distance"""

        HU = np.dot(self.F, self.G) + np.dot(self.T, self.V)

        # update basis
        self.F = (self.F *
                  np.dot(self.X, self.G.T) /
                  np.dot(HU, self.F.T))
        self.F = np.maximum(self.F, EPS)

        HU = np.dot(self.F, self.G) + np.dot(self.T, self.V)

        # update activation
        self.G = (self.G *
                  np.dot(self.F.T, self.X) /
                  np.dot(self.F.T, HU))
        self.G = np.maximum(self.G, EPS)

        HU = np.dot(self.F, self.G) + np.dot(self.T, self.V)

        self.V = (self.V *
                  np.dot(self.T.T, self.X) /
                  np.dot(self.T.T, HU))
        self.V = np.maximum(self.V, EPS)

        self.H = np.c_[self.F, self.T]
        self.U = np.r_[self.G, self.V]

        # cost function
        HU = np.dot(self.H, self.U)

        return np.sum((self.X - HU)**2)

    def _kl_divergence(self):
        """cost function: KL divergence"""

        HU = np.dot(self.F, self.G) + np.dot(self.T, self.V)

        # update basis
        self.F = (self.F *
                  np.dot(self.X / HU, self.G.T) /
                  np.dot(self._e, self.G.T))
        self.F = np.maximum(self.F, EPS)

        HU = np.dot(self.F, self.G) + np.dot(self.T, self.V)

        # update activation
        self.G = (self.G *
                  np.dot(self.F.T, self.X / HU) /
                  np.dot(self.F.T, self._e))
        self.G = np.maximum(self.G, EPS)

        HU = np.dot(self.F, self.G) + np.dot(self.T, self.V)

        self.V = (self.V *
                  np.dot(self.T.T, self.X / HU) /
                  np.dot(self.T.T, self._e))
        self.V = np.maximum(self.V, EPS)

        self.H = np.c_[self.F, self.T]
        self.U = np.r_[self.G, self.V]

        # cost function
        HU = np.dot(self.H, self.U)

        return np.sum(self.X * np.log(self.X / HU) - self.X + HU)

    def _is_divergence(self):
        """cost function: IS divergence"""

        HU = np.dot(self.F, self.G) + np.dot(self.T, self.V)

        # update basis
        self.F = (self.F *
                  np.sqrt(np.dot(self.X * np.power(HU, -2), self.G.T) /
                          np.dot(np.power(HU, -1), self.G.T)))
        self.F = np.maximum(self.F, EPS)

        HU = np.dot(self.F, self.G) + np.dot(self.T, self.V)

        # update activation
        self.G = (self.G *
                  np.sqrt(np.dot(self.F.T, self.X * np.power(HU, -2)) /
                          np.dot(self.F.T, np.power(HU, -1))))
        self.G = np.maximum(self.G, EPS)

        HU = np.dot(self.F, self.G) + np.dot(self.T, self.V)

        self.V = (self.V *
                  np.sqrt(np.dot(self.T.T, self.X * np.power(HU, -2)) /
                          np.dot(self.T.T, np.power(HU, -1))))
        self.V = np.maximum(self.V, EPS)

        self.H = np.c_[self.F, self.T]
        self.U = np.r_[self.G, self.V]

        # cost function
        HU = np.dot(self.H, self.U)

        return np.sum(self.X / HU - np.log(self.X / HU) - 1)

    def _beta_divergence(self):
        """cost function: beta divergence"""

        HU = np.dot(self.F, self.G) + np.dot(self.T, self.V)

        # update basis
        self.F = (self.F *
                  (np.dot(self.X *
                          np.power(HU, self.beta - 2), self.G.T) /
                   np.dot(np.power(HU, self.beta - 1), self.G.T)) ** phi)
        self.F = np.maximum(self.F, EPS)

        HU = np.dot(self.F, self.G) + np.dot(self.T, self.V)

        # update activation
        self.G = (self.G *
                  (np.dot(self.F.T, self.X *
                          np.power(HU, self.beta - 2)) /
                   np.dot(self.F.T, np.power(HU, self.beta - 1))) ** phi)
        self.G = np.maximum(self.G, EPS)

        HU = np.dot(self.F, self.G) + np.dot(self.T, self.V)

        self.V = (self.V *
                  (np.dot(self.T.T, self.X *
                          np.power(HU, self.beta - 2)) /
                   np.dot(self.T.T, np.power(HU, self.beta - 1))) ** phi)
        self.V = np.maximum(self.V, EPS)

        self.H = np.c_[self.F, self.T]
        self.U = np.r_[self.G, self.V]

        # cost function
        HU = np.dot(self.H, self.U)

        return (np.sum(abs(self.X**(self.beta) / self.beta * (self.beta - 1) +
                           HU**(self.beta) / self.beta - self.X *
                           HU**(self.beta - 1) / (self.beta - 1))))
###############################################################################
