import numpy as np
from time import time

EPS = np.spacing(1)

class ILRMA(object):
    """
    Independent Low-Rank Matrix Factorization: ILRMA

    Parameters:
    -----------
    data : array-like, dtype=complex, shape (I, J, M)
        observed multichannel complex-valued signals

    m : int
        number of bases for each source

    initialize : str, (default: 'identity')
        Initialize rule of W_i
        - identity: initialize with identity matrix
        - random: initialize with complex-random values

    tol : float, (default: 0.1)
        The convergence threshold

    n_iter : int, (default: 100)
        The number of iterations to perfom

    seed : int or None, (default: None)
        random seed for initialize

    interval : int, (default: 10)
        Number of iteration done before the next print.

    Attributes:
    -----------
    EPS: array-like, shape ()
        Machine epsilon 2.2204460492503131e-16 for avoid zero divie

    row: int
        row of input matrix

    column: int
        column of input matrix

    nb: int
        number of basis

    _e: np.ndarray, shape (I, J)
        ones matirix whose size is the same with input data matirix

    cost: np.ndarray, shape (iter)
        convergence of cost function

    W : numpy.ndarray, dtype=complex, shape (I, N, M)
        complex-valued N * M demixing matrix

    Returns:
    --------
    estim : numpy.ndarray, dtype=complex, shape (I, J, N, M)
        estimated complex-valued sources

    References:
    -----------
    [1] D. Kitamura, "Algorithms for Independent Low-Rank Matrix Analysis,"
    """
    def __init__(self, data, m, initialize='identity', tol=0.1,
                 n_iter=100, seed=None, interval=10):

         self.X = data
         self.row, self.column, self.n_mics = np.shape(self.X)
         self.n_components = self.n_mics  # fix to the determined situation
         self.nb = m
         self.tol = tol
         self.iter = n_iter
         self.cost = np.zeros(self.iter, dtype="float64")
         self.initialize = initialize[:1].lower()
         self.interval = interval
         self.seed = seed

         self._set_params()

    def _set_params(self):
        # random seed
        np.random.seed(self.seed)

        # Initialize W_i with identity matrix or complex-valued random values
        if self.initialize == 'i':   # initialize == 'identity'
            self.W = np.array([np.eye(self.n_components, self.n_mics,
                                      dtype=complex).T for _ in range(self.row)])
        elif self.initialize == 'r':  # initialize == 'random'
            self.W = (np.random.rand(self.row, self.n_components, self.n_mics)
                      + 1j * np.random.rand(self.row, self.n_components, self.n_mics))
        else:
            raise ValueError('initialize must be "identity" or "random"')

        # T and V with nonnegative random vlaues
        self.T = np.random.rand(self.n_components, self.row, self.nb)
        self.V = np.random.rand(self.n_components, self.nb, self.column)

        # Initial estimated sources
        self.Y = np.einsum("inm,ijm->ijn", self.W, self.X)

        # Initial power spectrograms of estimated sources
        # shape (self.n_components, self.row, self.column)
        self.P = np.square(np.abs(self.Y)).transpose((2, 0, 1))

        # Initial model spectrograms
        self.R = np.einsum("nik,nkj->nij", self.T, self.V)

        self.converge = False     # convergence

    def update(self):
        # time start
        self._set_iter()

        print("ILRMA update")
        for it in range(self.iter):

            prev_Y = self.Y.copy()   # previous value of y

            for n in range(self.n_components):
                # Update of basis matrix
                self.T[n] = (self.T[n] *
                             np.sqrt(np.dot(self.P[n] * self.R[n] ** (-2),
                                            self.V[n].T)
                                     / np.dot(self.R[n] ** (-1), self.V[n].T)))
                self.T[n] = np.maximum(self.T[n], EPS)

                # New model spectrograms
                self.R[n] = np.dot(self.T[n], self.V[n])

                # Update of activation matrix
                self.V[n] = (self.V[n] *
                             np.sqrt(np.dot(self.T[n].T,
                                            self.P[n] * self.R[n] ** (-2))
                                     / np.dot(self.T[n].T, self.R[n] ** (-1))))
                self.V[n] = np.maximum(self.V[n], EPS)

                # New model spectrograms
                self.R[n] = np.dot(self.T[n], self.V[n])

                for i in range(self.row):
                    # X_i:: is J * M matrix, R_i:n is J * 1 vector,
                    # U_i,n is M * M matrix
                    R_i = self.R[n, i].reshape((-1, 1))
                    temp = np.dot(self.X[i].T.conj(),
                                  self.X[i]
                                  * np.dot(R_i ** (-1),
                                           np.ones((1, self.n_mics))))
                    U_in = 1 / self.column * temp.T

                    # Update of demixing filter
                    e_n = np.zeros(self.n_components)
                    e_n[n] = 1
                    w_in = np.linalg.solve(np.dot(self.W[i], U_in), e_n)

                    # Normalization of demixing filter
                    temp = np.dot(w_in.T.conj(), np.dot(U_in, w_in))
                    w_in = np.dot(w_in, temp ** (-1 / 2))

                    self.W[i, n] = w_in

            # New estimated sources
            self.Y = np.einsum("inm,ijm->ijn", self.W, self.X)

            # New power spectrograms of estimated sources
            self.P = np.square(np.abs(self.Y)).transpose((2, 0, 1))

            # apply normalization to all n
            # Normalization coefficient
            lambda_n = np.sqrt(np.mean(self.P, axis=(1, 2)))
            # Normalization of demixing filter
            self.W = self.W / lambda_n[np.newaxis, :, np.newaxis]
            # Normalization of separated power spectrograms
            self.P = self.P / lambda_n[:, np.newaxis, np.newaxis] ** 2
            # Normalization of model spectrograms
            ## not implement the empirical knowledges
            self.R = self.R / lambda_n[:, np.newaxis, np.newaxis] ** 2
            # Normalization of basis matrix
            self.T = self.T / lambda_n[:, np.newaxis, np.newaxis] ** 2

            self.cost[it] = np.linalg.norm(np.abs(prev_Y - self.Y))

            if self.cost[it] < self.tol:
                self.converge = True
                break

            self._print_iter_prog(it, self.interval)

        self._print_iter_end()

        # Back Projection Technique
        self._back_projection()

        return self.estim

    def _back_projection(self):
        """Back-projection technique"""
        self.estim = np.zeros((self.row, self.column, self.n_components,
                               self.n_mics), dtype=complex)

        for i, (W_i, y_i) in enumerate(zip(self.W, self.Y)):
            for j, y_ij in enumerate(y_i):
                for n in range(len(y_ij)):
                    e_n = np.zeros_like(y_ij)
                    e_n[n] = 1
                    self.estim[i, j, n] = np.linalg.solve(W_i, e_n * y_ij)

    def _set_iter(self):
        """set initial values of iteration"""
        self._prev_time = time()
        self._iter_prev_time = self._prev_time

    def _print_iter_prog(self, n_iter, interval):
        """print iteration progress"""
        if n_iter % interval == 0:
            cur_time = time()
            cur_cost = self.cost[n_iter]
            print((" Iteration {0:d}\t time {1:.5f}s\t cost change {2:.4f}"
                   ).format(n_iter, cur_time - self._iter_prev_time, cur_cost))
            self._iter_prev_time = cur_time

    def _print_iter_end(self):
        """print messege on the end of iteration"""
        print((" converged {0} time {1:.5f}s\t cost convergence {2:.4f}"
               ).format(self.converge, time() - self._prev_time, self.cost[-1]))
