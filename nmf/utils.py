# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

EPS = np.spacing(1)


def plot_basis(basis, fs=48000, show=True):
    """
    show spectrum of basis matrix

    Parameters:
    -----------
    basis: np.ndarray, shape (I, m)
        input basis matrix
        basis matrix must be Non-negative

    fs: int (default: 48000)
        sampling frequency
    """

    if not np.isreal(basis).all():
        basis = np.abs(basis)**2  # powerSpectrum

    faxis = np.linspace(0, fs / 2, basis.shape[0])  # Frequency
    nb = np.shape(basis)[1]

    sns.set_style('white')

    for it in range(nb):
        n = it % 5
        if n == 0:
            plt.figure(figsize=(12, 6))
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.subplot(5, 1, n + 1)
        plt.subplots_adjust(wspace=0.1, hspace=0.9)
        plt.plot(faxis, 10 * np.log10(basis[:, it] + EPS))
        plt.title("basis" + str(it + 1))
        plt.axis([1, fs // 2, -100, 0])

    if show is True:
        plt.show()


def plot_activation(activation, fs=48000, show=True):
    """
    show spectrum of activation matrix

    Parameters:
    -----------
    activation: np.ndarray, shape (I, m)
        input activation matrix
        activation matrix must be Non-negative

    fs: int (default: 48000)
        sampling frequency
    """

    if not np.isreal(activation).all():
        activation = np.abs(activation)**2  # powerSpectrum

    taxis = np.linspace(0, fs / 2, activation.shape[1])  # Frequency
    nb = np.shape(activation)[0]

    sns.set_style('white')

    for it in range(nb):
        n = it % 5
        if n == 0:
            plt.figure(figsize=(12, 6))
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.subplot(5, 1, n + 1)
        plt.subplots_adjust(wspace=0.1, hspace=0.9)
        plt.plot(taxis, 10 * np.log10(activation[it, :] + EPS))
        plt.title("activation" + str(it + 1))
        # plt.axis([1, 8000, -80, 50])

    if show is True:
        plt.show()


def plot_cost(cost, iter):
    """
    show convergence curve of cost function

    Parameters:
    -----------
    cost: np.ndarray, shape (iter)
        convergence of cost function

    iter: int
        number of iteration for algorithm

    """

    plt.figure()
    sns.set_style('white')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.semilogy(cost)
    plt.xlim([0, iter])
    plt.xlabel("Iteration")
    plt.ylabel("cost function")
