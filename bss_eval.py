# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def source(self, estim, true):
    """
    measurement of the separation quality for estimated source signals

    Parameters:
    -----------
      estim: estimated sound source
       true: true sound source

    Returns:
    --------
       SDR: Sources to Distortion Ratio
       SIR: Sources to Interferences Ratio
       SAR: Sources to Artifacts Ratio
      perm: best ordering of estimated source

    """


def decomp_mtifilt(self, estim, true, index, length):
    """
    Decomposition of an estimated source into four components

    Parameters:
    -----------
      estim: estimated sound source
       true: true sound source
      index: source index of true sound source
     length: length of the multichannel time-invariant filters

    Returns:
    --------
      s_target: wanted sources
      e_interf: interferences from non-wanted sources
       e_noise: remaining sensor noise
       e_artif: "burbling" artifacts and error terms

    """


def source_crit(self, s_target, e_interf, e_noise, e_artif):
    """
    measurement of the separation quality for a given source

    Parameters:
    -----------
      s_target: wanted sources
      e_interf: interferences from non-wanted sources
       e_noise: remaining sensor noise
       e_artif: "burbling" artifacts and error terms

    Returns:
    --------
      SDR: Sources to Distortion Ratio
      SIR: Sources to Interferences Ratio
      SAR: Sources to Artifacts Ratio

    """


def orthogonal_project(self, estim, true, length):
    """
    computation of the orthogonal projection

    Parameters:
    -----------
       estim: estimated sound source
        true: true sound source
      length: length of the multichannel time-invariant filters
    Returns:
    --------
      sproj: orthogonal projection from sound source

     """
