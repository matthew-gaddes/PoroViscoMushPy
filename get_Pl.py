#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:03:03 2025

@author: matthew
"""

import numpy as np
import pdb
# from your_module import euler_inversion
# from your_module import Laplace_Pl

def get_Pl(alpha, phi_o, r_ratio, KsMr, KfMr, KlMr, MmMr,
           delta, TdTr, tinj, t, timescale=1):
    """
    get_Pl(alpha, phi_o, r_ratio, KsMr, KfMr, KlMr, MmMr,
           delta, TdTr, tinj, t, timescale=1)
    
    Calculates the core fluid pressure (Pl) normalized by Mr, the rigidity of the crust.

    Parameters
    ----------
    alpha : float
    phi_o : float
    r_ratio : float
        Ratio of core volume to total volume, i.e., (r^3 / R^3).
        When r_ratio = 1 => liquid chamber end member.
    KsMr, KfMr, KlMr, MmMr : float
        Ratios between various stiffness/rigidity parameters (MmMr: mush rigidity).
    delta : float
        Ratio between injected volume and pre-injection volume (core magma).
    TdTr : float
        Ratio between diffusion and relaxation time.
        - poroelastic end member: TdTr = 0
        - viscoelastic end member: TdTr = ∞
    tinj : float
        Injection time, normalized by the chosen characteristic timescale.
    t : array_like
        1D array of times (normalized).
        If timescale = 0, it refers to relaxation time.
        If timescale = 1, it refers to diffusion time.
    timescale : int, optional
        0 or 1, indicating which time scale is used (default is 1).

    Returns
    -------
    Pl_t : ndarray
        1D NumPy array of length len(t), giving the core fluid pressure at each time.
    """
    
    from euler_inversion import euler_inversion
    from Laplace_Pl import Laplace_Pl
    
    # Ensure t is a 1D NumPy array
    t_arr = np.array(t, dtype=float).flatten()

    # Define the Laplace transform function
    def fun(s):
        return Laplace_Pl(s, alpha, phi_o, r_ratio, KsMr, KfMr, KlMr,
                          MmMr, delta, TdTr, tinj, timescale)

    # Perform inverse Laplace transform using euler_inversion
    Pl_t = euler_inversion(fun, t_arr)

    return Pl_t
