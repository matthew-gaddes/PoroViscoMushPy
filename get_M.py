#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:00:11 2025

@author: matthew
"""

import numpy as np
# from your_module import euler_inversion
# from your_module import Laplace_M

def get_M(alpha, phi_o, r_ratio, KsMr, KfMr, KlMr, MmMr,
          delta, TdTr, tinj, t, timescale=1):
    """
    get_M(alpha, phi_o, r_ratio, KsMr, KfMr, KlMr, MmMr, delta, TdTr, tinj, t, timescale=1)
    
    Calculates M_leak (normalized by M0, the pre-injection core magma).

    Parameters
    ----------
    alpha : float
    phi_o : float
    r_ratio : float
        Ratio of core volume to total volume, i.e., (r^3 / R^3).
        r_ratio=1 corresponds to a liquid chamber end member.
    KsMr, KfMr, KlMr, MmMr : float
        Ratios between specific stiffness/rigidity parameters 
        (e.g., MmMr: mush rigidity).
    delta : float
        Ratio between injected volume and pre-injection volume.
    TdTr : float
        Ratio between diffusion and relaxation time.
        - Poroelastic end member: TdTr = 0
        - Viscoelastic end member: TdTr = ∞
    tinj : float
        Injection time, normalized by the chosen characteristic timescale.
    t : array_like
        1D array of times (normalized by the chosen timescale).
        If timescale = 0, it refers to relaxation time.
        If timescale = 1, it refers to diffusion time.
    timescale : int, optional
        0 or 1, indicating which time scale is used (default is 1).
    
    Returns
    -------
    M_t : ndarray
        A 1D NumPy array giving the normalized M(t) for each time in t.
    """
    from euler_inversion import euler_inversion
    from Laplace_M import Laplace_M
    
    # Ensure t is a 1D NumPy array
    t_arr = np.array(t, dtype=float).flatten()

    # Define the function handle for the Laplace transform
    def fun(s):
        return Laplace_M(s, alpha, phi_o, r_ratio, KsMr, KfMr, KlMr,
                         MmMr, delta, TdTr, tinj, timescale)
    
    # Perform the inverse Laplace transform via euler_inversion
    M_t = euler_inversion(fun, t_arr)
    
    return M_t
