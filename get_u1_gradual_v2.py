#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:07:01 2025

@author: matthew
"""

import numpy as np
# from your_module import euler_inversion
# from your_module import Laplace_gradual_u1_v2

def get_u1_gradual_v2(alpha, phi_o, r_ratio, KsMr, KfMr, KlMr, MmMr,
                      delta, TdTr, tinj, t, timescale=1):
    """
    get_u1_gradual_v2(alpha, phi_o, r_ratio, KsMr, KfMr, KlMr, MmMr,
                      delta, TdTr, tinj, t, timescale=1)

    Calculates the displacement (u1) at the chamber wall (r=1), normalized by R0.

    Parameters
    ----------
    alpha : float
    phi_o : float
    r_ratio : float
        Ratio of core volume to total volume, i.e. (r^3 / R^3).
        If r_ratio=1, this is the "liquid chamber end member."
    KsMr, KfMr, KlMr, MmMr : float
        Ratios between material properties (e.g., mush rigidity).
    delta : float
        Ratio between injected volume and pre-injection volume (core magma).
    TdTr : float
        Ratio between diffusion and relaxation time.
        - poroelastic end member: TdTr = 0
        - viscoelastic end member: TdTr = ∞
    tinj : float
        Injection time, normalized by the chosen characteristic timescale.
    t : array_like
        1D array of times (normalized by the chosen timescale).
        If timescale=0 => relaxation time
        If timescale=1 => diffusion time
    timescale : int, optional
        0 or 1, indicating which timescale is used (default=1).

    Returns
    -------
    u1 : ndarray
        1D NumPy array of length len(t). The displacement at r=1, 
        normalized by the chamber radius R0, for each time in t.
    """
    from euler_inversion import euler_inversion
    from Laplace_gradual_u1_v2 import Laplace_gradual_u1_v2
    
    # Convert t to a 1D NumPy array
    t_arr = np.array(t, dtype=float).flatten()

    # Define the Laplace transform function
    def fun(s):
        return Laplace_gradual_u1_v2(s, alpha, phi_o, r_ratio,
                                     KsMr, KfMr, KlMr, MmMr,
                                     delta, TdTr, tinj, timescale)
    
    # Use the euler_inversion routine to invert the Laplace transform
    u1 = euler_inversion(fun, t_arr)

    return u1
