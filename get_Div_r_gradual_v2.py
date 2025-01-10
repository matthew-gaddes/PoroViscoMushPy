#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 13:42:11 2025

@author: matthew
"""

import numpy as np
# from your_module import euler_inversion    # Make sure to import euler_inversion
# from your_module import Laplace_gradual_Div_r_v2  # The Laplace transform function you'll define

def get_Div_r_gradual_v2(alpha, phi_o, r_ratio, KsMr, KfMr, KlMr, MmMr,
                         delta, TdTr, tinj, t, timescale, r):
    """
    get_Div_r_gradual_v2(alpha, phi_o, r_ratio, KsMr, KfMr, KlMr, MmMr, delta,
                         TdTr, tinj, t, timescale, r)

    This code calculates the divergence of displacement.
    
    Parameters
    ----------
    alpha : float
    phi_o : float
    r_ratio : float
        Core volume / total volume ratio (e.g., r^3 / R^3).
    KsMr, KfMr, KlMr, MmMr : float
        Ratios between stiffness or rigidity parameters.
        (Mm: mush rigidity)
    delta : float
        Ratio between injected volume and pre-injection volume (core magma).
    TdTr : float
        Ratio between diffusion and relaxation time.
        poroelastic end member: TdTr=0
        viscoelastic end member: TdTr=∞
    tinj : float
        Injection time, normalized by characteristic time scale.
    t : array_like
        1D array of times (normalized by characteristic time scale).
        If timescale = 0, uses relaxation time as characteristic time scale.
        If timescale = 1, uses diffusion time as characteristic time scale.
    timescale : int
        0 or 1 indicating which time scale is used.
    r : array_like
        1D array of radii (normalized by chamber radius).

    Returns
    -------
    Div_r : 2D ndarray of shape (len(r), len(t))
        The divergence of displacement at each combination of r and t.
    """
    from euler_inversion import euler_inversion
    from Laplace_gradual_Div_r_v2 import Laplace_gradual_Div_r_v2
    
    # Ensure t and r are NumPy arrays
    t_arr = np.array(t, dtype=float).flatten()
    r_arr = np.array(r, dtype=float).flatten()

    # Initialize Div_r with ones, same shape as MATLAB: (numel(r), numel(t))
    Div_r = np.ones((r_arr.size, t_arr.size), dtype=float)
    
    # Loop over each radius
    for i in range(r_arr.size):
        # Define the Laplace transform function handle for each r[i].
        def fun(s):
            return Laplace_gradual_Div_r_v2(s, alpha, phi_o, r_ratio,
                                            KsMr, KfMr, KlMr, MmMr,
                                            delta, TdTr, tinj, timescale,
                                            r_arr[i])
        
        # Perform Euler inversion for each time in t.
        Div_r[i, :] = euler_inversion(fun, t_arr)
    
    return Div_r
