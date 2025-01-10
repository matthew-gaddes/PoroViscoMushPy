#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:01:50 2025

@author: matthew
"""

import numpy as np
# from your_module import euler_inversion
# from your_module import Laplace_Pf

def get_Pf(alpha, phi_o, r_ratio, KsMr, KfMr, KlMr, MmMr,
           delta, TdTr, tinj, t, timescale, r):
    """
    get_Pf(alpha, phi_o, r_ratio, KsMr, KfMr, KlMr, MmMr,
           delta, TdTr, tinj, t, timescale, r)
    
    Calculates pore pressure (Pf) at radial position r, normalized by Mr (rigidity of crust).
    
    Parameters
    ----------
    alpha : float
    phi_o : float
    r_ratio : float
        Ratio of core volume to total volume, i.e., (r^3 / R^3).
        When r_ratio = 1 => liquid chamber end member.
    KsMr, KfMr, KlMr, MmMr : float
        Ratios between various stiffness/rigidity parameters.
        (MmMr: mush rigidity)
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
        If timescale=0, uses relaxation time;
        if timescale=1, uses diffusion time.
    timescale : int
        0 or 1, indicating which time scale is used.
    r : array_like
        1D array of radial positions (normalized by chamber radius R).
        Range from (r0 / R) up to 1.
    
    Returns
    -------
    Pf_r : 2D ndarray, shape (len(r), len(t))
        The pore pressure at each combination of r[i] and t[j].
        Each row corresponds to a given r[i], and each column to a time t[j].
    """
    from euler_inversion import euler_inversion
    from Laplace_Pf import Laplace_Pf
    
    # Convert t and r to NumPy arrays, ensuring they're 1D
    t_arr = np.array(t, dtype=float).flatten()
    r_arr = np.array(r, dtype=float).flatten()
    
    # Initialize the result array
    Pf_r = np.ones((r_arr.size, t_arr.size), dtype=float)
    
    # Loop over each radial position
    for i in range(r_arr.size):
        # Define the Laplace transform function handle for this r[i].
        def fun(s):
            return Laplace_Pf(s, alpha, phi_o, r_ratio, KsMr, KfMr, KlMr,
                              MmMr, delta, TdTr, tinj, timescale, r_arr[i])
        
        # Perform Euler inversion for each time in t.
        Pf_r[i, :] = euler_inversion(fun, t_arr)
    
    return Pf_r
