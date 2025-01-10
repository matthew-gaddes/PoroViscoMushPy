#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 13:59:02 2025

@author: matthew
"""

import numpy as np
# from your_module import euler_inversion
# from your_module import Laplace_Dsurfacez

def get_Dsurfacez(rho, d, alpha, phi_o, r_ratio, KsMr, KfMr, KlMr, MmMr,
                  delta, TdTr, tinj, t, timescale=1):
    """
    get_Dsurfacez(rho, d, alpha, phi_o, r_ratio, KsMr, KfMr, KlMr, MmMr,
                  delta, TdTr, tinj, t, timescale=1)
    
    Calculates the time-derivative of the vertical surface displacement.
    
    Parameters
    ----------
    rho : float
        Horizontal distance (normalized by R, the chamber radius).
    d : float
        Depth of the chamber's center (normalized by R).
    alpha : float
    phi_o : float
    r_ratio : float
        Core volume / total volume ratio (e.g., r^3 / R^3).
        If r_ratio = 1, that indicates a liquid chamber end member.
    KsMr, KfMr, KlMr, MmMr : float
        Ratios between various stiffness/rigidity parameters.
        (MmMr: mush rigidity).
    delta : float
        Ratio between injected volume and pre-injection volume (core magma).
    TdTr : float
        Ratio between diffusion and relaxation time.
        - poroelastic end member: TdTr = 0
        - viscoelastic end member: TdTr = ∞
    tinj : float
        Injection time, normalized by the characteristic time scale.
    t : array_like
        1D array of times (normalized). If timescale=0, uses relaxation time 
        as the characteristic scale; if timescale=1, uses the diffusion time.
    timescale : int, optional
        0 or 1, indicating which time scale is used. Default is 1.
    
    Returns
    -------
    Dsurface_z : ndarray
        1D array of length len(t). The time-derivative of the vertical 
        surface displacement at each time in t.
    """
    # Ensure t is a NumPy array (1D).
    t_arr = np.array(t, dtype=float).flatten()
    
    # Define the function handle for Laplace transform.
    # MATLAB code: funz=@(s) Laplace_Dsurfacez(s, rho, d, alpha, phi_o, r_ratio, KsMr, KfMr, KlMr, MmMr, delta, TdTr, tinj, timescale);
    def funz(s):
        return Laplace_Dsurfacez(s, rho, d, alpha, phi_o, r_ratio,
                                 KsMr, KfMr, KlMr, MmMr, delta,
                                 TdTr, tinj, timescale)

    # Perform the inverse Laplace transform using euler_inversion.
    Dsurface_z = euler_inversion(funz, t_arr)

    return Dsurface_z
