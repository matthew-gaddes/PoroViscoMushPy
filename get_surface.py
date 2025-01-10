#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:04:16 2025

@author: matthew
"""

import numpy as np
# from your_module import euler_inversion
# from your_module import Laplace_surfacez, Laplace_surfacerho

def get_surface(rho, d, alpha, phi_o, r_ratio, KsMr, KfMr, KlMr, MmMr,
                delta, TdTr, tinj, t, timescale=1):
    """
    get_surface(rho, d, alpha, phi_o, r_ratio, KsMr, KfMr, KlMr, MmMr,
                delta, TdTr, tinj, t, timescale=1)

    Calculates the surface displacement (both vertical and radial components),
    normalized by R0 (the characteristic radius).

    Parameters
    ----------
    rho : float
        Horizontal distance (normalized by R0).
    d : float
        Depth of the chamber center (normalized by R0).
    alpha : float
    phi_o : float
    r_ratio : float
        Core volume / total volume ratio, i.e., (r^3 / R^3).
    KsMr, KfMr, KlMr, MmMr : float
        Ratios between stiffness/rigidity parameters (MmMr: mush rigidity).
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
        - If timescale=0, uses relaxation time
        - If timescale=1, uses diffusion time
    timescale : int, optional
        0 or 1, indicating which timescale is used. Default is 1.

    Returns
    -------
    surface_z : ndarray
        1D array of length len(t) giving the vertical surface displacement
        over time, normalized by R0.
    surface_rho : ndarray
        1D array of length len(t) giving the radial surface displacement
        over time, normalized by R0.
    """
    from euler_inversion import euler_inversion
    from Laplace_surfacez import Laplace_surfacez
    from Laplace_surfacerho import Laplace_surfacerho
    
    # Ensure t is a 1D NumPy array
    t_arr = np.array(t, dtype=float).flatten()

    # Define function handle for the Laplace transform of surface_z
    def funz(s):
        return Laplace_surfacez(s, rho, d, alpha, phi_o, r_ratio,
                                KsMr, KfMr, KlMr, MmMr,
                                delta, TdTr, tinj, timescale)

    # Invert to get the vertical surface displacement
    surface_z = euler_inversion(funz, t_arr)

    # Define function handle for the Laplace transform of surface_rho
    def funrho(s):
        return Laplace_surfacerho(s, rho, d, alpha, phi_o, r_ratio,
                                  KsMr, KfMr, KlMr, MmMr,
                                  delta, TdTr, tinj, timescale)

    # Invert to get the radial surface displacement
    surface_rho = euler_inversion(funrho, t_arr)

    return surface_z, surface_rho
