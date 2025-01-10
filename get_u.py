#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:05:49 2025

@author: matthew
"""

import numpy as np
# from your_module import get_Div_r_gradual_v2, get_u1_gradual_v2

def get_u(alpha, phi_o, r_ratio, KsMr, KfMr, KlMr, MmMr,
          delta, TdTr, tinj, t, timescale, r):
    """
    get_u(alpha, phi_o, r_ratio, KsMr, KfMr, KlMr, MmMr,
          delta, TdTr, tinj, t, timescale, r)

    Calculates the radial displacement (u), normalized by R0 (the chamber radius).
    
    Parameters
    ----------
    alpha : float
    phi_o : float
    r_ratio : float
        Ratio of core volume to total volume (r^3 / R^3).
    KsMr, KfMr, KlMr, MmMr : float
        Ratios between various stiffness/rigidity parameters (MmMr: mush rigidity).
    delta : float
        Ratio between injected volume and pre-injection volume (core magma).
    TdTr : float
        Ratio between diffusion and relaxation time:
            - poroelastic end member: TdTr = 0
            - viscoelastic end member: TdTr = ∞
    tinj : float
        Injection time, normalized by the chosen characteristic timescale.
    t : array_like
        1D array of times (normalized by chosen timescale).
        If timescale=0 => relaxation time
        If timescale=1 => diffusion time
    timescale : int
        0 or 1, indicating which timescale is used.
    r : array_like
        1D array of radial positions, normalized by R (the chamber radius).
    
    Returns
    -------
    u : 2D ndarray
        Displacement (normalized by R) of shape (len(r), len(t)).
        Each row corresponds to a given r[i], each column to time t[j].
    r : 1D ndarray
        Same radial positions as input, returned for convenience.
    t : 1D ndarray
        Same times as input, returned for convenience.

    Notes
    -----
    The translation preserves the MATLAB logic:
      1. Obtain the radial divergence `Div` from get_Div_r_gradual_v2.
      2. Obtain `u1_t` from get_u1_gradual_v2 (one particular radial solution).
      3. Perform an integral-like accumulation over r to form `a1`.
      4. Combine `u1_t` and `a1` to form the final displacement `u`.
    """
    from get_Div_r_gradual_v2 import get_Div_r_gradual_v2
    from get_u1_gradual_v2 import get_u1_gradual_v2

    # Convert inputs to NumPy arrays (flatten ensures 1D)
    r_arr = np.array(r, dtype=float).flatten()
    t_arr = np.array(t, dtype=float).flatten()
    
    # 1) Compute the radial divergence for each r, t
    Div = get_Div_r_gradual_v2(alpha, phi_o, r_ratio, KsMr, KfMr, KlMr,
                               MmMr, delta, TdTr, tinj, t_arr, timescale, r_arr)
    # Shape of Div: (len(r), len(t))

    # 2) Compute u1_t for times t (the "reference" displacement at r=1, typically)
    u1_t = get_u1_gradual_v2(alpha, phi_o, r_ratio, KsMr, KfMr, KlMr,
                             MmMr, delta, TdTr, tinj, t_arr, timescale)
    # In MATLAB, we do:  u1_t = u1_t';
    # If get_u1_gradual_v2 returns a (len(t),) shape, we can keep it as is.
    # Otherwise, transpose/flatten if needed:
    u1_t = np.array(u1_t, dtype=float).flatten()  # ensure shape (len(t),)

    # Prepare the output array u, same shape as Div
    u = np.ones((r_arr.size, t_arr.size), dtype=float)

    # 3) Calculate the integral-like accumulation: int_r^1(Div) * r^2 dr
    # First define the radial step (assuming uniform spacing in r)
    if len(r_arr) > 1:
        dr = r_arr[1] - r_arr[0]
    else:
        # Edge case: if there's only one r value, set dr=0 or something consistent
        dr = 0.0

    # In MATLAB:
    #   a1 = Div .* (r.^2)' * dr;
    #   for ii=1:numel(r)-1
    #       a1(ii,:) = sum(a1(ii:end-1,:),1);
    #   end
    #   a1(end,:) = 0*a1(end,:);
    #
    # Let's translate that step by step.

    # Multiply each row of Div by r^2 (row-by-row) and by dr
    # (r^2).reshape(-1,1) makes shape (len(r),1) so it can multiply Div (len(r), len(t)) row-wise
    a1 = Div * (r_arr**2).reshape(-1, 1) * dr  # shape: (len(r), len(t))

    # Accumulate sums from row i through row (end-1), then store in row i
    # This is effectively a "top-down" partial cumulative sum, ignoring the last row in each sum.
    for i in range(r_arr.size - 1):
        # sum over rows from i to end-1
        # axis=0 => sum across rows
        a1[i, :] = np.sum(a1[i:-1, :], axis=0)
    
    # Then set the last row to zero
    a1[-1, :] = 0.0

    # 4) Combine u1_t and a1 to form final displacement
    # In MATLAB:
    #   for ii=1:numel(r)
    #       u(ii,:) = u1_t/(r(ii)^2) - a1(ii,:)/(r(ii)^2);
    #   end
    #
    # We'll do that in Python by looping (or vectorizing).
    for i in range(r_arr.size):
        denom = (r_arr[i] ** 2)
        # Each column is time, so we do an elementwise operation
        u[i, :] = u1_t / denom - a1[i, :] / denom
    
    # Return u, along with r and t for convenience
    return u, r_arr, t_arr
