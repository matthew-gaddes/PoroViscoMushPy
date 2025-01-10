#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 13:36:05 2025

@author: matthew
"""

import numpy as np
import pdb

def euler_inversion(f_s, t, M=32):
    """
    euler_inversion(f_s, t, M=32)
    
    Returns an approximation to the inverse Laplace transform of the function
    handle f_s evaluated at each value in t (1D array) using the Euler method.
    
    Parameters
    ----------
    f_s : callable
        A function handle that computes F(s), the Laplace transform. Must be
        callable on arrays of s values (vectorized).
    t : array_like
        1D array of times at which to evaluate the inverse Laplace transform.
    M : int, optional
        Number of terms to sum for each t (default = 32).
        
    Returns
    -------
    ilt : ndarray
        1D array of the same length as t, giving the approximate inverse
        Laplace transform at each time in t.
    
    Notes
    -----
    This Python version mirrors the logic of the MATLAB code:
    
        function ilt = euler_inversion(f_s, t, M)
        % ...
    
    Reference:
        Abate, Joseph, and Ward Whitt. "A Unified Framework for Numerically 
        Inverting Laplace Transforms." INFORMS Journal of Computing, 
        vol. 18.4 (2006): 408-421.
    """
    # Convert t to a NumPy array if it isn't already.
    t = np.array(t, dtype=float).flatten()
    if t.ndim != 1:
        raise ValueError("Input times, t, must be a 1D array.")
    
    # Helper function for the binomial-like product in the original code.
    def bnml(n, z):
        # bnml = @(n, z) prod((n-(z-(1:z)))./(1:z)) in MATLAB
        # We replicate that logic directly below:
        numerator = 1.0
        denominator = 1.0
        for i in range(1, z + 1):
            numerator *= (n - (z - i))
            denominator *= i
        return numerator / denominator

    # Build xi array
    xi = np.zeros(2 * M + 1, dtype=float)
    xi[0] = 0.5
    xi[1] = 1.0
    xi[M] = 1.0
    xi[2*M] = 2.0 ** (-M)
    
    # Fill middle portion of xi (corresponding to k=1 to M-1 in the for-loop)
    for k in range(1, M):
        # In MATLAB: xi(2*M-k + 1) = xi(2*M-k + 2) + 2^-M * bnml(M, k)
        # Note: Python is 0-based, so we adjust indexing:
        idx = 2*M - k
        xi[idx] = xi[idx + 1] + (2.0 ** (-M)) * bnml(M, k)
    
    # Create the k array
    k_vals = np.arange(0, 2*M+1, dtype=float)
    
    # Compute beta and eta
    beta = (M * np.log(10) / 3.0) + 1j * np.pi * k_vals
    # In MATLAB: eta = (1 - mod(k,2)*2) .* xi
    # => mod(k,2) is 0 or 1. If 0, (1 - 0*2)=1; if 1, (1-2)= -1
    # => This flips sign for odd k.
    eta = (1.0 - 2.0 * (k_vals % 2)) * xi
    
    # We want to evaluate f_s(beta / t) for each t. We'll make a grid:
    # In MATLAB: [beta_mesh, t_mesh] = meshgrid(beta, t)
    # => t goes along rows, beta along columns
    beta_mesh, t_mesh = np.meshgrid(beta, t, indexing='xy')  # shape: (len(beta), len(t))
    # But we actually want to keep the same orientation as MATLAB sum(..., 2).
    # We'll adapt accordingly. We'll handle it carefully below.
    
    # Similarly for eta:
    eta_mesh = np.meshgrid(eta, t, indexing='xy')[0]  # same shape as beta_mesh
    
    # Evaluate f_s at all points in beta_mesh / t_mesh
    # We'll do elementwise division first:
    # note that in the original implementation, this is not done explicitly
    # (i.e. s_vals is not computed)
    s_vals = beta_mesh / t_mesh  # shape = (2M+1, len(t))
    

    
    # If f_s is truly vectorized, we can call it once on the entire s_vals.
    # Otherwise, you may need np.vectorize.
    f_s_vectorized = np.vectorize(f_s)
    F_vals = f_s_vectorized(s_vals)  # shape = (2M+1, len(t)) if vectorized
    
    # Now, we take real(...) of that:
    real_F_vals = np.real(F_vals)
    
    # Multiply by eta_mesh, then sum over k (the 'row' in this shape),
    # because in MATLAB sum(..., 2) means "sum across columns for each row".
    # Here, each 'row' is a single beta index, so we sum along axis=0
    # or axis=1 depending on how meshgrid was used.
    #
    # We used indexing='xy', which yields:
    #   beta_mesh.shape = (2M+1, len(t))
    #   t_mesh.shape    = (2M+1, len(t))
    # Summation "over k" => sum along axis=0 in this shape, leaving len(t) elements.
    
    summation = np.sum(eta_mesh * real_F_vals, axis=1)  # shape: (len(t),)
    
    # Finally, multiply by 10^(M/3) / t (elementwise).
    # 10^(M/3) is the same as 10**(M/3.0).
    factor = (10.0 ** (M / 3.0)) / t
    
    # `factor` is shape (len(t),), while `summation` is (len(t),).
    # We'll do elementwise multiply:
    ilt = factor * summation
    
    return ilt
