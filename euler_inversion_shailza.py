#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 19:08:39 2025

@author: matthew
"""

import pdb
import numpy as np

def euler_inversion(f_s, t, M=32):
    
    """
    """
    
    print("Usaing Shailza's version of the euler inversion")
    
    # Convert t to a NumPy array if it isn't already.
    t = np.array(t, dtype=float).flatten()
    if t.ndim != 1:
        raise ValueError("Input times, t, must be a 1D array.")
    # Helper function for the binomial-like product in the original code.
    def bnml(n, z):
        numerator = 1.0
        denominator = 1.0
        for i in range(1, z + 1):
            numerator *= (n - (z - i))
            denominator *= i
        return numerator / denominator
 
    # Correct initialization of xi
    xi = np.zeros(2 * M + 1, dtype=float)
    xi[0] = 0.5
    xi[1:M+1] = 1.0       # Set xi[1] to xi[M] to 1.0
    xi[M+1:2*M] = 0.0      # Ensure xi[M+1] to xi[2*M-1] are 0.0
    xi[2*M] = 2.0 ** (-M)
    
    pdb.set_trace()
    import matplotlib.pyplot as plt
    f, ax = plt.subplots(1);  ax.matshow(xi[:, np.newaxis], aspect = 0.01)
    
    # Fill middle portion of xi (corresponding to k=1 to M-1 in the for-loop)
    for k in range(1, M):
        idx = 2*M - k
        xi[idx] = xi[idx + 1] + (2.0 ** (-M)) * bnml(M, k)
    # Create the k array
    k_vals = np.arange(0, 2*M+1, dtype=float)
    # Compute beta and eta
    beta = (M * np.log(10) / 3.0) + 1j * np.pi * k_vals
    eta = (1.0 - 2.0 * (k_vals % 2)) * xi
    # Create meshgrid
    beta_mesh, t_mesh = np.meshgrid(beta, t, indexing='xy')  # shape: (len(t), 2*M+1)
    eta_mesh = np.meshgrid(eta, t, indexing='xy')[0]        # same shape as beta_mesh
    # Evaluate f_s at all points in beta_mesh / t_mesh
    s_vals = beta_mesh / t_mesh
    F_vals = np.vectorize(f_s)(s_vals)  # Ensure f_s is properly vectorized
    real_F_vals = np.real(F_vals)
    # Summation over k (axis=1)
    summation = np.sum(eta_mesh * real_F_vals, axis=1)  # shape: (len(t),)
    # Final multiplication
    factor = (10.0 ** (M / 3.0)) / t
    ilt = factor * summation
    return ilt