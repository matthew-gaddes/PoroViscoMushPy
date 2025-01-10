#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 15:44:16 2025

@author: matthew
"""

#!/usr/bin/env python3

"""
replicate_bash_command.py

This script replicates the logic of your MATLAB script:

1) Load parameters from a .mat file (../data/parameter_example).
2) Compute Pl_t (core fluid pressure), plot & save.
3) Compute M_t (magma transport), plot & save.
4) Compute surface displacement (z, rho) for d=10, plot & save.
5) Compute displacement (u) for r in [0.5..1], save to a .mat file.
6) Compute pore pressure (Pf_r), save to a .mat file.

Equivalent to the original MATLAB script that calls get_Pl, get_M, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat



# 1) Import your Python versions of these functions:
#    (You must have them already translated/implemented.)
# Example:
# from my_laplace_module import get_Pl, get_M, get_surface, get_u, get_Pf

def main():
    """
    """
    
    from get_Pl import get_Pl
    from get_M import get_M
    from get_surface import get_surface
    from get_u import get_u
    from get_Pf import get_Pf
    # -----------------------------------------------------------------------
    # 1) Load the parameter file (like `load('../data/parameter_example')` in MATLAB).
    #    This returns a dictionary of variables. We assume the .mat file
    #    contains alpha, phi_o, r_ratio, KsMr, KfMr, KlMr, MmMr, delta,
    #    TdTr, tinj, t, timescale, etc. all as arrays or scalars.
    # -----------------------------------------------------------------------
    data = loadmat('data/parameter_example.mat')
    
    
    # Extract variables from the dictionary:
    # (Keys might differ depending on how they were saved; 
    #  often MATLAB struct fields appear as data['alpha'], etc.)
    alpha    = data['alpha'].flatten()[0]
    phi_o    = data['phi_o'].flatten()[0]
    r_ratio  = data['r_ratio'].flatten()[0]
    KsMr     = data['KsMr'].flatten()[0]
    KfMr     = data['KfMr'].flatten()[0]
    KlMr     = data['KlMr'].flatten()[0]
    MmMr     = data['MmMr'].flatten()[0]
    delta    = data['delta'].flatten()[0]
    TdTr     = data['TdTr'].flatten()[0]
    tinj     = data['tinj'].flatten()[0]
    timescale= data['timescale'].flatten()[0]
    # t is presumably an arraysummation = sum(eta_mesh .* real(arrayfun(f_s, beta_mesh./t_mesh)), 2) of times
    t        = data['t'].flatten()

    # -----------------------------------------------------------------------
    # 2) Compute Pl_t and save the figure "pressure_evolution.png"
    # -----------------------------------------------------------------------
    # In Python, your get_Pl(...) function should accept these arguments
    # and return a NumPy array for Pl_t.
    Pl_t = get_Pl(alpha, phi_o, r_ratio,
                  KsMr, KfMr, KlMr, MmMr,
                  delta, TdTr, tinj, t, timescale)

    plt.figure()
    plt.plot(t, Pl_t, label='Pl(t)')
    plt.xlabel('normalized time')
    plt.ylabel('normalized presummation = sum(eta_mesh .* real(arrayfun(f_s, beta_mesh./t_mesh)), 2)ssure')
    plt.title('Pressure Evolution')
    plt.legend()
    plt.savefig('results/pressure_evolution.png', dpi=150)
    plt.close()

    # -----------------------------------------------------------------------
    # 3) Compute M_t, plot M_t/delta, save "magma_transport.png"
    # -----------------------------------------------------------------------
    M_t = get_M(alpha, phi_o, r_ratio,
                KsMr, KfMr, KlMr, MmMr,
                delta, TdTr, tinj, t, timescale)

    plt.figure()
    plt.plot(t, M_t/delta, label='M(t)/delta')
    plt.xlabel('normalized time')
    plt.ylabel('magma transport M/δ')
    plt.title('Magma Transport')
    plt.legend()
    plt.savefig('results/magma_transport.png', dpi=150)
    plt.close()

    # -----------------------------------------------------------------------
    # 4) Compute surface displacement for d=10, plot "ground_elevation.png"
    #    In MATLAB: [surface_z, surface_rho] = get_surface(0,10, ...)
    # -----------------------------------------------------------------------
    #  Suppose your get_surface(...) in Python is similarly defined.
    surface_z, surface_rho = get_surface(0.0, 10.0,
                                         alpha, phi_o, r_ratio,
                                         KsMr, KfMr, KlMr, MmMr,
                                         delta, TdTr, tinj, t, timescale)
    plt.figure()
    plt.plot(t, surface_z, label='surface_z')
    plt.xlabel('normalized time')
    plt.ylabel('normalized ground elevation')
    plt.title('depth d = 10 R_o')
    plt.legend()
    plt.savefig('results/ground_elevation.png', dpi=150)
    plt.close()

    # -----------------------------------------------------------------------
    # 5) Compute displacement for r = linspace(0.5, 1, 25), save to "deformation.mat"
    # -----------------------------------------------------------------------
    r_array = np.linspace(0.5, 1.0, 25)
    # In MATLAB: [u, r_out, t_out] = get_u(...)
    # We'll assume the Python version returns (u, r_out, t_out) similarly.
    u, r_out, t_out = get_u(alpha, phi_o, r_ratio,
                            KsMr, KfMr, KlMr, MmMr,
                            delta, TdTr, tinj, t, timescale,
                            r_array)
    # Save displacement data to .mat
    savemat('results/deformation.mat', {
        'u': u,
        'r_out': r_out,
        't_out': t_out
    })

    # -----------------------------------------------------------------------
    # 6) Compute pore pressure Pf_r, save to "pore_pressure.mat"
    # -----------------------------------------------------------------------
    Pf_r = get_Pf(alpha, phi_o, r_ratio,
                   KsMr, KfMr, KlMr, MmMr,
                   delta, TdTr, tinj, t, timescale,
                   r_array)

    savemat('results/pore_pressure.mat', {
        'Pf_r': Pf_r,
        'r': r_array,
        't': t
    })

    print('All computations and figure saves are complete!')

# Entry point when running this script
if __name__ == '__main__':
    main()
