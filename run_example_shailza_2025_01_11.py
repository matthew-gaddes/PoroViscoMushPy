#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 18:18:53 2025

@author: matthew
"""

import pdb

import numpy as np
import matplotlib
#matplotlib.use("TkAgg")  # or "Qt5Agg" if you have PyQt5/PySide2 installed
import matplotlib.pyplot as plt
from scipy.io import savemat
# 1) Import your Python versions of these functions:
from get_Pl import get_Pl
from get_M import get_M
from get_surface import get_surface
from get_u import get_u
from get_Pf import get_Pf
from scipy.io import loadmat
 
 
 
def main():
    """
    Main script to load parameters from parameter_example.mat, compute Pl_t, M_t,
    surface displacement, deformation, and pore pressure. The only change is
    in the parameters passed to get_surface, replicating the first case from
    replicate_matlab_case1 for delta=0.01 and tinj=1.0, etc.
    """
    # -----------------------------------------------------------------------
    # 1) Import your translated/implemented functions
    # -----------------------------------------------------------------------
    from get_Pl import get_Pl
    from get_M import get_M
    from get_surface import get_surface
    from get_u import get_u
    from get_Pf import get_Pf
 
    # -----------------------------------------------------------------------
    # 2) Load the parameter file (parameter_example.mat) for global parameters
    # -----------------------------------------------------------------------
    #data = loadmat('/home/bfxk361/Downloads/PoroViscoMushPy-main/data/parameter_example.mat')
    data = loadmat('data/parameter_example.mat')
    
    # Extract variables from the dictionary.
    alpha     = data['alpha'].flatten()[0]
    phi_o     = data['phi_o'].flatten()[0]
    r_ratio   = data['r_ratio'].flatten()[0]
    KsMr      = data['KsMr'].flatten()[0]
    KfMr      = data['KfMr'].flatten()[0]
    KlMr      = data['KlMr'].flatten()[0]
    MmMr      = data['MmMr'].flatten()[0]
    delta     = data['delta'].flatten()[0]
    TdTr      = data['TdTr'].flatten()[0]
    tinj      = data['tinj'].flatten()[0]
    timescale = data['timescale'].flatten()[0]
    t         = data['t'].flatten()
 
    # -----------------------------------------------------------------------
    # 3) Compute Pl(t) and save figure
    # -----------------------------------------------------------------------
    Pl_t = get_Pl(alpha, phi_o, r_ratio,
                  KsMr, KfMr, KlMr, MmMr,
                  delta, TdTr, tinj, t, timescale)
 
    plt.figure()
    plt.plot(t, Pl_t, label='Pl(t)')
    plt.xlabel('normalized time')
    plt.ylabel('normalized pressure')
    plt.title('Pressure Evolution')
    plt.legend()
    plt.savefig('results/pressure_evolution.png', dpi=150)
    plt.close()
 
    # -----------------------------------------------------------------------
    # 4) Compute M(t), plot M(t)/delta, save figure
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
 
    t_Tr_case1 = 15
    num_points_case1 = 300
    t_case1 = np.linspace(0, t_Tr_case1, num_points_case1)
 
    # Overridden parameters (First Case)
    KsMr_c1  = 0.1
    KfMr_c1  = 0.9
    alpha_c1 = 0.7
    phi_o_c1 = 0.9
    r_ratio_c1 = 6
    d_c1 = 5.0 / 3.0
    delta_c1 = 0.01
    TdTr_c1 = 10.0
    tinj_c1 = 1.0
    timescale_c1 = 1

    pdb.set_trace()
 
    # Compute dimensionless surface displacement with the overridden parameters
    surface_z_c1, surface_rho_c1 = get_surface(rho=0.0,
                                              d=d_c1,
                                              alpha=alpha_c1,
                                              phi_o=phi_o_c1,
                                              r_ratio=r_ratio_c1,
                                              KsMr=KsMr_c1,
                                              KfMr=KfMr_c1,
                                              KlMr=KlMr,     # still from loaded data, adjust if needed
                                              MmMr=MmMr,     # still from loaded data, adjust if needed
                                              delta=delta_c1,
                                              TdTr=TdTr_c1,
                                              tinj=tinj_c1,
                                              t=t_case1,
                                              timescale=timescale_c1)
 
    # Convert dimensionless displacement to meters
    surface_z_meters = surface_z_c1 * 3000.0
 
    # Convert dimensionless time to days
    days = 24 * 3600
    Tr   = 100 * days         # as in replicate_matlab_case1
    Td   = TdTr_c1 * Tr
    t_days = t_case1 * Td / days
 
    # Plot ground elevation vs time in days
    plt.figure(figsize=(8, 5))
    plt.plot(t_days, surface_z_meters, linewidth=2, label='surface displacement')
    plt.xlabel('Time [days]')
    plt.ylabel('Displacement [m]')
    plt.title('Case 1: Uplift + Subsidence')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/ground_elevation_case1.png', dpi=150)
    plt.close()
 
    # -----------------------------------------------------------------------
    # 6) Compute displacement for r = linspace(0.5, 1, 25), save to .mat
    #    (Unchanged parameters for get_u)
    # -----------------------------------------------------------------------
    r_array = np.linspace(0.5, 1.0, 25)
    u, r_out, t_out = get_u(alpha, phi_o, r_ratio,
                            KsMr, KfMr, KlMr, MmMr,
                            delta, TdTr, tinj, t, timescale,
                            r_array)
    savemat('results/deformation.mat', {
        'u': u,
        'r_out': r_out,
        't_out': t_out
    })
 
    # -----------------------------------------------------------------------
    # 7) Compute pore pressure Pf_r, save to .mat
    #    (Unchanged parameters for get_Pf)
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
 
# Entry point
if __name__ == '__main__':
    main()