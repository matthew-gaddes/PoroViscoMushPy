#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 19:08:18 2025

@author: matthew
"""

import numpy as np
import matplotlib
#matplotlib.use("TkAgg")  # or "Qt5Agg", etc.
import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy.io import loadmat
 
# Import your Python versions of these functions:
from get_Pl import get_Pl
from get_M import get_M
from get_surface import get_surface
from get_u import get_u
from get_Pf import get_Pf
 
def main():
    """
    Modified main script to replicate 'Case 1' parameters for *all* computations,
    Total time period of 10 years (3600 days),
    sampled every 6 days => 600 data points (plus the initial point => 601).
    """
    # -----------------------------------------------------------------------
    # 1) Define "Case 1" parameters
    # -----------------------------------------------------------------------
    alpha_c1   = 0.7
    phi_o_c1   = 0.9
    r_ratio_c1 = 6.0
    KsMr_c1    = 0.1
    KfMr_c1    = 0.9
 
    # Load KlMr, MmMr from parameter_example.mat (as in original code)
    # data = loadmat('/home/bfxk361/Downloads/PoroViscoMushPy-main/data/parameter_example.mat')
    data = loadmat('data/parameter_example.mat')

    
    KlMr   = data['KlMr'].flatten()[0]  # e.g., 1.0
    MmMr   = data['MmMr'].flatten()[0]  # e.g., 100.0
 
    delta_c1   = 0.01
    TdTr_c1    = 10.0   # ratio of diffusion timescale to relaxation timescale
    tinj_c1    = 1.0
    timescale_c1 = 1
 
    # Depth of injection center (dimensionless units)
    d_c1 = 5.0/3.0
    
    
    pdb.set_trace()
 
    # -----------------------------------------------------------------------
    # 2) Define the real (physical) time array in days (0 -> 3600 days in steps of 6)
    #    => total of 3600 days ~ 10 years
    # -----------------------------------------------------------------------
    # We'll use np.arange so that we get exactly 0, 6, 12, ... 3600
    # This yields 601 time points.
    t_days_array = np.arange(0, 3600+1, 6)  # from day 0 to day 3600 in steps of 6
 
    # Number of seconds in a day
    seconds_per_day = 24 * 3600
 
    # "Relaxation timescale" in seconds (Tr)
    # original code had: Tr = 100 * days
    Tr = 50.0 * seconds_per_day
    #Tr =  seconds_per_day
    # Diffusion timescale in seconds
    # If timescale=1, then Td = (TdTr_c1)*Tr
    Td = TdTr_c1 * Tr  # e.g., 10.0 * (100 days in seconds) = 1000 days in seconds
 
    # Convert from real time (days) to dimensionless time
    # t_case1 = t [seconds] / Td
    # But t [seconds] = (t_days * seconds_per_day)
    t_case1 = (t_days_array * seconds_per_day) / Td
 
    # -----------------------------------------------------------------------
    # 3) Compute Pl(t) with the "Case 1" parameters, plot vs. t_days_array
    #    (which goes up to 3600 days)
    # -----------------------------------------------------------------------
    Pl_t_case1 = get_Pl(alpha_c1, phi_o_c1, r_ratio_c1,
                        KsMr_c1, KfMr_c1, KlMr, MmMr,
                        delta_c1, TdTr_c1, tinj_c1,
                        t_case1, timescale_c1)
 
    plt.figure()
    plt.plot(t_days_array, Pl_t_case1, label='Pl(t) - Case 1', linewidth=2)
    plt.xlabel('Time [days]')
    plt.ylabel('Normalized Pressure, P_l (dimensionless)')
    plt.title('Case 1: Core Pressure Evolution (0-10 yrs)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/pressure_evolution_case1.png', dpi=150)
    plt.close()
 
    # -----------------------------------------------------------------------
    # 4) Compute M(t) with the "Case 1" parameters, plot M(t)/delta
    # -----------------------------------------------------------------------
    M_t_case1 = get_M(alpha_c1, phi_o_c1, r_ratio_c1,
                      KsMr_c1, KfMr_c1, KlMr, MmMr,
                      delta_c1, TdTr_c1, tinj_c1,
                      t_case1, timescale_c1)
 
    plt.figure()
    plt.plot(t_days_array, M_t_case1/delta_c1, label='M(t)/delta - Case 1', linewidth=2)
    plt.xlabel('Time [days]')
    plt.ylabel('Magma transport M / δ (dimensionless)')
    plt.title('Case 1: Magma Transport (0-10 yrs)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/magma_transport_case1.png', dpi=150)
    plt.close()
 
    # -----------------------------------------------------------------------
    # 5) Compute dimensionless surface displacement (get_surface)
    #    and convert to *millimeters* for plotting
    # -----------------------------------------------------------------------
    surface_z_dimless, surface_rho_c1 = get_surface(
        rho=0.0,
        d=d_c1,
        alpha=alpha_c1,
        phi_o=phi_o_c1,
        r_ratio=r_ratio_c1,
        KsMr=KsMr_c1,
        KfMr=KfMr_c1,
        KlMr=KlMr,
        MmMr=MmMr,
        delta=delta_c1,
        TdTr=TdTr_c1,
        tinj=tinj_c1,
        t=t_case1,
        timescale=timescale_c1
    )
 
    # Assume R = 3000 m => 3000 * 1000 mm = 3,000,000 mm
    # dimensionless displacement * 3000 m => (in mm) multiply by 3,000,000
    surface_z_mm = surface_z_dimless * 3000.0 * 1000.0
 
    # Plot ground elevation (in mm) vs. time in days
    plt.figure(figsize=(8, 5))
    plt.plot(t_days_array, surface_z_mm, linewidth=2, label='Surface displacement')
    plt.xlabel('Time [days]')
    plt.ylabel('Displacement [mm]')
    plt.title('Case 1: Surface Uplift + Subsidence (0-10 yrs)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/ground_elevation_case1.png', dpi=150)
    plt.close()
 
    # -----------------------------------------------------------------------
    # 6) Compute displacement for r in [0.5, 1.0], using "Case 1" params
    #    Convert final time displacement to mm for a radial plot
    # -----------------------------------------------------------------------
    r_array = np.linspace(0.5, 1.0, 25)
    u_case1_dimless, r_out_case1, t_out_case1 = get_u(
        alpha_c1, phi_o_c1, r_ratio_c1,
        KsMr_c1, KfMr_c1, KlMr, MmMr,
        delta_c1, TdTr_c1, tinj_c1,
        t_case1, timescale_c1,
        r_array
    )
 
    # Convert dimensionless displacement to mm
    # if dimensionless displacement = 1 => 3000 m => 3,000,000 mm
    u_case1_mm = u_case1_dimless * 3000.0 * 1000.0
 
    # Plot displacement vs. r at the final time index ( -1 )
    plt.figure()
    plt.plot(r_out_case1, u_case1_mm[:, -1], 'o-', label='u(r) at final time (in mm)')
    plt.xlabel('r / R (dimensionless)')
    plt.ylabel('Displacement [mm]')
    plt.title('Case 1: Displacement vs. Radius (final time, ~10 yrs)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/displacement_case1.png', dpi=150)
    plt.close()
 
    # Save displacement results to .mat
    savemat('results/deformation_case1.mat', {
        'u_dimless': u_case1_dimless,
        'u_mm': u_case1_mm,
        'r_out': r_out_case1,
        't_out_days': t_days_array,
        't_out_dimless': t_out_case1
    })
 
    # -----------------------------------------------------------------------
    # 7) Compute pore pressure Pf(r,t), save to .mat, "Case 1" parameters
    #    (kept dimensionless)
    # -----------------------------------------------------------------------
    Pf_r_case1 = get_Pf(
        alpha_c1, phi_o_c1, r_ratio_c1,
        KsMr_c1, KfMr_c1, KlMr, MmMr,
        delta_c1, TdTr_c1, tinj_c1,
        t_case1, timescale_c1,
        r_array
    )
 
    # Plot Pf vs. r at the final time
    plt.figure()
    plt.plot(r_array, Pf_r_case1[:, -1], 's-', label='P_f(r) at final time')
    plt.xlabel('r / R (dimensionless)')
    plt.ylabel('Pore Pressure (dimensionless)')
    plt.title('Case 1: Pore Pressure vs. Radius (final time, ~10 yrs)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/pore_pressure_case1.png', dpi=150)
    plt.close()
 
    # Save pore pressure results
    savemat('results/pore_pressure_case1.mat', {
        'Pf_r': Pf_r_case1,
        'r': r_array,
        't_days': t_days_array,
        't_dimless': t_case1
    })
 
    print('All modified "Case 1" computations (10-year timescale, mm displacement) complete!')
 
if __name__ == '__main__':
    main()