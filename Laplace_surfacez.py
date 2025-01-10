#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 15:25:50 2025

@author: matthew
"""

import numpy as np

def Laplace_surfacez(s, rho, d, alpha, phi_o, r_ratio,
                     KsMr, KfMr, KlMr, MmMr,
                     delta, TdTr, tinj, timescale):
    """
    Laplace_surfacez(
        s, rho, d, alpha, phi_o, r_ratio,
        KsMr, KfMr, KlMr, MmMr,
        delta, TdTr, tinj, timescale
    )

    Computes the Laplace-domain expression for the vertical (z) component
    of surface deformation, normalized by R0 (the characteristic chamber radius).

    Corresponds to the MATLAB function:
        Laplace_surfacez.m
    which is used by get_surface(...).m for the time-domain solution.

    Parameters
    ----------
    s : float
        Laplace variable.
    rho : float
        Horizontal distance from the chamber axis (normalized by R0).
    d : float
        Depth of the chamber center (normalized by R0).
    alpha : float
    phi_o : float
    r_ratio : float
        Ratio of core volume to total volume, i.e., (r^3 / R^3).
    KsMr, KfMr, KlMr, MmMr : float
        Ratios of stiffness/rigidity parameters (crust, fluid, mush, etc.).
    delta : float
        Ratio of injected volume to pre-injection volume (core magma).
    TdTr : float or np.inf
        Ratio of diffusion time to relaxation time.
          - 0   => poroelastic end member
          - ∞   => viscoelastic end member
    tinj : float
        Injection time, normalized by the chosen timescale (0 => instantaneous).
    timescale : int
        0 => relaxation timescale (s1 = -1, R = TdTr)
        1 => diffusion timescale  (s1 = -TdTr, R = 1)

    Returns
    -------
    surface_z_s : float
        Laplace transform of the vertical surface displacement,
        normalized by R0.

    Notes
    -----
    1) If timescale=0 => s1=-1,  R=TdTr
       If timescale=1 => s1=-TdTr, R=1
    2) If tinj=0 => instantaneous injection => f4=KlMr*delta/s
       else => gradual injection => f4=KlMr*delta*(1 - exp(-s*tinj)) / (s^2 * tinj)
    3) If TdTr=∞ => set Int_s=0 (viscoelastic end member).
    4) The final expression is
         surface_z_s = d * 0.75 * 4 * u1_s * (d^2 + rho^2)^(-3/2),
       which can be simplified since 0.75*4 = 3 => 
         surface_z_s = 3 * d * u1_s * (d^2 + rho^2)^(-3/2).
    5) This function is not vectorized by default. For multiple s or rho values,
       consider looping or np.vectorize.
    """

    # 1) Compute intermediate stiffness parameters
    KfKs = KfMr / KsMr
    KuMr = (1 - alpha)*KsMr + alpha**2 * KfKs * KsMr / (phi_o + (alpha - phi_o)*KfKs)
    KmMr = (1 - alpha)*KsMr

    # 2) Determine s1, R based on timescale
    if timescale == 0:
        # Relaxation timescale
        s1 = -1.0
        R  = TdTr
    elif timescale == 1:
        # Diffusion timescale
        s1 = -TdTr
        R  = 1.0
    else:
        raise ValueError("timescale must be 0 or 1.")

    # 3) Additional intermediate variables
    s2 = KuMr * s1 / (KuMr + (4.0/3.0)*MmMr)
    s3 = KmMr * s1 / (KmMr + (4.0/3.0)*MmMr)
    s4 = s1 / (1.0 - MmMr)

    A2 = -(KuMr - KmMr) / (alpha * (KuMr + (4.0/3.0)*MmMr))
    C1 = 4.0 * r_ratio * MmMr * (KuMr - KmMr) / (alpha*(KuMr + (4.0/3.0)*MmMr))
    C2 = (4.0/3.0)*(r_ratio - 1.0)*MmMr / (KuMr + (4.0/3.0)*MmMr)
    C3 = -4.0 * r_ratio * MmMr
    D1 = -3.0 * KlMr * r_ratio * (1.0 + A2)
    D2 =  KlMr * (r_ratio - 1.0) / (KuMr + (4.0/3.0)*MmMr)
    E0 =  4.0 * (MmMr - 1.0)

    s5 = 3.0 * KlMr * r_ratio * s1 / (3.0*KlMr*r_ratio - C3)
    s6 = (s2 + D2*s1) / (C2 + D2 + 1.0)
    s9 = (s2 + A2*s1) / (1.0 + A2)
    s7 = D1*s9 / (C1 + D1)
    s8 = ((A2 - D2)*E0*s4 + s2*3.0*KlMr*r_ratio) / ((A2 - D2)*E0 + 3.0*KlMr*r_ratio)

    A1 = (KuMr - KmMr)*(KmMr + (4.0/3.0)*MmMr) / (alpha**2*(KuMr + (4.0/3.0)*MmMr))
    B1 = -(KuMr - KmMr)*(r_ratio**(2.0/3.0)) / (alpha*(KuMr + (4.0/3.0)*MmMr))
    B2 = -(r_ratio**(2.0/3.0) - r_ratio**(-1.0/3.0))*(1.0/3.0)/(KuMr + (4.0/3.0)*MmMr)

    F2 = (A2 - D2)*E0 + 3.0*KlMr*r_ratio
    F3 = -D1

    # 4) Injection function f4
    if tinj == 0:
        # Instantaneous injection
        f4 = KlMr * delta / s
    else:
        # Gradual injection
        f4 = KlMr * delta * (1.0 - np.exp(-s*tinj)) / (s**2 * tinj)

    G1 = C1 + D1
    Ga = E0*(C2 + D2 + 1.0)
    Gb = C3 - 3.0*KlMr*r_ratio
    ro = r_ratio**(-1.0/3.0)
    g2 = Ga + Gb

    A = (
        4.0*KlMr*(r_ratio - 1.0)
        + KuMr*(4.0 + 3.0*KlMr*r_ratio)
    ) / (
        4.0*KlMr*(r_ratio - 1.0)
        + KuMr*(4.0 + 3.0*KlMr*r_ratio)
        + 4.0*MmMr*(KlMr + KuMr*(r_ratio - 1.0) + 4.0*r_ratio/3.0)
    )
    s10 = A * s1

    # 5) Define J1, J2, J3
    So = R * s * (s - s2) / (s - s3)

    J1 = (
        f4*(Ga + Gb)*(s - s2)*(s - s10)
        + F2*f4*(s - s2)*(s - s8)
    )
    J2 = (
        (A1*(Ga + Gb)/ro)*So*(s - s3)*(s - s10)
        + F3*(Ga + Gb)*(1.0 - ro)*(s - s9)*(s - s10)
        - F2*G1*(1.0 - ro)*(s - s7)*(s - s8)
    )
    J3 = (
        F3*(Ga + Gb)*(s - s9)*(s - s10)*(So*ro - 1.0)
        - F2*G1*(s - s7)*(s - s8)*(So*ro - 1.0)
        - (A1*(Ga + Gb)/ro)*So*(s - s3)*(s - s10)
    )

    # 6) Hyperbolic term => Int_s
    up_M = (
        (So*ro - 1.0)*np.sinh(np.sqrt(So)*(1.0 - ro))
        + np.sqrt(So)*(1.0 - ro)*np.cosh(np.sqrt(So)*(1.0 - ro))
    )
    down_M = (
        np.sqrt(So)*J2*np.cosh(np.sqrt(So)*(1.0 - ro))
        + J3*np.sinh(np.sqrt(So)*(1.0 - ro))
    )

    if TdTr == float('inf'):
        Int_s = 0.0
    else:
        if np.isclose(down_M, 0.0):
            Int_s = 0.0
        else:
            Int_s = J1 * up_M / down_M

    # 7) Boundary displacement at r=1 => u1_s
    u1_s = (
        -f4*(s - s2)/((Ga + Gb)*(s - s10))
        - G1*(s - s7)*Int_s/((Ga + Gb)*(s - s10))
    )

    # 8) Final vertical surface displacement in Laplace domain
    #    surface_z_s = d * 0.75 * 4 * u1_s * (d^2 + rho^2)^(-3/2)
    #    0.75 * 4 = 3, so it is effectively 3*d*u1_s*(d^2 + rho^2)^(-3/2).
    surface_z_s = d * 0.75 * 4.0 * u1_s * (d**2 + rho**2)**(-1.5)

    return surface_z_s
