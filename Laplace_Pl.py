#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 15:20:57 2025

@author: matthew
"""

import numpy as np

def Laplace_Pl(s, alpha, phi_o, r_ratio,
               KsMr, KfMr, KlMr, MmMr,
               delta, TdTr, tinj, timescale):
    """
    Laplace_Pl(s, alpha, phi_o, r_ratio, KsMr, KfMr, KlMr, MmMr,
               delta, TdTr, tinj, timescale)

    Computes the Laplace-domain solution for the core fluid pressure (Pl),
    normalized by Mr (the crust rigidity).

    This corresponds to the MATLAB function:
        Laplace_Pl.m
    which is used in get_Pl(...).m for the time-domain solution.

    Parameters
    ----------
    s : float
        Laplace variable.
    alpha : float
    phi_o : float
    r_ratio : float
        Ratio of core volume to total volume, i.e. (r^3 / R^3).
    KsMr, KfMr, KlMr, MmMr : float
        Ratios of stiffness/rigidity parameters.
    delta : float
        Ratio of injected volume to pre-injection volume (core magma).
    TdTr : float or np.inf
        Ratio between diffusion and relaxation times:
          - 0   => poroelastic end member
          - ∞   => viscoelastic end member
    tinj : float
        Injection time, normalized by the chosen characteristic timescale.
        (tinj=0 => instantaneous injection)
    timescale : int
        0 => relaxation timescale
        1 => diffusion timescale

    Returns
    -------
    Pl_s : float
        Laplace transform of the core fluid pressure, normalized by Mr.
    
    Notes
    -----
    1. If timescale=0 => s1 = -1,  R = TdTr
       If timescale=1 => s1 = -TdTr, R = 1
    2. If tinj=0 => instantaneous injection:    f4 = KlMr*delta/s
       else => gradual injection over tinj:    f4 = KlMr*delta*(1 - exp(-s*tinj)) / (s^2 * tinj)
    3. If TdTr=∞ => viscoelastic end member => sets Int_s=0
    4. The final expression for Pl_s is built from:
         f4  - KlMr*(3*r_ratio*Int_s + 3*Ro*u0_s)
       where u1_s and u0_s are intermediate solutions for displacement, and
       Int_s is a hyperbolic function ratio from the symbolic derivation.
    5. This function is written for scalar s. For arrays, loop or vectorize.
    """

    # 1) Basic parameter definitions
    KfKs = KfMr / KsMr
    KuMr = (1 - alpha)*KsMr + alpha**2 * KfKs * KsMr / (phi_o + (alpha - phi_o)*KfKs)
    KmMr = (1 - alpha) * KsMr

    # 2) Determine s1, R from timescale
    if timescale == 0:
        # Relaxation timescale
        s1 = -1.0
        R  = TdTr
    elif timescale == 1:
        # Diffusion timescale
        s1 = -TdTr
        R  = 1.0
    else:
        raise ValueError("timescale must be 0 (relaxation) or 1 (diffusion).")

    # 3) Intermediate variables
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

    A1 = (KuMr - KmMr)*(KmMr + (4.0/3.0)*MmMr) / (alpha**2 * (KuMr + (4.0/3.0)*MmMr))
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
    Ro = r_ratio**(1.0/3.0)  # new variable from the code

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

    # 5) Define So, J1, J2, J3
    So = R*s*(s - s2) / (s - s3)

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

    # 6) Hyperbolic terms => Int_s
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

    # 7) Displacements u1_s, u0_s
    #    The code solves for the boundary displacement (u1_s) and a second term (u0_s).
    u1_s = (
        -f4*(s - s2) / ((Ga + Gb)*(s - s10))
        - G1*(s - s7)*Int_s / ((Ga + Gb)*(s - s10))
    )

    u0_s = (
        B1*(s - s1)*Int_s/(s - s2)
        + (B2*E0*(s - s4)/(s - s2) + Ro**2)*u1_s
    )

    # 8) Final expression for Pl_s
    #    Pl_s = f4 - KlMr*(3*r_ratio*Int_s + 3*Ro*u0_s)
    Pl_s = f4 - KlMr*(3.0*r_ratio*Int_s + 3.0*Ro*u0_s)

    return Pl_s
