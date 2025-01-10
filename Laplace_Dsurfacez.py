#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:13:42 2025

@author: matthew
"""

import numpy as np

def Laplace_Dsurfacez(s, rho, d, alpha, phi_o, r_ratio,
                      KsMr, KfMr, KlMr, MmMr,
                      delta, TdTr, tinj, timescale):
    """
    Laplace_Dsurfacez(s, rho, d, alpha, phi_o, r_ratio,
                      KsMr, KfMr, KlMr, MmMr,
                      delta, TdTr, tinj, timescale)

    Computes the Laplace-domain expression for the time derivative of 
    the vertical surface displacement (proxy for strain rate), assuming:
    
      - Poisson's ratio for the crust = 0.25
      - rho, d, and displacements normalized by R0
      - 's' is the Laplace variable
      - tinj is injection time, normalized by the chosen time scale
      - timescale = 0 => relaxation time scale
      - timescale = 1 => diffusion time scale
      - TdTr is the ratio between diffusion and relaxation times
        (poroelastic end member => TdTr=0, viscoelastic => TdTr=∞)

    Returns
    -------
    Dsurface_z_s : float
        Laplace transform of the time derivative of vertical surface displacement.
    """

    # 1) Compute some intermediate ratios
    KfKs   = KfMr / KsMr
    KuMr   = (1 - alpha)*KsMr + alpha**2 * KfKs * KsMr / (phi_o + (alpha - phi_o)*KfKs)
    KmMr   = (1 - alpha)*KsMr

    # 2) Determine s1, R based on the timescale flag
    #    timescale=0 => use relaxation timescale
    #    timescale=1 => use diffusion timescale
    if timescale == 0:  # relaxation
        s1 = -1.0
        R  = TdTr
    elif timescale == 1:  # diffusion
        s1 = -TdTr
        R  = 1.0
    else:
        raise ValueError("timescale must be 0 (relaxation) or 1 (diffusion).")

    # 3) Additional intermediate variables
    s2 = KuMr * s1 / (KuMr + (4.0/3.0)*MmMr)
    s3 = KmMr * s1 / (KmMr + (4.0/3.0)*MmMr)
    s4 = s1 / (1.0 - MmMr)
    A2 = -(KuMr - KmMr) / (alpha*(KuMr + (4.0/3.0)*MmMr))
    C1 = 4.0*r_ratio*MmMr*(KuMr - KmMr) / (alpha*(KuMr + (4.0/3.0)*MmMr))
    C2 = (4.0/3.0)*(r_ratio - 1.0)*MmMr / (KuMr + (4.0/3.0)*MmMr)
    C3 = -4.0*r_ratio*MmMr
    D1 = -3.0*KlMr*r_ratio*(1.0 + A2)
    D2 = KlMr*(r_ratio - 1.0)/(KuMr + (4.0/3.0)*MmMr)
    E0 = 4.0*(MmMr - 1.0)
    s5 = 3.0*KlMr*r_ratio*s1 / (3.0*KlMr*r_ratio - C3)
    s6 = (s2 + D2*s1) / (C2 + D2 + 1.0)
    s9 = (s2 + A2*s1) / (1.0 + A2)
    s7 = D1*s9 / (C1 + D1)
    s8 = ((A2 - D2)*E0*s4 + s2*3.0*KlMr*r_ratio) / ((A2 - D2)*E0 + 3.0*KlMr*r_ratio)

    A1 = (KuMr - KmMr)*(KmMr + (4.0/3.0)*MmMr) / (alpha**2*(KuMr + (4.0/3.0)*MmMr))
    B1 = -(KuMr - KmMr)*(r_ratio**(2.0/3.0)) / (alpha*(KuMr + (4.0/3.0)*MmMr))
    B2 = -(r_ratio**(2.0/3.0) - r_ratio**(-1.0/3.0))*(1.0/3.0) / (KuMr + (4.0/3.0)*MmMr)
    F2 = (A2 - D2)*E0 + 3.0*KlMr*r_ratio
    F3 = -D1

    # 4) f4 term depends on whether tinj=0 or not
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

    A  = (4.0*KlMr*(r_ratio - 1.0) + KuMr*(4.0 + 3.0*KlMr*r_ratio)) / \
         (4.0*KlMr*(r_ratio - 1.0) + KuMr*(4.0 + 3.0*KlMr*r_ratio) + 4.0*MmMr*(KlMr + KuMr*(r_ratio - 1.0) + 4.0*r_ratio/3.0))

    s10 = A * s1

    So = R*s*(s - s2) / (s - s3)

    # 5) J1, J2, J3 expressions
    J1 = f4*(Ga + Gb)*(s - s2)*(s - s10) + F2*f4*(s - s2)*(s - s8)

    J2 = (A1*(Ga + Gb)/ro)*So*(s - s3)*(s - s10) \
         + F3*(Ga + Gb)*(1.0 - ro)*(s - s9)*(s - s10) \
         - F2*G1*(1.0 - ro)*(s - s7)*(s - s8)

    J3 = F3*(Ga + Gb)*(s - s9)*(s - s10)*(So*ro - 1.0) \
         - F2*G1*(s - s7)*(s - s8)*(So*ro - 1.0) \
         - (A1*(Ga + Gb)/ro)*So*(s - s3)*(s - s10)

    # 6) up_M and down_M for Int_s
    up_M   = (So*ro - 1.0)*np.sinh(np.sqrt(So)*(1.0 - ro)) \
             + np.sqrt(So)*(1.0 - ro)*np.cosh(np.sqrt(So)*(1.0 - ro))

    down_M = np.sqrt(So)*J2*np.cosh(np.sqrt(So)*(1.0 - ro)) \
             + J3*np.sinh(np.sqrt(So)*(1.0 - ro))

    # Int_s is zero if TdTr==∞ (viscoelastic end member)
    if TdTr == float('inf'):
        Int_s = 0.0
    else:
        # Avoid division by zero: if down_M ~ 0, you may need special handling
        Int_s = 0.0 if np.isclose(down_M, 0.0) else (J1 * up_M / down_M)

    # 7) Final expression for u1_s
    u1_s = -f4*(s - s2)/((Ga + Gb)*(s - s10)) \
           - G1*(s - s7)*Int_s/((Ga + Gb)*(s - s10))

    # 8) Convert u1_s into derivative of vertical surface displacement
    #    Dsurface_z_s = s*d*0.75*4 * u1_s * (d^2 + rho^2)^(-3/2)
    #    Note: 0.75 * 4 = 3, so the factor is s*d*3.
    Dsurface_z_s = s * d * 0.75 * 4.0 * u1_s * (d**2 + rho**2)**(-1.5)

    return Dsurface_z_s
