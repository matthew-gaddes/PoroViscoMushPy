#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 15:20:30 2025

@author: matthew
"""

import numpy as np

def Laplace_Pf(s, alpha, phi_o, r_ratio,
               KsMr, KfMr, KlMr, MmMr,
               delta, TdTr, tinj, timescale, r):
    """
    Laplace_Pf(s, alpha, phi_o, r_ratio, KsMr, KfMr, KlMr, MmMr,
               delta, TdTr, tinj, timescale, r)
    
    Computes the Laplace-domain solution for pore pressure Pf(r),
    normalized by Mr (the crust rigidity), at radial position r.

    Corresponds to the MATLAB function:
        Laplace_Pf.m
    used by get_Pf.m (time-domain code).

    Parameters
    ----------
    s : float
        Laplace variable.
    alpha : float
    phi_o : float
    r_ratio : float
        Ratio (core volume) / (total volume) = (r^3 / R^3).
    KsMr, KfMr, KlMr, MmMr : float
        Ratios of stiffness/rigidity parameters.
        (e.g., mush rigidity, crust rigidity, fluid/bulk moduli, etc.)
    delta : float
        Ratio of injected volume to pre-injection volume (core magma).
    TdTr : float or np.inf
        Ratio between diffusion time and relaxation time.
        - 0   => poroelastic end member
        - ∞   => viscoelastic end member
    tinj : float
        Injection time (normalized).
        If tinj=0 => instantaneous injection
        else => gradual injection over tinj.
    timescale : int
        - 0 => relaxation timescale
        - 1 => diffusion timescale
    r : float
        Radial position (normalized by chamber radius R). 0 < r <= 1

    Returns
    -------
    Pf_r_s : float
        Laplace transform of the pore pressure at radius r, 
        normalized by Mr (the crust rigidity).

    Notes
    -----
    1) If timescale=0, then s1=-1 and R=TdTr.
       If timescale=1, then s1=-TdTr and R=1.
    2) If tinj=0 => f4=KlMr*delta / s  (instantaneous injection)
       else => f4=KlMr*delta*(1 - exp(-s*tinj)) / (s^2 * tinj) (gradual).
    3) If TdTr=∞ => viscoelastic end member => sets m_r_s=0 and Int_s=0.
    4) The final Pf_r_s is built via a combination of hyperbolic terms (sinh, cosh)
       and partial sums J1, J2, J3 from the symbolic derivation.
    5) Non-vectorized by default. For arrays, wrap or vectorize as needed.
    """

    # --- 1) Basic definitions ---
    KfKs = KfMr / KsMr
    KuMr = (1 - alpha)*KsMr + alpha**2 * KfKs * KsMr / (phi_o + (alpha - phi_o)*KfKs)
    KmMr = (1 - alpha)*KsMr

    # --- 2) Determine s1, R based on timescale ---
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

    # --- 3) Intermediate variables ---
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
    s7 = D1 * s9 / (C1 + D1)
    s8 = ((A2 - D2)*E0*s4 + s2*3.0*KlMr*r_ratio) / ((A2 - D2)*E0 + 3.0*KlMr*r_ratio)

    A1 = (KuMr - KmMr)*(KmMr + (4.0/3.0)*MmMr) / (alpha**2*(KuMr + (4.0/3.0)*MmMr))
    B1 = -(KuMr - KmMr)*(r_ratio**(2.0/3.0)) / (alpha*(KuMr + (4.0/3.0)*MmMr))
    B2 = -(r_ratio**(2.0/3.0) - r_ratio**(-1.0/3.0))*(1.0/3.0)/(KuMr + (4.0/3.0)*MmMr)

    F2 = (A2 - D2)*E0 + 3.0*KlMr*r_ratio
    F3 = -D1

    # --- 4) Injection function f4 ---
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

    # The factor A => s10
    A = (
        4.0*KlMr*(r_ratio - 1.0) 
        + KuMr*(4.0 + 3.0*KlMr*r_ratio)
    ) / (
        4.0*KlMr*(r_ratio - 1.0)
        + KuMr*(4.0 + 3.0*KlMr*r_ratio)
        + 4.0*MmMr*(KlMr + KuMr*(r_ratio - 1.0) + 4.0*r_ratio/3.0)
    )
    s10 = A * s1

    # --- 5) So, J1, J2, J3 ---
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

    # --- 6) Int_s and hyperbolic terms ---
    # up_M, down_M
    up_M = ((So*ro - 1.0)*np.sinh(np.sqrt(So)*(1.0 - ro))
            + np.sqrt(So)*(1.0 - ro)*np.cosh(np.sqrt(So)*(1.0 - ro)))
    down_M = (np.sqrt(So)*J2*np.cosh(np.sqrt(So)*(1.0 - ro))
              + J3*np.sinh(np.sqrt(So)*(1.0 - ro)))

    Int_s = 0.0
    if not np.isclose(down_M, 0.0):
        Int_s = J1 * up_M / down_M

    # The code next defines up, down for "m_r_s":
    up = (np.sqrt(So)*np.cosh(np.sqrt(So)*(1.0 - r))
          - np.sinh(np.sqrt(So)*(1.0 - r)))
    down = (np.sqrt(So)*J2*np.cosh(np.sqrt(So)*(1.0 - ro))
            + J3*np.sinh(np.sqrt(So)*(1.0 - ro))

            # The original code reuses J2, J3 with ro in the hyperbolic terms.
            # So "down" remains the same as down_M, but we'll keep it named "down"
            # for clarity.
    )

    # m_r_s = So*J1*up / (r*down)
    # If TdTr=∞ => viscoelastic => m_r_s=0, Int_s=0
    if TdTr == float('inf'):
        m_r_s = 0.0
        Int_s = 0.0
    else:
        if np.isclose(down, 0.0):
            m_r_s = 0.0
        else:
            m_r_s = So * J1 * up / (r * down)

    # --- 7) Pf_r_s expression ---
    Pf_r_s = (
        A1*m_r_s*(s - s3)/(s - s2)
        - A2*f4*E0*(s - s4)/((Ga + Gb)*(s - s10))
        - A2*G1*E0*(s - s7)*(s - s4)*Int_s/((Ga + Gb)*(s - s10)*(s - s2))
    )

    return Pf_r_s
