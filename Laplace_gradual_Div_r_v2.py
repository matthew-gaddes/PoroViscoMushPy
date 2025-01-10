#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:16:32 2025

@author: matthew
"""
import numpy as np

def Laplace_gradual_Div_r_v2(s, alpha, phi_o, r_ratio,
                             KsMr, KfMr, KlMr, MmMr,
                             delta, TdTr, tinj, timescale, r):
    """
    Laplace_gradual_Div_r_v2(s, alpha, phi_o, r_ratio, KsMr, KfMr, KlMr, MmMr,
                             delta, TdTr, tinj, timescale, r)

    Computes the Laplace-domain solution for the radial divergence (Div_r).
    This is part of a gradual injection model. The result is used in
    get_Div_r_gradual_v2.m to obtain the time-domain divergence via inverse
    Laplace transform.

    Parameters
    ----------
    s : float
        Laplace variable.
    alpha : float
    phi_o : float
    r_ratio : float
        Core-to-total volume ratio, (r^3/R^3).
    KsMr, KfMr, KlMr, MmMr : float
        Ratios of various stiffnesses/rigidities (mush rigidity, crust rigidity, etc.).
    delta : float
        Ratio of injected volume to pre-injection volume.
    TdTr : float or np.inf
        Ratio of diffusion time to relaxation time.
        - If TdTr = 0   => poroelastic end member
        - If TdTr = ∞   => viscoelastic end member
    tinj : float
        Injection time, normalized by whichever timescale is chosen.
    timescale : int
        0 => relaxation timescale
        1 => diffusion timescale
    r : float
        Radial position (normalized by chamber radius R).

    Returns
    -------
    Div_r_s : float
        Laplace transform of the radial divergence at radius `r`.

    Notes
    -----
    - If `tinj = 0`, this indicates instantaneous injection (delta function in time).
    - If `TdTr = np.inf`, the viscoelastic end member sets m_r_s = 0 and Int_s = 0.
    - The function is not vectorized by default. If calling with arrays of `s`,
      consider using numpy vectorization or looping externally.
    """

    # 1) Compute some intermediate ratios
    KfKs = KfMr / KsMr
    KuMr = (1 - alpha) * KsMr + alpha**2 * KfKs * KsMr / (phi_o + (alpha - phi_o) * KfKs)
    KmMr = (1 - alpha) * KsMr

    # 2) Determine s1, R based on timescale
    if timescale == 0:
        # Use relaxation timescale
        s1 = -1.0
        R  = TdTr
    elif timescale == 1:
        # Use diffusion timescale
        s1 = -TdTr
        R  = 1.0
    else:
        raise ValueError("timescale must be 0 (relaxation) or 1 (diffusion).")

    # 3) Define intermediate variables
    s2 = KuMr * s1 / (KuMr + (4.0/3.0)*MmMr)
    s3 = KmMr * s1 / (KmMr + (4.0/3.0)*MmMr)
    s4 = s1 / (1.0 - MmMr)

    A2 = -(KuMr - KmMr) / (alpha * (KuMr + (4.0/3.0)*MmMr))
    C1 =  4.0 * r_ratio * MmMr * (KuMr - KmMr) / (alpha * (KuMr + (4.0/3.0)*MmMr))
    C2 = (4.0/3.0) * (r_ratio - 1.0) * MmMr / (KuMr + (4.0/3.0)*MmMr)
    C3 = -4.0 * r_ratio * MmMr
    D1 = -3.0 * KlMr * r_ratio * (1.0 + A2)
    D2 =  KlMr * (r_ratio - 1.0) / (KuMr + (4.0/3.0)*MmMr)
    E0 =  4.0 * (MmMr - 1.0)

    s5 = 3.0 * KlMr * r_ratio * s1 / (3.0 * KlMr * r_ratio - C3)
    s6 = (s2 + D2*s1) / (C2 + D2 + 1.0)
    s9 = (s2 + A2*s1) / (1.0 + A2)

    s7 = D1 * s9 / (C1 + D1)
    s8 = ((A2 - D2) * E0 * s4 + s2 * 3.0 * KlMr * r_ratio) / ((A2 - D2) * E0 + 3.0 * KlMr * r_ratio)

    A1 = (KuMr - KmMr) * (KmMr + (4.0/3.0)*MmMr) / (alpha**2 * (KuMr + (4.0/3.0)*MmMr))
    B1 = -(KuMr - KmMr) * (r_ratio**(2.0/3.0)) / (alpha * (KuMr + (4.0/3.0)*MmMr))
    B2 = -(r_ratio**(2.0/3.0) - r_ratio**(-1.0/3.0)) * (1.0/3.0) / (KuMr + (4.0/3.0)*MmMr)

    F2 = (A2 - D2)*E0 + 3.0 * KlMr * r_ratio
    F3 = -D1

    # 4) Injection function f4
    #    If tinj=0 => instantaneous injection => f4 = KlMr*delta/s
    #    Otherwise => gradual injection => f4 = KlMr*delta(1-e^{-s*tinj}) / (s^2 * tinj)
    if tinj == 0:
        f4 = KlMr * delta / s
    else:
        f4 = KlMr * delta * (1.0 - np.exp(-s * tinj)) / (s**2 * tinj)

    G1 = C1 + D1
    Ga = E0 * (C2 + D2 + 1.0)
    Gb = C3 - 3.0 * KlMr * r_ratio
    ro = r_ratio**(-1.0/3.0)  # ro = r_o / R_o
    g2 = Ga + Gb

    # A is used for defining s10
    A = (4.0*KlMr*(r_ratio - 1.0) + KuMr*(4.0 + 3.0*KlMr*r_ratio)) / \
        (4.0*KlMr*(r_ratio - 1.0) + KuMr*(4.0 + 3.0*KlMr*r_ratio)
         + 4.0*MmMr*(KlMr + KuMr*(r_ratio - 1.0) + 4.0*r_ratio/3.0))

    s10 = A * s1

    # So = R*s*(s-s2)/(s-s3)
    So = R * s * (s - s2) / (s - s3)

    # 5) J1, J2, J3 expressions
    J1 = f4 * (Ga + Gb) * (s - s2) * (s - s10) + F2 * f4 * (s - s2) * (s - s8)

    J2 = ((A1*(Ga + Gb)/ro) * So * (s - s3)*(s - s10)
          + F3*(Ga + Gb)*(1.0 - ro)*(s - s9)*(s - s10)
          - F2*G1*(1.0 - ro)*(s - s7)*(s - s8))

    J3 = (F3*(Ga + Gb)*(s - s9)*(s - s10)*(So*ro - 1.0)
          - F2*G1*(s - s7)*(s - s8)*(So*ro - 1.0)
          - (A1*(Ga + Gb)/ro)*So*(s - s3)*(s - s10))

    # Hyperbolic terms for Int_s
    # up_M, down_M
    up_M = ((So*ro - 1.0)*np.sinh(np.sqrt(So)*(1.0 - ro))
            + np.sqrt(So)*(1.0 - ro)*np.cosh(np.sqrt(So)*(1.0 - ro)))

    down_M = (np.sqrt(So)*J2*np.cosh(np.sqrt(So)*(1.0 - ro))
              + J3*np.sinh(np.sqrt(So)*(1.0 - ro)))

    # Int_s = J1 * up_M / down_M
    # except if TdTr == Inf => set them = 0 (viscoelastic end member)
    if TdTr == float('inf'):
        m_r_s = 0.0
        Int_s = 0.0
    else:
        Int_s = 0.0 if np.isclose(down_M, 0.0) else (J1 * up_M / down_M)

        # 6) Additional hyperbolic terms for m_r_s
        #    up = sqrt(So)*cosh(sqrt(So)*(1-r)) - sinh(sqrt(So)*(1-r))
        #    down = same as down_M but with ro in the cosh(...) replaced by r
        #    Actually, from the original code: 
        #      up = sqrt(So)*cosh(...) - sinh(...)
        #      down = sqrt(So)*J2*cosh(...) + J3*sinh(...), at ro replaced by r
        #    However, the code reuses down with ro => that’s correct for the boundary ro,
        #    then does the ratio for the actual r. The original code just does:
        #
        #      up = sqrt(So)*cosh(sqrt(So)*(1-r)) - sinh(sqrt(So)*(1-r));
        #      m_r_s = So*J1*up/(r*down);
        #
        # We'll do that literally:

        up = (np.sqrt(So)*np.cosh(np.sqrt(So)*(1.0 - r))
              - np.sinh(np.sqrt(So)*(1.0 - r)))

        down = down_M  # Because the code reuses the same 'down_M'

        m_r_s = 0.0
        if not np.isclose(down, 0.0):
            m_r_s = So * J1 * up / (r * down)

    # 7) Pf_r_s
    Pf_r_s = (A1*m_r_s*(s - s3)/(s - s2)
              - A2*f4*E0*(s - s4)/((Ga + Gb)*(s - s10))
              - A2*G1*E0*(s - s7)*(s - s4)*Int_s / ((Ga + Gb)*(s - s10)*(s - s2)))

    # 8) Final divergence
    Div_r_s = m_r_s / alpha - alpha*Pf_r_s / (KuMr - KmMr)

    return Div_r_s
