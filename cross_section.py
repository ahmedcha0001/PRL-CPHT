import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mainn import *
from read_F import *

# Possile configurations
configurations = [('q', 'q', 'g'), ('g', 'q', 'q'), ('g', 'g', 'g')]


#Hard factors

def K_ag_cd_1(a, c, d, u, s, t, ubar, sbar, tbar):
    """Calculate the K^(1)_ag->cd kernel."""

    if a == 'q' and c == 'q' and d == 'g':
        # qg → qg
        return -(s**2 + u**2) / (2 * t**2) - (u**2 + s**2) / (N_c**2 * t**2)
    elif a == 'g' and c == 'q' and d == 'q':
        # gg → qq̅
        return (t**2 + u**2)**2 / (2 * N_c * s**2 * t * u)
    elif a == 'g' and c == 'g' and d == 'g':
        # gg → gg
        return 2 * N_c * (s**2 - t * u) * (t**2 + u**2) / (C_F * t**2 * u**2 * s**2)
    else:
        raise ValueError("Invalid parton types for K^(1)_ag->cd kernel")

def K_ag_cd_2(a, c, d, u, s, t):
    """Calculate the K^(2)_ag->cd kernel."""
    N_c = 3  # Number of colors in QCD
    C_F = 4/3  # Casimir operator for SU(3)

    if a == 'q' and c == 'q' and d == 'g':
        # qg → qg
        return -(C_F * s * (s**2 + u**2)) / (N_c * t**2 * u)
    elif a == 'g' and c == 'q' and d == 'q':
        # gg → qq̅
        return -t**2 / (2 * C_F * N_c**2 * s**2) + u**2 / (2 * C_F * N_c**2 * s**2)
    elif a == 'g' and c == 'g' and d == 'g':
        # gg → gg
        return 2 * N_c * (s**2 - t * u)**2 / (C_F * t * u * s**2)
    else:
        raise ValueError("Invalid parton types for K^(2)_ag->cd kernel")
    
#Phis
def Phi_ag_cd_1(a, c, d, x1, x2, kt, mu):
    """Calculate the Phi^(1)_ag->cd function."""
    # Hard factors
    if a == 'q' and c == 'q' and d == 'g':
        # qg → qg
        return F_qg_1
    elif a == 'g' and c == 'q' and d == 'q':
        # gg → qq̅
        return F_gg_1
    elif a == 'g' and c == 'g' and d == 'g':
        # gg → gg
        return 0.5*(F_gg_1+F_gg_6)
    else:
        raise ValueError("Invalid parton types for Phi^(1)_ag->cd function")
    
def Phi_ag_cd_2(a, c, d, x1, x2, kt, mu):
    """Calculate the Phi^(2)_ag->cd function."""
    # Hard factors
    if a == 'q' and c == 'q' and d == 'g':
        # qg → qg
        return F_qg_2
    elif a == 'g' and c == 'q' and d == 'q':
        # gg → qq̅
        return -N_c**2*F_gg_2
    elif a == 'g' and c == 'g' and d == 'g':
        # gg → gg
        return F_gg_2+F_gg_6
    else:
        raise ValueError("Invalid parton types for Phi^(2)_ag->cd function")

