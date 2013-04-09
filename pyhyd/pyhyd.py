"""Functions pertaining to hydraulic analysis.

Note that SI units are used for all quantities 

"""

import numpy as np
import scipy.optimize as sp_opt
np.seterr(all='raise')

g = 9.81

def x_sec_area(D):
    """Cross-sectional area of a uniform pipe

    Keyword arguments:
    D -- internal diameter (m)
     
    """
    if np.any(D <=0):
        raise ValueError("Non-positive internal pipe diam.")
    return np.pi * np.power((0.5 * D), 2)

def dyn_visc(T=10.):
    """Dynamic viscosity of water (N.s.m^-2)

    Keyword arguments:
    T -- temperature (degC) (defaults to 10degC)

    See http://en.wikipedia.org/wiki/Viscosity#Viscosity_of_water for eq.

    """
    if np.any(T <= 0) or np.any(T >=100):
        raise ValueError("Cannot calc dynamic viscosity: temperature outside range [0,100].")
    A = 2.414*(10**-5) # Pa.s
    B = 247.8 # K 
    C = 140.0 # K
    return A * np.power(10, (B / (T + 273.15 - C)))

def reynolds(D, Q, T = 10.0, den = 1000.0):
    """Reynolds number
    
    Keyword arguments:
    D -- internal diameter (m)
    Q -- flow (m^3s^-1)
    T -- temperature; defaults to 10degC)
    den -- density defaults to 1000kg/m^3
     
    """
    if np.any(D <=0):
        raise ValueError("Non-positive internal pipe diam.")
    if np.any(den <=0):
        raise ValueError("Non-positive fluid density.")
    return ((np.abs(Q) / x_sec_area(D)) * D) / (dyn_visc(T) / (den+0.))

def _friction_factor(D, Q, k_s, T = 10.0, den = 1000.0, warn=False):
    # Helper function; see friction_factor
    if np.any(k_s < 0):
        raise ValueError("Negative pipe roughness.")
    Re = reynolds(D, Q, T, den)
    if Re == 0:
        f = 0
    elif Re < 2000:
        f = 64 / Re
    elif 2000 <= Re < 4000:
        y3 = -0.86859 * np.log((k_s / (3.7 * D)) + (5.74 / (4000**0.9)))
        y2 = (k_s / (3.7 * D)) + (5.74 / np.power(Re, 0.9))
        fa = np.power(y3, -2)
        fb = fa * (2 - (0.00514215 / (y2*y3)))
        r = Re / 2000.
        x4 = r * (0.032 - (3. * fa) + (0.5 * fb))
        x3 = -0.128 + (13. * fa) - (2.0 * fb)
        x2 =  0.128 - (17. * fa) + (2.5 * fb)
        x1 = (7 * fa) - fb
        f = x1 + r * (x2 + r * (x3 + x4))
    elif Re >= 4000:
        if warn:
            if k_s < 4e-4 or k_s > 5e-2:
                raise ValueError("Swamee Jain approx to Colebrook White not valid for turb flow as " + 
                                 "k_s={}m (outside range [0.004,0.05])".format(k_s))
            if Re > 1e7:
                raise ValueError("Swamee Jain approx to Colebrook White not valid for turb flow as " + 
                                 "Re={} (greater than 10,000,000)".format(Re))
        f = 0.25 / np.power((np.log10((k_s / (3.7 * D)) + (5.74 / np.power(Re, 0.9)))), 2)
    return f
_friction_factor = np.vectorize(_friction_factor)

def friction_factor(D, Q, k_s, T = 10.0, den = 1000.0, warn=False):
    """Darcy-Weisbach friction factor.
    
    Keyword arguments:
    D -- internal diameter (m)
    Q -- flow (m^3s^-1)
    k_s     -- roughness height (m)
    T -- temperature; defaults to 10degC)
    den -- density defaults to 1000kg/m^3
    warn -- warn if the Swamee Jain formula is inappropriate 
            due to k_s outside [0.004,0.05] or Re > 10e7

    Laminar flow:      Hagen-Poiseuille formula
    Transitional flow: cubic interpolation from Moody Diagram 
                       for transition region as per the EPANET2 manual 
                       (in turn taken from Dunlop (1991))
    Turbulent flow: Swamee-Jain approximation of implicit Colebrook-White equation
     
    """
    return _friction_factor(D, Q, k_s, T, den, warn)

def hyd_grad(D, Q, k_s, T=10.0, den=1000.0):
    """Headloss per unit length of pipe (in m).

    Keyword arguments:
    D -- internal diameter (m)
    Q -- flow (m^3s^-1)
    k_s -- roughness height (m)
    T -- temperature; defaults to 10degC)
    den -- density defaults to 1000kg/m^3

    """
    if np.any(den <= 0):
        raise ValueError("Non-positive fluid density.")
    if np.any(D <= 0):
        raise ValueError("Non-positive internal pipe diam.")
    f = friction_factor(D, Q, k_s, T, den)
    vel_sq = np.power((Q / x_sec_area(D)), 2)
    return (f * vel_sq) / (D * 2 * g)

def shear_stress(D, Q, k_s, T = 10.0, den = 1000.0):
    """Hydraulic shear stress at pipe wall (in Pa).

    Keyword arguments:
    D -- internal diameter (m)
    Q -- flow (m^3s^-1)
    k_s -- roughness height (m)
    T -- temperature; defaults to 10degC)
    den -- density defaults to 1000kg/m^3

    """
    if np.any(den <= 0):
        raise ValueError("Non-positive density")
    if np.any(D <= 0):
        raise ValueError("Non-positive internal pipe diam.")
    return den * g * (D / 4.0) * hyd_grad(D, Q, k_s, T, den)

def _flow_from_shear(D, tau_a, k_s, T, den):
    # Helper function required for numerically finding flow from shear stress
    # as np.vectorize cannot handle optional arguments
    return sp_opt.fminbound(
        lambda Q: np.absolute(shear_stress(D, Q, k_s, T, den) - tau_a), 
        x1 = 0e-10, 
        x2 = 100, 
        disp = 0)
_flow_from_shear_v = np.vectorize(_flow_from_shear)

def flow_from_shear(D, tau_a, k_s, T=10., den=1000.):
    """Numerically find pipe flow given shear stress.

    Keyword arguments:
    D -- internal diameter (m)
    tau_a -- shear stress (Pa)
    k_s -- roughness height (m)
    T -- temperature; defaults to 10degC)
    den -- density defaults to 1000kg/m^3

    """
    if np.any(den <= 0):
        raise ValueError("Non-positive density")
    if np.any(D <= 0):
        raise ValueError("Non-positive internal pipe diam.")
    if np.any(tau_a <= 0):
        raise ValueError("Non-positive shear stress.")
    return _flow_from_shear_v(D, tau_a, k_s, T, den)

def settling_velocity(den_part, D_part, T=10., den_fluid=1000.):
    """Settling velocity of a particle in a fluid (Stokes' Law)

    Keyword arguments:
    den_part -- density of the particle (kg m^-3) (sensible values 1000 to 1300)
    D_part -- particle diameter (m) (sensible values 1x10^-6 m to 250x10^-6 m) 
    T -- temperature (degC) (defaults to 10 degC)
    den_fluid -- density of the fluid (kg m^-3) (defaults to 1000)

    Assumptions:
     - The fluid is infinitely deep

    """
    if np.any(den_part <= 0):
        raise ValueError("Non-positive particle density.")
    if np.any(D_part <= 0):
        raise ValueError("Non-positive particle diameter.")
    if np.any(den_fluid <= 0):
        raise ValueError("Non-positive fluid density.")
    return (2 / 9.) * ((den_part - den_fluid) / dyn_visc(T)) * g * np.power((D_part/2.), 2)

def turnover_time(D, Q, L):
    """Time taken for fluid to traverse pipe at a given flow rate.

    Keyword arguments:
    D -- internal diameter (m)
    Q -- flow (m^3s^-1)
    L -- pipe length (m)

    """
    if np.any(L <= 0):
        raise ValueError("Non-positive pipe length.")

    return L / (np.abs(Q + 0.0) / x_sec_area(D))

def bed_shear_velocity(D, Q, k_s, T = 10.0, den = 1000.0):
    """Bed shear velocity (c.f. bed shear stress).
 
    Keyword arguments:
    D -- internal diameter (m)
    Q -- flow (m^3s^-1)
    k_s     -- roughness height (m)
    T -- temperature; defaults to 10degC)
    den -- density defaults to 1000kg/m^3

    """
    if np.any(D <= 0):
        raise ValueError("Non-positive pipe diam.")
    S_0 = hyd_grad(D, Q, k_s, T, den)
    return np.sqrt(g * (D / 4.0) * S_0)
