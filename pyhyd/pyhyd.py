"""Functions pertaining to hydraulic analysis.

Note that SI units are used for all quantities 

"""

import numpy as np
np.seterr(all='raise')

g = 9.81

def x_sec_area(D):
    """Cross-sectional area of a uniform pipe

    Keyword arguments:
    D -- internal diameter (m)
     
    """
    if D <=0:
        raise ValueError("Cannot find pipe x-sectional area as diameter is negative.")
    return np.pi * ((0.5 * D)**2)

def dyn_visc(T=10.):
    """Dynamic viscosity of water (N.s.m^-2)

    Keyword arguments:
    T -- temperature (degC) (defaults to 10degC)

    See http://en.wikipedia.org/wiki/Viscosity#Viscosity_of_water for eq.

    """
    if T <= 0 or T >=100:
        raise ValueError("Cannot calc dynamic viscosity: temperature outside range [0,100].")
    A = 2.414*(10**-5) # Pa.s
    B = 247.8 # K 
    C = 140.0 # K
    return A * 10**(B / (T + 273.15 - C))

def reynolds(D, Q, T = 10.0, den = 1000.0):
    """Reynolds number
    
    Keyword arguments:
    D -- internal diameter (m)
    Q -- flow (m^3s^-1)
    T -- temperature; defaults to 10degC)
    den -- density defaults to 1000kg/m^3
     
    """
    return ((Q / x_sec_area(D)) * D) / (dyn_visc(T) / (den+0.))

def friction_factor(D, Q, k_s, T = 10.0, den = 1000.0):
    """Darcy-Weisbach friction factor.
    
    Keyword arguments:
    D -- internal diameter (m)
    Q -- flow (m^3s^-1)
    k_s     -- roughness height (m)
    T -- temperature; defaults to 10degC)
    den -- density defaults to 1000kg/m^3
     
    """
    Re = reynolds(D, Q, T, den)
    if Re == 0:
        f = 0
    elif Re < 2000:
        # Hagen-Poiseuille formula for laminar flow
        f = 64 / Re
    elif 2000 <= Re < 4000:
        # Cubic interpolation from Moody Diagram for transition region
        # as per the EPANET2 manual (in turn taken from Dunlop (1991))
        y3 = -0.86859 * np.log((k_s / (3.7 * D)) + (5.74 / (4000**0.9)))
        y2 = (k_s / (3.7 * D)) + (5.74 / (Re**0.9))
        fa = y3**-2
        fb = fa * (2 - (0.00514215 / (y2*y3)))
        r = Re / 2000.
        x4 = r * (0.032 - (3. * fa) + (0.5 * fb))
        x3 = -0.128 + (13. * fa) - (2.0 * fb)
        x2 =  0.128 - (17. * fa) + (2.5 * fb)
        x1 = (7 * fa) - fb
        f = x1 + r * (x2 + r * (x3 + x4))
    elif Re >= 4000:
        if k_s < 4e-4 or k_s > 5e-2:
            raise ValueError("Swamee Jain approx to Colebrook White not valid for turb flow as " + 
                             "k_s={}m (outside range [0.004,0.05])".format(k_s))
        if k_s < 5e3 or Re > 1e7:
            raise ValueError("Swamee Jain approx to Colebrook White not valid for turb flow as " + 
                             "Re={} (outside range [5,000,10,000,000])".format(Re))
        if k_s < 0 or Re < 0:
            print k_s, Re
        f = 0.25 / (np.log10((k_s / (3.7 * D)) + (5.74 / (Re**0.9))))**2
    return f

def hyd_grad(D, Q, k_s, T=10.0, den=1000.0):
    """Headloss per unit length of pipe (in m).

    Keyword arguments:
    D -- internal diameter (m)
    Q -- flow (m^3s^-1)
    k_s     -- roughness height (m)
    T -- temperature; defaults to 10degC)
    den -- density defaults to 1000kg/m^3

    """
    if den <= 0:
        raise ValueError("Invalid density of {} kg/m^3".format(den))
    if D <= 0:
        raise ValueError("Invalid internal pipe diam of {} m".format(D))
    if Q < 0:
        raise ValueError("Invalid pipe flow of {} m^3/s".format(Q))
    f = friction_factor(D, Q, k_s, T, den)
    S_0 = (f / (D+0.)) * (((Q / x_sec_area(D))**2.0) / (2.0 * g))
    return S_0

def shear_stress(D, Q, k_s, T = 10.0, den = 1000.0):
    """Hydraulic shear stress at pipe wall (in Pa).

    Keyword arguments:
    D -- internal diameter (m)
    Q -- flow (m^3s^-1)
    k_s     -- roughness height (m)
    T -- temperature; defaults to 10degC)
    den -- density defaults to 1000kg/m^3

    """
    if den <= 0:
        raise ValueError("Invalid density of {} kg/m^3".format(den))
    if D <= 0:
        raise ValueError("Invalid internal pipe diam of {} m".format(D))
    return den * g * (D / 4.0) * hyd_grad(D, Q, k_s, T, den)

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
    if den_part <= 0:
        raise ValueError("Invalid particle density of {} kg/m^3".format(den_part))
    if D_part <= 0:
        raise ValueError("Invalid particle diameter of {} m".format(D_part))
    if den_fluid <= 0:
        raise ValueError("Invalid fluid density of {} kg/m^3".format(den_fluid))
    return (2 / 9.) * ((den_part - den_fluid) / dyn_visc(T)) * g * ((D_part/2.)**2)

def turnover_time(D, Q, L):
    """Time taken for fluid to traverse pipe at a given flow rate.

    Keyword arguments:
    D -- internal diameter (m)
    Q -- flow (m^3s^-1)
    L -- pipe length (m)

    """
    if Q <= 0:
        raise ValueError("Invalid pipe flow of {} m^3/s".format(Q))
    if L <= 0:
        raise ValueError("Invalid pipe length of {} m".format(L))

    return L / ((Q+0.) / x_sec_area(D))
turnover_time = np.vectorize(turnover_time)

def bed_shear_velocity(D, Q, k_s, T = 10.0, den = 1000.0):
    """Bed shear velocity (c.f. bed shear stress).
 
    Keyword arguments:
    D -- internal diameter (m)
    Q -- flow (m^3s^-1)
    k_s     -- roughness height (m)
    T -- temperature; defaults to 10degC)
    den -- density defaults to 1000kg/m^3

    """
    if D <= 0:
        raise ValueError("Invalid internal pipe diam of {} m".format(D))
    S_0 = hyd_grad(D, Q, k_s, T, den)
    return np.sqrt(g * (D / 4.0) * S_0)
