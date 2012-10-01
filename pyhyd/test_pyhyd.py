# Module to test:
import pyhyd as ph

import numpy as np
from numpy.testing import *

# TEST ph.x_sec_area_pos(D)
def test_x_sec_area_pos():
    assert_almost_equal(ph.x_sec_area(0.1), 0.00785398, decimal = 8)
def test_x_sec_area_zero():
    assert_raises(ValueError, ph.x_sec_area, 0)
def test_x_sec_area_neg():
    assert_raises(ValueError, ph.x_sec_area, -1.)
def test_x_sec_area_pos_vec():
    diams = np.array((0.1, 2))
    areas = np.array((0.00785398,  3.14159265))
    assert_array_almost_equal(ph.x_sec_area(diams), areas, decimal = 8)
def test_x_sec_area_bad_vec():
    diams = np.array((0.1, 0))
    assert_raises(ValueError, ph.x_sec_area, diams)
def test_x_sec_area_bad_vec2():
    diams = np.array((0.1, 0))
    assert_raises(ValueError, ph.x_sec_area, diams)

#TEST ph.dyn_visc(T=10.)
def test_dyn_visc_valid_scalar():
    pass
def test_dyn_visc_too_low_scalar():
    assert_raises(ValueError, ph.dyn_visc, 0)
def test_dyn_visc_too_high_scalar():
    assert_raises(ValueError, ph.dyn_visc, 100)
def test_dyn_visc_valid_vec():
    pass
def test_dyn_visc_invalid_vec():
    pass

#TEST ph.reynolds(D, Q, T = 10.0, den = 1000.0)
def test_reynolds():
    D = 0.02
    Q = 15 * np.pi * ((D/2.)**2)
    Re = 336912
    assert_almost_equal(int(ph.reynolds(D = D, Q = Q, T = 25, den = 1000)), Re)
# NEED MORE TESTS

#TEST friction_factor(D, Q, k_s, T = 10.0, den = 1000.0, warn=False)
def test_friction_factor():
    Q = 0.001
    D = 0.05
    k_s = 0.003
    f_epanet = 0.079
    f_ph = ph.friction_factor(D, Q, k_s, T = 10.0, den = 1000.0, warn=False)
    assert_almost_equal(f_ph, f_epanet, decimal = 3)
#NEED MORE TESTS

# TEST ph.hyd_grad(D, Q, k_s, T=10.0, den=1000.0)
def test_hyd_grad():
    Q = 0.001
    D = 0.05
    k_s = 0.003
    hyd_grad_ph = ph.hyd_grad(D, Q, k_s, T=10.0, den=1000.0)
    # NB with EPANET the 'unit headloss' is the headloss per 1000m if using SI units
    hyd_grad_epanet = 21.01 / 1000.
    assert_almost_equal(hyd_grad_ph, hyd_grad_epanet, decimal = 3)

#def shear_stress(D, Q, k_s, T = 10.0, den = 1000.0)
#def flow_from_shear(D, tau_a, k_s, T=10., den=1000.)
#def settling_velocity(den_part, D_part, T=10., den_fluid=1000.)
#TEST ph.turnover_time(D, Q, L)
def test_turnover_time():
    D = 0.1
    Q = 0.001
    L = 100.
    V = Q / (np.pi * (D/2.)**2)
    assert_equal(ph.turnover_time(D, Q, L), L / V)
#MORE TESTS NEEDED

#def bed_shear_velocity(D, Q, k_s, T = 10.0, den = 1000.0)
