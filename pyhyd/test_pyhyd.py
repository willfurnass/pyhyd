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

#def dyn_visc(T=10.)
#def reynolds(D, Q, T = 10.0, den = 1000.0)
#def friction_factor(D, Q, k_s, T = 10.0, den = 1000.0, warn=False)
#def hyd_grad(D, Q, k_s, T=10.0, den=1000.0)
#def shear_stress(D, Q, k_s, T = 10.0, den = 1000.0)
#def _flow_from_shear(D, tau_a, k_s, T, den)
#def flow_from_shear(D, tau_a, k_s, T=10., den=1000.)
#def settling_velocity(den_part, D_part, T=10., den_fluid=1000.)
#def turnover_time(D, Q, L)
#def bed_shear_velocity(D, Q, k_s, T = 10.0, den = 1000.0)
