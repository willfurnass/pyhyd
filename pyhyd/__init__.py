from __future__ import absolute_import

from .pyhyd import x_sec_area, dyn_visc, reynolds, friction_factor, hyd_grad, shear_stress, settling_velocity, turnover_time, bed_shear_velocity, g, flow_from_shear, flow_unit_conv, hyd_grad_hw, hw_C_to_cw_k_s
from . import version

import sys
import os
import nose


def run_nose(verbose=False):
    nose_argv = sys.argv
    nose_argv += ['--detailed-errors', '--exe']
    if verbose:
        nose_argv.append('-v')
    initial_dir = os.getcwd()
    my_package_file = os.path.abspath(__file__)
    my_package_dir = os.path.dirname(my_package_file)
    os.chdir(my_package_dir)
    try:
        nose.run(argv=nose_argv)
    finally:
        os.chdir(initial_dir)
