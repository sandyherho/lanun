"""
lanun: 2D Lagrangian Particle Transport for Idealized Ocean Basins

A Numba-accelerated Python library for simulating two-dimensional Lagrangian
particle transport in semi-enclosed ocean basins using Bell's incompressible flow.

The velocity field represents wind-driven recirculation in a closed basin:

    u = -(U₀/2) sin²(πx/L) sin(2πy/L)
    v =  (U₀/2) sin²(πy/L) sin(2πx/L)

This flow is analytically incompressible (∇·v = 0) with zero velocity at all
domain boundaries, making it ideal for studying chaotic advection and stirring.

Features:
    - Numba JIT compilation for fast particle advection
    - Parallel execution with prange
    - Bilinear interpolation (partition of unity)
    - Midpoint (RK2) time integration
    - CF-compliant NetCDF output
    - Comprehensive Lagrangian diagnostics

Example:
    >>> from lanun import BasinSystem, LagrangianSolver
    >>> basin = BasinSystem(Lx=50e3, Ly=50e3, U0=0.3)
    >>> solver = LagrangianSolver(nx=101, ny=101)
    >>> result = solver.solve(basin, total_time=7*86400, dt=300)

Note:
    This is a 2D idealized model using a prescribed velocity field.
    It does not solve the primitive equations.

Authors:
    Sandy H. S. Herho <sandy.herho@ronininstitute.org>
    Faruq Khadami <fkhadami@itb.ac.id>
    Iwan P. Anwar <iwanpanwar@itb.ac.id>
    Dasapta E. Irawan <dasaptaerwin@itb.ac.id>

License: MIT
"""

__version__ = "0.0.2"
__author__ = "Sandy H. S. Herho, Faruq Khadami, Iwan P. Anwar, Dasapta E. Irawan"
__email__ = "sandy.herho@ronininstitute.org"
__license__ = "MIT"

from .core.velocity import BasinSystem, VelocityField
from .core.particles import ParticleSet, deploy_particles
from .core.solver import LagrangianSolver
from .core.diagnostics import (
    compute_mass_conservation,
    compute_area_perimeter,
    compute_circularity,
    compute_variance_decay,
    compute_center_of_mass,
    compute_reversibility_error,
    compute_all_diagnostics,
)
from .io.config_manager import ConfigManager
from .io.data_handler import DataHandler

__all__ = [
    # Core classes
    "BasinSystem",
    "VelocityField",
    "ParticleSet",
    "LagrangianSolver",
    # Particle functions
    "deploy_particles",
    # Diagnostics
    "compute_mass_conservation",
    "compute_area_perimeter",
    "compute_circularity",
    "compute_variance_decay",
    "compute_center_of_mass",
    "compute_reversibility_error",
    "compute_all_diagnostics",
    # IO
    "ConfigManager",
    "DataHandler",
]
