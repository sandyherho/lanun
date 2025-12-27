"""
lanun: Numba-Accelerated Lagrangian Transport for Idealized Ocean Basins

A high-performance Python library for simulating Lagrangian particle transport
in semi-enclosed ocean basins using Bell's incompressible flow.

The velocity field represents wind-driven recirculation:
    u = -(U₀/2) sin²(πx/L) sin(2πy/L)
    v =  (U₀/2) sin²(πy/L) sin(2πx/L)

Features:
    - Numba JIT compilation for fast particle advection
    - Parallel execution with prange
    - Bilinear interpolation (partition of unity)
    - Midpoint (RK2) time integration
    - CF-compliant NetCDF output with oceanographic conventions
    - Comprehensive Lagrangian diagnostics
    - Beautiful dark-themed visualizations

Example:
    >>> from lanun import BasinSystem, LagrangianSolver
    >>> basin = BasinSystem(Lx=50e3, Ly=50e3, U0=0.3)
    >>> solver = LagrangianSolver(nx=101, ny=101)
    >>> result = solver.solve(basin, total_time=7*86400, dt=300)

Authors:
    Sandy H. S. Herho <sandy.herho@email.ucr.edu>
    Faiz R. Fajary <faizrohman@itb.ac.id>
    Iwan P. Anwar <iwanpanwar@itb.ac.id>
    Faruq Khadami <fkhadami@itb.ac.id>

License: MIT
"""

__version__ = "0.0.1"
__author__ = "Sandy H. S. Herho, Iwan P. Anwar, Faruq Khadami"
__email__ = "sandy.herho@email.ucr.edu"
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
