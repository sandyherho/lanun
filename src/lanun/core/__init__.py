"""Core solver components for Lagrangian transport."""

from .velocity import BasinSystem, VelocityField
from .particles import ParticleSet, deploy_particles
from .solver import LagrangianSolver
from .diagnostics import (
    compute_mass_conservation,
    compute_area_perimeter,
    compute_circularity,
    compute_variance_decay,
    compute_center_of_mass,
    compute_reversibility_error,
    compute_all_diagnostics,
)

__all__ = [
    "BasinSystem",
    "VelocityField",
    "ParticleSet",
    "deploy_particles",
    "LagrangianSolver",
    "compute_mass_conservation",
    "compute_area_perimeter",
    "compute_circularity",
    "compute_variance_decay",
    "compute_center_of_mass",
    "compute_reversibility_error",
    "compute_all_diagnostics",
]
