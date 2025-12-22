"""
Lagrangian Diagnostics for Transport Quality Assessment.

Implements comprehensive metrics for evaluating:
    - Mass conservation
    - Area/volume preservation
    - Shape distortion (circularity)
    - Mixing efficiency (variance decay)
    - Reversibility error
    - Center of mass drift

All metrics are dimensionless where possible for cross-case comparison.
"""

import numpy as np
from numba import njit, prange
from typing import Dict, Any, Optional, Tuple

from .particles import ParticleSet


def compute_mass_conservation(
    tracer_initial: np.ndarray,
    tracer_final: np.ndarray,
    dx: float,
    dy: float,
    particles_per_cell: int
) -> Dict[str, float]:
    """
    Compute tracer mass conservation error.
    
    Mass = Σ C_p * A_p where A_p is effective area per particle.
    
    Args:
        tracer_initial: Initial tracer on particles
        tracer_final: Final tracer on particles
        dx, dy: Cell sizes [m]
        particles_per_cell: Particles per cell side
    
    Returns:
        Dictionary with mass metrics
    """
    area_per_particle = (dx * dy) / (particles_per_cell ** 2)
    
    mass_initial = np.sum(tracer_initial) * area_per_particle
    mass_final = np.sum(tracer_final) * area_per_particle
    
    if abs(mass_initial) > 1e-15:
        relative_error = (mass_final - mass_initial) / mass_initial
    else:
        relative_error = 0.0
    
    return {
        'mass_initial': float(mass_initial),
        'mass_final': float(mass_final),
        'mass_error_absolute': float(mass_final - mass_initial),
        'mass_error_relative': float(relative_error),
    }


@njit(cache=True)
def _shoelace_area(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute polygon area using shoelace formula.
    
    A = (1/2) |Σ (x_i * y_{i+1} - x_{i+1} * y_i)|
    """
    n = len(x)
    area = 0.0
    
    for i in range(n):
        j = (i + 1) % n
        area += x[i] * y[j]
        area -= x[j] * y[i]
    
    return abs(area) / 2.0


@njit(cache=True)
def _perimeter(x: np.ndarray, y: np.ndarray) -> float:
    """Compute polygon perimeter."""
    n = len(x)
    perim = 0.0
    
    for i in range(n):
        j = (i + 1) % n
        perim += np.sqrt((x[j] - x[i])**2 + (y[j] - y[i])**2)
    
    return perim


def compute_area_perimeter(
    boundary_x: np.ndarray,
    boundary_y: np.ndarray
) -> Dict[str, float]:
    """
    Compute area and perimeter of particle boundary.
    
    Args:
        boundary_x: x-coordinates of boundary particles (ordered)
        boundary_y: y-coordinates of boundary particles (ordered)
    
    Returns:
        Dictionary with area and perimeter
    """
    area = _shoelace_area(boundary_x, boundary_y)
    perimeter = _perimeter(boundary_x, boundary_y)
    
    return {
        'area': float(area),
        'perimeter': float(perimeter),
    }


def compute_circularity(area: float, perimeter: float) -> float:
    """
    Compute circularity (isoperimetric quotient).
    
    C = 4πA / P²
    
    C = 1 for a circle, C < 1 for stretched shapes.
    
    Args:
        area: Region area
        perimeter: Region perimeter
    
    Returns:
        Circularity value [0, 1]
    """
    if perimeter > 1e-15:
        return 4 * np.pi * area / (perimeter ** 2)
    return 0.0


def compute_variance_decay(
    tracer_initial: np.ndarray,
    tracer_current: np.ndarray
) -> Dict[str, float]:
    """
    Compute intensity of segregation (normalized variance).
    
    I_s = σ²(t) / σ²(0)
    
    Perfect mixing: I_s → 0
    No mixing: I_s = 1
    
    Args:
        tracer_initial: Initial tracer distribution
        tracer_current: Current tracer distribution
    
    Returns:
        Dictionary with variance metrics
    """
    var_initial = np.var(tracer_initial)
    var_current = np.var(tracer_current)
    
    if var_initial > 1e-15:
        intensity_segregation = var_current / var_initial
    else:
        intensity_segregation = 1.0
    
    return {
        'variance_initial': float(var_initial),
        'variance_current': float(var_current),
        'intensity_segregation': float(intensity_segregation),
        'mixing_efficiency': float(1.0 - intensity_segregation),
    }


def compute_center_of_mass(
    x: np.ndarray,
    y: np.ndarray,
    tracer: np.ndarray
) -> Dict[str, float]:
    """
    Compute tracer-weighted center of mass.
    
    x_cm = Σ(C_i * x_i) / Σ(C_i)
    
    Args:
        x, y: Particle positions
        tracer: Tracer values
    
    Returns:
        Dictionary with center of mass coordinates
    """
    total_tracer = np.sum(tracer)
    
    if total_tracer > 1e-15:
        x_cm = np.sum(tracer * x) / total_tracer
        y_cm = np.sum(tracer * y) / total_tracer
    else:
        x_cm = np.mean(x)
        y_cm = np.mean(y)
    
    return {
        'x_cm': float(x_cm),
        'y_cm': float(y_cm),
        'total_tracer': float(total_tracer),
    }


def compute_reversibility_error(
    initial_x: np.ndarray,
    initial_y: np.ndarray,
    final_x: np.ndarray,
    final_y: np.ndarray,
    L: float
) -> Dict[str, float]:
    """
    Compute reversibility error after forward+backward advection.
    
    E_rev = mean(|X_final - X_initial|)
    
    Args:
        initial_x, initial_y: Initial particle positions
        final_x, final_y: Final positions after forward+reverse
        L: Characteristic length scale for normalization
    
    Returns:
        Dictionary with reversibility metrics
    """
    displacement = np.sqrt((final_x - initial_x)**2 + (final_y - initial_y)**2)
    
    mean_error = np.mean(displacement)
    max_error = np.max(displacement)
    rms_error = np.sqrt(np.mean(displacement**2))
    
    return {
        'reversibility_mean': float(mean_error),
        'reversibility_max': float(max_error),
        'reversibility_rms': float(rms_error),
        'reversibility_normalized': float(mean_error / L),
        'reversibility_max_normalized': float(max_error / L),
    }


def compute_stretching_factor(
    perimeter_initial: float,
    perimeter_current: float
) -> float:
    """
    Compute stretching factor λ = P(t) / P(0).
    
    λ > 1 indicates stretching/filamentation.
    
    Args:
        perimeter_initial: Initial boundary perimeter
        perimeter_current: Current boundary perimeter
    
    Returns:
        Stretching factor
    """
    if perimeter_initial > 1e-15:
        return perimeter_current / perimeter_initial
    return 1.0


@njit(cache=True, parallel=True)
def compute_particle_dispersion(
    x: np.ndarray,
    y: np.ndarray,
    x0: np.ndarray,
    y0: np.ndarray
) -> Tuple[float, float]:
    """
    Compute mean and variance of particle displacements.
    
    Args:
        x, y: Current positions
        x0, y0: Initial positions
    
    Returns:
        Tuple of (mean_displacement, variance_displacement)
    """
    n = len(x)
    displacements = np.zeros(n, dtype=np.float64)
    
    for i in prange(n):
        displacements[i] = np.sqrt((x[i] - x0[i])**2 + (y[i] - y0[i])**2)
    
    mean_disp = np.mean(displacements)
    var_disp = np.var(displacements)
    
    return mean_disp, var_disp


def compute_boundary_particles(
    x: np.ndarray,
    y: np.ndarray,
    tracer: np.ndarray,
    threshold: float,
    n_boundary: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract boundary particles above tracer threshold.
    
    Uses convex hull approximation for ordering.
    
    Args:
        x, y: Particle positions
        tracer: Tracer values
        threshold: Concentration threshold for "inside"
        n_boundary: Number of boundary particles to return
    
    Returns:
        Tuple of (boundary_x, boundary_y) ordered arrays
    """
    # Find particles above threshold
    mask = tracer >= threshold
    x_high = x[mask]
    y_high = y[mask]
    
    if len(x_high) < 3:
        return np.array([]), np.array([])
    
    # Compute center
    cx = np.mean(x_high)
    cy = np.mean(y_high)
    
    # Compute angles from center
    angles = np.arctan2(y_high - cy, x_high - cx)
    
    # Sort by angle
    sort_idx = np.argsort(angles)
    x_sorted = x_high[sort_idx]
    y_sorted = y_high[sort_idx]
    
    # Subsample to n_boundary points
    n = len(x_sorted)
    if n > n_boundary:
        indices = np.linspace(0, n - 1, n_boundary, dtype=int)
        return x_sorted[indices], y_sorted[indices]
    
    return x_sorted, y_sorted


def compute_all_diagnostics(
    result: 'SimulationResult',
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compute all Lagrangian diagnostics for a simulation result.
    
    Args:
        result: SimulationResult from solver
        verbose: Print diagnostic summary
    
    Returns:
        Dictionary with all computed metrics
    """
    from .solver import SimulationResult
    
    diagnostics = {}
    
    basin = result.basin
    vf = result.velocity_field
    
    # Mass conservation
    mass_metrics = compute_mass_conservation(
        result.particle_tracer[0, :],
        result.particle_tracer[-1, :],
        vf.dx,
        vf.dy,
        result.config['particles_per_cell']
    )
    diagnostics.update(mass_metrics)
    
    # Variance decay / mixing
    variance_metrics = compute_variance_decay(
        result.particle_tracer[0, :],
        result.particle_tracer[-1, :]
    )
    diagnostics.update(variance_metrics)
    
    # Center of mass initial and final
    cm_initial = compute_center_of_mass(
        result.particle_x[0, :],
        result.particle_y[0, :],
        result.particle_tracer[0, :]
    )
    cm_final = compute_center_of_mass(
        result.particle_x[-1, :],
        result.particle_y[-1, :],
        result.particle_tracer[-1, :]
    )
    
    diagnostics['x_cm_initial'] = cm_initial['x_cm']
    diagnostics['y_cm_initial'] = cm_initial['y_cm']
    diagnostics['x_cm_final'] = cm_final['x_cm']
    diagnostics['y_cm_final'] = cm_final['y_cm']
    
    # Center of mass drift
    cm_drift = np.sqrt(
        (cm_final['x_cm'] - cm_initial['x_cm'])**2 +
        (cm_final['y_cm'] - cm_initial['y_cm'])**2
    )
    diagnostics['cm_drift'] = float(cm_drift)
    diagnostics['cm_drift_normalized'] = float(cm_drift / basin.Lx)
    
    # Particle dispersion
    mean_disp, var_disp = compute_particle_dispersion(
        result.particle_x[-1, :],
        result.particle_y[-1, :],
        result.particle_x[0, :],
        result.particle_y[0, :]
    )
    diagnostics['mean_displacement'] = float(mean_disp)
    diagnostics['variance_displacement'] = float(var_disp)
    diagnostics['mean_displacement_normalized'] = float(mean_disp / basin.Lx)
    
    # Boundary shape metrics (if tracer is concentrated)
    max_tracer = np.max(result.particle_tracer[0, :])
    threshold = 0.5 * max_tracer
    
    # Initial boundary
    bx_init, by_init = compute_boundary_particles(
        result.particle_x[0, :],
        result.particle_y[0, :],
        result.particle_tracer[0, :],
        threshold
    )
    
    # Final boundary
    bx_final, by_final = compute_boundary_particles(
        result.particle_x[-1, :],
        result.particle_y[-1, :],
        result.particle_tracer[-1, :],
        threshold
    )
    
    if len(bx_init) > 2 and len(bx_final) > 2:
        ap_init = compute_area_perimeter(bx_init, by_init)
        ap_final = compute_area_perimeter(bx_final, by_final)
        
        diagnostics['area_initial'] = ap_init['area']
        diagnostics['area_final'] = ap_final['area']
        diagnostics['perimeter_initial'] = ap_init['perimeter']
        diagnostics['perimeter_final'] = ap_final['perimeter']
        
        diagnostics['circularity_initial'] = compute_circularity(
            ap_init['area'], ap_init['perimeter']
        )
        diagnostics['circularity_final'] = compute_circularity(
            ap_final['area'], ap_final['perimeter']
        )
        
        diagnostics['stretching_factor'] = compute_stretching_factor(
            ap_init['perimeter'], ap_final['perimeter']
        )
        
        # Area conservation error (should be 0 for incompressible)
        if ap_init['area'] > 1e-10:
            diagnostics['area_error_relative'] = float(
                (ap_final['area'] - ap_init['area']) / ap_init['area']
            )
        else:
            diagnostics['area_error_relative'] = 0.0
    
    # Store in result
    result.diagnostics = diagnostics
    
    if verbose:
        print("\n" + "=" * 60)
        print("LAGRANGIAN DIAGNOSTICS")
        print("=" * 60)
        print(f"  Mass conservation error: {diagnostics['mass_error_relative']:.2e}")
        print(f"  Variance decay (I_s): {diagnostics['intensity_segregation']:.4f}")
        print(f"  Mean displacement: {diagnostics['mean_displacement_normalized']:.4f} L")
        print(f"  CM drift: {diagnostics['cm_drift_normalized']:.4f} L")
        if 'stretching_factor' in diagnostics:
            print(f"  Stretching factor: {diagnostics['stretching_factor']:.2f}")
            print(f"  Circularity (final): {diagnostics['circularity_final']:.4f}")
            print(f"  Area error: {diagnostics['area_error_relative']:.2e}")
        print("=" * 60)
    
    return diagnostics
