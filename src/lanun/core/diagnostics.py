"""
Lagrangian Diagnostics for Transport Quality Assessment.
=========================================================

Physics Background:
------------------
In purely Lagrangian transport (no diffusion), particles carry tracer values
unchanged along trajectories. This means:

1. MASS is exactly conserved (trivially, since C_p is constant)
2. MATERIAL AREA is conserved for incompressible flow (∇·v = 0)
3. VARIANCE ON PARTICLES is constant (I_s = 1 always without diffusion)
4. PERIMETER GROWS exponentially for chaotic flows (stirring)

The challenge: computing area/perimeter for non-convex, filamented distributions.

Solution approach:
- Use convex hull for "spread extent" (well-defined, but overestimates area)
- Track particle-pair separation for stretching (FTLE-like diagnostic)
- Compute mesh-projected variance for "effective mixing" (includes numerical diffusion)
- Use proper Lagrangian deformation metrics

References:
    - Ottino, J. M. (1989). The Kinematics of Mixing. Cambridge University Press.
    - Haller, G. (2015). Lagrangian Coherent Structures. Annu. Rev. Fluid Mech. 47:137-162.
"""

import numpy as np
from numba import njit, prange
from typing import Dict, Any, Optional, Tuple
from scipy.spatial import ConvexHull


# =============================================================================
# MASS CONSERVATION
# =============================================================================

def compute_mass_conservation(
    tracer_initial: np.ndarray,
    tracer_final: np.ndarray,
    dx: float,
    dy: float,
    particles_per_cell: int
) -> Dict[str, float]:
    """
    Compute tracer mass conservation error.
    
    For purely Lagrangian transport, this should be EXACTLY zero
    since particles carry their tracer values unchanged.
    
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


# =============================================================================
# CONVEX HULL BASED METRICS (robust for any distribution)
# =============================================================================

def compute_convex_hull_metrics(
    x: np.ndarray,
    y: np.ndarray,
    tracer: np.ndarray,
    threshold_fraction: float = 0.1
) -> Dict[str, float]:
    """
    Compute convex hull area and perimeter for tracer patch.
    
    The convex hull is the smallest convex polygon containing all points.
    For filamented distributions:
        - Hull AREA measures "spread extent" (increases with stirring)
        - Hull PERIMETER measures "envelope size"
    
    NOTE: Hull area ≠ material area. For incompressible flow, material area
    is conserved, but hull area grows as the blob filaments and spreads.
    
    Args:
        x, y: Particle positions
        tracer: Tracer values
        threshold_fraction: Fraction of (max-min) above min to define "high tracer"
    
    Returns:
        Dictionary with hull metrics
    """
    # Identify particles in the tracer patch
    tracer_min = np.min(tracer)
    tracer_max = np.max(tracer)
    threshold = tracer_min + threshold_fraction * (tracer_max - tracer_min)
    
    mask = tracer >= threshold
    x_patch = x[mask]
    y_patch = y[mask]
    
    n_points = len(x_patch)
    
    if n_points < 3:
        return {
            'hull_area': 0.0,
            'hull_perimeter': 0.0,
            'hull_n_vertices': 0,
            'hull_n_points': n_points,
        }
    
    # Stack coordinates for scipy
    points = np.column_stack([x_patch, y_patch])
    
    try:
        hull = ConvexHull(points)
        
        return {
            'hull_area': float(hull.volume),  # In 2D, volume = area
            'hull_perimeter': float(hull.area),  # In 2D, area = perimeter
            'hull_n_vertices': len(hull.vertices),
            'hull_n_points': n_points,
        }
    except Exception:
        # Degenerate case (collinear points)
        return {
            'hull_area': 0.0,
            'hull_perimeter': 0.0,
            'hull_n_vertices': 0,
            'hull_n_points': n_points,
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


# =============================================================================
# PARTICLE-PAIR STRETCHING (FTLE-like diagnostic)
# =============================================================================

@njit(cache=True, parallel=True)
def _compute_pair_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute all pairwise distances (upper triangle only)."""
    n = len(x)
    n_pairs = n * (n - 1) // 2
    distances = np.zeros(n_pairs, dtype=np.float64)
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            distances[idx] = d
            idx += 1
    
    return distances


def compute_stretching_statistics(
    x_init: np.ndarray,
    y_init: np.ndarray,
    x_final: np.ndarray,
    y_final: np.ndarray,
    tracer: np.ndarray,
    threshold_fraction: float = 0.5,
    max_sample: int = 200
) -> Dict[str, float]:
    """
    Compute stretching statistics from particle-pair separations.
    
    This is a finite-time Lyapunov exponent (FTLE) like diagnostic.
    
    For chaotic advection, nearby particles separate exponentially:
        d(t) ~ d(0) * exp(λt)
    
    where λ is the Lyapunov exponent.
    
    The stretching factor λ_eff = <d_final> / <d_initial> measures
    how much material lines have been stretched on average.
    
    Args:
        x_init, y_init: Initial particle positions
        x_final, y_final: Final particle positions
        tracer: Tracer values (to identify patch)
        threshold_fraction: Fraction to define tracer patch
        max_sample: Maximum particles to sample (for efficiency)
    
    Returns:
        Dictionary with stretching metrics
    """
    # Identify tracer patch particles
    tracer_min = np.min(tracer)
    tracer_max = np.max(tracer)
    threshold = tracer_min + threshold_fraction * (tracer_max - tracer_min)
    
    mask = tracer >= threshold
    indices = np.where(mask)[0]
    n_patch = len(indices)
    
    if n_patch < 10:
        return {
            'stretching_factor_mean': 1.0,
            'stretching_factor_std': 0.0,
            'stretching_factor_max': 1.0,
            'stretching_factor_close_pairs': 1.0,
            'lyapunov_proxy': 0.0,
            'n_pairs_used': 0,
        }
    
    # Subsample if too many particles
    if n_patch > max_sample:
        np.random.seed(42)  # Reproducibility
        indices = np.random.choice(indices, max_sample, replace=False)
        n_patch = max_sample
    
    # Get positions for patch particles
    xi = x_init[indices]
    yi = y_init[indices]
    xf = x_final[indices]
    yf = y_final[indices]
    
    # Compute pairwise distances
    d_init = _compute_pair_distances(xi, yi)
    d_final = _compute_pair_distances(xf, yf)
    
    # Avoid division by zero for initially coincident particles
    valid = d_init > 1e-10
    if np.sum(valid) < 10:
        return {
            'stretching_factor_mean': 1.0,
            'stretching_factor_std': 0.0,
            'stretching_factor_max': 1.0,
            'stretching_factor_close_pairs': 1.0,
            'lyapunov_proxy': 0.0,
            'n_pairs_used': 0,
        }
    
    d_init_valid = d_init[valid]
    d_final_valid = d_final[valid]
    
    # Compute stretching factors for each pair
    stretch_factors = d_final_valid / d_init_valid
    
    # Also compute for initially close pairs only (more FTLE-like)
    close_threshold = np.percentile(d_init_valid, 25)  # Bottom quartile
    close_mask = d_init_valid < close_threshold
    
    if np.sum(close_mask) > 10:
        stretch_close = d_final_valid[close_mask] / d_init_valid[close_mask]
        stretch_close_mean = np.mean(stretch_close)
    else:
        stretch_close_mean = np.mean(stretch_factors)
    
    return {
        'stretching_factor_mean': float(np.mean(stretch_factors)),
        'stretching_factor_std': float(np.std(stretch_factors)),
        'stretching_factor_max': float(np.max(stretch_factors)),
        'stretching_factor_close_pairs': float(stretch_close_mean),
        'lyapunov_proxy': float(np.log(stretch_close_mean)) if stretch_close_mean > 0 else 0.0,
        'n_pairs_used': int(np.sum(valid)),
    }


# =============================================================================
# VARIANCE AND MIXING METRICS
# =============================================================================

def compute_variance_decay(
    tracer_initial: np.ndarray,
    tracer_current: np.ndarray
) -> Dict[str, float]:
    """
    Compute intensity of segregation (normalized variance) for PARTICLES.
    
    I_s = σ²(t) / σ²(0)
    
    CRITICAL PHYSICS:
    For purely Lagrangian transport without diffusion, each particle 
    carries its tracer value unchanged. Therefore:
    
        I_s = 1.0 ALWAYS (this is correct, not a bug!)
    
    Stirring ≠ Mixing. Real mixing requires molecular diffusion.
    
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


def compute_mesh_variance_decay(
    mesh_tracer_initial: np.ndarray,
    mesh_tracer_final: np.ndarray
) -> Dict[str, float]:
    """
    Compute variance decay on the MESH (includes numerical diffusion).
    
    When particles are projected to a mesh, averaging occurs within cells.
    This acts like numerical diffusion and causes mesh variance to decrease
    even though particle variance remains constant.
    
    The mesh-based I_s captures "effective mixing" due to sub-grid stirring.
    
    Args:
        mesh_tracer_initial: Initial tracer field on mesh
        mesh_tracer_final: Final tracer field on mesh
    
    Returns:
        Dictionary with mesh variance metrics
    """
    var_init = np.var(mesh_tracer_initial)
    var_final = np.var(mesh_tracer_final)
    
    if var_init > 1e-15:
        I_s_mesh = var_final / var_init
    else:
        I_s_mesh = 1.0
    
    return {
        'variance_mesh_initial': float(var_init),
        'variance_mesh_final': float(var_final),
        'intensity_segregation_mesh': float(I_s_mesh),
        'effective_mixing': float(max(0, 1.0 - I_s_mesh)),
    }


# =============================================================================
# CENTER OF MASS AND DISPERSION
# =============================================================================

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


@njit(cache=True, parallel=True)
def compute_particle_dispersion(
    x: np.ndarray,
    y: np.ndarray,
    x0: np.ndarray,
    y0: np.ndarray
) -> Tuple[float, float, float]:
    """
    Compute statistics of particle displacements.
    
    Args:
        x, y: Current positions
        x0, y0: Initial positions
    
    Returns:
        Tuple of (mean_displacement, std_displacement, max_displacement)
    """
    n = len(x)
    displacements = np.zeros(n, dtype=np.float64)
    
    for i in prange(n):
        displacements[i] = np.sqrt((x[i] - x0[i])**2 + (y[i] - y0[i])**2)
    
    return np.mean(displacements), np.std(displacements), np.max(displacements)


def compute_radius_of_gyration(
    x: np.ndarray,
    y: np.ndarray,
    tracer: np.ndarray,
    threshold_fraction: float = 0.1
) -> Dict[str, float]:
    """
    Compute radius of gyration for tracer patch.
    
    R_g² = <(r - r_cm)²> = Σ C_i |r_i - r_cm|² / Σ C_i
    
    This measures the "spread" of the tracer distribution
    and increases as the blob filaments and disperses.
    
    Unlike convex hull area, R_g is well-defined for any distribution.
    
    Args:
        x, y: Particle positions
        tracer: Tracer values
        threshold_fraction: Fraction to define tracer patch
    
    Returns:
        Dictionary with radius of gyration
    """
    # Identify tracer patch
    tracer_min = np.min(tracer)
    tracer_max = np.max(tracer)
    threshold = tracer_min + threshold_fraction * (tracer_max - tracer_min)
    
    mask = tracer >= threshold
    x_patch = x[mask]
    y_patch = y[mask]
    tracer_patch = tracer[mask]
    
    if len(x_patch) < 3:
        return {'radius_of_gyration': 0.0}
    
    # Tracer-weighted center
    total_c = np.sum(tracer_patch)
    x_cm = np.sum(tracer_patch * x_patch) / total_c
    y_cm = np.sum(tracer_patch * y_patch) / total_c
    
    # Radius of gyration (tracer-weighted)
    r2 = (x_patch - x_cm)**2 + (y_patch - y_cm)**2
    rg2 = np.sum(tracer_patch * r2) / total_c
    rg = np.sqrt(rg2)
    
    return {
        'radius_of_gyration': float(rg),
    }


# =============================================================================
# REVERSIBILITY ERROR
# =============================================================================

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


# =============================================================================
# LEGACY FUNCTIONS (for backward compatibility)
# =============================================================================

@njit(cache=True)
def _shoelace_area(x: np.ndarray, y: np.ndarray) -> float:
    """Compute polygon area using shoelace formula."""
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
    
    DEPRECATED: Use compute_convex_hull_metrics instead for robust results.
    
    Args:
        boundary_x: x-coordinates of boundary particles (ordered)
        boundary_y: y-coordinates of boundary particles (ordered)
    
    Returns:
        Dictionary with area and perimeter
    """
    if len(boundary_x) < 3:
        return {'area': 0.0, 'perimeter': 0.0}
    
    area = _shoelace_area(boundary_x, boundary_y)
    perimeter = _perimeter(boundary_x, boundary_y)
    
    return {
        'area': float(area),
        'perimeter': float(perimeter),
    }


def compute_stretching_factor(
    perimeter_initial: float,
    perimeter_current: float
) -> float:
    """
    Compute stretching factor λ = P(t) / P(0).
    
    DEPRECATED: Use compute_stretching_statistics for robust FTLE-like metric.
    
    Args:
        perimeter_initial: Initial boundary perimeter
        perimeter_current: Current boundary perimeter
    
    Returns:
        Stretching factor
    """
    if perimeter_initial > 1e-15:
        return perimeter_current / perimeter_initial
    return 1.0


def compute_boundary_particles(
    x: np.ndarray,
    y: np.ndarray,
    tracer: np.ndarray,
    threshold: float,
    n_boundary: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract boundary particles using convex hull.
    
    NOTE: This now uses convex hull instead of angle-sorting
    to ensure robust results for non-convex distributions.
    
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
    
    # Use convex hull vertices as boundary
    points = np.column_stack([x_high, y_high])
    
    try:
        hull = ConvexHull(points)
        boundary_indices = hull.vertices
        
        bx = x_high[boundary_indices]
        by = y_high[boundary_indices]
        
        return bx, by
        
    except Exception:
        return np.array([]), np.array([])


# =============================================================================
# MAIN DIAGNOSTIC FUNCTION
# =============================================================================

def compute_all_diagnostics(
    result: 'SimulationResult',
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compute all Lagrangian diagnostics for a simulation result.
    
    This function computes physically meaningful metrics that are
    well-defined even for highly filamented tracer distributions.
    
    Args:
        result: SimulationResult from solver
        verbose: Print diagnostic summary
    
    Returns:
        Dictionary with all computed metrics
    """
    diagnostics = {}
    
    basin = result.basin
    vf = result.velocity_field
    
    # =========================================================================
    # 1. MASS CONSERVATION (exact for Lagrangian)
    # =========================================================================
    mass_metrics = compute_mass_conservation(
        result.particle_tracer[0, :],
        result.particle_tracer[-1, :],
        vf.dx,
        vf.dy,
        result.config['particles_per_cell']
    )
    diagnostics.update(mass_metrics)
    
    # =========================================================================
    # 2. PARTICLE VARIANCE (should be constant for Lagrangian)
    # =========================================================================
    variance_metrics = compute_variance_decay(
        result.particle_tracer[0, :],
        result.particle_tracer[-1, :]
    )
    diagnostics.update(variance_metrics)
    
    # =========================================================================
    # 3. MESH VARIANCE (captures numerical diffusion)
    # =========================================================================
    mesh_var_metrics = compute_mesh_variance_decay(
        result.mesh_tracer[0, :, :],
        result.mesh_tracer[-1, :, :]
    )
    diagnostics.update(mesh_var_metrics)
    
    # =========================================================================
    # 4. CONVEX HULL METRICS (robust for any shape)
    # =========================================================================
    hull_init = compute_convex_hull_metrics(
        result.particle_x[0, :],
        result.particle_y[0, :],
        result.particle_tracer[0, :]
    )
    hull_final = compute_convex_hull_metrics(
        result.particle_x[-1, :],
        result.particle_y[-1, :],
        result.particle_tracer[-1, :]
    )
    
    diagnostics['hull_area_initial'] = hull_init['hull_area']
    diagnostics['hull_area_final'] = hull_final['hull_area']
    diagnostics['hull_perimeter_initial'] = hull_init['hull_perimeter']
    diagnostics['hull_perimeter_final'] = hull_final['hull_perimeter']
    
    # Hull area ratio (measures spreading, NOT material area conservation)
    if hull_init['hull_area'] > 1e-10:
        diagnostics['hull_area_ratio'] = hull_final['hull_area'] / hull_init['hull_area']
    else:
        diagnostics['hull_area_ratio'] = 1.0
    
    # Hull circularity
    diagnostics['hull_circularity_initial'] = compute_circularity(
        hull_init['hull_area'], hull_init['hull_perimeter']
    )
    diagnostics['hull_circularity_final'] = compute_circularity(
        hull_final['hull_area'], hull_final['hull_perimeter']
    )
    
    # =========================================================================
    # 5. STRETCHING (FTLE-like)
    # =========================================================================
    stretch_metrics = compute_stretching_statistics(
        result.particle_x[0, :],
        result.particle_y[0, :],
        result.particle_x[-1, :],
        result.particle_y[-1, :],
        result.particle_tracer[0, :]
    )
    diagnostics.update(stretch_metrics)
    
    # =========================================================================
    # 6. CENTER OF MASS
    # =========================================================================
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
    
    cm_drift = np.sqrt(
        (cm_final['x_cm'] - cm_initial['x_cm'])**2 +
        (cm_final['y_cm'] - cm_initial['y_cm'])**2
    )
    diagnostics['cm_drift'] = float(cm_drift)
    diagnostics['cm_drift_normalized'] = float(cm_drift / basin.Lx)
    
    # =========================================================================
    # 7. RADIUS OF GYRATION
    # =========================================================================
    rg_init = compute_radius_of_gyration(
        result.particle_x[0, :],
        result.particle_y[0, :],
        result.particle_tracer[0, :]
    )
    rg_final = compute_radius_of_gyration(
        result.particle_x[-1, :],
        result.particle_y[-1, :],
        result.particle_tracer[-1, :]
    )
    
    diagnostics['radius_of_gyration_initial'] = rg_init['radius_of_gyration']
    diagnostics['radius_of_gyration_final'] = rg_final['radius_of_gyration']
    
    if rg_init['radius_of_gyration'] > 1e-10:
        diagnostics['rg_growth_ratio'] = (
            rg_final['radius_of_gyration'] / rg_init['radius_of_gyration']
        )
    else:
        diagnostics['rg_growth_ratio'] = 1.0
    
    # =========================================================================
    # 8. PARTICLE DISPERSION
    # =========================================================================
    mean_disp, std_disp, max_disp = compute_particle_dispersion(
        result.particle_x[-1, :],
        result.particle_y[-1, :],
        result.particle_x[0, :],
        result.particle_y[0, :]
    )
    
    diagnostics['mean_displacement'] = float(mean_disp)
    diagnostics['std_displacement'] = float(std_disp)
    diagnostics['max_displacement'] = float(max_disp)
    diagnostics['mean_displacement_normalized'] = float(mean_disp / basin.Lx)
    
    # =========================================================================
    # 9. LEGACY COMPATIBILITY
    # =========================================================================
    # Map new metrics to old names for backward compatibility
    diagnostics['stretching_factor'] = diagnostics['stretching_factor_mean']
    diagnostics['circularity_initial'] = diagnostics['hull_circularity_initial']
    diagnostics['circularity_final'] = diagnostics['hull_circularity_final']
    diagnostics['area_initial'] = diagnostics['hull_area_initial']
    diagnostics['area_final'] = diagnostics['hull_area_final']
    diagnostics['perimeter_initial'] = diagnostics['hull_perimeter_initial']
    diagnostics['perimeter_final'] = diagnostics['hull_perimeter_final']
    
    # Area "error" is now interpreted as spreading ratio - 1
    diagnostics['area_error_relative'] = diagnostics['hull_area_ratio'] - 1.0
    
    # =========================================================================
    # Store in result
    # =========================================================================
    result.diagnostics = diagnostics
    
    # =========================================================================
    # PRINT SUMMARY
    # =========================================================================
    if verbose:
        print("\n" + "=" * 70)
        print("LAGRANGIAN DIAGNOSTICS")
        print("=" * 70)
        
        print("\n[CONSERVATION]")
        print(f"  Mass error (relative): {diagnostics['mass_error_relative']:.2e}")
        
        print("\n[MIXING & STIRRING]")
        print(f"  I_s (particles): {diagnostics['intensity_segregation']:.4f}")
        print(f"    → Expected: 1.0 (no diffusion = no mixing on particles)")
        print(f"  I_s (mesh): {diagnostics['intensity_segregation_mesh']:.4f}")
        print(f"    → < 1.0 due to projection averaging")
        print(f"  Effective mixing: {diagnostics['effective_mixing']:.4f}")
        
        print("\n[STRETCHING (FTLE-like)]")
        print(f"  Mean stretching factor: {diagnostics['stretching_factor_mean']:.2f}")
        print(f"  Max stretching factor: {diagnostics['stretching_factor_max']:.2f}")
        print(f"  Lyapunov proxy (ln λ): {diagnostics['lyapunov_proxy']:.3f}")
        
        print("\n[SPATIAL EXTENT]")
        print(f"  Hull area ratio: {diagnostics['hull_area_ratio']:.2f}")
        print(f"    → > 1 means tracer has spread (not a conservation error)")
        print(f"  Hull circularity: {diagnostics['hull_circularity_initial']:.3f} → {diagnostics['hull_circularity_final']:.3f}")
        print(f"  R_g growth: {diagnostics['rg_growth_ratio']:.2f}")
        
        print("\n[TRANSPORT]")
        print(f"  Mean displacement: {diagnostics['mean_displacement_normalized']:.4f} L")
        print(f"  CM drift: {diagnostics['cm_drift_normalized']:.4f} L")
        
        print("=" * 70)
    
    return diagnostics
