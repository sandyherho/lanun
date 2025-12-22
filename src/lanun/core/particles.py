"""
Lagrangian Particle System for Tracer Transport.

Implements particle deployment, advection, and mesh-particle projections
using Numba for high performance.

Key features:
    - Uniform particle deployment with configurable density
    - Bilinear interpolation weights (partition of unity)
    - Midpoint (RK2) advection scheme
    - Particle→mesh projection via weighted bincount
    - Boundary constraint handling
"""

import numpy as np
from numba import njit, prange
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any


@dataclass
class ParticleSet:
    """
    Container for Lagrangian particle data.
    
    Attributes:
        x: Particle x-positions [m]
        y: Particle y-positions [m]
        tracer: Tracer concentration on particles
        n_particles: Total number of particles
        cell_i: Cell index in x for each particle
        cell_j: Cell index in y for each particle
        w1, w2, w3, w4: Bilinear interpolation weights
    """
    x: np.ndarray
    y: np.ndarray
    tracer: np.ndarray
    n_particles: int = field(init=False)
    cell_i: np.ndarray = field(init=False)
    cell_j: np.ndarray = field(init=False)
    w1: np.ndarray = field(init=False)
    w2: np.ndarray = field(init=False)
    w3: np.ndarray = field(init=False)
    w4: np.ndarray = field(init=False)
    
    def __post_init__(self):
        self.n_particles = len(self.x)
        self.cell_i = np.zeros(self.n_particles, dtype=np.int32)
        self.cell_j = np.zeros(self.n_particles, dtype=np.int32)
        self.w1 = np.zeros(self.n_particles, dtype=np.float64)
        self.w2 = np.zeros(self.n_particles, dtype=np.float64)
        self.w3 = np.zeros(self.n_particles, dtype=np.float64)
        self.w4 = np.zeros(self.n_particles, dtype=np.float64)
    
    def copy(self) -> 'ParticleSet':
        """Create a deep copy of the particle set."""
        return ParticleSet(
            x=self.x.copy(),
            y=self.y.copy(),
            tracer=self.tracer.copy()
        )


def deploy_particles(
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    x_min: float,
    y_min: float,
    particles_per_cell_side: int = 4,
    tracer_init: Optional[np.ndarray] = None
) -> ParticleSet:
    """
    Deploy particles uniformly across the domain.
    
    Args:
        nx: Number of cells in x
        ny: Number of cells in y
        dx: Cell width in x [m]
        dy: Cell width in y [m]
        x_min: Minimum x-coordinate [m]
        y_min: Minimum y-coordinate [m]
        particles_per_cell_side: Number of particles per cell side
        tracer_init: Optional initial tracer values
    
    Returns:
        ParticleSet with deployed particles
    """
    pps = particles_per_cell_side
    particles_per_cell = pps * pps
    n_cells_x = nx - 1
    n_cells_y = ny - 1
    n_particles = particles_per_cell * n_cells_x * n_cells_y
    
    # Create particle positions
    x_temp = x_min + (np.arange(n_cells_x * pps) + 0.5) * dx / pps
    y_temp = y_min + (np.arange(n_cells_y * pps) + 0.5) * dy / pps
    
    xp, yp = np.meshgrid(x_temp, y_temp, indexing='ij')
    xp = xp.reshape(n_particles)
    yp = yp.reshape(n_particles)
    
    # Initialize tracer
    if tracer_init is None:
        tracer = np.zeros(n_particles, dtype=np.float64)
    else:
        tracer = tracer_init.copy()
    
    return ParticleSet(x=xp, y=yp, tracer=tracer)


def initialize_gaussian_tracer(
    particles: ParticleSet,
    center_x: float,
    center_y: float,
    sigma: float,
    background: float = 0.0,
    amplitude: float = 1.0
) -> None:
    """
    Initialize Gaussian tracer anomaly on particles.
    
    C(x,y) = background + amplitude * exp(-((x-x0)² + (y-y0)²) / (2σ²))
    
    Args:
        particles: ParticleSet to initialize
        center_x: x-coordinate of anomaly center [m]
        center_y: y-coordinate of anomaly center [m]
        sigma: Width of Gaussian [m]
        background: Background concentration
        amplitude: Anomaly amplitude
    """
    r2 = (particles.x - center_x)**2 + (particles.y - center_y)**2
    particles.tracer[:] = background + amplitude * np.exp(-r2 / (2 * sigma**2))


@njit(cache=True, parallel=True)
def locate_particles(
    px: np.ndarray,
    py: np.ndarray,
    x_min: float,
    y_min: float,
    dx: float,
    dy: float,
    nx: int,
    ny: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find cell indices for all particles.
    
    Args:
        px, py: Particle positions
        x_min, y_min: Domain minimum coordinates
        dx, dy: Cell sizes
        nx, ny: Number of cells
    
    Returns:
        Tuple of (cell_i, cell_j) integer arrays
    """
    n_particles = len(px)
    cell_i = np.zeros(n_particles, dtype=np.int32)
    cell_j = np.zeros(n_particles, dtype=np.int32)
    
    for p in prange(n_particles):
        ci = int((px[p] - x_min) / dx)
        cj = int((py[p] - y_min) / dy)
        
        # Clamp to valid range
        cell_i[p] = max(0, min(ci, nx - 2))
        cell_j[p] = max(0, min(cj, ny - 2))
    
    return cell_i, cell_j


@njit(cache=True, parallel=True)
def compute_bilinear_weights(
    px: np.ndarray,
    py: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    cell_i: np.ndarray,
    cell_j: np.ndarray,
    dx: float,
    dy: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute bilinear interpolation weights.
    
    Weights satisfy partition of unity: w1 + w2 + w3 + w4 = 1
    
    Node layout in cell:
        (i, j+1) --- (i+1, j+1)
           |    p       |
        (i, j) ----- (i+1, j)
    
    Args:
        px, py: Particle positions
        X, Y: Mesh coordinate arrays
        cell_i, cell_j: Cell indices for particles
        dx, dy: Cell sizes
    
    Returns:
        Tuple of (w1, w2, w3, w4) weight arrays
    """
    n_particles = len(px)
    
    w1 = np.zeros(n_particles, dtype=np.float64)
    w2 = np.zeros(n_particles, dtype=np.float64)
    w3 = np.zeros(n_particles, dtype=np.float64)
    w4 = np.zeros(n_particles, dtype=np.float64)
    
    for p in prange(n_particles):
        i = cell_i[p]
        j = cell_j[p]
        
        # Local coordinates [0, 1]
        xi = (px[p] - X[i]) / dx
        eta = (py[p] - Y[j]) / dy
        
        # Clamp to [0, 1] for numerical safety
        xi = max(0.0, min(1.0, xi))
        eta = max(0.0, min(1.0, eta))
        
        # Bilinear shape functions
        w1[p] = (1.0 - xi) * (1.0 - eta)  # node (i, j)
        w2[p] = xi * (1.0 - eta)          # node (i+1, j)
        w3[p] = (1.0 - xi) * eta          # node (i, j+1)
        w4[p] = xi * eta                  # node (i+1, j+1)
    
    return w1, w2, w3, w4


@njit(cache=True, parallel=True)
def interpolate_to_particles(
    field: np.ndarray,
    cell_i: np.ndarray,
    cell_j: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    w3: np.ndarray,
    w4: np.ndarray
) -> np.ndarray:
    """
    Interpolate mesh field to particle positions.
    
    F_p = w1*F[i,j] + w2*F[i+1,j] + w3*F[i,j+1] + w4*F[i+1,j+1]
    
    Args:
        field: 2D mesh field array
        cell_i, cell_j: Cell indices
        w1, w2, w3, w4: Bilinear weights
    
    Returns:
        1D array of field values at particles
    """
    n_particles = len(cell_i)
    result = np.zeros(n_particles, dtype=np.float64)
    
    for p in prange(n_particles):
        i = cell_i[p]
        j = cell_j[p]
        
        result[p] = (
            w1[p] * field[i, j] +
            w2[p] * field[i + 1, j] +
            w3[p] * field[i, j + 1] +
            w4[p] * field[i + 1, j + 1]
        )
    
    return result


@njit(cache=True)
def project_to_mesh(
    tracer_p: np.ndarray,
    cell_i: np.ndarray,
    cell_j: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    w3: np.ndarray,
    w4: np.ndarray,
    nx: int,
    ny: int
) -> np.ndarray:
    """
    Project particle tracer values to mesh using weighted averaging.
    
    F[i,j] = Σ_p (w_p * C_p) / Σ_p w_p
    
    Args:
        tracer_p: Tracer values on particles
        cell_i, cell_j: Cell indices
        w1, w2, w3, w4: Bilinear weights
        nx, ny: Mesh dimensions
    
    Returns:
        2D mesh array of projected tracer
    """
    n_particles = len(tracer_p)
    
    field = np.zeros((nx, ny), dtype=np.float64)
    weight_sum = np.zeros((nx, ny), dtype=np.float64)
    
    for p in range(n_particles):
        i = cell_i[p]
        j = cell_j[p]
        c = tracer_p[p]
        
        # Distribute to four corners
        field[i, j] += w1[p] * c
        field[i + 1, j] += w2[p] * c
        field[i, j + 1] += w3[p] * c
        field[i + 1, j + 1] += w4[p] * c
        
        weight_sum[i, j] += w1[p]
        weight_sum[i + 1, j] += w2[p]
        weight_sum[i, j + 1] += w3[p]
        weight_sum[i + 1, j + 1] += w4[p]
    
    # Normalize
    for i in range(nx):
        for j in range(ny):
            if weight_sum[i, j] > 1e-10:
                field[i, j] /= weight_sum[i, j]
            else:
                field[i, j] = 0.0
    
    return field


@njit(cache=True, parallel=True)
def advect_midpoint(
    px: np.ndarray,
    py: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    dt: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    dx: float,
    dy: float,
    nx: int,
    ny: int,
    direction: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Advect particles using midpoint (RK2) scheme.
    
    Midpoint method:
        1. x* = x + (dt/2) * v(x)
        2. x_new = x + dt * v(x*)
    
    Args:
        px, py: Particle positions
        vx, vy: Velocity field on mesh
        X, Y: Mesh coordinates
        dt: Time step [s]
        x_min, x_max: Domain bounds in x
        y_min, y_max: Domain bounds in y
        dx, dy: Cell sizes
        nx, ny: Mesh dimensions
        direction: 1.0 for forward, -1.0 for reverse
    
    Returns:
        Tuple of (new_px, new_py)
    """
    n_particles = len(px)
    new_px = np.zeros(n_particles, dtype=np.float64)
    new_py = np.zeros(n_particles, dtype=np.float64)
    
    for p in prange(n_particles):
        x = px[p]
        y = py[p]
        
        # Step 1: Interpolate velocity at current position
        ci = int((x - x_min) / dx)
        cj = int((y - y_min) / dy)
        ci = max(0, min(ci, nx - 2))
        cj = max(0, min(cj, ny - 2))
        
        xi = (x - X[ci]) / dx
        eta = (y - Y[cj]) / dy
        xi = max(0.0, min(1.0, xi))
        eta = max(0.0, min(1.0, eta))
        
        w1 = (1.0 - xi) * (1.0 - eta)
        w2 = xi * (1.0 - eta)
        w3 = (1.0 - xi) * eta
        w4 = xi * eta
        
        u = (w1 * vx[ci, cj] + w2 * vx[ci+1, cj] + 
             w3 * vx[ci, cj+1] + w4 * vx[ci+1, cj+1])
        v = (w1 * vy[ci, cj] + w2 * vy[ci+1, cj] + 
             w3 * vy[ci, cj+1] + w4 * vy[ci+1, cj+1])
        
        # Step 2: Compute midpoint position
        x_mid = x + direction * 0.5 * dt * u
        y_mid = y + direction * 0.5 * dt * v
        
        # Clamp midpoint to domain
        x_mid = max(x_min, min(x_max, x_mid))
        y_mid = max(y_min, min(y_max, y_mid))
        
        # Step 3: Interpolate velocity at midpoint
        ci = int((x_mid - x_min) / dx)
        cj = int((y_mid - y_min) / dy)
        ci = max(0, min(ci, nx - 2))
        cj = max(0, min(cj, ny - 2))
        
        xi = (x_mid - X[ci]) / dx
        eta = (y_mid - Y[cj]) / dy
        xi = max(0.0, min(1.0, xi))
        eta = max(0.0, min(1.0, eta))
        
        w1 = (1.0 - xi) * (1.0 - eta)
        w2 = xi * (1.0 - eta)
        w3 = (1.0 - xi) * eta
        w4 = xi * eta
        
        u_mid = (w1 * vx[ci, cj] + w2 * vx[ci+1, cj] + 
                 w3 * vx[ci, cj+1] + w4 * vx[ci+1, cj+1])
        v_mid = (w1 * vy[ci, cj] + w2 * vy[ci+1, cj] + 
                 w3 * vy[ci, cj+1] + w4 * vy[ci+1, cj+1])
        
        # Step 4: Full step using midpoint velocity
        new_x = x + direction * dt * u_mid
        new_y = y + direction * dt * v_mid
        
        # Constrain to domain
        new_px[p] = max(x_min + 1e-10, min(x_max - 1e-10, new_x))
        new_py[p] = max(y_min + 1e-10, min(y_max - 1e-10, new_y))
    
    return new_px, new_py


@njit(cache=True, parallel=True)
def constrain_particles(
    px: np.ndarray,
    py: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    dx: float,
    dy: float,
    pps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constrain particles to domain boundaries.
    
    Args:
        px, py: Particle positions
        x_min, x_max: Domain bounds in x
        y_min, y_max: Domain bounds in y
        dx, dy: Cell sizes
        pps: Particles per cell side
    
    Returns:
        Tuple of (constrained_px, constrained_py)
    """
    n_particles = len(px)
    offset = 0.5 / pps
    
    new_px = np.zeros(n_particles, dtype=np.float64)
    new_py = np.zeros(n_particles, dtype=np.float64)
    
    for p in prange(n_particles):
        x = px[p]
        y = py[p]
        
        if x < x_min:
            x = x_min + offset * dx
        elif x > x_max:
            x = x_max - offset * dx
        
        if y < y_min:
            y = y_min + offset * dy
        elif y > y_max:
            y = y_max - offset * dy
        
        new_px[p] = x
        new_py[p] = y
    
    return new_px, new_py
