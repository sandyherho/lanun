"""
Velocity Field Definition for Semi-Enclosed Ocean Basin.

Implements Bell's incompressible thinning flow representing wind-driven
recirculation in a coastal embayment or marginal sea.

Velocity field (dimensional):
    u* = -(U₀/2) sin²(πx*/L) sin(2πy*/L)
    v* =  (U₀/2) sin²(πy*/L) sin(2πx*/L)

Properties:
    - Incompressible: ∇·v = 0 (verified analytically)
    - Zero at boundaries: v(0,y) = v(L,y) = v(x,0) = v(x,L) = 0
    - Maximum velocity at intermediate positions
    - Creates chaotic stirring and filamentation

Stream function:
    ψ = (U₀L/4π) sin²(πx/L) sin²(πy/L)

References:
    Bell, J. B., Colella, P., & Glaz, H. M. (1989). J. Comput. Phys., 85(2), 257-283.
"""

import numpy as np
from numba import njit, prange
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class BasinSystem:
    """
    Semi-enclosed ocean basin configuration.
    
    Represents a coastal embayment, marginal sea, or lake with
    wind-driven recirculation.
    
    Attributes:
        Lx: Basin width in x-direction [m]
        Ly: Basin width in y-direction [m]
        H: Basin depth [m] (z positive downward from surface)
        U0: Maximum velocity magnitude [m/s]
        tracer_name: Name of the tracer (e.g., "Chlorophyll-a")
        tracer_units: Units of tracer concentration
        tracer_background: Background tracer concentration
        tracer_anomaly: Initial anomaly amplitude
        rho0: Reference density [kg/m³]
    
    Example:
        >>> basin = BasinSystem(
        ...     Lx=50e3, Ly=50e3, H=100.0, U0=0.3,
        ...     tracer_name="Chlorophyll-a", tracer_units="mg/m³"
        ... )
    """
    Lx: float = 50000.0  # [m]
    Ly: float = 50000.0  # [m]
    H: float = 100.0     # [m]
    U0: float = 0.3      # [m/s]
    tracer_name: str = "Tracer"
    tracer_units: str = "kg/m³"
    tracer_background: float = 0.0
    tracer_anomaly: float = 1.0
    rho0: float = 1025.0  # [kg/m³]
    
    @property
    def T_advect(self) -> float:
        """Advective timescale T = L/U₀ [s]."""
        return self.Lx / self.U0
    
    @property
    def T_circ(self) -> float:
        """Circulation timescale T_circ = πL/U₀ [s]."""
        return np.pi * self.Lx / self.U0
    
    @property
    def T_circ_days(self) -> float:
        """Circulation timescale [days]."""
        return self.T_circ / 86400.0
    
    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio Ly/Lx."""
        return self.Ly / self.Lx
    
    def __repr__(self) -> str:
        return (
            f"BasinSystem(Lx={self.Lx/1e3:.1f} km, Ly={self.Ly/1e3:.1f} km, "
            f"H={self.H:.0f} m, U0={self.U0:.2f} m/s, "
            f"T_circ={self.T_circ_days:.1f} days)"
        )
    
    def describe(self) -> str:
        """Return detailed description of the basin."""
        return f"""
Semi-Enclosed Ocean Basin
=========================
Domain:
  Lx = {self.Lx/1e3:.1f} km (x-extent)
  Ly = {self.Ly/1e3:.1f} km (y-extent)
  H  = {self.H:.1f} m (depth)
  Aspect ratio = {self.aspect_ratio:.2f}

Flow Parameters:
  U₀ = {self.U0:.3f} m/s (max velocity)
  T_advect = {self.T_advect/3600:.1f} hours
  T_circ = {self.T_circ_days:.2f} days

Tracer:
  Name: {self.tracer_name}
  Units: {self.tracer_units}
  Background: {self.tracer_background} {self.tracer_units}
  Anomaly: {self.tracer_anomaly} {self.tracer_units}

Reference density: {self.rho0} kg/m³
"""


class VelocityField:
    """
    Bell's incompressible velocity field for semi-enclosed basin.
    
    Creates and manages the velocity field on a structured mesh.
    
    Attributes:
        basin: BasinSystem configuration
        nx: Number of grid points in x
        ny: Number of grid points in y
        X: 1D array of x-coordinates [m]
        Y: 1D array of y-coordinates [m]
        vx: 2D velocity field u(x,y) [m/s]
        vy: 2D velocity field v(x,y) [m/s]
    """
    
    def __init__(self, basin: BasinSystem, nx: int = 101, ny: int = 101):
        """
        Initialize velocity field on structured mesh.
        
        Args:
            basin: BasinSystem configuration
            nx: Number of grid points in x-direction
            ny: Number of grid points in y-direction
        """
        self.basin = basin
        self.nx = nx
        self.ny = ny
        
        # Grid spacing
        self.dx = basin.Lx / (nx - 1)
        self.dy = basin.Ly / (ny - 1)
        
        # Coordinate arrays
        self.X = np.linspace(0, basin.Lx, nx)
        self.Y = np.linspace(0, basin.Ly, ny)
        
        # Initialize velocity field
        self.vx, self.vy = self._compute_velocity_field()
        
        # Compute derived fields
        self.psi = self._compute_stream_function()
        self.vorticity = self._compute_vorticity()
        self.divergence = self._compute_divergence()
    
    def _compute_velocity_field(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Bell's velocity field on the mesh.
        
        Returns:
            Tuple of (vx, vy) arrays with shape (nx, ny)
        """
        Lx, Ly, U0 = self.basin.Lx, self.basin.Ly, self.basin.U0
        
        vx = np.zeros((self.nx, self.ny), dtype=np.float64)
        vy = np.zeros((self.nx, self.ny), dtype=np.float64)
        
        for i in range(self.nx):
            for j in range(self.ny):
                x, y = self.X[i], self.Y[j]
                vx[i, j], vy[i, j] = _bell_velocity(x, y, Lx, Ly, U0)
        
        return vx, vy
    
    def _compute_stream_function(self) -> np.ndarray:
        """
        Compute stream function ψ.
        
        ψ = (U₀L/4π) sin²(πx/L) sin²(πy/L)
        """
        Lx, Ly, U0 = self.basin.Lx, self.basin.Ly, self.basin.U0
        
        psi = np.zeros((self.nx, self.ny), dtype=np.float64)
        
        for i in range(self.nx):
            for j in range(self.ny):
                x, y = self.X[i], self.Y[j]
                psi[i, j] = (U0 * Lx / (4 * np.pi)) * \
                           np.sin(np.pi * x / Lx)**2 * \
                           np.sin(np.pi * y / Ly)**2
        
        return psi
    
    def _compute_vorticity(self) -> np.ndarray:
        """
        Compute vertical vorticity ω = ∂v/∂x - ∂u/∂y.
        """
        # Use central differences
        dvdx = np.zeros_like(self.vy)
        dudy = np.zeros_like(self.vx)
        
        dvdx[1:-1, :] = (self.vy[2:, :] - self.vy[:-2, :]) / (2 * self.dx)
        dudy[:, 1:-1] = (self.vx[:, 2:] - self.vx[:, :-2]) / (2 * self.dy)
        
        return dvdx - dudy
    
    def _compute_divergence(self) -> np.ndarray:
        """
        Compute horizontal divergence ∇·v = ∂u/∂x + ∂v/∂y.
        
        Should be ~0 for incompressible flow.
        """
        dudx = np.zeros_like(self.vx)
        dvdy = np.zeros_like(self.vy)
        
        dudx[1:-1, :] = (self.vx[2:, :] - self.vx[:-2, :]) / (2 * self.dx)
        dvdy[:, 1:-1] = (self.vy[:, 2:] - self.vy[:, :-2]) / (2 * self.dy)
        
        return dudx + dvdy
    
    def velocity_at(self, x: float, y: float) -> Tuple[float, float]:
        """
        Compute velocity at arbitrary point (x, y).
        
        Args:
            x: x-coordinate [m]
            y: y-coordinate [m]
        
        Returns:
            Tuple (u, v) velocity components [m/s]
        """
        return _bell_velocity(x, y, self.basin.Lx, self.basin.Ly, self.basin.U0)
    
    def max_velocity(self) -> float:
        """Return maximum velocity magnitude."""
        speed = np.sqrt(self.vx**2 + self.vy**2)
        return float(np.max(speed))
    
    def check_incompressibility(self) -> float:
        """
        Check divergence-free condition.
        
        Returns:
            Maximum absolute divergence (should be ~0)
        """
        return float(np.max(np.abs(self.divergence)))


@njit(cache=True)
def _bell_velocity(x: float, y: float, Lx: float, Ly: float, U0: float) -> Tuple[float, float]:
    """
    Compute Bell's velocity at point (x, y).
    
    Args:
        x: x-coordinate [m]
        y: y-coordinate [m]
        Lx: Domain width in x [m]
        Ly: Domain width in y [m]
        U0: Maximum velocity [m/s]
    
    Returns:
        Tuple (u, v) velocity components [m/s]
    """
    pi = np.pi
    
    # Normalized coordinates
    x_norm = pi * x / Lx
    y_norm = pi * y / Ly
    
    # Bell's flow
    u = -0.5 * U0 * np.sin(x_norm)**2 * np.sin(2 * y_norm)
    v =  0.5 * U0 * np.sin(y_norm)**2 * np.sin(2 * x_norm)
    
    return u, v


@njit(cache=True, parallel=True)
def compute_velocity_field_numba(
    X: np.ndarray,
    Y: np.ndarray,
    Lx: float,
    Ly: float,
    U0: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute velocity field using Numba parallelization.
    
    Args:
        X: 1D array of x-coordinates
        Y: 1D array of y-coordinates
        Lx: Domain width in x
        Ly: Domain width in y
        U0: Maximum velocity
    
    Returns:
        Tuple of (vx, vy) arrays
    """
    nx = len(X)
    ny = len(Y)
    
    vx = np.zeros((nx, ny), dtype=np.float64)
    vy = np.zeros((nx, ny), dtype=np.float64)
    
    for i in prange(nx):
        for j in range(ny):
            vx[i, j], vy[i, j] = _bell_velocity(X[i], Y[j], Lx, Ly, U0)
    
    return vx, vy
