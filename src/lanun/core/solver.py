"""
Lagrangian Transport Solver for Semi-Enclosed Ocean Basins.

Main solver class that coordinates:
    - Velocity field initialization
    - Particle deployment and advection
    - Mesh-particle projections
    - Time stepping with progress tracking
    - Output data collection
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from tqdm import tqdm

from .velocity import BasinSystem, VelocityField
from .particles import (
    ParticleSet,
    deploy_particles,
    initialize_gaussian_tracer,
    locate_particles,
    compute_bilinear_weights,
    interpolate_to_particles,
    project_to_mesh,
    advect_midpoint,
    constrain_particles,
)


@dataclass
class SimulationResult:
    """
    Container for simulation output data.
    
    Attributes:
        time: Time array [s]
        particle_x: Particle x-positions over time
        particle_y: Particle y-positions over time
        particle_tracer: Tracer values on particles over time
        mesh_tracer: Tracer field on mesh over time
        basin: Basin configuration
        velocity_field: Velocity field object
        initial_particles: Initial particle positions
        final_particles: Final particle positions
        diagnostics: Dictionary of computed diagnostics
    """
    time: np.ndarray
    particle_x: np.ndarray
    particle_y: np.ndarray
    particle_tracer: np.ndarray
    mesh_tracer: np.ndarray
    basin: BasinSystem
    velocity_field: VelocityField
    initial_particles: ParticleSet
    final_particles: ParticleSet
    diagnostics: Dict[str, Any]
    config: Dict[str, Any]


class LagrangianSolver:
    """
    Lagrangian particle transport solver.
    
    Solves passive tracer transport using the Particle-in-Cell method
    with Bell's incompressible velocity field.
    
    Attributes:
        nx: Number of mesh points in x
        ny: Number of mesh points in y
        particles_per_cell: Particles per cell side (total = pps²)
    
    Example:
        >>> solver = LagrangianSolver(nx=101, ny=101, particles_per_cell=4)
        >>> basin = BasinSystem(Lx=50e3, Ly=50e3, U0=0.3)
        >>> result = solver.solve(basin, total_time=7*86400, dt=300)
    """
    
    def __init__(
        self,
        nx: int = 101,
        ny: int = 101,
        particles_per_cell: int = 4
    ):
        """
        Initialize solver.
        
        Args:
            nx: Number of mesh points in x-direction
            ny: Number of mesh points in y-direction
            particles_per_cell: Number of particles per cell side
        """
        self.nx = nx
        self.ny = ny
        self.particles_per_cell = particles_per_cell
        
        # Will be set during solve()
        self.velocity_field: Optional[VelocityField] = None
        self.particles: Optional[ParticleSet] = None
    
    def solve(
        self,
        basin: BasinSystem,
        total_time: float,
        dt: float,
        output_interval: int = 10,
        tracer_center_x: Optional[float] = None,
        tracer_center_y: Optional[float] = None,
        tracer_sigma: Optional[float] = None,
        reverse_flow: bool = False,
        verbose: bool = True
    ) -> SimulationResult:
        """
        Run Lagrangian transport simulation.
        
        Args:
            basin: BasinSystem configuration
            total_time: Total simulation time [s]
            dt: Time step [s]
            output_interval: Save output every N steps
            tracer_center_x: Initial tracer center x [m] (default: Lx/4)
            tracer_center_y: Initial tracer center y [m] (default: Ly/2)
            tracer_sigma: Initial tracer width [m] (default: Lx/10)
            reverse_flow: If True, reverse the flow direction
            verbose: Print progress information
        
        Returns:
            SimulationResult with all output data
        """
        # Initialize velocity field
        if verbose:
            print(f"[1/5] Initializing velocity field ({self.nx}×{self.ny} mesh)...")
        
        self.velocity_field = VelocityField(basin, self.nx, self.ny)
        
        # Check CFL condition
        max_v = self.velocity_field.max_velocity()
        cfl = max_v * dt / min(self.velocity_field.dx, self.velocity_field.dy)
        
        if verbose:
            print(f"      Max velocity: {max_v:.4f} m/s")
            print(f"      CFL number: {cfl:.3f}")
            if cfl > 0.5:
                print(f"      WARNING: CFL > 0.5, consider reducing dt")
        
        # Deploy particles
        if verbose:
            print(f"[2/5] Deploying particles ({self.particles_per_cell}² per cell)...")
        
        self.particles = deploy_particles(
            nx=self.nx,
            ny=self.ny,
            dx=self.velocity_field.dx,
            dy=self.velocity_field.dy,
            x_min=0.0,
            y_min=0.0,
            particles_per_cell_side=self.particles_per_cell
        )
        
        if verbose:
            print(f"      Total particles: {self.particles.n_particles:,}")
        
        # Initialize tracer
        if verbose:
            print("[3/5] Initializing tracer distribution...")
        
        if tracer_center_x is None:
            tracer_center_x = basin.Lx / 4
        if tracer_center_y is None:
            tracer_center_y = basin.Ly / 2
        if tracer_sigma is None:
            tracer_sigma = basin.Lx / 10
        
        initialize_gaussian_tracer(
            self.particles,
            center_x=tracer_center_x,
            center_y=tracer_center_y,
            sigma=tracer_sigma,
            background=basin.tracer_background,
            amplitude=basin.tracer_anomaly
        )
        
        # Store initial state
        initial_particles = self.particles.copy()
        
        # Prepare output arrays
        n_steps = int(total_time / dt)
        n_outputs = n_steps // output_interval + 1
        
        time_out = np.zeros(n_outputs, dtype=np.float64)
        px_out = np.zeros((n_outputs, self.particles.n_particles), dtype=np.float64)
        py_out = np.zeros((n_outputs, self.particles.n_particles), dtype=np.float64)
        tracer_p_out = np.zeros((n_outputs, self.particles.n_particles), dtype=np.float64)
        tracer_m_out = np.zeros((n_outputs, self.nx, self.ny), dtype=np.float64)
        
        # Store initial state
        time_out[0] = 0.0
        px_out[0, :] = self.particles.x
        py_out[0, :] = self.particles.y
        tracer_p_out[0, :] = self.particles.tracer
        
        # Project initial tracer to mesh
        self._update_particle_locations()
        tracer_m_out[0, :, :] = project_to_mesh(
            self.particles.tracer,
            self.particles.cell_i,
            self.particles.cell_j,
            self.particles.w1,
            self.particles.w2,
            self.particles.w3,
            self.particles.w4,
            self.nx,
            self.ny
        )
        
        # Time integration
        if verbose:
            print(f"[4/5] Running advection ({n_steps} steps, {total_time/86400:.2f} days)...")
        
        direction = -1.0 if reverse_flow else 1.0
        output_idx = 1
        
        iterator = tqdm(
            range(1, n_steps + 1),
            desc="      Advecting",
            disable=not verbose,
            ncols=70,
            unit="step"
        )
        
        for step in iterator:
            # Advect particles
            new_px, new_py = advect_midpoint(
                self.particles.x,
                self.particles.y,
                self.velocity_field.vx,
                self.velocity_field.vy,
                self.velocity_field.X,
                self.velocity_field.Y,
                dt,
                0.0,
                basin.Lx,
                0.0,
                basin.Ly,
                self.velocity_field.dx,
                self.velocity_field.dy,
                self.nx,
                self.ny,
                direction
            )
            
            # Constrain to domain
            self.particles.x, self.particles.y = constrain_particles(
                new_px,
                new_py,
                0.0,
                basin.Lx,
                0.0,
                basin.Ly,
                self.velocity_field.dx,
                self.velocity_field.dy,
                self.particles_per_cell
            )
            
            # Save output
            if step % output_interval == 0:
                current_time = step * dt
                time_out[output_idx] = current_time
                px_out[output_idx, :] = self.particles.x
                py_out[output_idx, :] = self.particles.y
                tracer_p_out[output_idx, :] = self.particles.tracer
                
                # Project to mesh
                self._update_particle_locations()
                tracer_m_out[output_idx, :, :] = project_to_mesh(
                    self.particles.tracer,
                    self.particles.cell_i,
                    self.particles.cell_j,
                    self.particles.w1,
                    self.particles.w2,
                    self.particles.w3,
                    self.particles.w4,
                    self.nx,
                    self.ny
                )
                
                output_idx += 1
        
        # Store final state
        final_particles = self.particles.copy()
        
        # Build config dictionary
        config = {
            'nx': self.nx,
            'ny': self.ny,
            'particles_per_cell': self.particles_per_cell,
            'total_time': total_time,
            'dt': dt,
            'output_interval': output_interval,
            'tracer_center_x': tracer_center_x,
            'tracer_center_y': tracer_center_y,
            'tracer_sigma': tracer_sigma,
            'reverse_flow': reverse_flow,
            'n_steps': n_steps,
            'n_outputs': output_idx,
            'cfl': cfl,
        }
        
        if verbose:
            print(f"[5/5] Simulation complete!")
            print(f"      Outputs saved: {output_idx}")
        
        return SimulationResult(
            time=time_out[:output_idx],
            particle_x=px_out[:output_idx],
            particle_y=py_out[:output_idx],
            particle_tracer=tracer_p_out[:output_idx],
            mesh_tracer=tracer_m_out[:output_idx],
            basin=basin,
            velocity_field=self.velocity_field,
            initial_particles=initial_particles,
            final_particles=final_particles,
            diagnostics={},
            config=config
        )
    
    def solve_with_reversal(
        self,
        basin: BasinSystem,
        total_time: float,
        dt: float,
        output_interval: int = 10,
        verbose: bool = True,
        **kwargs
    ) -> Tuple[SimulationResult, SimulationResult]:
        """
        Run forward simulation then reverse to test reversibility.
        
        Args:
            basin: BasinSystem configuration
            total_time: Total time for EACH direction [s]
            dt: Time step [s]
            output_interval: Save output every N steps
            verbose: Print progress
            **kwargs: Additional arguments for solve()
        
        Returns:
            Tuple of (forward_result, reverse_result)
        """
        if verbose:
            print("=" * 60)
            print("FORWARD ADVECTION")
            print("=" * 60)
        
        forward_result = self.solve(
            basin=basin,
            total_time=total_time,
            dt=dt,
            output_interval=output_interval,
            reverse_flow=False,
            verbose=verbose,
            **kwargs
        )
        
        if verbose:
            print("\n" + "=" * 60)
            print("REVERSE ADVECTION")
            print("=" * 60)
        
        # Start reverse from final forward position
        self.particles = forward_result.final_particles.copy()
        
        # We need to solve again but with particles at final positions
        # and reversed flow
        reverse_result = self._solve_reverse(
            basin=basin,
            total_time=total_time,
            dt=dt,
            output_interval=output_interval,
            initial_particles=forward_result.final_particles,
            original_initial=forward_result.initial_particles,
            verbose=verbose
        )
        
        return forward_result, reverse_result
    
    def _solve_reverse(
        self,
        basin: BasinSystem,
        total_time: float,
        dt: float,
        output_interval: int,
        initial_particles: ParticleSet,
        original_initial: ParticleSet,
        verbose: bool
    ) -> SimulationResult:
        """Internal method for reverse advection."""
        # Use existing velocity field
        n_steps = int(total_time / dt)
        n_outputs = n_steps // output_interval + 1
        
        # Reset particles to end-of-forward positions
        self.particles = initial_particles.copy()
        
        time_out = np.zeros(n_outputs, dtype=np.float64)
        px_out = np.zeros((n_outputs, self.particles.n_particles), dtype=np.float64)
        py_out = np.zeros((n_outputs, self.particles.n_particles), dtype=np.float64)
        tracer_p_out = np.zeros((n_outputs, self.particles.n_particles), dtype=np.float64)
        tracer_m_out = np.zeros((n_outputs, self.nx, self.ny), dtype=np.float64)
        
        # Store initial (which is end of forward)
        time_out[0] = 0.0
        px_out[0, :] = self.particles.x
        py_out[0, :] = self.particles.y
        tracer_p_out[0, :] = self.particles.tracer
        
        self._update_particle_locations()
        tracer_m_out[0, :, :] = project_to_mesh(
            self.particles.tracer,
            self.particles.cell_i,
            self.particles.cell_j,
            self.particles.w1,
            self.particles.w2,
            self.particles.w3,
            self.particles.w4,
            self.nx,
            self.ny
        )
        
        output_idx = 1
        
        iterator = tqdm(
            range(1, n_steps + 1),
            desc="      Reversing",
            disable=not verbose,
            ncols=70,
            unit="step"
        )
        
        for step in iterator:
            new_px, new_py = advect_midpoint(
                self.particles.x,
                self.particles.y,
                self.velocity_field.vx,
                self.velocity_field.vy,
                self.velocity_field.X,
                self.velocity_field.Y,
                dt,
                0.0,
                basin.Lx,
                0.0,
                basin.Ly,
                self.velocity_field.dx,
                self.velocity_field.dy,
                self.nx,
                self.ny,
                -1.0  # Reverse direction
            )
            
            self.particles.x, self.particles.y = constrain_particles(
                new_px,
                new_py,
                0.0,
                basin.Lx,
                0.0,
                basin.Ly,
                self.velocity_field.dx,
                self.velocity_field.dy,
                self.particles_per_cell
            )
            
            if step % output_interval == 0:
                current_time = step * dt
                time_out[output_idx] = current_time
                px_out[output_idx, :] = self.particles.x
                py_out[output_idx, :] = self.particles.y
                tracer_p_out[output_idx, :] = self.particles.tracer
                
                self._update_particle_locations()
                tracer_m_out[output_idx, :, :] = project_to_mesh(
                    self.particles.tracer,
                    self.particles.cell_i,
                    self.particles.cell_j,
                    self.particles.w1,
                    self.particles.w2,
                    self.particles.w3,
                    self.particles.w4,
                    self.nx,
                    self.ny
                )
                
                output_idx += 1
        
        final_particles = self.particles.copy()
        
        config = {
            'nx': self.nx,
            'ny': self.ny,
            'particles_per_cell': self.particles_per_cell,
            'total_time': total_time,
            'dt': dt,
            'output_interval': output_interval,
            'reverse_flow': True,
            'n_steps': n_steps,
            'n_outputs': output_idx,
        }
        
        return SimulationResult(
            time=time_out[:output_idx],
            particle_x=px_out[:output_idx],
            particle_y=py_out[:output_idx],
            particle_tracer=tracer_p_out[:output_idx],
            mesh_tracer=tracer_m_out[:output_idx],
            basin=basin,
            velocity_field=self.velocity_field,
            initial_particles=initial_particles,
            final_particles=final_particles,
            diagnostics={},
            config=config
        )
    
    def _update_particle_locations(self) -> None:
        """Update cell indices and weights for all particles."""
        self.particles.cell_i, self.particles.cell_j = locate_particles(
            self.particles.x,
            self.particles.y,
            0.0,
            0.0,
            self.velocity_field.dx,
            self.velocity_field.dy,
            self.nx,
            self.ny
        )
        
        (
            self.particles.w1,
            self.particles.w2,
            self.particles.w3,
            self.particles.w4
        ) = compute_bilinear_weights(
            self.particles.x,
            self.particles.y,
            self.velocity_field.X,
            self.velocity_field.Y,
            self.particles.cell_i,
            self.particles.cell_j,
            self.velocity_field.dx,
            self.velocity_field.dy
        )
