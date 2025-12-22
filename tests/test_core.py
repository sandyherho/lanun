"""
Comprehensive tests for lanun core functionality.

Run with: pytest tests/ -v
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from lanun import BasinSystem, VelocityField, LagrangianSolver
from lanun import (
    deploy_particles,
    compute_mass_conservation,
    compute_area_perimeter,
    compute_circularity,
    compute_variance_decay,
    compute_center_of_mass,
    compute_reversibility_error,
    compute_all_diagnostics,
)
from lanun.core.particles import (
    ParticleSet,
    locate_particles,
    compute_bilinear_weights,
    interpolate_to_particles,
    project_to_mesh,
    advect_midpoint,
    constrain_particles,
    initialize_gaussian_tracer,
)
from lanun.core.velocity import _bell_velocity, compute_velocity_field_numba
from lanun.io.config_manager import ConfigManager
from lanun.io.data_handler import DataHandler


class TestBasinSystem:
    """Test ocean basin system definition."""
    
    def test_default_parameters(self):
        """Test default parameter values."""
        basin = BasinSystem()
        assert basin.Lx == 50000.0
        assert basin.Ly == 50000.0
        assert basin.H == 100.0
        assert basin.U0 == 0.3
        assert basin.rho0 == 1025.0
    
    def test_custom_parameters(self):
        """Test custom parameter values."""
        basin = BasinSystem(
            Lx=100000.0,
            Ly=80000.0,
            H=200.0,
            U0=0.5,
            tracer_name='Chlorophyll',
            tracer_units='mg/m³'
        )
        assert basin.Lx == 100000.0
        assert basin.Ly == 80000.0
        assert basin.H == 200.0
        assert basin.U0 == 0.5
        assert basin.tracer_name == 'Chlorophyll'
    
    def test_advective_timescale(self):
        """Test advective timescale computation."""
        basin = BasinSystem(Lx=50000.0, U0=0.5)
        expected = 50000.0 / 0.5
        assert basin.T_advect == expected
    
    def test_circulation_timescale(self):
        """Test circulation timescale computation."""
        basin = BasinSystem(Lx=50000.0, U0=0.5)
        expected = np.pi * 50000.0 / 0.5
        assert basin.T_circ == expected
    
    def test_circulation_days(self):
        """Test circulation time in days."""
        basin = BasinSystem(Lx=50000.0, U0=0.3)
        expected_days = basin.T_circ / 86400.0
        assert abs(basin.T_circ_days - expected_days) < 1e-10
    
    def test_aspect_ratio(self):
        """Test aspect ratio computation."""
        basin = BasinSystem(Lx=50000.0, Ly=30000.0)
        assert basin.aspect_ratio == 0.6
    
    def test_repr(self):
        """Test string representation."""
        basin = BasinSystem()
        repr_str = repr(basin)
        assert "BasinSystem" in repr_str
        assert "50.0 km" in repr_str
    
    def test_describe(self):
        """Test detailed description."""
        basin = BasinSystem()
        desc = basin.describe()
        assert "Semi-Enclosed Ocean Basin" in desc
        assert "U₀" in desc


class TestVelocityField:
    """Test Bell's velocity field implementation."""
    
    def test_velocity_field_creation(self, small_basin):
        """Test velocity field initialization."""
        vf = VelocityField(small_basin, nx=21, ny=21)
        
        assert vf.nx == 21
        assert vf.ny == 21
        assert vf.vx.shape == (21, 21)
        assert vf.vy.shape == (21, 21)
    
    def test_velocity_at_boundaries_is_zero(self, small_basin):
        """Test that velocity is zero at boundaries."""
        vf = VelocityField(small_basin, nx=51, ny=51)
        
        # Left and right boundaries
        assert np.allclose(vf.vx[0, :], 0.0, atol=1e-10)
        assert np.allclose(vf.vx[-1, :], 0.0, atol=1e-10)
        assert np.allclose(vf.vy[0, :], 0.0, atol=1e-10)
        assert np.allclose(vf.vy[-1, :], 0.0, atol=1e-10)
        
        # Top and bottom boundaries
        assert np.allclose(vf.vx[:, 0], 0.0, atol=1e-10)
        assert np.allclose(vf.vx[:, -1], 0.0, atol=1e-10)
        assert np.allclose(vf.vy[:, 0], 0.0, atol=1e-10)
        assert np.allclose(vf.vy[:, -1], 0.0, atol=1e-10)
    
    def test_incompressibility(self, small_basin):
        """Test that divergence is near zero (incompressible)."""
        vf = VelocityField(small_basin, nx=51, ny=51)
        
        max_div = vf.check_incompressibility()
        assert max_div < 1e-6
    
    def test_max_velocity(self, small_basin):
        """Test maximum velocity magnitude."""
        vf = VelocityField(small_basin, nx=51, ny=51)
        
        max_v = vf.max_velocity()
        # Maximum should be close to U0/2 (Bell's flow property)
        assert max_v <= small_basin.U0 / 2 + 0.01
        assert max_v > 0
    
    def test_stream_function_exists(self, small_basin):
        """Test stream function computation."""
        vf = VelocityField(small_basin, nx=21, ny=21)
        
        assert vf.psi.shape == (21, 21)
        assert np.isfinite(vf.psi).all()
        
        # Stream function should be zero at boundaries
        assert np.allclose(vf.psi[0, :], 0.0, atol=1e-10)
        assert np.allclose(vf.psi[-1, :], 0.0, atol=1e-10)
    
    def test_vorticity_exists(self, small_basin):
        """Test vorticity computation."""
        vf = VelocityField(small_basin, nx=21, ny=21)
        
        assert vf.vorticity.shape == (21, 21)
        assert np.isfinite(vf.vorticity).all()


class TestBellVelocity:
    """Test the Numba-compiled Bell velocity function."""
    
    def test_velocity_at_center(self):
        """Test velocity at domain center."""
        Lx, Ly, U0 = 1000.0, 1000.0, 1.0
        u, v = _bell_velocity(500.0, 500.0, Lx, Ly, U0)
        
        assert np.isfinite(u)
        assert np.isfinite(v)
    
    def test_velocity_at_origin(self):
        """Test velocity at origin is zero."""
        Lx, Ly, U0 = 1000.0, 1000.0, 1.0
        u, v = _bell_velocity(0.0, 0.0, Lx, Ly, U0)
        
        assert abs(u) < 1e-10
        assert abs(v) < 1e-10
    
    def test_velocity_at_corner(self):
        """Test velocity at corner is zero."""
        Lx, Ly, U0 = 1000.0, 1000.0, 1.0
        u, v = _bell_velocity(Lx, Ly, Lx, Ly, U0)
        
        assert abs(u) < 1e-10
        assert abs(v) < 1e-10
    
    def test_velocity_field_numba(self):
        """Test Numba-parallelized velocity field computation."""
        X = np.linspace(0, 1000, 21)
        Y = np.linspace(0, 1000, 21)
        
        vx, vy = compute_velocity_field_numba(X, Y, 1000.0, 1000.0, 1.0)
        
        assert vx.shape == (21, 21)
        assert vy.shape == (21, 21)
        assert np.isfinite(vx).all()
        assert np.isfinite(vy).all()


class TestParticles:
    """Test particle deployment and operations."""
    
    def test_deploy_particles(self):
        """Test uniform particle deployment."""
        particles = deploy_particles(
            nx=11, ny=11,
            dx=100.0, dy=100.0,
            x_min=0.0, y_min=0.0,
            particles_per_cell_side=2
        )
        
        assert particles.n_particles == 10 * 10 * 4  # (nx-1)*(ny-1)*pps²
        assert len(particles.x) == particles.n_particles
        assert len(particles.y) == particles.n_particles
        assert len(particles.tracer) == particles.n_particles
    
    def test_particle_positions_in_domain(self):
        """Test that particles are within domain."""
        particles = deploy_particles(
            nx=11, ny=11,
            dx=100.0, dy=100.0,
            x_min=0.0, y_min=0.0,
            particles_per_cell_side=2
        )
        
        assert np.all(particles.x >= 0)
        assert np.all(particles.x <= 1000)
        assert np.all(particles.y >= 0)
        assert np.all(particles.y <= 1000)
    
    def test_gaussian_tracer_initialization(self):
        """Test Gaussian tracer initialization."""
        particles = deploy_particles(
            nx=11, ny=11,
            dx=100.0, dy=100.0,
            x_min=0.0, y_min=0.0,
            particles_per_cell_side=2
        )
        
        initialize_gaussian_tracer(
            particles,
            center_x=500.0,
            center_y=500.0,
            sigma=100.0,
            background=0.0,
            amplitude=1.0
        )
        
        # Maximum should be near center
        max_idx = np.argmax(particles.tracer)
        assert abs(particles.x[max_idx] - 500.0) < 100.0
        assert abs(particles.y[max_idx] - 500.0) < 100.0
        
        # Maximum value should be close to amplitude
        assert particles.tracer[max_idx] <= 1.0 + 0.01
    
    def test_particle_copy(self):
        """Test particle set deep copy."""
        particles = deploy_particles(
            nx=11, ny=11,
            dx=100.0, dy=100.0,
            x_min=0.0, y_min=0.0,
            particles_per_cell_side=2
        )
        
        particles_copy = particles.copy()
        
        # Modify original
        particles.x[0] = -999.0
        
        # Copy should be unchanged
        assert particles_copy.x[0] != -999.0


class TestParticleOperations:
    """Test particle location and interpolation."""
    
    def test_locate_particles(self):
        """Test particle cell location."""
        px = np.array([50.0, 150.0, 250.0])
        py = np.array([50.0, 150.0, 250.0])
        
        cell_i, cell_j = locate_particles(
            px, py,
            x_min=0.0, y_min=0.0,
            dx=100.0, dy=100.0,
            nx=11, ny=11
        )
        
        assert np.array_equal(cell_i, np.array([0, 1, 2]))
        assert np.array_equal(cell_j, np.array([0, 1, 2]))
    
    def test_bilinear_weights_sum_to_one(self):
        """Test partition of unity for bilinear weights."""
        px = np.array([50.0, 75.0, 25.0])
        py = np.array([50.0, 75.0, 25.0])
        X = np.linspace(0, 1000, 11)
        Y = np.linspace(0, 1000, 11)
        
        cell_i, cell_j = locate_particles(
            px, py, 0.0, 0.0, 100.0, 100.0, 11, 11
        )
        
        w1, w2, w3, w4 = compute_bilinear_weights(
            px, py, X, Y, cell_i, cell_j, 100.0, 100.0
        )
        
        weight_sum = w1 + w2 + w3 + w4
        assert np.allclose(weight_sum, 1.0, atol=1e-10)
    
    def test_interpolate_to_particles(self):
        """Test field interpolation to particles."""
        # Create a simple linear field
        field = np.zeros((11, 11))
        for i in range(11):
            for j in range(11):
                field[i, j] = i + j  # Linear gradient
        
        px = np.array([50.0, 150.0])
        py = np.array([50.0, 150.0])
        X = np.linspace(0, 1000, 11)
        Y = np.linspace(0, 1000, 11)
        
        cell_i, cell_j = locate_particles(px, py, 0.0, 0.0, 100.0, 100.0, 11, 11)
        w1, w2, w3, w4 = compute_bilinear_weights(px, py, X, Y, cell_i, cell_j, 100.0, 100.0)
        
        values = interpolate_to_particles(field, cell_i, cell_j, w1, w2, w3, w4)
        
        assert len(values) == 2
        assert np.isfinite(values).all()
    
    def test_project_to_mesh(self):
        """Test particle to mesh projection."""
        tracer_p = np.array([1.0, 2.0, 3.0, 4.0])
        cell_i = np.array([0, 0, 1, 1], dtype=np.int32)
        cell_j = np.array([0, 1, 0, 1], dtype=np.int32)
        w1 = np.array([0.25, 0.25, 0.25, 0.25])
        w2 = np.array([0.25, 0.25, 0.25, 0.25])
        w3 = np.array([0.25, 0.25, 0.25, 0.25])
        w4 = np.array([0.25, 0.25, 0.25, 0.25])
        
        field = project_to_mesh(tracer_p, cell_i, cell_j, w1, w2, w3, w4, 5, 5)
        
        assert field.shape == (5, 5)
        assert np.isfinite(field).all()


class TestAdvection:
    """Test particle advection schemes."""
    
    def test_advect_midpoint_stationary(self):
        """Test that particles don't move in zero velocity."""
        px = np.array([500.0, 500.0])
        py = np.array([500.0, 500.0])
        vx = np.zeros((11, 11))
        vy = np.zeros((11, 11))
        X = np.linspace(0, 1000, 11)
        Y = np.linspace(0, 1000, 11)
        
        new_px, new_py = advect_midpoint(
            px, py, vx, vy, X, Y,
            dt=60.0,
            x_min=0.0, x_max=1000.0,
            y_min=0.0, y_max=1000.0,
            dx=100.0, dy=100.0,
            nx=11, ny=11,
            direction=1.0
        )
        
        assert np.allclose(new_px, px, atol=1e-8)
        assert np.allclose(new_py, py, atol=1e-8)
    
    def test_advect_midpoint_stays_in_domain(self, small_basin):
        """Test that advected particles stay in domain."""
        vf = VelocityField(small_basin, nx=21, ny=21)
        
        # Random particles
        np.random.seed(42)
        px = np.random.uniform(100, small_basin.Lx - 100, 100)
        py = np.random.uniform(100, small_basin.Ly - 100, 100)
        
        new_px, new_py = advect_midpoint(
            px, py, vf.vx, vf.vy, vf.X, vf.Y,
            dt=60.0,
            x_min=0.0, x_max=small_basin.Lx,
            y_min=0.0, y_max=small_basin.Ly,
            dx=vf.dx, dy=vf.dy,
            nx=vf.nx, ny=vf.ny,
            direction=1.0
        )
        
        assert np.all(new_px >= 0)
        assert np.all(new_px <= small_basin.Lx)
        assert np.all(new_py >= 0)
        assert np.all(new_py <= small_basin.Ly)
    
    def test_advect_reverse_direction(self, small_basin):
        """Test reverse advection."""
        vf = VelocityField(small_basin, nx=21, ny=21)
        
        px = np.array([small_basin.Lx / 2])
        py = np.array([small_basin.Ly / 2])
        
        # Forward step
        px1, py1 = advect_midpoint(
            px, py, vf.vx, vf.vy, vf.X, vf.Y,
            dt=60.0,
            x_min=0.0, x_max=small_basin.Lx,
            y_min=0.0, y_max=small_basin.Ly,
            dx=vf.dx, dy=vf.dy,
            nx=vf.nx, ny=vf.ny,
            direction=1.0
        )
        
        # Backward step
        px2, py2 = advect_midpoint(
            px1, py1, vf.vx, vf.vy, vf.X, vf.Y,
            dt=60.0,
            x_min=0.0, x_max=small_basin.Lx,
            y_min=0.0, y_max=small_basin.Ly,
            dx=vf.dx, dy=vf.dy,
            nx=vf.nx, ny=vf.ny,
            direction=-1.0
        )
        
        # Should return close to original (within numerical error)
        assert np.allclose(px2, px, rtol=0.1)
        assert np.allclose(py2, py, rtol=0.1)
    
    def test_constrain_particles(self):
        """Test particle boundary constraints."""
        px = np.array([-10.0, 500.0, 1010.0])
        py = np.array([500.0, -10.0, 500.0])
        
        new_px, new_py = constrain_particles(
            px, py,
            x_min=0.0, x_max=1000.0,
            y_min=0.0, y_max=1000.0,
            dx=100.0, dy=100.0,
            pps=4
        )
        
        assert np.all(new_px >= 0)
        assert np.all(new_px <= 1000)
        assert np.all(new_py >= 0)
        assert np.all(new_py <= 1000)


class TestLagrangianSolver:
    """Test the main Lagrangian solver."""
    
    def test_solver_initialization(self):
        """Test solver initialization."""
        solver = LagrangianSolver(nx=51, ny=51, particles_per_cell=4)
        
        assert solver.nx == 51
        assert solver.ny == 51
        assert solver.particles_per_cell == 4
    
    def test_basic_solve(self, small_basin, small_solver):
        """Test basic simulation run."""
        result = small_solver.solve(
            basin=small_basin,
            total_time=600.0,  # 10 minutes
            dt=60.0,           # 1 minute
            output_interval=2,
            verbose=False
        )
        
        assert 'time' in result.__dict__
        assert 'particle_x' in result.__dict__
        assert 'particle_y' in result.__dict__
        assert 'particle_tracer' in result.__dict__
        assert 'mesh_tracer' in result.__dict__
    
    def test_result_shapes(self, small_basin, small_solver):
        """Test output array shapes."""
        result = small_solver.solve(
            basin=small_basin,
            total_time=600.0,
            dt=60.0,
            output_interval=2,
            verbose=False
        )
        
        n_outputs = len(result.time)
        n_particles = result.particle_x.shape[1]
        
        assert result.particle_x.shape == (n_outputs, n_particles)
        assert result.particle_y.shape == (n_outputs, n_particles)
        assert result.particle_tracer.shape == (n_outputs, n_particles)
        assert result.mesh_tracer.shape == (n_outputs, small_solver.nx, small_solver.ny)
    
    def test_particles_in_domain(self, small_basin, small_solver):
        """Test that particles stay in domain."""
        result = small_solver.solve(
            basin=small_basin,
            total_time=600.0,
            dt=60.0,
            output_interval=2,
            verbose=False
        )
        
        assert np.all(result.particle_x >= 0)
        assert np.all(result.particle_x <= small_basin.Lx)
        assert np.all(result.particle_y >= 0)
        assert np.all(result.particle_y <= small_basin.Ly)
    
    def test_tracer_conservation(self, small_basin, small_solver):
        """Test tracer mass conservation."""
        result = small_solver.solve(
            basin=small_basin,
            total_time=600.0,
            dt=60.0,
            output_interval=2,
            verbose=False
        )
        
        # Tracer values should not change (passive advection)
        initial_tracer = result.particle_tracer[0, :]
        final_tracer = result.particle_tracer[-1, :]
        
        assert np.allclose(initial_tracer, final_tracer, atol=1e-10)


class TestDiagnostics:
    """Test Lagrangian diagnostic computations."""
    
    def test_mass_conservation(self):
        """Test mass conservation computation."""
        tracer_init = np.array([1.0, 2.0, 3.0, 4.0])
        tracer_final = np.array([1.0, 2.0, 3.0, 4.0])
        
        metrics = compute_mass_conservation(
            tracer_init, tracer_final,
            dx=100.0, dy=100.0,
            particles_per_cell=2
        )
        
        assert 'mass_initial' in metrics
        assert 'mass_final' in metrics
        assert 'mass_error_relative' in metrics
        assert abs(metrics['mass_error_relative']) < 1e-10
    
    def test_area_perimeter(self):
        """Test area and perimeter computation."""
        # Square boundary
        bx = np.array([0.0, 100.0, 100.0, 0.0])
        by = np.array([0.0, 0.0, 100.0, 100.0])
        
        metrics = compute_area_perimeter(bx, by)
        
        assert abs(metrics['area'] - 10000.0) < 1.0
        assert abs(metrics['perimeter'] - 400.0) < 1.0
    
    def test_circularity(self):
        """Test circularity computation."""
        # Perfect circle: C = 1
        # Square: C ≈ π/4 ≈ 0.785
        area = 100.0
        perimeter = 40.0  # Square with side 10
        
        c = compute_circularity(area, perimeter)
        
        expected = 4 * np.pi * 100.0 / (40.0 ** 2)
        assert abs(c - expected) < 1e-10
    
    def test_variance_decay(self):
        """Test variance decay computation."""
        tracer_init = np.array([0.0, 1.0, 2.0, 3.0])
        tracer_current = np.array([0.5, 1.0, 1.5, 2.0])  # More uniform
        
        metrics = compute_variance_decay(tracer_init, tracer_current)
        
        assert 'variance_initial' in metrics
        assert 'variance_current' in metrics
        assert 'intensity_segregation' in metrics
        assert 'mixing_efficiency' in metrics
        
        # Current variance should be less
        assert metrics['intensity_segregation'] < 1.0
    
    def test_center_of_mass(self):
        """Test center of mass computation."""
        x = np.array([0.0, 100.0, 0.0, 100.0])
        y = np.array([0.0, 0.0, 100.0, 100.0])
        tracer = np.array([1.0, 1.0, 1.0, 1.0])  # Uniform
        
        metrics = compute_center_of_mass(x, y, tracer)
        
        assert abs(metrics['x_cm'] - 50.0) < 1e-10
        assert abs(metrics['y_cm'] - 50.0) < 1e-10
    
    def test_reversibility_error(self):
        """Test reversibility error computation."""
        init_x = np.array([0.0, 100.0])
        init_y = np.array([0.0, 100.0])
        final_x = np.array([1.0, 101.0])  # Small displacement
        final_y = np.array([1.0, 101.0])
        
        metrics = compute_reversibility_error(
            init_x, init_y, final_x, final_y, L=1000.0
        )
        
        assert 'reversibility_mean' in metrics
        assert 'reversibility_normalized' in metrics
        assert metrics['reversibility_mean'] > 0
    
    def test_all_diagnostics(self, small_basin, small_solver):
        """Test comprehensive diagnostics computation."""
        result = small_solver.solve(
            basin=small_basin,
            total_time=600.0,
            dt=60.0,
            output_interval=2,
            verbose=False
        )
        
        diagnostics = compute_all_diagnostics(result, verbose=False)
        
        # Check key metrics exist
        assert 'mass_error_relative' in diagnostics
        assert 'intensity_segregation' in diagnostics
        assert 'mean_displacement' in diagnostics


class TestConfigManager:
    """Test configuration file handling."""
    
    def test_load_config(self):
        """Test loading configuration from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("# Test config\n")
            f.write("Lx = 50000.0\n")
            f.write("Ly = 50000.0\n")
            f.write("U0 = 0.3\n")
            f.write("nx = 101\n")
            f.write("save_gif = false\n")
            f.write("scenario_name = Test\n")
            config_path = f.name
        
        config = ConfigManager.load(config_path)
        
        assert config['Lx'] == 50000.0
        assert config['U0'] == 0.3
        assert config['nx'] == 101
        assert config['save_gif'] == False
        assert config['scenario_name'] == 'Test'
        
        Path(config_path).unlink()
    
    def test_default_config(self):
        """Test default configuration."""
        config = ConfigManager.get_default_config('case1')
        
        assert 'Lx' in config
        assert 'U0' in config
        assert 'nx' in config
        assert config['Lx'] == 50000.0
    
    def test_all_default_configs(self):
        """Test all 4 default configurations."""
        for case in ['case1', 'case2', 'case3', 'case4']:
            config = ConfigManager.get_default_config(case)
            
            assert 'Lx' in config
            assert 'U0' in config
            assert 'tracer_name' in config
            assert config['Lx'] > 0
            assert config['U0'] > 0
    
    def test_save_config(self):
        """Test saving configuration to file."""
        config = {'Lx': 50000.0, 'U0': 0.3, 'save_gif': True}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            config_path = f.name
        
        ConfigManager.save(config, config_path)
        
        loaded = ConfigManager.load(config_path)
        assert loaded['Lx'] == 50000.0
        assert loaded['save_gif'] == True
        
        Path(config_path).unlink()
    
    def test_validate_config(self):
        """Test configuration validation."""
        valid_config = {
            'Lx': 50000.0,
            'Ly': 50000.0,
            'U0': 0.3,
            'nx': 101,
            'ny': 101,
            'dt': 300.0
        }
        
        assert ConfigManager.validate_config(valid_config) == True
    
    def test_validate_config_missing_param(self):
        """Test validation fails with missing parameter."""
        invalid_config = {'Lx': 50000.0}  # Missing required params
        
        with pytest.raises(ValueError):
            ConfigManager.validate_config(invalid_config)


class TestDataHandler:
    """Test data saving functionality."""
    
    def test_save_diagnostics_csv(self):
        """Test diagnostics CSV saving."""
        diagnostics = {
            'mass_error_relative': 1e-10,
            'intensity_segregation': 0.95,
            'stretching_factor': 2.5
        }
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            filepath = f.name
        
        DataHandler.save_diagnostics_csv(filepath, diagnostics)
        
        assert Path(filepath).exists()
        
        import pandas as pd
        df = pd.read_csv(filepath)
        assert 'Metric' in df.columns
        assert 'Value' in df.columns
        
        Path(filepath).unlink()
    
    def test_save_netcdf(self, small_basin, small_solver):
        """Test NetCDF saving."""
        result = small_solver.solve(
            basin=small_basin,
            total_time=600.0,
            dt=60.0,
            output_interval=2,
            verbose=False
        )
        
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
            filepath = f.name
        
        config = {'scenario_name': 'Test'}
        DataHandler.save_netcdf(filepath, result, config)
        
        assert Path(filepath).exists()
        
        # Check content
        from netCDF4 import Dataset
        with Dataset(filepath, 'r') as nc:
            assert 'time' in nc.variables
            assert 'x' in nc.variables
            assert 'y' in nc.variables
            assert 'depth' in nc.variables
            assert 'tracer' in nc.variables
            
            # Check oceanographic z convention
            depth_var = nc.variables['depth']
            assert depth_var.positive == 'down'
        
        Path(filepath).unlink()


class TestNumericalAccuracy:
    """Test numerical accuracy and conservation properties."""
    
    def test_cfl_condition(self, small_basin):
        """Test CFL number computation."""
        vf = VelocityField(small_basin, nx=21, ny=21)
        max_v = vf.max_velocity()
        
        dt = 60.0
        dx = min(vf.dx, vf.dy)
        cfl = max_v * dt / dx
        
        # Should be less than 1 for stability
        assert cfl < 1.0 or max_v < 1e-10
    
    def test_long_integration_stability(self, small_basin):
        """Test stability over longer integration."""
        solver = LagrangianSolver(nx=21, ny=21, particles_per_cell=2)
        
        result = solver.solve(
            basin=small_basin,
            total_time=3600.0,  # 1 hour
            dt=60.0,
            output_interval=10,
            verbose=False
        )
        
        # All values should be finite
        assert np.isfinite(result.particle_x).all()
        assert np.isfinite(result.particle_y).all()
        assert np.isfinite(result.particle_tracer).all()
        assert np.isfinite(result.mesh_tracer).all()
    
    def test_tracer_bounds(self, small_basin):
        """Test that tracer stays within physical bounds."""
        solver = LagrangianSolver(nx=21, ny=21, particles_per_cell=2)
        
        result = solver.solve(
            basin=small_basin,
            total_time=600.0,
            dt=60.0,
            output_interval=2,
            verbose=False
        )
        
        # Tracer should stay within initial range (passive advection)
        init_min = result.particle_tracer[0, :].min()
        init_max = result.particle_tracer[0, :].max()
        
        assert result.particle_tracer.min() >= init_min - 1e-10
        assert result.particle_tracer.max() <= init_max + 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
