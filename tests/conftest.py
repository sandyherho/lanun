"""Pytest configuration and fixtures for lanun tests."""

import pytest
import numpy as np


@pytest.fixture(scope="session")
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42


@pytest.fixture
def default_basin_params():
    """Default basin parameters for testing."""
    return {
        'Lx': 50000.0,
        'Ly': 50000.0,
        'H': 100.0,
        'U0': 0.3,
        'tracer_name': 'Test Tracer',
        'tracer_units': 'kg/m³',
        'tracer_background': 0.0,
        'tracer_anomaly': 1.0,
    }


@pytest.fixture
def default_solver_params():
    """Default solver parameters for testing."""
    return {
        'nx': 51,
        'ny': 51,
        'particles_per_cell': 4,
    }


@pytest.fixture
def default_integration_params():
    """Default integration parameters for testing."""
    return {
        'total_time': 3600.0,  # 1 hour
        'dt': 60.0,            # 1 minute
        'output_interval': 10,
    }


@pytest.fixture
def small_basin():
    """Create a small basin for quick tests."""
    from lanun import BasinSystem
    return BasinSystem(
        Lx=10000.0,
        Ly=10000.0,
        H=50.0,
        U0=0.1,
        tracer_name='Test',
        tracer_units='kg/m³',
        tracer_background=0.0,
        tracer_anomaly=1.0
    )


@pytest.fixture
def small_solver():
    """Create a small solver for quick tests."""
    from lanun import LagrangianSolver
    return LagrangianSolver(nx=21, ny=21, particles_per_cell=2)
