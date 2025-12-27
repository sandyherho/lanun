# `lanun`: Numba-Accelerated Lagrangian Transport for Idealized Ocean Basins

[![CI](https://github.com/sandyherho/lanun/actions/workflows/ci.yml/badge.svg)](https://github.com/sandyherho/lanun/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![NumPy](https://img.shields.io/badge/NumPy-%23013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?logo=scipy&logoColor=white)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?logo=Matplotlib&logoColor=black)](https://matplotlib.org/)
[![Numba](https://img.shields.io/badge/Numba-%2300A3E0.svg)](https://numba.pydata.org/)
[![netCDF4](https://img.shields.io/badge/netCDF4-%23004B87.svg)](https://unidata.github.io/netcdf4-python/)

Numba-accelerated Python library for simulating Lagrangian particle transport in idealized semi-enclosed ocean basins using Bell's incompressible flow.

> *This library is named after the **Lanun** (also known as Iranun or Illanun), legendary seafarers and maritime warriors of the Sulu and Celebes Seas in Southeast Asia. From the 16th to 19th centuries, they dominated maritime trade routes across the Indonesian archipelago, known for their exceptional navigation skills and understanding of ocean currents. This library honors their legacy as masters of the sea.*

## Physical Model

The library simulates passive tracer transport (nutrients, phytoplankton, sediments) in a semi-enclosed basin using Bell's incompressible thinning flow:

$$u = -\frac{U_0}{2} \sin^2\left(\frac{\pi x}{L}\right) \sin\left(\frac{2\pi y}{L}\right)$$

$$v = \frac{U_0}{2} \sin^2\left(\frac{\pi y}{L}\right) \sin\left(\frac{2\pi x}{L}\right)$$

**Properties:**
- Incompressible: $\nabla \cdot \mathbf{v} = 0$ (mass conserved exactly)
- Zero velocity at boundaries (coastal no-slip condition)
- Maximum stirring at intermediate distances (chaotic advection)
- Stream function: $\psi = \frac{U_0 L}{4\pi} \sin^2(\pi x/L) \sin^2(\pi y/L)$

## Installation

```bash
pip install lanun                    # From PyPI (when published)
pip install poetry && poetry install # Development mode
```

## Quick Start

**CLI:**
```bash
lanun case1                    # Coastal Embayment
lanun case2                    # Marginal Sea
lanun case3                    # Volcanic Lake  
lanun case4                    # Estuary Plume
lanun --all                    # Run all 4 cases
```

**Python API:**
```python
from lanun import BasinSystem, LagrangianSolver, compute_all_diagnostics

# Define basin (50 km coastal embayment)
basin = BasinSystem(
    Lx=50e3, Ly=50e3,           # [m] domain size
    H=100.0,                     # [m] depth
    U0=0.3,                      # [m/s] max velocity
    tracer_name="Chlorophyll-a",
    tracer_units="mg/m³"
)

# Initialize solver
solver = LagrangianSolver(nx=101, ny=101, particles_per_cell=4)

# Run simulation
result = solver.solve(
    basin=basin,
    total_time=7*86400,          # 7 days
    dt=300.0,                    # 5 min timestep
    output_interval=100
)

# Compute diagnostics
diagnostics = compute_all_diagnostics(result)
print(f"Mass conservation error: {diagnostics['mass_error']:.2e}")
print(f"Stretching factor: {diagnostics['stretching_factor']:.2f}")
```

## Test Cases

| Case | System | $L$ [km] | $U_0$ [m/s] | $T_{circ}$ | Tracer | Depth |
|:----:|:-------|:--------:|:-----------:|:----------:|:-------|:-----:|
| 1 | Coastal Embayment | 50 | 0.30 | 6.1 days | Chlorophyll-a [mg/m³] | 100 m |
| 2 | Marginal Sea | 500 | 0.10 | 182 days | DIC [μmol/kg] | 1000 m |
| 3 | Volcanic Lake | 30 | 0.05 | 21.8 days | Temperature [°C] | 500 m |
| 4 | Estuary Plume | 20 | 0.50 | 1.5 days | Sediment [mg/L] | 20 m |

## Lagrangian Diagnostics

| Metric | Symbol | Description |
|:-------|:------:|:------------|
| **Mass Conservation** | $\Delta M/M_0$ | Relative tracer mass error |
| **Area Conservation** | $\Delta A/A_0$ | Material area change (should be 0) |
| **Stretching Factor** | $\lambda = P(t)/P_0$ | Perimeter growth rate |
| **Circularity** | $\mathcal{C} = 4\pi A/P^2$ | Shape distortion (1 = circle) |
| **Variance Decay** | $I_s = \sigma^2(t)/\sigma^2_0$ | Mixing intensity |
| **Reversibility Error** | $E_{rev}$ | Mean particle displacement after reversal |
| **Center of Mass Drift** | $\Delta \mathbf{X}_{cm}$ | Spurious drift detection |

## Numerical Methods

- **Advection:** Midpoint (RK2) scheme - 2nd order accurate
- **Interpolation:** Bilinear shape functions (partition of unity)
- **Projection:** Weighted bincount for particle→mesh
- **Parallelization:** Numba `@njit(parallel=True)` with `prange`

## Output Files

- **NetCDF:** CF-1.8 compliant with oceanographic z-convention (0 at surface, positive down)
- **CSV:** Trajectory and diagnostic metrics
- **GIF:** Beautiful dark-themed animations
- **PNG:** Publication-quality static figures

## Coordinate Convention

Following oceanographic standards:
- **x**: Eastward [m]
- **y**: Northward [m]  
- **z**: Depth below surface [m] (0 at surface, positive downward)

## Authors

- Sandy H. S. Herho (UC Riverside)
- Nurjanna J. Trilaksono (ITB)
- Faiz R. Fajary (ITB)
- Iwan P. Anwar (ITB)
- Faruq Khadami (ITB)

## License

MIT © 2025 Sandy H. S. Herho, Nurjanna J. Trilaksono, Faiz R. Fajary, Iwan P. Anwar, Faruq Khadami

## Citation

```bibtex
@software{herho2025_lanun,
  title   = {{\texttt{lanun}: Numba-Accelerated Lagrangian Transport for Idealized Ocean Basins}},
  author  = {Herho, Sandy H. S. and Fajary, Faiz R. and Anwar, Iwan P. and Khadami, Faruq},
  year    = {2026},
  url     = {https://github.com/sandyherho/lanun}
}
```
