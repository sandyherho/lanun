# `lanun`: 2D Lagrangian Particle Transport for Idealized Ocean Basins

[![DOI](https://zenodo.org/badge/1120895962.svg)](https://doi.org/10.5281/zenodo.18070973)
[![PyPI version](https://img.shields.io/pypi/v/lanun.svg)](https://pypi.org/project/lanun/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![NumPy](https://img.shields.io/badge/NumPy-%23013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?logo=scipy&logoColor=white)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?logo=Matplotlib&logoColor=black)](https://matplotlib.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![netCDF4](https://img.shields.io/badge/netCDF4-%23004B87.svg)](https://unidata.github.io/netcdf4-python/)
[![Numba](https://img.shields.io/badge/Numba-%2300A3E0.svg?logo=numba&logoColor=white)](https://numba.pydata.org/)
[![Pillow](https://img.shields.io/badge/Pillow-%23000000.svg)](https://python-pillow.org/)
[![tqdm](https://img.shields.io/badge/tqdm-%23FFC107.svg)](https://tqdm.github.io/)


A Numba-accelerated Python library for simulating 2D Lagrangian particle transport in idealized semi-enclosed ocean basins using Bell's incompressible flow field. This is an idealized 2D model intended for process studies and educational purposes, not operational forecasting.

> The library takes its name from the *Lanun* (also Iranun or Illanun), maritime peoples of the Sulu and Celebes Seas. Historically, Lanun maritime activity (c. 1768–1848) constituted a sophisticated political economy centered on the Sultanate of Sulu rather than simple brigandage. In a region abundant in marine resources but critically short of labor, Iranun and Balangingi fleets operated as state-sponsored forces conducting systematic seasonal expeditions (*mangayau*) to acquire manpower for processing trepang, pearls, and other sea products bound for the China trade. Revenue from this commerce funded the Sultanate's acquisition of Western armaments. The European designation of "piracy" was largely a 19th-century colonial construct employed by British and Dutch authorities to criminalize traditional Malay maritime sovereignty and toll collection rights, thereby delegitimizing indigenous statecraft to facilitate the establishment of European trade monopolies. This library honors their legacy as accomplished navigators with deep understanding of regional ocean circulation.

<p align="center">
  <img src="https://github.com/sandyherho/lanun/blob/main/.assets/anim.gif" alt="Chlorophyll-a transport simulation" width="600">
</p>

## Overview

`lanun` provides a particle-in-cell framework for studying passive tracer dispersion in closed or semi-enclosed water bodies. The library is designed for idealized process studies rather than operational forecasting, making it suitable for:

- Understanding chaotic advection and stirring dynamics
- Testing Lagrangian diagnostic methods
- Educational demonstrations of transport phenomena
- Preliminary assessment of tracer dispersion timescales

**Important:** This is a **2D, depth-averaged** model using a **prescribed analytical velocity field**. It does not solve the primitive equations and should not be used for realistic ocean simulations.

## Mathematical Formulation

### Bell's Incompressible Flow

The velocity field implements the Bell-Colella-Glaz (1989) test case, an analytically incompressible flow commonly used for validating advection schemes.

**Velocity components:**

$$u = -\frac{U_0}{2} \sin^2\left(\frac{\pi x}{L}\right) \sin\left(\frac{2\pi y}{L}\right)$$

$$v = \frac{U_0}{2} \sin^2\left(\frac{\pi y}{L}\right) \sin\left(\frac{2\pi x}{L}\right)$$

**Incompressibility verification via tensor calculus:**

The velocity field satisfies $\nabla \cdot \mathbf{v} = 0$ identically. Adopting index notation with coordinates $x^1 = x$ and $x^2 = y$, the divergence is:

$$\partial_i v^i = \frac{\partial v^1}{\partial x^1} + \frac{\partial v^2}{\partial x^2} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}$$

Computing the partial derivatives:

$$\frac{\partial u}{\partial x} = -\frac{U_0}{2} \cdot 2\sin\left(\frac{\pi x}{L}\right)\cos\left(\frac{\pi x}{L}\right) \cdot \frac{\pi}{L} \cdot \sin\left(\frac{2\pi y}{L}\right) = -\frac{\pi U_0}{2L} \sin\left(\frac{2\pi x}{L}\right) \sin\left(\frac{2\pi y}{L}\right)$$

$$\frac{\partial v}{\partial y} = \frac{U_0}{2} \cdot 2\sin\left(\frac{\pi y}{L}\right)\cos\left(\frac{\pi y}{L}\right) \cdot \frac{\pi}{L} \cdot \sin\left(\frac{2\pi x}{L}\right) = \frac{\pi U_0}{2L} \sin\left(\frac{2\pi x}{L}\right) \sin\left(\frac{2\pi y}{L}\right)$$

The sum vanishes identically: $\partial_i v^i = 0$ $\forall (x,y) \in \Omega$.

**Stream function:**

The velocity derives from a stream function $\psi$ such that $v^i = \epsilon^{ij}\partial_j\psi$ (where $\epsilon^{ij}$ is the 2D Levi-Civita symbol):

$$\psi = \frac{U_0 L}{4\pi} \sin^2\left(\frac{\pi x}{L}\right) \sin^2\left(\frac{\pi y}{L}\right)$$

satisfying $u = \partial\psi/\partial y$ and $v = -\partial\psi/\partial x$.

**Boundary conditions:**

The velocity field naturally satisfies homogeneous Dirichlet conditions $v^i|_{\partial\Omega} = 0$ on all domain boundaries, representing impermeable walls.

**Characteristic scales:**

- Advective timescale: $T_{adv} = L/U_0$
- Circulation timescale: $T_{circ} = \pi L/U_0$

### Numerical Methods

| Component | Method | Properties |
|-----------|--------|------------|
| Time integration | Midpoint (RK2) | 2nd-order accurate, $O(\Delta t^2)$ global error |
| Spatial interpolation | Bilinear | Partition of unity ($\sum w_i = 1$), conservative |
| Particle-mesh projection | Weighted scatter | Introduces numerical diffusion at sub-grid scales |
| Parallelization | Numba `prange` | Efficient multi-threaded particle loops |

### Key Physical Insight: Stirring ≠ Mixing

In purely Lagrangian transport without molecular diffusion, particles carry their tracer values unchanged along trajectories. This means:

- **Particle variance is constant:** The intensity of segregation $I_s = \sigma^2(t)/\sigma^2(0) = 1$ always
- **Stirring creates filaments:** Material lines stretch exponentially, increasing interfacial area
- **True mixing requires diffusion:** Homogenization only occurs with $\kappa > 0$

The mesh-projected variance decreases due to numerical averaging during particle-to-grid projection, not physical mixing.

## Installation

```bash
# Using pip
pip install lanun

# Using poetry (recommended for development)
git clone https://github.com/sandyherho/lanun.git
cd lanun
poetry install --with dev
```

### Dependencies

- Python ≥ 3.9
- NumPy, SciPy, Matplotlib, Pandas
- Numba (JIT compilation)
- netCDF4 (CF-compliant output)
- Pillow, tqdm (visualization)

## Quick Start

```python
from lanun import BasinSystem, LagrangianSolver

# Define basin geometry and flow
basin = BasinSystem(
    Lx=50e3,           # 50 km domain
    Ly=50e3,
    H=100.0,           # 100 m depth (reference only, 2D model)
    U0=0.3,            # 30 cm/s max velocity
    tracer_name="Chlorophyll-a",
    tracer_units="mg/m³",
    tracer_background=0.5,
    tracer_anomaly=5.0
)

# Initialize solver
solver = LagrangianSolver(nx=101, ny=101, particles_per_cell=4)

# Run simulation (7 days)
result = solver.solve(
    basin=basin,
    total_time=7 * 86400,
    dt=300,
    output_interval=100
)

# Access results
print(f"Circulation time: {basin.T_circ_days:.1f} days")
print(f"Final particle positions: {result.particle_x[-1].shape}")
```

## Command Line Interface

```bash
# Run predefined test cases
lanun case1    # Coastal Embayment (Chlorophyll-a)
lanun case2    # Marginal Sea (DIC)
lanun case3    # Volcanic Lake (Temperature)
lanun case4    # Estuary Plume (Sediment)

# Run all cases
lanun --all

# Custom configuration
lanun --config my_config.txt

# Skip animation generation
lanun case1 --no-gif
```

## Test Cases

| Case | Domain | Tracer | $T_{circ}$ | Application |
|------|--------|--------|------------|-------------|
| Coastal Embayment | 50 km | Chlorophyll-a | 6 days | Bloom dispersion |
| Marginal Sea | 500 km | DIC | 182 days | Carbon transport |
| Volcanic Lake | 30 km | Temperature | 22 days | Hydrothermal mixing |
| Estuary Plume | 20 km | Sediment | 1.5 days | River discharge |

## Output Formats

- **NetCDF:** CF-1.8 compliant with oceanographic conventions
- **CSV:** Diagnostic time series
- **PNG:** Summary visualizations
- **GIF:** Animated tracer evolution

## Diagnostics

The library computes physically meaningful Lagrangian diagnostics:

- Mass conservation error (machine precision for Lagrangian transport)
- Intensity of segregation (particle-based and mesh-based)
- Convex hull spreading ratio and circularity
- Particle-pair stretching statistics (FTLE proxy)
- Center of mass drift
- Radius of gyration

## Limitations

This is an **idealized 2D model** with the following simplifications:

- 2D depth-averaged (no vertical structure)
- Prescribed analytical velocity field (no dynamics)
- No molecular diffusion (pure advection)
- No Coriolis force, stratification, or tidal forcing
- Idealized rectangular geometry with impermeable boundaries

For realistic ocean applications, consider [OceanParcels](https://oceanparcels.org/), [OpenDrift](https://opendrift.github.io/), or similar operational Lagrangian tools.

## Authors

- [Sandy H. S. Herho](mailto:sandy.herho@ronininstitute.org) (Ronin Institute)
- [Iwan P. Anwar](mailto:iwanpanwar@itb.ac.id) (ITB)
- [Faruq Khadami](mailto:fkhadami@itb.ac.id) (ITB)
- [Dasapta E. Irawan](mailto:dasaptaerwin@itb.ac.id) (ITB) 

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{herho2026lanun,
  author    = {Herho, Sandy H. S. and Khadami, Faruq and Anwar, Iwan P. and Irawan, Dasapta E.},
  title     = {{lanun}: {2D} {L}agrangian Particle Transport for Idealized Ocean Basins},
  year      = {2026},
  url       = {https://github.com/sandyherho/lanun},
  version   = {0.0.2}
}
```
