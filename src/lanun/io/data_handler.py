"""
Data Handler for Lagrangian Transport Simulations.

Saves results to:
    - CSV: Trajectory and diagnostic metrics
    - NetCDF: CF-1.8 compliant with oceanographic convention

Coordinate convention (oceanographic):
    - x: Eastward [m]
    - y: Northward [m]
"""

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class DataHandler:
    """Handle saving simulation data to various formats."""
    
    @staticmethod
    def save_trajectory_csv(
        filepath: str,
        result: 'SimulationResult',
        particle_subset: Optional[int] = 1000
    ):
        """
        Save particle trajectories to CSV.
        
        Args:
            filepath: Output file path
            result: SimulationResult from solver
            particle_subset: Number of particles to save (None = all)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        n_times = len(result.time)
        n_particles = result.particle_x.shape[1]
        
        if particle_subset is not None and particle_subset < n_particles:
            indices = np.linspace(0, n_particles - 1, particle_subset, dtype=int)
        else:
            indices = np.arange(n_particles)
        
        # Create long-form dataframe
        rows = []
        for t_idx in range(n_times):
            for p_idx in indices:
                rows.append({
                    'time_s': result.time[t_idx],
                    'time_days': result.time[t_idx] / 86400.0,
                    'particle_id': int(p_idx),
                    'x_m': result.particle_x[t_idx, p_idx],
                    'y_m': result.particle_y[t_idx, p_idx],
                    'tracer': result.particle_tracer[t_idx, p_idx],
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False, float_format='%.8e')
    
    @staticmethod
    def save_diagnostics_csv(filepath: str, diagnostics: Dict[str, Any]):
        """
        Save diagnostic metrics to CSV.
        
        Args:
            filepath: Output file path
            diagnostics: Dictionary of metrics
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        rows = []
        for key, value in sorted(diagnostics.items()):
            if isinstance(value, (int, float, bool)):
                rows.append({
                    'Metric': key,
                    'Value': value,
                    'Units': DataHandler._get_metric_units(key),
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
    
    @staticmethod
    def _get_metric_units(metric_name: str) -> str:
        """Get units for a metric."""
        units_map = {
            'mass_initial': 'tracer_units * m²',
            'mass_final': 'tracer_units * m²',
            'mass_error_absolute': 'tracer_units * m²',
            'mass_error_relative': 'dimensionless',
            'variance_initial': 'tracer_units²',
            'variance_current': 'tracer_units²',
            'intensity_segregation': 'dimensionless',
            'mixing_efficiency': 'dimensionless',
            'area_initial': 'm²',
            'area_final': 'm²',
            'perimeter_initial': 'm',
            'perimeter_final': 'm',
            'circularity_initial': 'dimensionless',
            'circularity_final': 'dimensionless',
            'stretching_factor': 'dimensionless',
            'area_error_relative': 'dimensionless',
            'x_cm_initial': 'm',
            'y_cm_initial': 'm',
            'x_cm_final': 'm',
            'y_cm_final': 'm',
            'cm_drift': 'm',
            'cm_drift_normalized': 'L',
            'mean_displacement': 'm',
            'mean_displacement_normalized': 'L',
            'reversibility_mean': 'm',
            'reversibility_normalized': 'L',
        }
        return units_map.get(metric_name, 'unknown')
    
    @staticmethod
    def save_netcdf(
        filepath: str,
        result: 'SimulationResult',
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Save complete simulation data to CF-compliant NetCDF.
        
        Uses oceanographic convention:
            - z positive downward from surface
            - CF-1.8 conventions
        
        Args:
            filepath: Output file path
            result: SimulationResult from solver
            config: Optional configuration dictionary
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        basin = result.basin
        vf = result.velocity_field
        
        n_time = len(result.time)
        n_particles = result.particle_x.shape[1]
        nx = vf.nx
        ny = vf.ny
        
        with Dataset(filepath, 'w', format='NETCDF4') as nc:
            # ================================================================
            # DIMENSIONS
            # ================================================================
            nc.createDimension('time', n_time)
            nc.createDimension('particle', n_particles)
            nc.createDimension('x', nx)
            nc.createDimension('y', ny)
            nc.createDimension('depth', 1)  # Single layer for 2D
            
            # ================================================================
            # COORDINATE VARIABLES
            # ================================================================
            
            # Time
            nc_time = nc.createVariable('time', 'f8', ('time',), zlib=True)
            nc_time[:] = result.time
            nc_time.units = 'seconds since simulation start'
            nc_time.long_name = 'time'
            nc_time.standard_name = 'time'
            nc_time.axis = 'T'
            nc_time.calendar = 'none'
            
            # Time in days (auxiliary)
            nc_time_days = nc.createVariable('time_days', 'f8', ('time',), zlib=True)
            nc_time_days[:] = result.time / 86400.0
            nc_time_days.units = 'days'
            nc_time_days.long_name = 'time in days'
            
            # X coordinate (mesh)
            nc_x = nc.createVariable('x', 'f8', ('x',), zlib=True)
            nc_x[:] = vf.X
            nc_x.units = 'm'
            nc_x.long_name = 'x-coordinate (eastward)'
            nc_x.standard_name = 'projection_x_coordinate'
            nc_x.axis = 'X'
            
            # Y coordinate (mesh)
            nc_y = nc.createVariable('y', 'f8', ('y',), zlib=True)
            nc_y[:] = vf.Y
            nc_y.units = 'm'
            nc_y.long_name = 'y-coordinate (northward)'
            nc_y.standard_name = 'projection_y_coordinate'
            nc_y.axis = 'Y'
            
            # Depth (oceanographic convention: 0 at surface, positive down)
            nc_depth = nc.createVariable('depth', 'f8', ('depth',), zlib=True)
            nc_depth[:] = [0.0]  # Surface layer
            nc_depth.units = 'm'
            nc_depth.long_name = 'depth below sea surface'
            nc_depth.standard_name = 'depth'
            nc_depth.axis = 'Z'
            nc_depth.positive = 'down'
            
            # Particle ID
            nc_pid = nc.createVariable('particle_id', 'i4', ('particle',), zlib=True)
            nc_pid[:] = np.arange(n_particles)
            nc_pid.long_name = 'particle identifier'
            
            # ================================================================
            # PARTICLE VARIABLES
            # ================================================================
            
            # Particle x-position
            nc_px = nc.createVariable(
                'particle_x', 'f8', ('time', 'particle'), zlib=True,
                chunksizes=(min(100, n_time), min(10000, n_particles))
            )
            nc_px[:] = result.particle_x
            nc_px.units = 'm'
            nc_px.long_name = 'particle x-position'
            nc_px.coordinates = 'time particle_id'
            
            # Particle y-position
            nc_py = nc.createVariable(
                'particle_y', 'f8', ('time', 'particle'), zlib=True,
                chunksizes=(min(100, n_time), min(10000, n_particles))
            )
            nc_py[:] = result.particle_y
            nc_py.units = 'm'
            nc_py.long_name = 'particle y-position'
            nc_py.coordinates = 'time particle_id'
            
            # Particle tracer
            nc_pt = nc.createVariable(
                'particle_tracer', 'f8', ('time', 'particle'), zlib=True,
                chunksizes=(min(100, n_time), min(10000, n_particles))
            )
            nc_pt[:] = result.particle_tracer
            nc_pt.units = basin.tracer_units
            nc_pt.long_name = f'particle {basin.tracer_name} concentration'
            nc_pt.coordinates = 'time particle_id'
            
            # ================================================================
            # MESH VARIABLES
            # ================================================================
            
            # Tracer field on mesh
            nc_tracer = nc.createVariable(
                'tracer', 'f8', ('time', 'y', 'x'), zlib=True,
                chunksizes=(min(10, n_time), ny, nx)
            )
            # Transpose from (time, x, y) to (time, y, x) for CF convention
            nc_tracer[:] = np.transpose(result.mesh_tracer, (0, 2, 1))
            nc_tracer.units = basin.tracer_units
            nc_tracer.long_name = f'{basin.tracer_name} concentration'
            nc_tracer.standard_name = 'mass_concentration_of_tracer_in_sea_water'
            nc_tracer.coordinates = 'time y x'
            nc_tracer.grid_mapping = 'crs'
            
            # Velocity field (static)
            nc_vx = nc.createVariable('u', 'f8', ('y', 'x'), zlib=True)
            nc_vx[:] = vf.vx.T  # Transpose for (y, x)
            nc_vx.units = 'm s-1'
            nc_vx.long_name = 'eastward sea water velocity'
            nc_vx.standard_name = 'eastward_sea_water_velocity'
            
            nc_vy = nc.createVariable('v', 'f8', ('y', 'x'), zlib=True)
            nc_vy[:] = vf.vy.T  # Transpose for (y, x)
            nc_vy.units = 'm s-1'
            nc_vy.long_name = 'northward sea water velocity'
            nc_vy.standard_name = 'northward_sea_water_velocity'
            
            # Stream function
            nc_psi = nc.createVariable('psi', 'f8', ('y', 'x'), zlib=True)
            nc_psi[:] = vf.psi.T
            nc_psi.units = 'm2 s-1'
            nc_psi.long_name = 'stream function'
            
            # Vorticity
            nc_vort = nc.createVariable('vorticity', 'f8', ('y', 'x'), zlib=True)
            nc_vort[:] = vf.vorticity.T
            nc_vort.units = 's-1'
            nc_vort.long_name = 'vertical vorticity'
            
            # ================================================================
            # DIAGNOSTICS (as scalar variables)
            # ================================================================
            if result.diagnostics:
                for key, value in result.diagnostics.items():
                    if isinstance(value, (int, float)):
                        var_name = f'diag_{key}'
                        nc_var = nc.createVariable(var_name, 'f8')
                        nc_var[()] = float(value)
                        nc_var.long_name = key.replace('_', ' ')
                        nc_var.units = DataHandler._get_metric_units(key)
            
            # ================================================================
            # COORDINATE REFERENCE SYSTEM
            # ================================================================
            crs = nc.createVariable('crs', 'i4')
            crs.grid_mapping_name = 'local_cartesian'
            crs.comment = 'Idealized basin coordinates'
            
            # ================================================================
            # GLOBAL ATTRIBUTES
            # ================================================================
            nc.title = f'Lagrangian Transport Simulation: {basin.tracer_name}'
            nc.institution = 'lanun'
            nc.source = 'lanun v0.0.1'
            nc.history = f'Created {datetime.now().isoformat()}'
            nc.Conventions = 'CF-1.8'
            nc.featureType = 'trajectory'
            
            # Basin parameters
            nc.basin_Lx_m = float(basin.Lx)
            nc.basin_Ly_m = float(basin.Ly)
            nc.basin_H_m = float(basin.H)
            nc.basin_U0_ms = float(basin.U0)
            nc.basin_T_circ_days = float(basin.T_circ_days)
            nc.basin_rho0_kgm3 = float(basin.rho0)
            
            # Tracer info
            nc.tracer_name = basin.tracer_name
            nc.tracer_units = basin.tracer_units
            nc.tracer_background = float(basin.tracer_background)
            nc.tracer_anomaly = float(basin.tracer_anomaly)
            
            # Simulation parameters
            if config:
                nc.scenario_name = config.get('scenario_name', 'unknown')
                nc.nx = int(config.get('nx', nx))
                nc.ny = int(config.get('ny', ny))
                nc.particles_per_cell = int(config.get('particles_per_cell', 4))
                nc.dt_s = float(config.get('dt', 0))
                nc.total_time_s = float(config.get('total_time', 0))
            
            nc.n_particles = n_particles
            nc.n_time_outputs = n_time
            
            # Authors
            nc.authors = 'Sandy H. S. Herho, Faruq Khadami, Iwan P. Anwar, Dasapta E. Irawan'
            nc.contact = 'sandy.herho@ronininstitute.org'
            nc.license = 'MIT'
    
    @staticmethod
    def save_comparison_csv(
        filepath: str,
        results: Dict[str, 'SimulationResult']
    ):
        """
        Save comparison table across multiple cases.
        
        Args:
            filepath: Output file path
            results: Dictionary mapping case names to results
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        rows = []
        for case_name, result in results.items():
            basin = result.basin
            diag = result.diagnostics
            
            row = {
                'Case': case_name,
                'L [km]': basin.Lx / 1e3,
                'U0 [m/s]': basin.U0,
                'T_circ [days]': basin.T_circ_days,
                'Tracer': basin.tracer_name,
                'Mass error': diag.get('mass_error_relative', np.nan),
                'Intensity segregation': diag.get('intensity_segregation', np.nan),
                'Stretching factor': diag.get('stretching_factor', np.nan),
                'Circularity (final)': diag.get('circularity_final', np.nan),
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False, float_format='%.4e')
