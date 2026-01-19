#!/usr/bin/env python
"""
Command Line Interface for lanun Lagrangian Transport Analyzer.

Usage:
    lanun case1              # Coastal Embayment (Chlorophyll-a)
    lanun case2              # Marginal Sea (DIC)
    lanun case3              # Volcanic Lake (Temperature)
    lanun case4              # Estuary Plume (Sediment)
    lanun --all              # Run all cases
    lanun --config path.txt  # Custom config
"""

import argparse
import sys
from pathlib import Path
import numpy as np

from .core.velocity import BasinSystem, VelocityField
from .core.solver import LagrangianSolver
from .core.diagnostics import compute_all_diagnostics
from .io.config_manager import ConfigManager
from .io.data_handler import DataHandler
from .visualization.animator import Animator
from .utils.logger import SimulationLogger
from .utils.timer import Timer


def print_header():
    """Print ASCII art header."""
    print("\n" + "=" * 70)
    print(" " * 12 + "lanun: 2D Lagrangian Transport for Ocean Basins")
    print(" " * 25 + "Version 0.0.2")
    print("=" * 70)
    print("\n  Numba-Accelerated Particle-in-Cell Transport Simulation")
    print("  Bell's Incompressible Flow | Bilinear Interpolation | Midpoint RK2")
    print("  License: MIT")
    print("=" * 70 + "\n")


def normalize_scenario_name(scenario_name: str) -> str:
    """Convert scenario name to clean filename format."""
    clean = scenario_name.lower()
    clean = clean.replace(' - ', '_')
    clean = clean.replace('-', '_')
    clean = clean.replace(' ', '_')
    
    while '__' in clean:
        clean = clean.replace('__', '_')
    
    clean = clean.rstrip('_')
    return clean


def run_scenario(
    config: dict,
    output_dir: str = "outputs",
    verbose: bool = True
):
    """Run a complete Lagrangian transport simulation scenario."""
    
    scenario_name = config.get('scenario_name', 'simulation')
    clean_name = normalize_scenario_name(scenario_name)
    
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'=' * 70}")
    
    logger = SimulationLogger(clean_name, "logs", verbose)
    timer = Timer()
    timer.start("total")
    
    try:
        # [1/7] Initialize basin
        with timer.time_section("basin_init"):
            if verbose:
                print("\n[1/7] Initializing ocean basin...")
            
            basin = BasinSystem(
                Lx=config.get('Lx', 50000.0),
                Ly=config.get('Ly', 50000.0),
                H=config.get('H', 100.0),
                U0=config.get('U0', 0.3),
                tracer_name=config.get('tracer_name', 'Tracer'),
                tracer_units=config.get('tracer_units', 'kg/m³'),
                tracer_background=config.get('tracer_background', 0.0),
                tracer_anomaly=config.get('tracer_anomaly', 1.0),
            )
            
            logger.log_basin(basin)
            
            if verbose:
                print(f"      {basin}")
        
        # [2/7] Initialize solver
        with timer.time_section("solver_init"):
            if verbose:
                print("\n[2/7] Initializing Lagrangian solver...")
            
            solver = LagrangianSolver(
                nx=config.get('nx', 101),
                ny=config.get('ny', 101),
                particles_per_cell=config.get('particles_per_cell', 4)
            )
            
            if verbose:
                print(f"      Mesh: {solver.nx}×{solver.ny}")
                print(f"      Particles per cell: {solver.particles_per_cell}²")
        
        # [3/7] Run simulation
        with timer.time_section("simulation"):
            if verbose:
                print("\n[3/7] Running Lagrangian advection...")
            
            total_time = config.get('total_time_days', 7.0) * 86400.0
            dt = config.get('dt', 300.0)
            
            # Tracer initial position (fractional)
            center_x = config.get('tracer_center_x_frac', 0.25) * basin.Lx
            center_y = config.get('tracer_center_y_frac', 0.50) * basin.Ly
            sigma = config.get('tracer_sigma_frac', 0.10) * basin.Lx
            
            result = solver.solve(
                basin=basin,
                total_time=total_time,
                dt=dt,
                output_interval=config.get('output_interval', 100),
                tracer_center_x=center_x,
                tracer_center_y=center_y,
                tracer_sigma=sigma,
                verbose=verbose
            )
            
            # Store config in result
            result.config.update(config)
        
        # [4/7] Compute diagnostics
        with timer.time_section("diagnostics"):
            if verbose:
                print("\n[4/7] Computing Lagrangian diagnostics...")
            
            diagnostics = compute_all_diagnostics(result, verbose=verbose)
            logger.log_diagnostics(diagnostics)
        
        # [5/7] Save CSV data
        with timer.time_section("csv_save"):
            if verbose:
                print("\n[5/7] Saving CSV data...")
            
            csv_dir = Path(output_dir) / "csv"
            csv_dir.mkdir(parents=True, exist_ok=True)
            
            # Diagnostics CSV
            diag_file = csv_dir / f"{clean_name}_diagnostics.csv"
            DataHandler.save_diagnostics_csv(str(diag_file), diagnostics)
            
            if verbose:
                print(f"      Saved: {diag_file}")
        
        # [6/7] Save NetCDF
        with timer.time_section("netcdf_save"):
            if verbose:
                print("\n[6/7] Saving NetCDF data...")
            
            nc_dir = Path(output_dir) / "netcdf"
            nc_dir.mkdir(parents=True, exist_ok=True)
            
            nc_file = nc_dir / f"{clean_name}.nc"
            DataHandler.save_netcdf(str(nc_file), result, config)
            
            if verbose:
                print(f"      Saved: {nc_file}")
        
        # [7/7] Generate visualizations
        with timer.time_section("visualization"):
            if verbose:
                print("\n[7/7] Generating visualizations...")
            
            animator = Animator(fps=15, dpi=150)
            
            # Static plot
            fig_dir = Path(output_dir) / "figs"
            fig_dir.mkdir(parents=True, exist_ok=True)
            
            png_file = fig_dir / f"{clean_name}_summary.png"
            animator.create_static_plot(result, str(png_file), diagnostics)
            
            if verbose:
                print(f"      Saved: {png_file}")
            
            # Animation
            if config.get('save_gif', True):
                gif_dir = Path(output_dir) / "gifs"
                gif_dir.mkdir(parents=True, exist_ok=True)
                
                gif_file = gif_dir / f"{clean_name}_animation.gif"
                animator.create_animation(
                    result, str(gif_file),
                    n_frames=config.get('animation_frames', 100),
                    duration_seconds=config.get('animation_duration', 10.0)
                )
        
        timer.stop("total")
        logger.log_timing(timer.get_times())
        
        # Summary
        if verbose:
            total_time = timer.times.get('total', 0)
            print(f"\n{'=' * 70}")
            print(f"SIMULATION COMPLETED")
            print(f"{'=' * 70}")
            print(f"  Mass conservation error: {diagnostics.get('mass_error_relative', 0):.2e}")
            print(f"  Intensity of segregation: {diagnostics.get('intensity_segregation', 0):.4f}")
            print(f"  Total time: {total_time:.2f} s")
            print(f"{'=' * 70}\n")
        
        return result, diagnostics
    
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"SIMULATION FAILED: {str(e)}")
            print(f"{'=' * 70}\n")
        
        raise
    
    finally:
        logger.finalize()


def main():
    """Main entry point for command-line interface."""
    parser = argparse.ArgumentParser(
        description='lanun: Lagrangian Transport for Ocean Basins',
        epilog='Example: lanun case1'
    )
    
    parser.add_argument(
        'case',
        nargs='?',
        choices=['case1', 'case2', 'case3', 'case4'],
        help='Test case to run (case1-4)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to custom configuration file'
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Run all test cases sequentially'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='outputs',
        help='Output directory for results (default: outputs)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet mode (minimal output)'
    )
    
    parser.add_argument(
        '--no-gif',
        action='store_true',
        help='Skip GIF animation generation'
    )
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    if verbose:
        print_header()
    
    # Custom config
    if args.config:
        config = ConfigManager.load(args.config)
        if args.no_gif:
            config['save_gif'] = False
        run_scenario(config, args.output_dir, verbose)
    
    # All cases
    elif args.all:
        for case_num in range(1, 5):
            case_name = f'case{case_num}'
            config = ConfigManager.get_default_config(case_name)
            if args.no_gif:
                config['save_gif'] = False
            run_scenario(config, args.output_dir, verbose)
    
    # Single case
    elif args.case:
        config = ConfigManager.get_default_config(args.case)
        if args.no_gif:
            config['save_gif'] = False
        run_scenario(config, args.output_dir, verbose)
    
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == '__main__':
    main()
