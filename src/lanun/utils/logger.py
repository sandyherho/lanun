"""Simulation logger for Lagrangian transport analysis."""

import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List


class SimulationLogger:
    """Logger for Lagrangian transport simulations."""
    
    def __init__(
        self,
        scenario_name: str,
        log_dir: str = "logs",
        verbose: bool = True
    ):
        """
        Initialize simulation logger.
        
        Args:
            scenario_name: Scenario name (for log filename)
            log_dir: Directory for log files
            verbose: Print messages to console
        """
        self.scenario_name = scenario_name
        self.log_dir = Path(log_dir)
        self.verbose = verbose
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean scenario name for filename
        clean_name = scenario_name.lower().replace(' ', '_').replace('-', '_')
        self.log_file = self.log_dir / f"{clean_name}.log"
        
        self.logger = self._setup_logger()
        self.warnings: List[str] = []
        self.errors: List[str] = []
    
    def _setup_logger(self) -> logging.Logger:
        """Configure Python logging."""
        logger = logging.getLogger(f"lanun_{self.scenario_name}")
        logger.setLevel(logging.DEBUG)
        logger.handlers = []
        
        handler = logging.FileHandler(self.log_file, mode='w')
        handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        return logger
    
    def info(self, msg: str):
        """Log informational message."""
        self.logger.info(msg)
    
    def warning(self, msg: str):
        """Log warning message."""
        self.logger.warning(msg)
        self.warnings.append(msg)
        
        if self.verbose:
            print(f"  WARNING: {msg}")
    
    def error(self, msg: str):
        """Log error message."""
        self.logger.error(msg)
        self.errors.append(msg)
        
        if self.verbose:
            print(f"  ERROR: {msg}")
    
    def log_basin(self, basin: 'BasinSystem'):
        """Log basin configuration."""
        self.info("=" * 70)
        self.info("LAGRANGIAN TRANSPORT SIMULATION")
        self.info(f"Scenario: {self.scenario_name}")
        self.info("=" * 70)
        self.info("")
        
        self.info("BASIN PARAMETERS:")
        self.info(f"  Lx = {basin.Lx/1e3:.1f} km")
        self.info(f"  Ly = {basin.Ly/1e3:.1f} km")
        self.info(f"  H = {basin.H:.1f} m")
        self.info(f"  U0 = {basin.U0:.3f} m/s")
        self.info(f"  T_circ = {basin.T_circ_days:.2f} days")
        
        self.info("")
        self.info("TRACER:")
        self.info(f"  Name: {basin.tracer_name}")
        self.info(f"  Units: {basin.tracer_units}")
        self.info(f"  Background: {basin.tracer_background}")
        self.info(f"  Anomaly: {basin.tracer_anomaly}")
        
        self.info("=" * 70)
    
    def log_config(self, config: Dict[str, Any]):
        """Log simulation configuration."""
        self.info("")
        self.info("SIMULATION PARAMETERS:")
        self.info(f"  nx × ny = {config.get('nx', '?')} × {config.get('ny', '?')}")
        self.info(f"  Particles per cell = {config.get('particles_per_cell', '?')}")
        self.info(f"  dt = {config.get('dt', '?')} s")
        self.info(f"  Total time = {config.get('total_time', 0)/86400:.2f} days")
        self.info(f"  Output interval = {config.get('output_interval', '?')} steps")
        
        cfl = config.get('cfl', None)
        if cfl is not None:
            self.info(f"  CFL number = {cfl:.3f}")
        
        self.info("=" * 70)
    
    def log_diagnostics(self, diagnostics: Dict[str, Any]):
        """Log diagnostic metrics."""
        self.info("")
        self.info("=" * 70)
        self.info("LAGRANGIAN DIAGNOSTICS")
        self.info("=" * 70)
        
        self.info("")
        self.info("CONSERVATION:")
        self.info(f"  Mass error (relative): {diagnostics.get('mass_error_relative', np.nan):.2e}")
        
        # Hull spreading ratio (formerly "area error") - this measures how much
        # the convex hull has expanded due to stirring, not a conservation error
        if 'hull_area_ratio' in diagnostics:
            self.info(f"  Hull spreading ratio: {diagnostics.get('hull_area_ratio', np.nan):.2f}")
        
        self.info("")
        self.info("MIXING:")
        self.info(f"  Intensity of segregation: {diagnostics.get('intensity_segregation', np.nan):.4f}")
        self.info(f"  Mixing efficiency: {diagnostics.get('mixing_efficiency', np.nan):.4f}")
        
        self.info("")
        self.info("DEFORMATION:")
        if 'stretching_factor' in diagnostics:
            self.info(f"  Stretching factor: {diagnostics.get('stretching_factor', np.nan):.2f}")
            self.info(f"  Circularity (initial): {diagnostics.get('circularity_initial', np.nan):.4f}")
            self.info(f"  Circularity (final): {diagnostics.get('circularity_final', np.nan):.4f}")
        
        self.info("")
        self.info("TRANSPORT:")
        self.info(f"  Mean displacement: {diagnostics.get('mean_displacement_normalized', np.nan):.4f} L")
        self.info(f"  CM drift: {diagnostics.get('cm_drift_normalized', np.nan):.4f} L")
        
        self.info("=" * 70)
    
    def log_timing(self, timing: Dict[str, float]):
        """Log timing breakdown."""
        self.info("")
        self.info("=" * 70)
        self.info("TIMING")
        self.info("=" * 70)
        
        for key, value in sorted(timing.items()):
            if key != 'total':
                self.info(f"  {key}: {value:.3f} s")
        
        self.info(f"  {'-' * 40}")
        total = timing.get('total', sum(timing.values()))
        self.info(f"  TOTAL: {total:.3f} s")
        
        self.info("=" * 70)
    
    def finalize(self):
        """Write final summary."""
        self.info("")
        self.info("=" * 70)
        self.info("SUMMARY")
        self.info("=" * 70)
        
        if self.errors:
            self.info(f"ERRORS: {len(self.errors)}")
            for i, err in enumerate(self.errors, 1):
                self.info(f"  {i}. {err}")
        else:
            self.info("ERRORS: None")
        
        if self.warnings:
            self.info(f"WARNINGS: {len(self.warnings)}")
            for i, warn in enumerate(self.warnings, 1):
                self.info(f"  {i}. {warn}")
        else:
            self.info("WARNINGS: None")
        
        self.info("")
        self.info(f"Log file: {self.log_file}")
        self.info(f"Completed: {datetime.now().isoformat()}")
        self.info("=" * 70)
