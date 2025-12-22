"""
Beautiful Visualization for Lagrangian Ocean Transport.

Creates publication-quality visualizations with dark ocean theme,
smooth animations, and professional aesthetics.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import warnings
import io

from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm

warnings.filterwarnings('ignore')


class Animator:
    """
    Create stunning visualizations for Lagrangian ocean transport.
    
    Features dark ocean-themed aesthetics with professional styling.
    """
    
    # Ocean dark theme color palette
    COLOR_BG = '#0A1628'        # Deep ocean blue-black
    COLOR_BG_LIGHTER = '#0F1D32'
    COLOR_BG_PANEL = '#152238'
    COLOR_OCEAN_DEEP = '#0A2463'
    COLOR_OCEAN_MID = '#1E5288'
    COLOR_OCEAN_LIGHT = '#3E92CC'
    COLOR_ACCENT_CYAN = '#00F5FF'
    COLOR_ACCENT_CORAL = '#FF6B6B'
    COLOR_ACCENT_GOLD = '#FFD93D'
    COLOR_ACCENT_GREEN = '#6BCB77'
    COLOR_GRID = '#1A3A5C'
    COLOR_TEXT = '#C8D4E3'
    COLOR_TITLE = '#FFFFFF'
    COLOR_LAND = '#2D3436'
    
    def __init__(self, fps: int = 15, dpi: int = 150):
        """
        Initialize animator.
        
        Args:
            fps: Frames per second for animations
            dpi: Resolution for output images
        """
        self.fps = fps
        self.dpi = dpi
        self._setup_style()
    
    def _setup_style(self):
        """Setup matplotlib ocean dark theme."""
        plt.style.use('dark_background')
        plt.rcParams.update({
            'figure.facecolor': self.COLOR_BG,
            'axes.facecolor': self.COLOR_BG_LIGHTER,
            'axes.edgecolor': self.COLOR_GRID,
            'axes.labelcolor': self.COLOR_TEXT,
            'axes.titlecolor': self.COLOR_TITLE,
            'xtick.color': self.COLOR_TEXT,
            'ytick.color': self.COLOR_TEXT,
            'text.color': self.COLOR_TEXT,
            'grid.color': self.COLOR_GRID,
            'grid.alpha': 0.3,
            'legend.facecolor': self.COLOR_BG_PANEL,
            'legend.edgecolor': self.COLOR_GRID,
            'font.family': 'sans-serif',
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'mathtext.fontset': 'cm',
        })
    
    def _create_ocean_cmap(self) -> LinearSegmentedColormap:
        """Create ocean-themed colormap for tracer."""
        colors = [
            self.COLOR_BG,
            self.COLOR_OCEAN_DEEP,
            self.COLOR_OCEAN_MID,
            self.COLOR_OCEAN_LIGHT,
            self.COLOR_ACCENT_CYAN,
            '#FFFFFF'
        ]
        return LinearSegmentedColormap.from_list('ocean_tracer', colors, N=256)
    
    def _create_chlorophyll_cmap(self) -> LinearSegmentedColormap:
        """Create chlorophyll-themed colormap (blue to green)."""
        colors = [
            '#0A1628',
            '#0A2463',
            '#1E5288',
            '#2E8B57',
            '#32CD32',
            '#ADFF2F'
        ]
        return LinearSegmentedColormap.from_list('chlorophyll', colors, N=256)
    
    def _get_tracer_cmap(self, tracer_name: str) -> LinearSegmentedColormap:
        """Get appropriate colormap for tracer type."""
        if 'chlorophyll' in tracer_name.lower():
            return self._create_chlorophyll_cmap()
        elif 'temperature' in tracer_name.lower():
            return cm.get_cmap('RdYlBu_r')
        elif 'sediment' in tracer_name.lower():
            return cm.get_cmap('YlOrBr')
        else:
            return self._create_ocean_cmap()
    
    def create_static_plot(
        self,
        result: 'SimulationResult',
        filepath: str,
        diagnostics: Optional[Dict[str, Any]] = None
    ):
        """
        Create comprehensive static visualization.
        
        Layout: 2×3 panels
        - Row 1: Initial tracer, Streamlines, Final tracer
        - Row 2: Time series, Diagnostics, Velocity magnitude
        
        Args:
            result: SimulationResult from solver
            filepath: Output file path
            diagnostics: Optional diagnostics dictionary
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        basin = result.basin
        vf = result.velocity_field
        
        fig = plt.figure(figsize=(18, 12), facecolor=self.COLOR_BG)
        
        # Title
        fig.suptitle(
            f'{basin.tracer_name} Transport in Semi-Enclosed Basin\n'
            f'$L$ = {basin.Lx/1e3:.0f} km, $U_0$ = {basin.U0:.2f} m/s, '
            f'$T_{{circ}}$ = {basin.T_circ_days:.1f} days',
            fontsize=16, fontweight='bold', color=self.COLOR_TITLE, y=0.98
        )
        
        cmap = self._get_tracer_cmap(basin.tracer_name)
        
        # Convert coordinates to km for display
        X_km = vf.X / 1e3
        Y_km = vf.Y / 1e3
        
        # Tracer bounds
        vmin = basin.tracer_background
        vmax = basin.tracer_background + basin.tracer_anomaly
        
        # ====== Panel 1: Initial tracer ======
        ax1 = fig.add_subplot(231, facecolor=self.COLOR_BG_LIGHTER)
        
        tracer_init = result.mesh_tracer[0, :, :].T  # (y, x)
        im1 = ax1.pcolormesh(
            X_km, Y_km, tracer_init,
            cmap=cmap, vmin=vmin, vmax=vmax, shading='auto'
        )
        ax1.set_xlabel('$x$ [km]', fontweight='bold')
        ax1.set_ylabel('$y$ [km]', fontweight='bold')
        ax1.set_title('Initial ($t$ = 0)', fontweight='bold')
        ax1.set_aspect('equal')
        cbar1 = plt.colorbar(im1, ax=ax1, pad=0.02)
        cbar1.set_label(f'{basin.tracer_name} [{basin.tracer_units}]')
        
        # ====== Panel 2: Streamlines ======
        ax2 = fig.add_subplot(232, facecolor=self.COLOR_BG_LIGHTER)
        
        # Plot stream function contours
        psi_scaled = vf.psi.T * 1e3  # Scale for visibility
        ax2.contour(
            X_km, Y_km, psi_scaled,
            levels=20, colors=self.COLOR_OCEAN_LIGHT, linewidths=0.5, alpha=0.7
        )
        
        # Overlay streamplot
        ax2.streamplot(
            X_km, Y_km, vf.vx.T, vf.vy.T,
            color=self.COLOR_ACCENT_CYAN, linewidth=0.8,
            density=1.5, arrowsize=0.8, arrowstyle='->'
        )
        
        ax2.set_xlabel('$x$ [km]', fontweight='bold')
        ax2.set_ylabel('$y$ [km]', fontweight='bold')
        ax2.set_title('Velocity Field (Bell\'s Flow)', fontweight='bold')
        ax2.set_aspect('equal')
        
        # ====== Panel 3: Final tracer ======
        ax3 = fig.add_subplot(233, facecolor=self.COLOR_BG_LIGHTER)
        
        tracer_final = result.mesh_tracer[-1, :, :].T
        im3 = ax3.pcolormesh(
            X_km, Y_km, tracer_final,
            cmap=cmap, vmin=vmin, vmax=vmax, shading='auto'
        )
        
        t_final_days = result.time[-1] / 86400
        ax3.set_xlabel('$x$ [km]', fontweight='bold')
        ax3.set_ylabel('$y$ [km]', fontweight='bold')
        ax3.set_title(f'Final ($t$ = {t_final_days:.1f} days)', fontweight='bold')
        ax3.set_aspect('equal')
        cbar3 = plt.colorbar(im3, ax=ax3, pad=0.02)
        cbar3.set_label(f'{basin.tracer_name} [{basin.tracer_units}]')
        
        # ====== Panel 4: Time evolution of variance ======
        ax4 = fig.add_subplot(234, facecolor=self.COLOR_BG_LIGHTER)
        
        time_days = result.time / 86400
        variances = np.var(result.particle_tracer, axis=1)
        var_normalized = variances / variances[0] if variances[0] > 0 else variances
        
        ax4.plot(time_days, var_normalized, color=self.COLOR_ACCENT_CYAN, lw=2)
        ax4.fill_between(time_days, 0, var_normalized, 
                        color=self.COLOR_ACCENT_CYAN, alpha=0.2)
        ax4.set_xlabel('Time [days]', fontweight='bold')
        ax4.set_ylabel('$I_s = \\sigma^2(t)/\\sigma^2(0)$', fontweight='bold')
        ax4.set_title('Intensity of Segregation', fontweight='bold')
        ax4.set_ylim(0, 1.1)
        ax4.grid(True, alpha=0.3)
        
        # ====== Panel 5: Diagnostics ======
        ax5 = fig.add_subplot(235, facecolor=self.COLOR_BG_LIGHTER)
        ax5.axis('off')
        
        diag = diagnostics if diagnostics else result.diagnostics
        
        info_lines = []
        info_lines.append("SIMULATION PARAMETERS")
        info_lines.append("─" * 35)
        info_lines.append(f"Domain: {basin.Lx/1e3:.0f} × {basin.Ly/1e3:.0f} km")
        info_lines.append(f"Depth: {basin.H:.0f} m")
        info_lines.append(f"Max velocity: {basin.U0:.3f} m/s")
        info_lines.append(f"Circulation time: {basin.T_circ_days:.2f} days")
        info_lines.append("")
        info_lines.append("LAGRANGIAN DIAGNOSTICS")
        info_lines.append("─" * 35)
        
        if diag:
            info_lines.append(f"Mass error: {diag.get('mass_error_relative', 0):.2e}")
            info_lines.append(f"Final I_s: {diag.get('intensity_segregation', 0):.4f}")
            info_lines.append(f"Mean displacement: {diag.get('mean_displacement_normalized', 0):.4f} L")
            if 'stretching_factor' in diag:
                info_lines.append(f"Stretching: {diag.get('stretching_factor', 1):.2f}×")
                info_lines.append(f"Circularity: {diag.get('circularity_final', 0):.4f}")
        
        info_text = "\n".join(info_lines)
        ax5.text(
            0.1, 0.95, info_text,
            transform=ax5.transAxes,
            fontsize=11, fontfamily='monospace',
            color=self.COLOR_TEXT,
            verticalalignment='top',
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor=self.COLOR_BG_PANEL,
                edgecolor=self.COLOR_GRID,
                alpha=0.9
            )
        )
        
        # ====== Panel 6: Velocity magnitude ======
        ax6 = fig.add_subplot(236, facecolor=self.COLOR_BG_LIGHTER)
        
        speed = np.sqrt(vf.vx**2 + vf.vy**2).T * 100  # cm/s
        im6 = ax6.pcolormesh(
            X_km, Y_km, speed,
            cmap='viridis', shading='auto'
        )
        ax6.set_xlabel('$x$ [km]', fontweight='bold')
        ax6.set_ylabel('$y$ [km]', fontweight='bold')
        ax6.set_title('Speed $|\\mathbf{v}|$', fontweight='bold')
        ax6.set_aspect('equal')
        cbar6 = plt.colorbar(im6, ax=ax6, pad=0.02)
        cbar6.set_label('Speed [cm/s]')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        plt.savefig(
            filepath, dpi=self.dpi,
            facecolor=self.COLOR_BG, edgecolor='none',
            bbox_inches='tight'
        )
        plt.close(fig)
    
    def create_animation(
        self,
        result: 'SimulationResult',
        filepath: str,
        n_frames: Optional[int] = None,
        duration_seconds: float = 10.0
    ):
        """
        Create animated GIF of tracer evolution.
        
        Args:
            result: SimulationResult from solver
            filepath: Output file path
            n_frames: Number of frames (None = use all outputs)
            duration_seconds: Target duration in seconds
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        basin = result.basin
        vf = result.velocity_field
        
        n_outputs = len(result.time)
        if n_frames is None or n_frames > n_outputs:
            n_frames = n_outputs
        
        # Select frames
        frame_indices = np.linspace(0, n_outputs - 1, n_frames, dtype=int)
        
        cmap = self._get_tracer_cmap(basin.tracer_name)
        
        X_km = vf.X / 1e3
        Y_km = vf.Y / 1e3
        
        vmin = basin.tracer_background
        vmax = basin.tracer_background + basin.tracer_anomaly
        
        frames = []
        
        print(f"      Generating {n_frames} frames...")
        
        for idx in tqdm(frame_indices, desc="      Rendering", ncols=70):
            fig = plt.figure(figsize=(10, 8), facecolor=self.COLOR_BG, dpi=100)
            ax = fig.add_subplot(111, facecolor=self.COLOR_BG_LIGHTER)
            
            tracer = result.mesh_tracer[idx, :, :].T
            
            im = ax.pcolormesh(
                X_km, Y_km, tracer,
                cmap=cmap, vmin=vmin, vmax=vmax, shading='auto'
            )
            
            # Streamlines overlay (light gray for visibility)
            ax.streamplot(
                X_km, Y_km, vf.vx.T, vf.vy.T,
                color='#404040', linewidth=0.3, density=1.0,
                arrowsize=0.5
            )
            
            t_days = result.time[idx] / 86400
            ax.set_title(
                f'{basin.tracer_name} Transport\n$t$ = {t_days:.2f} days',
                fontsize=14, fontweight='bold', color=self.COLOR_TITLE
            )
            ax.set_xlabel('$x$ [km]', fontweight='bold')
            ax.set_ylabel('$y$ [km]', fontweight='bold')
            ax.set_aspect('equal')
            
            cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
            cbar.set_label(f'{basin.tracer_name} [{basin.tracer_units}]')
            
            # Info box
            info = f'$L$ = {basin.Lx/1e3:.0f} km\n$U_0$ = {basin.U0:.2f} m/s'
            ax.text(
                0.02, 0.98, info,
                transform=ax.transAxes,
                fontsize=9, fontfamily='monospace',
                color=self.COLOR_TEXT,
                verticalalignment='top',
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor=self.COLOR_BG_PANEL,
                    edgecolor=self.COLOR_GRID,
                    alpha=0.8
                )
            )
            
            fig.tight_layout()
            
            # Render to buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100,
                       facecolor=self.COLOR_BG, edgecolor='none')
            buf.seek(0)
            frame_img = Image.open(buf).copy()
            frames.append(frame_img)
            buf.close()
            plt.close(fig)
        
        # Save GIF
        frame_duration_ms = int(duration_seconds * 1000 / n_frames)
        
        print(f"      Saving GIF ({n_frames} frames)...")
        
        frames[0].save(
            str(filepath),
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration_ms,
            loop=0,
            optimize=True
        )
        
        print(f"      ✓ Saved: {filepath.name}")
    
    def create_comparison_plot(
        self,
        results: Dict[str, 'SimulationResult'],
        filepath: str
    ):
        """
        Create comparison plot across multiple cases.
        
        Args:
            results: Dictionary mapping case names to results
            filepath: Output file path
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        n_cases = len(results)
        fig, axes = plt.subplots(2, n_cases, figsize=(5*n_cases, 10),
                                facecolor=self.COLOR_BG)
        
        if n_cases == 1:
            axes = axes.reshape(2, 1)
        
        for i, (name, result) in enumerate(results.items()):
            basin = result.basin
            vf = result.velocity_field
            
            cmap = self._get_tracer_cmap(basin.tracer_name)
            
            X_km = vf.X / 1e3
            Y_km = vf.Y / 1e3
            
            vmin = basin.tracer_background
            vmax = basin.tracer_background + basin.tracer_anomaly
            
            # Initial
            ax_init = axes[0, i]
            ax_init.set_facecolor(self.COLOR_BG_LIGHTER)
            tracer_init = result.mesh_tracer[0, :, :].T
            ax_init.pcolormesh(X_km, Y_km, tracer_init, cmap=cmap,
                              vmin=vmin, vmax=vmax, shading='auto')
            ax_init.set_title(f'{name}\nInitial', fontweight='bold', fontsize=10)
            ax_init.set_aspect('equal')
            ax_init.set_xlabel('$x$ [km]')
            ax_init.set_ylabel('$y$ [km]')
            
            # Final
            ax_final = axes[1, i]
            ax_final.set_facecolor(self.COLOR_BG_LIGHTER)
            tracer_final = result.mesh_tracer[-1, :, :].T
            im = ax_final.pcolormesh(X_km, Y_km, tracer_final, cmap=cmap,
                                     vmin=vmin, vmax=vmax, shading='auto')
            
            t_days = result.time[-1] / 86400
            ax_final.set_title(f'Final ($t$ = {t_days:.1f} days)', fontweight='bold', fontsize=10)
            ax_final.set_aspect('equal')
            ax_final.set_xlabel('$x$ [km]')
            ax_final.set_ylabel('$y$ [km]')
        
        fig.suptitle('Lagrangian Transport Comparison', fontsize=16, 
                    fontweight='bold', color=self.COLOR_TITLE, y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        plt.savefig(filepath, dpi=self.dpi, facecolor=self.COLOR_BG,
                   edgecolor='none', bbox_inches='tight')
        plt.close(fig)
