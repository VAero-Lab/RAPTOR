"""
Unified Visualization Module
============================

Provides publication-quality visualizations for:
    - Optimization convergence curves
    - Path evolution during optimization (snapshots)
    - 2D flight path over DEM contour map
    - 3D flight path over DEM surface
    - Energy/SOC profiles along optimized paths
    - Multi-objective Pareto front plots
    - Scenario comparison dashboards
    - Airspace map with regulatory zones
    - Path vs ceiling altitude bands
    - Topology comparison (airspace ON vs OFF)
    - Constraint penalty budget
    - Multi-leg mission SOC profile
    - Stall envelope
    - Three-path progressive comparison
    - Vehicle comparison
    - A* vs DE comparison

Author: Victor (LUAS-EPN / KU Leuven)
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Polygon, FancyArrowPatch
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as pe

from .dem import DEMInterface
from .path import FlightPath
from .energy import MissionEnergyResult, analyze_path_energy, AircraftEnergyParams
from .terrain import TerrainAnalyzer
from .config import MissionConstraints
from .airspace import AirspaceManager, AirspaceZone, ZoneType, CircularZone, PolygonalZone


# ═══════════════════════════════════════════════════════════════════════════
# STYLE DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════

_COLORS = {
    'energy': '#2196F3',
    'time': '#FF5722',
    'multi': '#4CAF50',
    'initial': '#9E9E9E',
    'terrain': '#795548',
    'clearance': '#F44336',
    'feasible': '#4CAF50',
    'infeasible': '#F44336',
}

_MODE_COLORS = {
    'energy': '#2196F3',
    'time': '#FF5722',
    'multi': '#4CAF50',
}

_SEGMENT_COLORS = {
    'VTOL_ASCEND': '#E91E63',
    'VTOL_DESCEND': '#9C27B0',
    'TRANSITION': '#FF9800',
    'FW_CLIMB': '#2196F3',
    'FW_CRUISE': '#4CAF50',
    'FW_DESCEND': '#00BCD4',
}

_ZONE_COLORS = {
    ZoneType.PROHIBITED:    ('#D32F2F', 0.35),
    ZoneType.RESTRICTED:    ('#FF5722', 0.25),
    ZoneType.AERODROME_CTR: ('#F44336', 0.20),
    ZoneType.ALTITUDE_LIMIT:('#FFC107', 0.08),
    ZoneType.POPULATED_AREA:('#FF9800', 0.25),
    ZoneType.ECOLOGICAL:    ('#4CAF50', 0.30),
    ZoneType.TEMPORAL:      ('#9C27B0', 0.20),
}

_ZONE_HATCHES = {
    ZoneType.PROHIBITED:    '///',
    ZoneType.AERODROME_CTR: '\\\\',
    ZoneType.ECOLOGICAL:    'xxx',
    ZoneType.POPULATED_AREA: '...',
}

_CONFIG_COLORS = {
    'A': '#2196F3',
    'B': '#FF9800',
    'C': '#4CAF50',
    'D': '#F44336',
}

plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 9,
    'font.family': 'sans-serif',
    'axes.titlesize': 11,
    'axes.labelsize': 9,
    'legend.fontsize': 7,
    'figure.facecolor': 'white',
})


# ═══════════════════════════════════════════════════════════════════════════
# CONVERGENCE PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_convergence(
    results: List,
    title: str = "Optimization Convergence",
    save_path: str = None,
) -> plt.Figure:
    """
    Plot convergence curves for one or more optimization results.

    Parameters
    ----------
    results : list of OptimizationResult
        Results from optimizer.optimize() calls.
    title : str
        Plot title.
    save_path : str, optional
        If provided, save figure to this path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: objective convergence
    ax = axes[0]
    for res in results:
        color = _MODE_COLORS.get(res.mode, '#333')
        label = f"{res.mode.upper()} (w_E={res.weights[0]:.1f})"
        ax.plot(res.convergence_history, color=color, label=label, linewidth=1.5)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Penalized Objective")
    ax.set_title("Objective Convergence")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Right: energy and time evolution
    ax2 = axes[1]
    bar_width = 0.35
    x = np.arange(len(results))

    # Initial values (same for all since same route)
    initial_e = [r.initial_energy_wh for r in results]
    initial_t = [r.initial_time_s / 60 for r in results]
    final_e = [r.energy_wh for r in results]
    final_t = [r.flight_time_s / 60 for r in results]

    ax2.bar(x - bar_width/2, initial_e, bar_width, label='Initial Energy',
            color='#BBDEFB', edgecolor='#1565C0', linewidth=0.5)
    ax2.bar(x + bar_width/2, final_e, bar_width, label='Optimized Energy',
            color='#2196F3', edgecolor='#0D47A1', linewidth=0.5)
    ax2.set_ylabel("Energy [Wh]", color='#2196F3')
    ax2.set_xticks(x)
    ax2.set_xticklabels([r.mode.upper() for r in results], fontsize=8)

    ax2t = ax2.twinx()
    ax2t.plot(x, initial_t, 'o--', color='#FFAB91', markersize=6,
              label='Initial Time')
    ax2t.plot(x, final_t, 's-', color='#FF5722', markersize=6,
              label='Optimized Time')
    ax2t.set_ylabel("Time [min]", color='#FF5722')

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2t.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=7)
    ax2.set_title("Energy & Time Comparison")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 2D PATH OVER DEM
# ═══════════════════════════════════════════════════════════════════════════

def plot_path_2d(
    dem: DEMInterface,
    paths: Dict[str, FlightPath],
    facilities: List = None,
    title: str = "Flight Paths over DEM",
    save_path: str = None,
    show_terrain_profile: bool = True,
) -> plt.Figure:
    """
    Plot flight paths on a 2D DEM contour map with terrain profiles.

    Parameters
    ----------
    dem : DEMInterface
        Terrain model.
    paths : dict
        Name → FlightPath mapping (e.g. {'Initial': path0, 'Optimized': path1}).
    facilities : list of Facility, optional
        Mark facility locations on the map.
    """
    n_cols = 2 if show_terrain_profile else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5.5))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # ── Left: Plan view (lat/lon contour map) ─────────────────────────────
    ax = axes[0]

    # Determine crop region from paths
    all_lats, all_lons = [], []
    for name, path in paths.items():
        wp = path.get_waypoints_array()
        all_lats.extend(wp[:, 0])
        all_lons.extend(wp[:, 1])

    lat_min, lat_max = min(all_lats), max(all_lats)
    lon_min, lon_max = min(all_lons), max(all_lons)
    margin = 0.02
    lat_min -= margin; lat_max += margin
    lon_min -= margin; lon_max += margin

    # Crop DEM for display
    lat_mask = (dem.lat_1d >= lat_min) & (dem.lat_1d <= lat_max)
    lon_mask = (dem.lon_1d >= lon_min) & (dem.lon_1d <= lon_max)
    lat_sub = dem.lat_1d[lat_mask]
    lon_sub = dem.lon_1d[lon_mask]
    elev_sub = dem.elev_grid[np.ix_(lat_mask, lon_mask)]

    LON, LAT = np.meshgrid(lon_sub, lat_sub)

    # Hillshade-style rendering
    ls = LightSource(azdeg=315, altdeg=35)
    rgb = ls.shade(elev_sub, cmap=cm.terrain, blend_mode='soft',
                   vmin=np.nanmin(elev_sub), vmax=np.nanmax(elev_sub))
    ax.imshow(rgb, extent=[lon_sub[0], lon_sub[-1], lat_sub[0], lat_sub[-1]],
              origin='lower', aspect='auto')

    # Contour lines
    levels = np.arange(
        np.nanmin(elev_sub) // 100 * 100,
        np.nanmax(elev_sub) + 200, 200
    )
    cs = ax.contour(LON, LAT, elev_sub, levels=levels, colors='k',
                    linewidths=0.3, alpha=0.4)
    ax.clabel(cs, inline=True, fontsize=5, fmt='%d')

    # Plot paths
    colors = list(_MODE_COLORS.values()) + ['#9C27B0', '#FFEB3B', '#795548']
    for idx, (name, path) in enumerate(paths.items()):
        wp = path.get_waypoints_array()
        color = colors[idx % len(colors)]
        ax.plot(wp[:, 1], wp[:, 0], '-', color=color, linewidth=2,
                label=name, alpha=0.9)
        # Start and end markers
        ax.plot(wp[0, 1], wp[0, 0], 'o', color=color, markersize=8,
                markeredgecolor='k', markeredgewidth=0.5)
        ax.plot(wp[-1, 1], wp[-1, 0], 's', color=color, markersize=8,
                markeredgecolor='k', markeredgewidth=0.5)

    # Facilities
    if facilities:
        for fac in facilities:
            if lat_min <= fac.lat <= lat_max and lon_min <= fac.lon <= lon_max:
                marker = '^' if fac.is_hangar else 'v'
                color = '#E91E63' if fac.is_hangar else '#FF9800'
                ax.plot(fac.lon, fac.lat, marker, color=color, markersize=9,
                        markeredgecolor='k', markeredgewidth=0.5, zorder=10)
                ax.annotate(fac.short_name, (fac.lon, fac.lat),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=6, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                    alpha=0.7, ec='none'))

    ax.set_xlabel("Longitude [°]")
    ax.set_ylabel("Latitude [°]")
    ax.set_title("Plan View (DEM)")
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.2)

    # ── Right: Terrain profile ────────────────────────────────────────────
    if show_terrain_profile and len(axes) > 1:
        ax2 = axes[1]

        for idx, (name, path) in enumerate(paths.items()):
            wp = path.get_waypoints_array()
            dist_km = wp[:, 4] / 1000.0
            alt = wp[:, 2]
            color = colors[idx % len(colors)]
            ax2.plot(dist_km, alt, '-', color=color, linewidth=1.5, label=name)

        # Terrain profile along the first path
        first_path = list(paths.values())[0]
        wp0 = first_path.get_waypoints_array()
        terrain_elevs = dem.elevation_batch(wp0[:, 0], wp0[:, 1])
        dist_km = wp0[:, 4] / 1000.0
        ax2.fill_between(dist_km, terrain_elevs, alpha=0.3,
                        color=_COLORS['terrain'], label='Terrain')
        ax2.plot(dist_km, terrain_elevs, '-', color=_COLORS['terrain'],
                linewidth=0.8, alpha=0.7)

        ax2.set_xlabel("Ground Distance [km]")
        ax2.set_ylabel("Altitude [m AMSL]")
        ax2.set_title("Altitude Profile")
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 3D PATH OVER DEM
# ═══════════════════════════════════════════════════════════════════════════

def plot_path_3d(
    dem: DEMInterface,
    paths: Dict[str, FlightPath],
    title: str = "3D Flight Path over DEM",
    save_path: str = None,
    elev_angle: float = 30,
    azim_angle: float = -60,
) -> plt.Figure:
    """
    Plot flight paths on a 3D DEM surface.

    Parameters
    ----------
    dem : DEMInterface
    paths : dict of name → FlightPath
    """
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Determine region
    all_lats, all_lons = [], []
    for path in paths.values():
        wp = path.get_waypoints_array()
        all_lats.extend(wp[:, 0]); all_lons.extend(wp[:, 1])

    margin = 0.015
    lat_min, lat_max = min(all_lats) - margin, max(all_lats) + margin
    lon_min, lon_max = min(all_lons) - margin, max(all_lons) + margin

    lat_mask = (dem.lat_1d >= lat_min) & (dem.lat_1d <= lat_max)
    lon_mask = (dem.lon_1d >= lon_min) & (dem.lon_1d <= lon_max)
    lat_sub = dem.lat_1d[lat_mask]
    lon_sub = dem.lon_1d[lon_mask]
    elev_sub = dem.elev_grid[np.ix_(lat_mask, lon_mask)]

    # Downsample for performance
    stride = max(1, len(lat_sub) // 80)
    lat_ds = lat_sub[::stride]
    lon_ds = lon_sub[::stride]
    elev_ds = elev_sub[::stride, ::stride]

    LON, LAT = np.meshgrid(lon_ds, lat_ds)

    # Plot terrain surface
    ls = LightSource(azdeg=315, altdeg=35)
    facecolors = ls.shade(elev_ds, cmap=cm.terrain, blend_mode='soft',
                          vmin=np.nanmin(elev_ds), vmax=np.nanmax(elev_ds))
    ax.plot_surface(LON, LAT, elev_ds, facecolors=facecolors,
                    rstride=1, cstride=1, antialiased=True,
                    shade=False, alpha=0.7)

    # Plot flight paths
    colors = list(_MODE_COLORS.values()) + ['#9C27B0', '#FFEB3B']
    for idx, (name, path) in enumerate(paths.items()):
        wp = path.get_waypoints_array()
        color = colors[idx % len(colors)]
        ax.plot(wp[:, 1], wp[:, 0], wp[:, 2], '-', color=color,
                linewidth=2.5, label=name, alpha=0.95)

        # Vertical lines to ground at key points (every 10th)
        for j in range(0, len(wp), max(1, len(wp) // 10)):
            terrain_z = dem.elevation(wp[j, 0], wp[j, 1])
            if not np.isnan(terrain_z):
                ax.plot([wp[j, 1], wp[j, 1]], [wp[j, 0], wp[j, 0]],
                       [terrain_z, wp[j, 2]], ':', color=color,
                       alpha=0.3, linewidth=0.5)

        # Start/end markers
        ax.scatter([wp[0, 1]], [wp[0, 0]], [wp[0, 2]], c=color,
                  marker='o', s=60, edgecolors='k', linewidths=0.5, zorder=10)
        ax.scatter([wp[-1, 1]], [wp[-1, 0]], [wp[-1, 2]], c=color,
                  marker='s', s=60, edgecolors='k', linewidths=0.5, zorder=10)

    ax.set_xlabel('Longitude [°]', fontsize=8, labelpad=8)
    ax.set_ylabel('Latitude [°]', fontsize=8, labelpad=8)
    ax.set_zlabel('Altitude [m]', fontsize=8, labelpad=8)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')
    ax.view_init(elev=elev_angle, azim=azim_angle)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# PATH EVOLUTION (SNAPSHOTS DURING OPTIMIZATION)
# ═══════════════════════════════════════════════════════════════════════════

def plot_path_evolution(
    dem: DEMInterface,
    initial_path: FlightPath,
    opt_result,
    ac: AircraftEnergyParams,
    title: str = "Path Evolution During Optimization",
    save_path: str = None,
    n_snapshots: int = 6,
) -> plt.Figure:
    """
    Show how the path evolves during optimization.

    Uses parameter snapshots recorded during DE generations.
    """
    snapshots = opt_result.parameter_history
    if len(snapshots) == 0:
        snapshots = [opt_result.parameter_vector]

    # Select evenly spaced snapshots
    n = min(n_snapshots, len(snapshots))
    indices = np.linspace(0, len(snapshots) - 1, n, dtype=int)

    n_rows = 2
    n_cols = (n + 1) // 2  # +1 for ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 7))
    axes_flat = axes.flatten()

    # Get terrain profile for background
    wp_init = initial_path.get_waypoints_array()
    terrain_elevs = dem.elevation_batch(wp_init[:, 0], wp_init[:, 1])

    # Clone path for parameter setting
    from .optimizer import PathOptimizer
    cloner = PathOptimizer.__new__(PathOptimizer)

    for panel_idx, snap_idx in enumerate(indices):
        if panel_idx >= len(axes_flat):
            break
        ax = axes_flat[panel_idx]

        theta = snapshots[snap_idx]

        # Reconstruct path with these parameters
        path_snap = cloner._clone_path(initial_path)
        try:
            path_snap.parameter_vector = theta
            wp = path_snap.get_waypoints_array()
            dist_km = wp[:, 4] / 1000.0

            # Terrain fill
            terrain_snap = dem.elevation_batch(wp[:, 0], wp[:, 1])
            ax.fill_between(dist_km, terrain_snap, alpha=0.3,
                           color=_COLORS['terrain'])
            ax.plot(dist_km, terrain_snap, '-', color=_COLORS['terrain'],
                   linewidth=0.5)

            # Flight path colored by segment type
            seg_starts = [0]
            for seg in path_snap.segments:
                seg_starts.append(seg_starts[-1] + len(seg.waypoints))

            for s_idx, seg in enumerate(path_snap.segments):
                start = seg_starts[s_idx]
                end = seg_starts[s_idx + 1]
                if end > len(dist_km):
                    end = len(dist_km)
                color = _SEGMENT_COLORS.get(seg.segment_type.value, '#333')
                ax.plot(dist_km[start:end], wp[start:end, 2], '-',
                       color=color, linewidth=1.5)

            gen_num = int(snap_idx / max(len(snapshots)-1, 1) *
                         opt_result.n_iterations) if len(snapshots) > 1 else opt_result.n_iterations
            ax.set_title(f"Gen {gen_num}", fontsize=9)
        except Exception:
            ax.set_title(f"Snapshot {snap_idx} (error)", fontsize=9)

        ax.set_xlabel("Distance [km]", fontsize=7)
        ax.set_ylabel("Alt [m]", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)

    # Hide unused panels
    for i in range(len(indices), len(axes_flat)):
        axes_flat[i].set_visible(False)

    # Legend
    patches = [mpatches.Patch(color=c, label=n.replace('_', ' '))
               for n, c in _SEGMENT_COLORS.items()]
    fig.legend(handles=patches, loc='lower center', ncol=6, fontsize=7)

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# ENERGY / SOC PROFILE
# ═══════════════════════════════════════════════════════════════════════════

def plot_energy_profile(
    energy_result: MissionEnergyResult,
    title: str = "Energy & SOC Profile",
    save_path: str = None,
) -> plt.Figure:
    """Plot power and SOC profiles along the optimized path."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    # ── Power profile ─────────────────────────────────────────────────────
    ax = axes[0]
    cum_time = 0
    for seg in energy_result.segments:
        t_start = cum_time
        t_end = cum_time + seg.duration
        color = _SEGMENT_COLORS.get(seg.segment_type, '#333')

        ax.barh(0, seg.duration, left=t_start, height=seg.P_elec / 1000,
                color=color, alpha=0.7, edgecolor='k', linewidth=0.3)
        ax.fill_between([t_start, t_end], 0, seg.P_elec,
                       color=color, alpha=0.5)
        ax.plot([t_start, t_end], [seg.P_elec, seg.P_elec],
               color=color, linewidth=1.5)

        # Label
        mid_t = (t_start + t_end) / 2
        if seg.duration > 5:
            ax.text(mid_t, seg.P_elec + 50, f"{seg.P_elec:.0f}W",
                   ha='center', va='bottom', fontsize=6, rotation=45)

        cum_time = t_end

    ax.set_ylabel("Electrical Power [W]")
    ax.set_title("Power Profile by Segment")
    ax.grid(True, alpha=0.3)

    # ── SOC profile ───────────────────────────────────────────────────────
    ax2 = axes[1]
    timeline = energy_result.battery_timeline
    times = [s.time for s in timeline]
    socs = [s.percent_remaining for s in timeline]

    ax2.plot(times, socs, '-', color='#2196F3', linewidth=2)
    ax2.axhline(y=15, color='r', linestyle='--', alpha=0.5, label='Min SOC (15%)')
    ax2.fill_between(times, socs, alpha=0.15, color='#2196F3')

    ax2.set_xlabel("Mission Time [s]")
    ax2.set_ylabel("Battery SOC [%]")
    ax2.set_title("Battery State of Charge")
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# PARETO FRONT
# ═══════════════════════════════════════════════════════════════════════════

def plot_pareto_front(
    results: List,
    title: str = "Pareto Front: Energy vs. Time",
    save_path: str = None,
) -> plt.Figure:
    """
    Plot the Pareto front from a weight sweep.

    Parameters
    ----------
    results : list of OptimizationResult
        From PathOptimizer.pareto_sweep().
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    energies = [r.energy_wh for r in results]
    times = [r.flight_time_s / 60 for r in results]
    w_energies = [r.weights[0] for r in results]
    feasible = [r.fully_feasible for r in results]

    # ── Left: Pareto front ────────────────────────────────────────────────
    ax = axes[0]
    for i, (e, t, w, f) in enumerate(zip(energies, times, w_energies, feasible)):
        color = cm.coolwarm(w)
        marker = 'o' if f else 'x'
        size = 80 if f else 60
        ax.scatter(t, e, c=[color], marker=marker, s=size,
                  edgecolors='k', linewidths=0.5, zorder=5)
        ax.annotate(f"w={w:.1f}", (t, e), xytext=(5, 5),
                   textcoords='offset points', fontsize=6)

    # Connect with line
    ax.plot(times, energies, '--', color='gray', alpha=0.5, linewidth=0.8)

    ax.set_xlabel("Flight Time [min]")
    ax.set_ylabel("Energy Consumption [Wh]")
    ax.set_title("Pareto Front")
    ax.grid(True, alpha=0.3)

    # Colorbar for weights
    sm = cm.ScalarMappable(cmap=cm.coolwarm,
                           norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('w_energy', fontsize=8)

    # ── Right: Trade-off details ──────────────────────────────────────────
    ax2 = axes[1]
    x = np.arange(len(results))

    ax2.bar(x - 0.2, energies, 0.35, label='Energy [Wh]',
            color='#2196F3', alpha=0.7)
    ax2r = ax2.twinx()
    ax2r.bar(x + 0.2, times, 0.35, label='Time [min]',
             color='#FF5722', alpha=0.7)

    ax2.set_xlabel("Weight Configuration")
    ax2.set_ylabel("Energy [Wh]", color='#2196F3')
    ax2r.set_ylabel("Time [min]", color='#FF5722')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"({r.weights[0]:.1f},{r.weights[1]:.1f})"
                         for r in results], fontsize=6, rotation=45)
    ax2.set_title("Trade-off by Weight")
    ax2.grid(True, alpha=0.3)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# SCENARIO COMPARISON DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

def plot_scenario_dashboard(
    scenario_results: Dict[str, List],
    title: str = "Scenario Optimization Dashboard",
    save_path: str = None,
) -> plt.Figure:
    """
    Dashboard comparing optimization results across scenario legs.

    Parameters
    ----------
    scenario_results : dict
        leg_key → list of OptimizationResult (from optimize_scenario).
    """
    n_legs = len(scenario_results)
    fig, axes = plt.subplots(2, max(n_legs, 2), figsize=(5 * n_legs, 8))
    if n_legs == 1:
        axes = axes.reshape(2, 1)

    for col, (leg_key, results) in enumerate(scenario_results.items()):
        if col >= axes.shape[1]:
            break

        # Top: Energy comparison
        ax = axes[0, col]
        modes = [r.mode for r in results]
        initial_e = [r.initial_energy_wh for r in results]
        opt_e = [r.energy_wh for r in results]

        x = np.arange(len(results))
        ax.bar(x - 0.2, initial_e, 0.35, label='Initial',
               color='#BBDEFB', edgecolor='#1565C0')
        ax.bar(x + 0.2, opt_e, 0.35, label='Optimized',
               color='#2196F3', edgecolor='#0D47A1')

        for i, r in enumerate(results):
            if r.energy_improvement_pct != 0:
                ax.annotate(f"{r.energy_improvement_pct:+.1f}%",
                           (i + 0.2, opt_e[i]), xytext=(0, 5),
                           textcoords='offset points', fontsize=6,
                           ha='center', fontweight='bold',
                           color='green' if r.energy_improvement_pct > 0 else 'red')

        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in modes], fontsize=7)
        ax.set_ylabel("Energy [Wh]")
        ax.set_title(leg_key.replace('_', ' '), fontsize=9)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

        # Bottom: Time comparison
        ax2 = axes[1, col]
        initial_t = [r.initial_time_s / 60 for r in results]
        opt_t = [r.flight_time_s / 60 for r in results]

        ax2.bar(x - 0.2, initial_t, 0.35, label='Initial',
                color='#FFCCBC', edgecolor='#BF360C')
        ax2.bar(x + 0.2, opt_t, 0.35, label='Optimized',
                color='#FF5722', edgecolor='#BF360C')

        for i, r in enumerate(results):
            if r.time_improvement_pct != 0:
                ax2.annotate(f"{r.time_improvement_pct:+.1f}%",
                            (i + 0.2, opt_t[i]), xytext=(0, 5),
                            textcoords='offset points', fontsize=6,
                            ha='center', fontweight='bold',
                            color='green' if r.time_improvement_pct > 0 else 'red')

        ax2.set_xticks(x)
        ax2.set_xticklabels([m.upper() for m in modes], fontsize=7)
        ax2.set_ylabel("Time [min]")
        ax2.legend(fontsize=6)
        ax2.grid(True, alpha=0.3)

    # Hide unused columns
    for col in range(len(scenario_results), axes.shape[1]):
        axes[0, col].set_visible(False)
        axes[1, col].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 1. AIRSPACE MAP — The "hero figure"
# ═══════════════════════════════════════════════════════════════════════════

def plot_airspace_map(
    dem: DEMInterface,
    airspace: AirspaceManager,
    paths: Dict[str, FlightPath] = None,
    routed_paths: Dict[str, 'RoutedPath'] = None,
    facilities: List = None,
    title: str = "Quito DMQ — Airspace Restrictions and Flight Paths",
    lat_range: Tuple[float, float] = None,
    lon_range: Tuple[float, float] = None,
    save_path: str = None,
    figsize: Tuple[float, float] = (10, 8),
) -> plt.Figure:
    """
    Plot DEM with colored airspace zones and flight paths overlaid.

    This is the primary publication figure showing the interaction
    between terrain, regulatory zones, and optimized flight corridors.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # ── Determine extent ──────────────────────────────────────────────
    if lat_range is None or lon_range is None:
        # Auto from DEM and paths
        lat_min = dem.metadata.lat_min
        lat_max = dem.metadata.lat_max
        lon_min = dem.metadata.lon_min
        lon_max = dem.metadata.lon_max

        if paths:
            for p in paths.values():
                wp = p.get_waypoints_array()
                lat_min = min(lat_min, np.min(wp[:, 0]) - 0.02)
                lat_max = max(lat_max, np.max(wp[:, 0]) + 0.02)
                lon_min = min(lon_min, np.min(wp[:, 1]) - 0.02)
                lon_max = max(lon_max, np.max(wp[:, 1]) + 0.02)
    else:
        lat_min, lat_max = lat_range
        lon_min, lon_max = lon_range

    # ── DEM base layer ────────────────────────────────────────────────
    lat_mask = (dem.lat_1d >= lat_min) & (dem.lat_1d <= lat_max)
    lon_mask = (dem.lon_1d >= lon_min) & (dem.lon_1d <= lon_max)
    lat_sub = dem.lat_1d[lat_mask]
    lon_sub = dem.lon_1d[lon_mask]
    elev_sub = dem.elev_grid[np.ix_(lat_mask, lon_mask)]

    LON, LAT = np.meshgrid(lon_sub, lat_sub)

    ls = LightSource(azdeg=315, altdeg=35)
    rgb = ls.shade(elev_sub, cmap=cm.terrain, blend_mode='soft',
                   vmin=np.nanmin(elev_sub) - 200,
                   vmax=np.nanmax(elev_sub) + 200)
    ax.imshow(rgb, extent=[lon_sub[0], lon_sub[-1], lat_sub[0], lat_sub[-1]],
              origin='lower', aspect='auto')

    # Contour lines
    levels = np.arange(
        np.nanmin(elev_sub) // 200 * 200,
        np.nanmax(elev_sub) + 300, 200
    )
    cs = ax.contour(LON, LAT, elev_sub, levels=levels, colors='k',
                    linewidths=0.3, alpha=0.3)
    ax.clabel(cs, inline=True, fontsize=5, fmt='%d')

    # ── Airspace zones ────────────────────────────────────────────────
    legend_patches = []
    for zone in airspace.active_zones():
        if zone.is_global:
            continue  # Don't plot global zones (AGL ceiling covers everything)

        color, alpha = _ZONE_COLORS.get(zone.zone_type, ('#999', 0.2))
        hatch = _ZONE_HATCHES.get(zone.zone_type, None)

        if isinstance(zone.geometry, CircularZone):
            cz = zone.geometry
            # Convert radius to approximate degree offset
            r_lat = cz.radius_m / 111320.0
            r_lon = cz.radius_m / (111320.0 * np.cos(np.radians(cz.center_lat)))

            # Draw as ellipse for aspect correction
            theta = np.linspace(0, 2 * np.pi, 100)
            elat = cz.center_lat + r_lat * np.sin(theta)
            elon = cz.center_lon + r_lon * np.cos(theta)

            ax.fill(elon, elat, color=color, alpha=alpha, zorder=2)
            if hatch:
                ax.fill(elon, elat, color='none', edgecolor=color,
                        hatch=hatch, alpha=0.4, zorder=2)
            ax.plot(elon, elat, '-', color=color, linewidth=1.0, alpha=0.7, zorder=2)

            # Label
            if (lat_min <= cz.center_lat <= lat_max and
                lon_min <= cz.center_lon <= lon_max):
                ax.text(cz.center_lon, cz.center_lat, zone.zone_id,
                       ha='center', va='center', fontsize=6, fontweight='bold',
                       color=color, alpha=0.8,
                       path_effects=[pe.withStroke(linewidth=2, foreground='white')])

        elif isinstance(zone.geometry, PolygonalZone):
            pz = zone.geometry
            verts = [(lon, lat) for lat, lon in pz.vertices]
            poly = Polygon(verts, closed=True, facecolor=color, alpha=alpha,
                          edgecolor=color, linewidth=1.2, zorder=2)
            ax.add_patch(poly)
            if hatch:
                poly_h = Polygon(verts, closed=True, facecolor='none',
                                edgecolor=color, hatch=hatch, alpha=0.4, zorder=2)
                ax.add_patch(poly_h)

            # Label at centroid
            c_lat = np.mean([v[0] for v in pz.vertices])
            c_lon = np.mean([v[1] for v in pz.vertices])
            if lat_min <= c_lat <= lat_max and lon_min <= c_lon <= lon_max:
                ax.text(c_lon, c_lat, zone.zone_id, ha='center', va='center',
                       fontsize=6, fontweight='bold', color=color, alpha=0.8,
                       path_effects=[pe.withStroke(linewidth=2, foreground='white')])

        legend_patches.append(
            plt.Rectangle((0, 0), 1, 1, fc=color, alpha=alpha,
                          label=f"{zone.zone_id} ({zone.zone_type.value})")
        )

    # ── Flight paths ──────────────────────────────────────────────────
    path_colors = ['#E91E63', '#00BCD4', '#CDDC39', '#7C4DFF', '#FF6E40']
    all_paths = {}
    if paths:
        all_paths.update(paths)
    if routed_paths:
        for name, rp in routed_paths.items():
            all_paths[name] = rp.flight_path

    for idx, (name, path) in enumerate(all_paths.items()):
        wp = path.get_waypoints_array()
        color = path_colors[idx % len(path_colors)]
        ax.plot(wp[:, 1], wp[:, 0], '-', color=color, linewidth=2.5,
                label=name, alpha=0.9, zorder=5,
                path_effects=[pe.withStroke(linewidth=4, foreground='white')])
        ax.plot(wp[0, 1], wp[0, 0], 'o', color=color, markersize=10,
                markeredgecolor='k', markeredgewidth=1, zorder=6)
        ax.plot(wp[-1, 1], wp[-1, 0], 's', color=color, markersize=10,
                markeredgecolor='k', markeredgewidth=1, zorder=6)

    # Plot intermediate waypoints for routed paths
    if routed_paths:
        for idx, (name, rp) in enumerate(routed_paths.items()):
            color = path_colors[idx % len(path_colors)]
            wp_pos = rp.waypoint_positions
            for i, (lat, lon) in enumerate(wp_pos[1:-1], 1):
                ax.plot(lon, lat, 'D', color=color, markersize=7,
                        markeredgecolor='k', markeredgewidth=0.8, zorder=6)
                ax.annotate(f'WP{i}', (lon, lat), xytext=(5, 5),
                           textcoords='offset points', fontsize=6,
                           fontweight='bold', color=color,
                           path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    # ── Facilities ────────────────────────────────────────────────────
    if facilities:
        for fac in facilities:
            if lat_min <= fac.lat <= lat_max and lon_min <= fac.lon <= lon_max:
                marker = '^' if getattr(fac, 'is_hangar', False) else 'v'
                fc = '#E91E63' if getattr(fac, 'is_hangar', False) else '#FFFFFF'
                ax.plot(fac.lon, fac.lat, marker, color=fc, markersize=9,
                        markeredgecolor='k', markeredgewidth=0.8, zorder=7)
                ax.annotate(getattr(fac, 'short_name', fac.name),
                           (fac.lon, fac.lat), xytext=(6, 6),
                           textcoords='offset points', fontsize=6,
                           fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                    alpha=0.85, ec='none'),
                           zorder=7)

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude [°]")
    ax.set_ylabel("Latitude [°]")
    ax.set_title(title, fontsize=12, fontweight='bold')

    # Combined legend
    handles, labels = ax.get_legend_handles_labels()
    handles += legend_patches
    ax.legend(handles=handles, loc='lower left', fontsize=6,
             framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 2. PATH VS CEILING — Altitude band visualization
# ═══════════════════════════════════════════════════════════════════════════

def plot_path_vs_ceiling(
    dem: DEMInterface,
    airspace: AirspaceManager,
    paths: Dict[str, FlightPath],
    title: str = "Flight Profile with Regulatory Ceiling",
    save_path: str = None,
) -> plt.Figure:
    """
    Plot altitude profiles with terrain floor and AGL ceiling bands.

    Shows the "flyable corridor" between terrain + clearance and the
    AGL regulatory ceiling. The path must stay within this band.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    path_colors = ['#E91E63', '#00BCD4', '#CDDC39', '#7C4DFF']

    # Use first path for terrain reference
    ref_path = list(paths.values())[0]
    wp0 = ref_path.get_waypoints_array()
    dist_km = wp0[:, 4] / 1000.0
    lats = wp0[:, 0]
    lons = wp0[:, 1]

    # Terrain profile
    terrain = dem.elevation_batch(lats, lons)

    # Compute AGL ceilings along the route
    agl_ceilings = np.full_like(terrain, np.nan)
    urban_ceilings = np.full_like(terrain, np.nan)

    for i in range(len(lats)):
        if np.isnan(terrain[i]):
            continue
        # General AGL ceiling
        agl_ceilings[i] = terrain[i] + 120.0

        # Check for urban/populated area ceiling
        for zone in airspace.active_zones():
            if zone.zone_type == ZoneType.POPULATED_AREA:
                if zone.horizontal_contains(lats[i], lons[i]):
                    if zone.is_agl:
                        urban_ceilings[i] = terrain[i] + zone.altitude_ceiling_m
                    break

    # Minimum clearance floor
    clearance_floor = terrain + 50.0  # 50m min AGL

    # ── Terrain fill ──────────────────────────────────────────────────
    ax.fill_between(dist_km, terrain, alpha=0.4, color='#8D6E63',
                   label='Terrain')
    ax.plot(dist_km, terrain, '-', color='#5D4037', linewidth=1.0)

    # ── Clearance floor ───────────────────────────────────────────────
    ax.plot(dist_km, clearance_floor, '--', color='#F44336', linewidth=0.8,
           alpha=0.6, label='Min clearance (50m AGL)')

    # ── AGL ceiling band ──────────────────────────────────────────────
    valid = ~np.isnan(agl_ceilings)
    ax.fill_between(dist_km[valid], agl_ceilings[valid],
                   np.full(np.sum(valid), np.nanmax(agl_ceilings) + 200),
                   alpha=0.12, color='#FFC107',
                   label='Above 120m AGL (restricted)')
    ax.plot(dist_km[valid], agl_ceilings[valid], '-', color='#F57F17',
           linewidth=1.5, alpha=0.8, label='120m AGL ceiling')

    # ── Urban ceiling ─────────────────────────────────────────────────
    urban_valid = ~np.isnan(urban_ceilings)
    if np.any(urban_valid):
        ax.fill_between(dist_km[urban_valid], urban_ceilings[urban_valid],
                       agl_ceilings[urban_valid],
                       alpha=0.20, color='#FF9800',
                       label='60m AGL urban ceiling')
        ax.plot(dist_km[urban_valid], urban_ceilings[urban_valid], '-',
               color='#E65100', linewidth=1.5, alpha=0.9)

    # ── Flyable corridor annotation ───────────────────────────────────
    mid_idx = len(dist_km) // 2
    if valid[mid_idx] and not np.isnan(terrain[mid_idx]):
        floor_val = clearance_floor[mid_idx]
        ceil_val = agl_ceilings[mid_idx]
        band = ceil_val - floor_val
        ax.annotate('', xy=(dist_km[mid_idx], ceil_val),
                   xytext=(dist_km[mid_idx], floor_val),
                   arrowprops=dict(arrowstyle='<->', color='#388E3C',
                                  lw=1.5, ls='-'))
        ax.text(dist_km[mid_idx] + 0.2, (floor_val + ceil_val) / 2,
               f'{band:.0f}m\nband', fontsize=7, color='#388E3C',
               va='center', fontweight='bold')

    # ── Flight paths ──────────────────────────────────────────────────
    for idx, (name, path) in enumerate(paths.items()):
        wp = path.get_waypoints_array()
        d_km = wp[:, 4] / 1000.0
        alt = wp[:, 2]
        color = path_colors[idx % len(path_colors)]
        ax.plot(d_km, alt, '-', color=color, linewidth=2.5, label=name,
               zorder=5, path_effects=[pe.withStroke(linewidth=4, foreground='white')])

    ax.set_xlabel("Ground Distance [km]")
    ax.set_ylabel("Altitude [m AMSL]")
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=7, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Set y-axis to show terrain nicely
    y_min = np.nanmin(terrain) - 100
    y_max = max(np.nanmax(agl_ceilings[valid]) + 300,
                max(wp[:, 2].max() for _, p in paths.items()
                    for wp in [p.get_waypoints_array()]) + 200)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 3. TOPOLOGY COMPARISON — Side by side WITH vs WITHOUT constraints
# ═══════════════════════════════════════════════════════════════════════════

def plot_topology_comparison(
    dem: DEMInterface,
    airspace: AirspaceManager,
    path_off: FlightPath,
    path_on: FlightPath,
    rp_off: 'RoutedPath' = None,
    rp_on: 'RoutedPath' = None,
    facilities: List = None,
    title: str = "Topology Change: Airspace OFF vs ON",
    save_path: str = None,
) -> plt.Figure:
    """
    Side-by-side comparison of optimal paths with and without
    airspace constraints. Shows both plan view and altitude profile.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    configs = [
        ('OFF (no airspace)', path_off, rp_off, '#2196F3'),
        ('ON (RDAC 101)',     path_on,  rp_on,  '#F44336'),
    ]

    for col, (label, path, rp, color) in enumerate(configs):
        # ── Plan view (top row) ───────────────────────────────────
        ax = axes[0, col]
        wp = path.get_waypoints_array()

        # DEM region
        margin = 0.025
        lat_min = np.min(wp[:, 0]) - margin
        lat_max = np.max(wp[:, 0]) + margin
        lon_min = np.min(wp[:, 1]) - margin
        lon_max = np.max(wp[:, 1]) + margin

        # Also include the other path to keep same extent
        other_path = path_on if col == 0 else path_off
        other_wp = other_path.get_waypoints_array()
        lat_min = min(lat_min, np.min(other_wp[:, 0]) - margin)
        lat_max = max(lat_max, np.max(other_wp[:, 0]) + margin)
        lon_min = min(lon_min, np.min(other_wp[:, 1]) - margin)
        lon_max = max(lon_max, np.max(other_wp[:, 1]) + margin)

        lat_mask = (dem.lat_1d >= lat_min) & (dem.lat_1d <= lat_max)
        lon_mask = (dem.lon_1d >= lon_min) & (dem.lon_1d <= lon_max)

        if np.any(lat_mask) and np.any(lon_mask):
            lat_sub = dem.lat_1d[lat_mask]
            lon_sub = dem.lon_1d[lon_mask]
            elev_sub = dem.elev_grid[np.ix_(lat_mask, lon_mask)]

            ls_obj = LightSource(azdeg=315, altdeg=35)
            rgb = ls_obj.shade(elev_sub, cmap=cm.terrain, blend_mode='soft',
                              vmin=np.nanmin(elev_sub) - 200,
                              vmax=np.nanmax(elev_sub) + 200)
            ax.imshow(rgb, extent=[lon_sub[0], lon_sub[-1],
                                   lat_sub[0], lat_sub[-1]],
                     origin='lower', aspect='auto')

        # Airspace zones (only for ON column)
        if col == 1:
            for zone in airspace.active_zones():
                if zone.is_global:
                    continue
                zc, za = _ZONE_COLORS.get(zone.zone_type, ('#999', 0.2))
                if isinstance(zone.geometry, CircularZone):
                    cz = zone.geometry
                    r_lat = cz.radius_m / 111320.0
                    r_lon = cz.radius_m / (111320.0 * np.cos(np.radians(cz.center_lat)))
                    theta = np.linspace(0, 2*np.pi, 80)
                    ax.fill(cz.center_lon + r_lon*np.cos(theta),
                           cz.center_lat + r_lat*np.sin(theta),
                           color=zc, alpha=za, zorder=2)
                elif isinstance(zone.geometry, PolygonalZone):
                    verts = [(lon, lat) for lat, lon in zone.geometry.vertices]
                    poly = Polygon(verts, closed=True, facecolor=zc, alpha=za,
                                  edgecolor=zc, linewidth=1, zorder=2)
                    ax.add_patch(poly)

        # Direct line (dashed gray)
        ax.plot([wp[0, 1], wp[-1, 1]], [wp[0, 0], wp[-1, 0]], '--',
               color='gray', linewidth=1, alpha=0.5, zorder=3)

        # Flight path
        ax.plot(wp[:, 1], wp[:, 0], '-', color=color, linewidth=2.5,
               zorder=5, path_effects=[pe.withStroke(linewidth=4, foreground='white')])
        ax.plot(wp[0, 1], wp[0, 0], 'o', color=color, markersize=10,
               markeredgecolor='k', markeredgewidth=1, zorder=6)
        ax.plot(wp[-1, 1], wp[-1, 0], 's', color=color, markersize=10,
               markeredgecolor='k', markeredgewidth=1, zorder=6)

        # Waypoint markers for routed paths
        if rp is not None:
            for i, (lat, lon) in enumerate(rp.waypoint_positions[1:-1], 1):
                ax.plot(lon, lat, 'D', color=color, markersize=7,
                       markeredgecolor='k', markeredgewidth=0.8, zorder=6)

        # Facilities
        if facilities:
            for fac in facilities:
                if lat_min <= fac.lat <= lat_max and lon_min <= fac.lon <= lon_max:
                    mk = '^' if getattr(fac, 'is_hangar', False) else 'v'
                    ax.plot(fac.lon, fac.lat, mk, color='white', markersize=8,
                           markeredgecolor='k', markeredgewidth=0.8, zorder=7)
                    ax.annotate(getattr(fac, 'short_name', fac.name),
                               (fac.lon, fac.lat), xytext=(5, 5),
                               textcoords='offset points', fontsize=5,
                               fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.15', fc='white',
                                        alpha=0.85, ec='none'))

        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_xlabel("Longitude [°]")
        ax.set_ylabel("Latitude [°]")
        ax.set_title(f"Plan View — {label}", fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.2)

        # Annotation box with metrics
        if rp is not None:
            topo = rp.topology_summary()
            from .energy import analyze_path_energy, AircraftEnergyParams as AEP
            aep = AEP()
            er = analyze_path_energy(path, aep)
            info = (f"E = {er.total_energy_wh:.0f} Wh\n"
                    f"t = {er.total_time/60:.1f} min\n"
                    f"LatDev = {topo['max_lateral_deviation_m']:.0f} m\n"
                    f"Stretch = {topo['route_stretch_factor']:.3f}×")
            ax.text(0.02, 0.98, info, transform=ax.transAxes,
                   fontsize=7, va='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', fc='white', alpha=0.9, ec='gray'))

        # ── Altitude profile (bottom row) ─────────────────────────
        ax2 = axes[1, col]
        d_km = wp[:, 4] / 1000.0

        # Terrain
        terrain = dem.elevation_batch(wp[:, 0], wp[:, 1])
        ax2.fill_between(d_km, terrain, alpha=0.35, color='#8D6E63')
        ax2.plot(d_km, terrain, '-', color='#5D4037', linewidth=0.8)

        # AGL ceiling
        valid = ~np.isnan(terrain)
        agl_ceil = terrain.copy()
        agl_ceil[valid] += 120.0
        ax2.plot(d_km[valid], agl_ceil[valid], '-', color='#F57F17',
                linewidth=1.2, alpha=0.7, label='120m AGL')
        ax2.fill_between(d_km[valid], agl_ceil[valid],
                        agl_ceil[valid] + 500, alpha=0.08, color='#FFC107')

        # Flight path
        ax2.plot(d_km, wp[:, 2], '-', color=color, linewidth=2.5,
                path_effects=[pe.withStroke(linewidth=4, foreground='white')])

        ax2.set_xlabel("Ground Distance [km]")
        ax2.set_ylabel("Altitude [m AMSL]")
        ax2.set_title(f"Altitude Profile — {label}", fontsize=10)
        ax2.legend(fontsize=6)
        ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 4. CONSTRAINT PENALTY BUDGET — Stacked bar chart
# ═══════════════════════════════════════════════════════════════════════════

def plot_constraint_budget(
    scenario_results: Dict[str, List[Dict]],
    title: str = "Constraint Penalty Budget — Energy Cost per Regulatory Layer",
    save_path: str = None,
) -> plt.Figure:
    """
    Stacked bar chart showing the energy cost of each constraint layer.

    Parameters
    ----------
    scenario_results : dict
        Keys = scenario labels, values = list of result dicts
        (from experiment 1), each with 'label' and 'E' keys.
        Expected order: A, B, C, D.
    """
    scenarios = list(scenario_results.keys())
    n_scenarios = len(scenarios)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: Absolute energy per config ──────────────────────────────
    ax = axes[0]
    x = np.arange(n_scenarios)
    width = 0.18
    config_labels = ['A: terrain', 'B: AGL', 'C: urban', 'D: full']
    config_colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336']

    for i, (clbl, ccol) in enumerate(zip(config_labels, config_colors)):
        vals = []
        for sname in scenarios:
            rows = scenario_results[sname]
            if i < len(rows):
                vals.append(rows[i]['E'])
            else:
                vals.append(0)
        ax.bar(x + i * width, vals, width, label=clbl, color=ccol, alpha=0.8,
              edgecolor='k', linewidth=0.3)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(scenarios, fontsize=7)
    ax.set_ylabel("Energy [Wh]")
    ax.set_title("Energy by Constraint Configuration")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    # ── Right: Incremental cost (delta per layer) ─────────────────────
    ax2 = axes[1]
    bar_width = 0.5
    layer_labels = ['B−A: AGL ceiling', 'C−B: Urban zones', 'D−C: Airport+eco']
    layer_colors = ['#FF9800', '#4CAF50', '#F44336']

    for s_idx, sname in enumerate(scenarios):
        rows = scenario_results[sname]
        bottom = 0
        energies = [r['E'] for r in rows]
        base = energies[0]

        for i in range(1, min(len(energies), 4)):
            delta = energies[i] - energies[i - 1]
            color = layer_colors[min(i - 1, len(layer_colors) - 1)]
            label = layer_labels[min(i - 1, len(layer_labels) - 1)] if s_idx == 0 else None

            # Color positive deltas (cost) normally, negative (savings) lighter
            if delta >= 0:
                ax2.bar(s_idx, delta, bar_width, bottom=bottom,
                       color=color, alpha=0.8, edgecolor='k', linewidth=0.3,
                       label=label)
            else:
                ax2.bar(s_idx, delta, bar_width, bottom=bottom,
                       color=color, alpha=0.3, edgecolor='k', linewidth=0.3,
                       hatch='///', label=label)
            bottom += delta

    ax2.axhline(y=0, color='k', linewidth=0.8)
    ax2.set_xticks(range(n_scenarios))
    ax2.set_xticklabels(scenarios, fontsize=7)
    ax2.set_ylabel("ΔEnergy [Wh] per layer")
    ax2.set_title("Incremental Constraint Cost")
    ax2.legend(fontsize=7, loc='best')
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 5. MISSION SOC PROFILE — Multi-leg battery timeline
# ═══════════════════════════════════════════════════════════════════════════

def plot_mission_soc(
    leg_results: List[Dict],
    title: str = "Multi-Point Mission — Battery SOC Profile",
    save_path: str = None,
) -> plt.Figure:
    """
    Plot SOC timeline across a multi-leg mission with leg boundaries.

    Parameters
    ----------
    leg_results : list of dict
        From experiment 3, with keys: 'label', 'E', 't_min',
        'soc_i', 'soc_f', 'dist_km', 'payload'.
    """
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), gridspec_kw={'height_ratios': [2, 1]})

    # ── SOC timeline (top) ────────────────────────────────────────────
    ax = axes[0]

    cum_time = 0
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']

    for i, r in enumerate(leg_results):
        t_start = cum_time
        t_end = cum_time + r['t_min']
        color = colors[i % len(colors)]

        # SOC line segment
        ax.plot([t_start, t_end], [r['soc_i'] * 100, r['soc_f'] * 100],
               '-', color=color, linewidth=3, zorder=5)
        ax.fill_between([t_start, t_end],
                       [r['soc_i'] * 100, r['soc_f'] * 100],
                       alpha=0.15, color=color)

        # Leg boundary
        if i > 0:
            ax.axvline(x=t_start, color='gray', linestyle=':', alpha=0.5)

        # Label
        mid_t = (t_start + t_end) / 2
        mid_soc = (r['soc_i'] + r['soc_f']) / 2 * 100
        short_label = r['label'].split(':')[-1].strip() if ':' in r['label'] else r['label']
        ax.text(mid_t, mid_soc + 3, short_label, ha='center', va='bottom',
               fontsize=7, fontweight='bold', color=color,
               bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8, ec='none'))

        # Energy annotation
        ax.text(mid_t, mid_soc - 4, f'{r["E"]:.0f} Wh',
               ha='center', va='top', fontsize=6, color=color)

        cum_time = t_end

    # Reserve line
    ax.axhline(y=15, color='#F44336', linestyle='--', linewidth=1.5,
              alpha=0.7, label='15% SOC reserve')
    ax.fill_between([0, cum_time], 0, 15, alpha=0.08, color='#F44336')

    ax.set_xlim(0, cum_time * 1.05)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Battery SOC [%]")
    ax.set_title("Battery State of Charge")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Payload / distance bar (bottom) ───────────────────────────────
    ax2 = axes[1]
    cum_time = 0
    for i, r in enumerate(leg_results):
        t_start = cum_time
        t_end = cum_time + r['t_min']
        color = colors[i % len(colors)]
        width = t_end - t_start

        # Payload bar
        ax2.barh(1, width, left=t_start, height=0.3,
                color=color, alpha=0.6, edgecolor='k', linewidth=0.3)
        ax2.text((t_start + t_end) / 2, 1,
                f'{r["payload"]:.1f} kg', ha='center', va='center', fontsize=6)

        # Distance bar
        ax2.barh(0.5, width, left=t_start, height=0.3,
                color=color, alpha=0.3, edgecolor='k', linewidth=0.3)
        ax2.text((t_start + t_end) / 2, 0.5,
                f'{r["dist_km"]:.1f} km', ha='center', va='center', fontsize=6)

        cum_time = t_end

    ax2.set_yticks([0.5, 1.0])
    ax2.set_yticklabels(['Distance', 'Payload'], fontsize=8)
    ax2.set_xlabel("Mission Time [min]")
    ax2.set_xlim(0, cum_time * 1.05)
    ax2.grid(True, alpha=0.3, axis='x')

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 6. STALL ENVELOPE — Feasible flight speed vs altitude
# ═══════════════════════════════════════════════════════════════════════════

def plot_stall_envelope(
    ac: AircraftEnergyParams = None,
    gamma_deg: float = 8.0,
    title: str = "Aerodynamic Flight Envelope at High Altitude",
    save_path: str = None,
) -> plt.Figure:
    """
    Plot the feasible flight speed envelope as a function of altitude,
    highlighting the stall boundary and safety margin.
    """
    if ac is None:
        ac = AircraftEnergyParams()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    altitudes = np.linspace(0, 5500, 200)

    # ── Left: V_stall and V_safe vs altitude ──────────────────────────
    ax = axes[0]

    for gamma, ls_style, lbl in [(0, '-', 'Level'), (8, '--', 'γ=8°'), (15, ':', 'γ=15°')]:
        v_stall = np.array([ac.stall_speed_at(alt, gamma) for alt in altitudes])
        v_safe = v_stall * 1.3
        ax.plot(v_stall, altitudes, ls_style, color='#F44336', linewidth=1.5,
               label=f'V_stall ({lbl})')
        ax.plot(v_safe, altitudes, ls_style, color='#FF9800', linewidth=1.5,
               label=f'V_safe 1.3× ({lbl})')

    # Cruise speed line
    ax.axvline(x=25, color='#2196F3', linewidth=2, alpha=0.7, label='V_cruise = 25 m/s')
    ax.axvline(x=15, color='#9E9E9E', linewidth=1, alpha=0.5, label='Old lower bound')

    # Fill infeasible region
    v_stall_level = np.array([ac.stall_speed_at(alt) for alt in altitudes])
    v_safe_level = v_stall_level * 1.3
    ax.fill_betweenx(altitudes, 0, v_safe_level, alpha=0.08, color='#F44336')

    # Quito altitude band
    ax.axhspan(2700, 3000, alpha=0.15, color='#2196F3', label='Quito base alt')
    ax.axhspan(3500, 4500, alpha=0.10, color='#9C27B0', label='Mountain crossing')

    ax.set_xlabel("Airspeed [m/s]")
    ax.set_ylabel("Altitude [m AMSL]")
    ax.set_title("Stall Speed vs Altitude")
    ax.set_xlim(10, 35)
    ax.legend(fontsize=6, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)

    # ── Right: Power vs altitude ──────────────────────────────────────
    ax2 = axes[1]

    from .energy import power_fw_cruise, power_fw_climb
    from .atmosphere import isa_density

    alts_coarse = np.linspace(0, 5000, 100)
    p_cruise = np.array([power_fw_cruise(ac, 25.0, alt) / ac.eta_prop
                         for alt in alts_coarse])
    p_climb = np.array([power_fw_climb(ac, 22.0, 8.0, alt) / ac.eta_prop
                        for alt in alts_coarse])
    rho = np.array([isa_density(alt) for alt in alts_coarse])

    ax2.plot(p_cruise, alts_coarse, '-', color='#2196F3', linewidth=2,
            label='P_cruise (V=25)')
    ax2.plot(p_climb, alts_coarse, '-', color='#F44336', linewidth=2,
            label='P_climb (V=22, γ=8°)')

    ax2_top = ax2.twiny()
    ax2_top.plot(rho, alts_coarse, '--', color='#9E9E9E', linewidth=1,
               label='ρ [kg/m³]')
    ax2_top.set_xlabel("Air density ρ [kg/m³]", fontsize=8, color='gray')
    ax2_top.tick_params(axis='x', labelcolor='gray', labelsize=7)

    ax2.axhspan(2700, 3000, alpha=0.15, color='#2196F3')
    ax2.axhspan(3500, 4500, alpha=0.10, color='#9C27B0')

    ax2.set_xlabel("Electrical Power [W]")
    ax2.set_ylabel("Altitude [m AMSL]")
    ax2.set_title("Power Requirements vs Altitude")
    ax2.legend(fontsize=7, loc='upper left')
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 7. THREE-PATH PROGRESSIVE COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

def _draw_dem_on_ax(ax, dem, lr, lo):
    """Render hillshaded DEM with contour lines (internal helper)."""
    lm = (dem.lat_1d >= lr[0]) & (dem.lat_1d <= lr[1])
    lnm = (dem.lon_1d >= lo[0]) & (dem.lon_1d <= lo[1])
    ls_, lo_ = dem.lat_1d[lm], dem.lon_1d[lnm]
    el = dem.elev_grid[np.ix_(lm, lnm)]
    LO, LA = np.meshgrid(lo_, ls_)
    li = LightSource(azdeg=315, altdeg=35)
    rgb = li.shade(el, cmap=cm.terrain, blend_mode='soft',
                   vmin=np.nanmin(el) - 200, vmax=np.nanmax(el) + 200)
    ax.imshow(rgb, extent=[lo_[0], lo_[-1], ls_[0], ls_[-1]],
              origin='lower', aspect='auto')
    cs = ax.contour(LO, LA, el,
                    levels=np.arange(np.nanmin(el) // 200 * 200,
                                     np.nanmax(el) + 300, 200),
                    colors='k', linewidths=0.3, alpha=0.3)
    ax.clabel(cs, inline=True, fontsize=5, fmt='%d')


def _draw_zones_on_ax(ax, airspace, lr, lo):
    """Overlay airspace zones on plan-view axis (internal helper)."""
    for zone in airspace.active_zones():
        if zone.is_global:
            continue
        zc, za = _ZONE_COLORS.get(zone.zone_type, ('#999', 0.2))
        if isinstance(zone.geometry, CircularZone):
            cz = zone.geometry
            rl = cz.radius_m / 111320.0
            rn = cz.radius_m / (111320.0 * np.cos(np.radians(cz.center_lat)))
            t = np.linspace(0, 2 * np.pi, 80)
            ax.fill(cz.center_lon + rn * np.cos(t),
                    cz.center_lat + rl * np.sin(t),
                    color=zc, alpha=za, zorder=2)
            ax.plot(cz.center_lon + rn * np.cos(t),
                    cz.center_lat + rl * np.sin(t),
                    '-', color=zc, lw=0.8, alpha=0.5, zorder=2)
            if (lr[0] < cz.center_lat < lr[1] and
                    lo[0] < cz.center_lon < lo[1]):
                ax.text(cz.center_lon, cz.center_lat,
                        zone.zone_id.split('_')[-1],
                        ha='center', va='center', fontsize=5,
                        color=zc, fontweight='bold',
                        path_effects=[pe.withStroke(linewidth=2,
                                                     foreground='white')])
        elif isinstance(zone.geometry, PolygonalZone):
            verts = [(lon, lat) for lat, lon in zone.geometry.vertices]
            ax.add_patch(Polygon(verts, closed=True, fc=zc, alpha=za,
                                 ec=zc, lw=1, zorder=2))


def plot_three_path_comparison(
    dem, airspace, orig, dest,
    fp_straight, fp_direct, fp_rdac,
    rp_direct=None, rp_rdac=None,
    e_straight=None, e_direct=None, e_rdac=None,
    tr_straight=None, tr_direct=None, tr_rdac=None,
    ar_straight=None, ar_direct=None, ar_rdac=None,
    title="Progressive Constraint Analysis",
    save_path=None,
):
    """Three-column comparison: straight / optimized direct / optimized RDAC."""
    from .terrain import TerrainAnalyzer
    from .config import MissionConstraints

    if tr_straight is None:
        ta = TerrainAnalyzer(dem, MissionConstraints())
        tr_straight = ta.analyze(fp_straight)
        tr_direct = ta.analyze(fp_direct)
        tr_rdac = ta.analyze(fp_rdac)
    if ar_straight is None:
        ar_straight = airspace.check_path(fp_straight)
        ar_direct = airspace.check_path(fp_direct)
        ar_rdac = airspace.check_path(fp_rdac)
    if e_straight is None:
        from .energy import AircraftEnergyParams as AEP, analyze_path_energy
        ac = AEP()
        e_straight = analyze_path_energy(fp_straight, ac)
        e_direct = analyze_path_energy(fp_direct, ac)
        e_rdac = analyze_path_energy(fp_rdac, ac)

    all_fps = [fp_straight, fp_direct, fp_rdac]
    all_lats = np.concatenate([f.get_waypoints_array()[:, 0] for f in all_fps])
    all_lons = np.concatenate([f.get_waypoints_array()[:, 1] for f in all_fps])
    lr = (min(all_lats) - 0.025, max(all_lats) + 0.025)
    lo = (min(all_lons) - 0.025, max(all_lons) + 0.025)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    configs = [
        ("Straight line\n(no optimization)", fp_straight, None,
         '#9E9E9E', e_straight, tr_straight, ar_straight, False),
        ("Optimized direct\n(no airspace)", fp_direct, rp_direct,
         '#2196F3', e_direct, tr_direct, ar_direct, False),
        ("Optimized routed\n(RDAC 101 active)", fp_rdac, rp_rdac,
         '#F44336', e_rdac, tr_rdac, ar_rdac, True),
    ]

    for col, (lbl, fp, rp, color, e_r, tr_r, ar_r, show_z) in enumerate(configs):
        ax = axes[0, col]
        _draw_dem_on_ax(ax, dem, lr, lo)
        if show_z:
            _draw_zones_on_ax(ax, airspace, lr, lo)
        ax.plot([orig.lon, dest.lon], [orig.lat, dest.lat],
                ':', color='gray', lw=1, alpha=0.6, zorder=3)
        wp = fp.get_waypoints_array()
        ax.plot(wp[:, 1], wp[:, 0], '-', color=color, lw=2.5,
                alpha=0.9, zorder=5,
                path_effects=[pe.withStroke(linewidth=4, foreground='white')])
        if rp is not None and hasattr(rp, 'n_intermediate') and rp.n_intermediate > 0:
            for lat, lon in rp.waypoint_positions[1:-1]:
                ax.plot(lon, lat, 'D', color=color, ms=7, mec='k', mew=0.8, zorder=6)
        ax.plot(orig.lon, orig.lat, 'o', color='#43A047', ms=12, mec='k', mew=1.5, zorder=7)
        ax.plot(dest.lon, dest.lat, 's', color='#E53935', ms=12, mec='k', mew=1.5, zorder=7)
        if col == 0:
            ax.annotate(getattr(orig, 'name', 'Origin'), (orig.lon, orig.lat),
                        xytext=(-8, 8), textcoords='offset points',
                        fontsize=6, fontweight='bold', color='#2E7D32',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.9, ec='#43A047'))
            ax.annotate(getattr(dest, 'name', 'Dest'), (dest.lon, dest.lat),
                        xytext=(8, -8), textcoords='offset points',
                        fontsize=6, fontweight='bold', color='#C62828',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.9, ec='#E53935'))
        ax.set_xlim(*lo); ax.set_ylim(*lr)
        ax.set_xlabel("Longitude [°]"); ax.set_ylabel("Latitude [°]")
        ax.set_title(lbl, fontsize=10, fontweight='bold'); ax.grid(True, alpha=0.2)
        info = f"E = {e_r.total_energy_wh:.0f} Wh\nt = {e_r.total_time / 60:.1f} min\nAirV = {ar_r.n_violations}"
        ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=7,
                va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', fc='white', alpha=0.9))
        f_label = "FEASIBLE" if ar_r.feasible else "INFEASIBLE"
        f_color = '#43A047' if ar_r.feasible else '#E53935'
        ax.text(0.98, 0.02, f_label, transform=ax.transAxes,
                fontsize=8, va='bottom', ha='right', fontweight='bold', color=f_color,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.9, ec=f_color, lw=1.5))

        ax2 = axes[1, col]
        dk = wp[:, 4] / 1000.0
        terr = dem.elevation_batch(wp[:, 0], wp[:, 1])
        v = ~np.isnan(terr)
        ax2.fill_between(dk[v], terr[v], alpha=0.35, color='#8D6E63')
        ax2.plot(dk[v], terr[v], '-', color='#5D4037', lw=0.8, label='Terrain')
        if show_z:
            ceil = terr.copy(); ceil[v] += 120
            ax2.plot(dk[v], ceil[v], '-', color='#F57F17', lw=1.2, alpha=0.7, label='120m AGL')
            ax2.fill_between(dk[v], ceil[v], ceil[v]+500, alpha=0.08, color='#FFC107')
        ax2.plot(dk, wp[:, 2], '-', color=color, lw=2.5, label='Flight path',
                 path_effects=[pe.withStroke(linewidth=4, foreground='white')])
        ax2.plot(dk[0], wp[0,2], 'o', color='#43A047', ms=10, mec='k', mew=1, zorder=6)
        ax2.plot(dk[-1], wp[-1,2], 's', color='#E53935', ms=10, mec='k', mew=1, zorder=6)
        ax2.set_xlabel("Ground Distance [km]"); ax2.set_ylabel("Altitude [m AMSL]")
        ax2.set_title("Altitude Profile", fontsize=9)
        ax2.legend(fontsize=6); ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 8. VEHICLE COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

def plot_vehicle_comparison(altitude=2850.0, airspeed=25.0,
                            title="Parametric Vehicle Comparison at High Altitude",
                            save_path=None):
    """Compare vehicle configs: stall envelope + performance bars."""
    from .vehicles import get_vehicle, compare_vehicles_at_altitude

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    alts = np.linspace(0, 5500, 100)
    v_colors = {'baseline': '#2196F3', 'heavy_cargo': '#F44336',
                'long_range': '#4CAF50', 'high_altitude': '#9C27B0'}
    for name, color in v_colors.items():
        ac_v = get_vehicle(name)
        vs_safe = [ac_v.stall_speed_at(a) * 1.3 for a in alts]
        axes[0].plot(vs_safe, alts, '-', color=color, lw=2, label=name)
    axes[0].axvline(x=airspeed, color='k', ls='--', alpha=0.5, label=f'V_cruise={airspeed}m/s')
    axes[0].axhspan(2700, 3000, alpha=0.1, color='#2196F3')
    axes[0].set_xlabel("V_safe (1.3×V_stall) [m/s]"); axes[0].set_ylabel("Altitude [m]")
    axes[0].set_title("Safe Speed vs Altitude"); axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)

    comp = compare_vehicles_at_altitude(altitude, airspeed)
    names = list(comp.keys()); x = np.arange(len(names))
    ranges = [comp[n]['range_cruise_km'] for n in names]
    hovers = [comp[n]['endurance_hover_min'] for n in names]
    axes[1].bar(x - 0.2, ranges, 0.35, color=[v_colors[n] for n in names], alpha=0.7)
    ax2 = axes[1].twinx()
    ax2.bar(x + 0.2, hovers, 0.35, color=[v_colors[n] for n in names], alpha=0.3, hatch='///')
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, fontsize=7, rotation=15)
    axes[1].set_ylabel("Cruise Range [km]"); ax2.set_ylabel("Hover Endurance [min]")
    axes[1].set_title(f"Performance at {altitude:.0f}m ASL"); axes[1].grid(True, alpha=0.3, axis='y')

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 9. A* vs DE COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

def plot_astar_vs_de(labels, e_astar, e_de, t_astar, t_de,
                     title="A* Grid vs DE Continuous Optimization",
                     save_path=None):
    """Bar chart comparing A* grid vs DE continuous optimizer."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(labels))
    axes[0].bar(x-0.2, e_astar, 0.35, label='A* grid', color='#FF9800', alpha=0.7, edgecolor='k', lw=0.3)
    axes[0].bar(x+0.2, e_de, 0.35, label='DE continuous', color='#2196F3', alpha=0.7, edgecolor='k', lw=0.3)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("Energy [Wh]"); axes[0].set_title("Energy"); axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3, axis='y')
    axes[1].bar(x-0.2, t_astar, 0.35, label='A* grid', color='#FF9800', alpha=0.7, edgecolor='k', lw=0.3)
    axes[1].bar(x+0.2, t_de, 0.35, label='DE continuous', color='#2196F3', alpha=0.7, edgecolor='k', lw=0.3)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("Time [min]"); axes[1].set_title("Time"); axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3, axis='y')
    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE: PLOT ALL APPLICABLE FIGURES
# ═══════════════════════════════════════════════════════════════════════════

def plot_all(
    dem: DEMInterface = None,
    paths: Dict[str, FlightPath] = None,
    facilities: List = None,
    energy_results: List = None,
    opt_results: List = None,
    initial_path: FlightPath = None,
    ac: AircraftEnergyParams = None,
    airspace: 'AirspaceManager' = None,
    # Three-path comparison data
    orig=None, dest=None,
    fp_straight: FlightPath = None,
    fp_direct: FlightPath = None,
    fp_rdac: FlightPath = None,
    rp_direct=None, rp_rdac=None,
    e_straight=None, e_direct=None, e_rdac=None,
    # A* vs DE data
    astar_labels=None, e_astar=None, e_de=None, t_astar=None, t_de=None,
    # Mission SOC data
    leg_data: List[Dict] = None,
    # Output
    save_dir: str = None,
    title_prefix: str = "",
) -> Dict[str, plt.Figure]:
    """
    Generate all applicable figures based on available data.

    Returns a dict of name → Figure for all figures produced.
    Only generates figures for which sufficient data was provided.
    Figures are saved to save_dir if provided, otherwise just returned.

    Parameters
    ----------
    dem : DEMInterface
        Terrain model (needed for most spatial plots).
    paths : dict of name → FlightPath
        For 2D/3D path plots.
    facilities : list
        Facility markers for map plots.
    energy_results : list of MissionEnergyResult
        For energy profile plots.
    opt_results : list of OptimizationResult
        For convergence plots.
    initial_path : FlightPath
        For path evolution plots.
    ac : AircraftEnergyParams
        Aircraft parameters (for stall envelope, evolution).
    airspace : AirspaceManager
        For airspace-aware plots.
    orig, dest : FacilityNode
        Origin/destination for three-path comparison.
    fp_straight, fp_direct, fp_rdac : FlightPath
        Three paths for progressive comparison.
    rp_direct, rp_rdac : RoutedPath
        Routed paths for comparison.
    e_straight, e_direct, e_rdac : MissionEnergyResult
        Energy results for comparison.
    astar_labels, e_astar, e_de, t_astar, t_de
        Data for A* vs DE comparison.
    leg_data : list of dict
        Multi-leg mission data for SOC profile.
    save_dir : str
        Directory to save figures. If None, figures are not saved.
    title_prefix : str
        Prefix for all figure titles.
    """
    import os
    figs = {}

    def _sp(name):
        """Build save path or return None."""
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            return os.path.join(save_dir, name)
        return None

    def _title(t):
        return f"{title_prefix}{t}" if title_prefix else t

    # 1. Convergence plot
    if opt_results:
        figs['convergence'] = plot_convergence(
            opt_results, title=_title("Optimization Convergence"),
            save_path=_sp('convergence.png'))

    # 2. 2D path plot
    if dem is not None and paths:
        figs['path_2d'] = plot_path_2d(
            dem, paths, facilities=facilities,
            title=_title("Flight Paths over DEM"),
            save_path=_sp('path_2d.png'))

    # 3. 3D path plot
    if dem is not None and paths:
        figs['path_3d'] = plot_path_3d(
            dem, paths, title=_title("3D Flight Path over DEM"),
            save_path=_sp('path_3d.png'))

    # 4. Path evolution
    if (dem is not None and initial_path is not None
            and opt_results and ac is not None):
        try:
            figs['path_evolution'] = plot_path_evolution(
                dem, initial_path, opt_results[0], ac,
                title=_title("Path Evolution During Optimization"),
                save_path=_sp('path_evolution.png'))
        except Exception:
            pass  # skip if snapshot data unavailable

    # 5. Energy profile (one per energy result)
    if energy_results:
        for i, er in enumerate(energy_results):
            suffix = f"_{i}" if len(energy_results) > 1 else ""
            figs[f'energy_profile{suffix}'] = plot_energy_profile(
                er, title=_title(f"Energy & SOC Profile{suffix}"),
                save_path=_sp(f'energy_profile{suffix}.png'))

    # 6. Pareto front
    if opt_results and len(opt_results) >= 3:
        try:
            figs['pareto'] = plot_pareto_front(
                opt_results, title=_title("Pareto Front: Energy vs. Time"),
                save_path=_sp('pareto.png'))
        except Exception:
            pass

    # 7. Airspace map
    if dem is not None and airspace is not None:
        figs['airspace_map'] = plot_airspace_map(
            dem, airspace, paths=paths, facilities=facilities,
            title=_title("Airspace Restrictions and Flight Paths"),
            save_path=_sp('airspace_map.png'))

    # 8. Path vs ceiling
    if dem is not None and airspace is not None and paths:
        figs['path_vs_ceiling'] = plot_path_vs_ceiling(
            dem, airspace, paths,
            title=_title("Flight Profile with Regulatory Ceiling"),
            save_path=_sp('path_vs_ceiling.png'))

    # 9. Topology comparison (airspace OFF vs ON)
    if (dem is not None and airspace is not None
            and fp_direct is not None and fp_rdac is not None):
        figs['topology_comparison'] = plot_topology_comparison(
            dem, airspace, fp_direct, fp_rdac,
            rp_off=rp_direct, rp_on=rp_rdac,
            facilities=facilities,
            title=_title("Topology Change: Airspace OFF vs ON"),
            save_path=_sp('topology_comparison.png'))

    # 10. Three-path progressive comparison
    if (dem is not None and airspace is not None
            and fp_straight is not None and fp_direct is not None
            and fp_rdac is not None and orig is not None and dest is not None):
        figs['three_path'] = plot_three_path_comparison(
            dem, airspace, orig, dest,
            fp_straight, fp_direct, fp_rdac,
            rp_direct=rp_direct, rp_rdac=rp_rdac,
            e_straight=e_straight, e_direct=e_direct, e_rdac=e_rdac,
            title=_title("Progressive Constraint Analysis"),
            save_path=_sp('three_path_comparison.png'))

    # 11. Stall envelope
    if ac is not None:
        figs['stall_envelope'] = plot_stall_envelope(
            ac, title=_title("Aerodynamic Flight Envelope"),
            save_path=_sp('stall_envelope.png'))

    # 12. A* vs DE comparison
    if (astar_labels is not None and e_astar is not None
            and e_de is not None and t_astar is not None and t_de is not None):
        figs['astar_vs_de'] = plot_astar_vs_de(
            astar_labels, e_astar, e_de, t_astar, t_de,
            title=_title("A* Grid vs DE Continuous"),
            save_path=_sp('astar_vs_de.png'))

    # 13. Mission SOC profile
    if leg_data:
        figs['mission_soc'] = plot_mission_soc(
            leg_data, title=_title("Multi-Point Mission — Battery SOC Profile"),
            save_path=_sp('mission_soc.png'))

    return figs
