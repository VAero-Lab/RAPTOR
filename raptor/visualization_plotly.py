"""
Interactive Visualization Module (Plotly)
==========================================

Provides interactive HTML visualizations for the same figures as
visualization.py (matplotlib). Both modules expose identical function
signatures; this one returns plotly.graph_objects.Figure objects and
saves to .html (interactive) rather than .png.

Figures:
    - Optimization convergence curves
    - 2D flight path over DEM contour map
    - 3D flight path over DEM surface
    - Path evolution during optimization (snapshots)
    - Energy/SOC profiles along optimized paths
    - Multi-objective Pareto front
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

Usage:
    from raptor.visualization_plotly import plot_path_2d, plot_all
    fig = plot_path_2d(dem, paths, save_path='out/paths.html')
    fig.show()

Author: Victor (LUAS-EPN / KU Leuven)
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import os
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .dem import DEMInterface
from .path import FlightPath
from .energy import MissionEnergyResult, analyze_path_energy, AircraftEnergyParams
from .airspace import AirspaceManager, ZoneType, CircularZone, PolygonalZone


# ═══════════════════════════════════════════════════════════════════════════
# STYLE DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════

_FONT = dict(family="Georgia, 'Times New Roman', serif", size=12, color='black')
_TITLE_FONT = dict(family="Georgia, 'Times New Roman', serif", size=14, color='black')
_AXIS_FONT = dict(family="Georgia, 'Times New Roman', serif", size=11)

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
    ZoneType.PROHIBITED:     ('#D32F2F', 0.35),
    ZoneType.RESTRICTED:     ('#FF5722', 0.25),
    ZoneType.AERODROME_CTR:  ('#F44336', 0.20),
    ZoneType.ALTITUDE_LIMIT: ('#FFC107', 0.08),
    ZoneType.POPULATED_AREA: ('#FF9800', 0.25),
    ZoneType.ECOLOGICAL:     ('#4CAF50', 0.30),
    ZoneType.TEMPORAL:       ('#9C27B0', 0.20),
}

_PATH_COLORS = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FFEB3B', '#795548']


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _save(fig: go.Figure, save_path: Optional[str]) -> None:
    """Save figure as .html (replaces .png extension if given)."""
    if not save_path:
        return
    html_path = (save_path[:-4] + '.html'
                 if save_path.lower().endswith('.png') else save_path)
    os.makedirs(os.path.dirname(os.path.abspath(html_path)), exist_ok=True)
    fig.write_html(
        html_path,
        include_plotlyjs='cdn',
        config={
            'toImageButtonOptions': {
                'format': 'svg',
                'filename': os.path.splitext(os.path.basename(html_path))[0],
                'height': None,
                'width': None,
                'scale': 1,
            },
            'modeBarButtonsToAdd': ['downloadImage'],
            'displaylogo': False,
        },
    )


def _layout(title: str = '', **kwargs) -> dict:
    """Standard layout dict."""
    return dict(
        title=dict(text=title, font=_TITLE_FONT, x=0.5, y=0.97),
        font=_FONT,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=70, r=40, t=60, b=55),
        **kwargs,
    )


def _zone_ring(lat_c: float, lon_c: float, radius_m: float,
               n: int = 80) -> Tuple[list, list]:
    """Return (lats, lons) ring of n points for a circular zone."""
    r_lat = radius_m / 111320.0
    r_lon = radius_m / (111320.0 * np.cos(np.radians(lat_c)))
    theta = np.linspace(0, 2 * np.pi, n)
    lats = (lat_c + r_lat * np.sin(theta)).tolist()
    lons = (lon_c + r_lon * np.cos(theta)).tolist()
    return lats, lons


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert '#RRGGBB' hex + alpha to 'rgba(r,g,b,a)' string."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha:.2f})'


def _crop_dem(dem: DEMInterface, lat_min: float, lat_max: float,
              lon_min: float, lon_max: float):
    """Return (lat_sub, lon_sub, elev_sub) cropped to bounding box."""
    lm = (dem.lat_1d >= lat_min) & (dem.lat_1d <= lat_max)
    nm = (dem.lon_1d >= lon_min) & (dem.lon_1d <= lon_max)
    return dem.lat_1d[lm], dem.lon_1d[nm], dem.elev_grid[np.ix_(lm, nm)]


def _path_bbox(paths: dict, margin: float = 0.02):
    """Return (lat_min, lat_max, lon_min, lon_max) for a dict of paths."""
    all_lats, all_lons = [], []
    for path in paths.values():
        wp = path.get_waypoints_array()
        all_lats.extend(wp[:, 0])
        all_lons.extend(wp[:, 1])
    return (min(all_lats) - margin, max(all_lats) + margin,
            min(all_lons) - margin, max(all_lons) + margin)


def _dem_heatmap(lat_sub, lon_sub, elev_sub, opacity: float = 0.8,
                 showscale: bool = False, name: str = 'Terrain') -> go.Heatmap:
    """Build a terrain heatmap trace."""
    return go.Heatmap(
        z=elev_sub,
        x=lon_sub,
        y=lat_sub,
        colorscale='earth',
        opacity=opacity,
        showscale=showscale,
        colorbar=dict(title=dict(text='Elev [m]', font=_FONT)) if showscale else None,
        name=name,
        hovertemplate='Lon: %{x:.4f}°<br>Lat: %{y:.4f}°<br>Elev: %{z:.0f} m<extra></extra>',
    )


def _dem_surface(lat_ds, lon_ds, elev_ds, opacity: float = 0.7) -> go.Surface:
    """Build a terrain surface trace for 3D plots."""
    LON, LAT = np.meshgrid(lon_ds, lat_ds)
    return go.Surface(
        z=elev_ds,
        x=LON,
        y=LAT,
        colorscale='earth',
        opacity=opacity,
        showscale=False,
        name='Terrain',
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.1,
                      roughness=0.5, fresnel=0.1),
        lightposition=dict(x=-1000, y=2000, z=5000),
        hovertemplate='Lon: %{x:.4f}°<br>Lat: %{y:.4f}°<br>Elev: %{z:.0f} m<extra></extra>',
    )


def _zone_traces(airspace: AirspaceManager, lat_range=None, lon_range=None,
                 show_labels: bool = True) -> List[go.Scatter]:
    """Build Scatter traces for all airspace zones."""
    traces = []
    for zone in airspace.active_zones():
        if zone.is_global:
            continue
        color, alpha = _ZONE_COLORS.get(zone.zone_type, ('#999999', 0.2))
        fill_color = _hex_to_rgba(color, alpha)
        line_color = _hex_to_rgba(color, min(alpha * 2, 0.8))

        if isinstance(zone.geometry, CircularZone):
            cz = zone.geometry
            if lat_range and lon_range:
                if not (lat_range[0] <= cz.center_lat <= lat_range[1] and
                        lon_range[0] <= cz.center_lon <= lon_range[1]):
                    continue
            lats, lons = _zone_ring(cz.center_lat, cz.center_lon, cz.radius_m)
            traces.append(go.Scatter(
                x=lons, y=lats, mode='lines',
                fill='toself', fillcolor=fill_color,
                line=dict(color=line_color, width=1.5),
                name=f'{zone.zone_id} ({zone.zone_type.value})',
                hoverinfo='name', showlegend=True,
            ))
            if show_labels:
                traces.append(go.Scatter(
                    x=[cz.center_lon], y=[cz.center_lat],
                    mode='text',
                    text=[zone.zone_id],
                    textfont=dict(family=_FONT['family'], size=9, color=color),
                    showlegend=False, hoverinfo='skip',
                ))

        elif isinstance(zone.geometry, PolygonalZone):
            pz = zone.geometry
            lons_p = [v[1] for v in pz.vertices] + [pz.vertices[0][1]]
            lats_p = [v[0] for v in pz.vertices] + [pz.vertices[0][0]]
            c_lat = np.mean([v[0] for v in pz.vertices])
            c_lon = np.mean([v[1] for v in pz.vertices])
            if lat_range and lon_range:
                if not (lat_range[0] <= c_lat <= lat_range[1] and
                        lon_range[0] <= c_lon <= lon_range[1]):
                    continue
            traces.append(go.Scatter(
                x=lons_p, y=lats_p, mode='lines',
                fill='toself', fillcolor=fill_color,
                line=dict(color=line_color, width=1.5),
                name=f'{zone.zone_id} ({zone.zone_type.value})',
                hoverinfo='name', showlegend=True,
            ))
            if show_labels:
                traces.append(go.Scatter(
                    x=[c_lon], y=[c_lat],
                    mode='text',
                    text=[zone.zone_id],
                    textfont=dict(family=_FONT['family'], size=9, color=color),
                    showlegend=False, hoverinfo='skip',
                ))
    return traces


# ═══════════════════════════════════════════════════════════════════════════
# 1. CONVERGENCE PLOT
# ═══════════════════════════════════════════════════════════════════════════

def plot_convergence(
    results: List,
    title: str = "Optimization Convergence",
    save_path: str = None,
) -> go.Figure:
    """Interactive convergence curves for one or more optimization results."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Objective Convergence', 'Energy & Time Comparison'),
        specs=[[{'secondary_y': False}, {'secondary_y': True}]],
    )

    # Left: convergence history
    for res in results:
        color = _MODE_COLORS.get(res.mode, '#333333')
        label = f"{res.mode.upper()} (w_E={res.weights[0]:.1f})"
        fig.add_trace(go.Scatter(
            y=res.convergence_history,
            mode='lines', name=label,
            line=dict(color=color, width=2),
        ), row=1, col=1)

    # Right: initial vs optimised energy bars + time lines
    modes = [r.mode.upper() for r in results]
    x_pos = list(range(len(results)))

    fig.add_trace(go.Bar(
        x=modes, y=[r.initial_energy_wh for r in results],
        name='Initial Energy', marker_color='#BBDEFB',
        marker_line_color='#1565C0', marker_line_width=0.8,
        offsetgroup='initial',
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        x=modes, y=[r.energy_wh for r in results],
        name='Optimized Energy', marker_color='#2196F3',
        marker_line_color='#0D47A1', marker_line_width=0.8,
        offsetgroup='opt',
    ), row=1, col=2)

    # Time lines on secondary y
    fig.add_trace(go.Scatter(
        x=modes, y=[r.initial_time_s / 60 for r in results],
        mode='lines+markers', name='Initial Time',
        line=dict(color='#FFAB91', dash='dash'), marker=dict(symbol='circle', size=8),
        yaxis='y3',
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=modes, y=[r.flight_time_s / 60 for r in results],
        mode='lines+markers', name='Optimized Time',
        line=dict(color='#FF5722'), marker=dict(symbol='square', size=8),
        yaxis='y3',
    ), row=1, col=2)

    fig.update_layout(
        **_layout(title),
        yaxis2=dict(title=dict(text='Energy [Wh]', font=dict(color='#2196F3', **_AXIS_FONT))),
        yaxis3=dict(title=dict(text='Time [min]', font=dict(color='#FF5722', **_AXIS_FONT)),
                    overlaying='y2', side='right'),
        yaxis=dict(title='Penalized Objective'),
        barmode='group',
    )
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 2. 2D PATH OVER DEM
# ═══════════════════════════════════════════════════════════════════════════

def plot_path_2d(
    dem: DEMInterface,
    paths: Dict[str, FlightPath],
    facilities: List = None,
    title: str = "Flight Paths over DEM",
    save_path: str = None,
    show_terrain_profile: bool = True,
) -> go.Figure:
    """Interactive 2D flight paths on DEM map with terrain profile."""
    n_cols = 2 if show_terrain_profile else 1
    col_titles = ['Plan View (DEM)', 'Altitude Profile'] if show_terrain_profile else ['Plan View (DEM)']
    fig = make_subplots(rows=1, cols=n_cols, subplot_titles=col_titles)

    lat_min, lat_max, lon_min, lon_max = _path_bbox(paths, margin=0.02)
    lat_sub, lon_sub, elev_sub = _crop_dem(dem, lat_min, lat_max, lon_min, lon_max)

    # DEM heatmap (plan view)
    fig.add_trace(_dem_heatmap(lat_sub, lon_sub, elev_sub, showscale=True), row=1, col=1)

    # Contour overlay
    LON_g, LAT_g = np.meshgrid(lon_sub, lat_sub)
    levels = np.arange(np.nanmin(elev_sub) // 100 * 100,
                       np.nanmax(elev_sub) + 200, 200).tolist()
    fig.add_trace(go.Contour(
        z=elev_sub, x=lon_sub, y=lat_sub,
        contours=dict(start=levels[0], end=levels[-1], size=200,
                      showlabels=True, labelfont=dict(size=8)),
        line=dict(width=0.5, color='black'),
        contours_coloring='none',
        showscale=False, name='Contours', hoverinfo='skip',
    ), row=1, col=1)

    # Flight paths
    for idx, (name, path) in enumerate(paths.items()):
        wp = path.get_waypoints_array()
        color = _PATH_COLORS[idx % len(_PATH_COLORS)]
        fig.add_trace(go.Scatter(
            x=wp[:, 1], y=wp[:, 0], mode='lines+markers',
            name=name, line=dict(color=color, width=2.5),
            marker=dict(size=[10 if i in (0, len(wp)-1) else 4
                              for i in range(len(wp))],
                        color=color, line=dict(color='black', width=0.5)),
            hovertemplate=f'{name}<br>Lon: %{{x:.4f}}°<br>Lat: %{{y:.4f}}°<extra></extra>',
        ), row=1, col=1)

    # Facilities
    if facilities:
        for fac in facilities:
            if lat_min <= fac.lat <= lat_max and lon_min <= fac.lon <= lon_max:
                is_hangar = getattr(fac, 'is_hangar', False)
                sym = 'triangle-up' if is_hangar else 'triangle-down'
                col = '#E91E63' if is_hangar else '#FF9800'
                fig.add_trace(go.Scatter(
                    x=[fac.lon], y=[fac.lat], mode='markers+text',
                    marker=dict(symbol=sym, size=12, color=col,
                                line=dict(color='black', width=1)),
                    text=[getattr(fac, 'short_name', fac.name)],
                    textposition='top right',
                    textfont=dict(family=_FONT['family'], size=9),
                    name=getattr(fac, 'name', ''),
                    showlegend=False,
                ), row=1, col=1)

    # Terrain profile (right panel)
    if show_terrain_profile:
        first_path = list(paths.values())[0]
        wp0 = first_path.get_waypoints_array()
        terrain_elevs = dem.elevation_batch(wp0[:, 0], wp0[:, 1])
        dist_km0 = wp0[:, 4] / 1000.0

        # Terrain fill
        fig.add_trace(go.Scatter(
            x=dist_km0.tolist() + dist_km0[::-1].tolist(),
            y=terrain_elevs.tolist() + [np.nanmin(terrain_elevs)] * len(terrain_elevs),
            fill='toself', fillcolor=_hex_to_rgba('#795548', 0.3),
            line=dict(color='#795548', width=1),
            name='Terrain', showlegend=True, hoverinfo='skip',
        ), row=1, col=2)

        for idx, (name, path) in enumerate(paths.items()):
            wp = path.get_waypoints_array()
            color = _PATH_COLORS[idx % len(_PATH_COLORS)]
            fig.add_trace(go.Scatter(
                x=wp[:, 4] / 1000.0, y=wp[:, 2],
                mode='lines', name=name,
                line=dict(color=color, width=2),
                showlegend=False,
                hovertemplate=f'{name}<br>Dist: %{{x:.1f}} km<br>Alt: %{{y:.0f}} m<extra></extra>',
            ), row=1, col=2)

    fig.update_xaxes(title_text='Longitude [°]', row=1, col=1)
    fig.update_yaxes(title_text='Latitude [°]', scaleanchor='x', scaleratio=1, row=1, col=1)
    if show_terrain_profile:
        fig.update_xaxes(title_text='Ground Distance [km]', row=1, col=2)
        fig.update_yaxes(title_text='Altitude [m AMSL]', row=1, col=2)

    fig.update_layout(**_layout(title), height=550)
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 3. 3D PATH OVER DEM
# ═══════════════════════════════════════════════════════════════════════════

def plot_path_3d(
    dem: DEMInterface,
    paths: Dict[str, FlightPath],
    title: str = "3D Flight Path over DEM",
    save_path: str = None,
    elev_angle: float = 30,
    azim_angle: float = -60,
) -> go.Figure:
    """Interactive 3D flight paths on DEM surface."""
    fig = go.Figure()

    lat_min, lat_max, lon_min, lon_max = _path_bbox(paths, margin=0.015)
    lat_sub, lon_sub, elev_sub = _crop_dem(dem, lat_min, lat_max, lon_min, lon_max)

    # Downsample for performance
    stride = max(1, len(lat_sub) // 80)
    lat_ds = lat_sub[::stride]
    lon_ds = lon_sub[::stride]
    elev_ds = elev_sub[::stride, ::stride]

    fig.add_trace(_dem_surface(lat_ds, lon_ds, elev_ds))

    # Flight paths
    for idx, (name, path) in enumerate(paths.items()):
        wp = path.get_waypoints_array()
        color = _PATH_COLORS[idx % len(_PATH_COLORS)]
        fig.add_trace(go.Scatter3d(
            x=wp[:, 1], y=wp[:, 0], z=wp[:, 2],
            mode='lines+markers',
            name=name,
            line=dict(color=color, width=5),
            marker=dict(size=[6 if i in (0, len(wp)-1) else 2
                              for i in range(len(wp))],
                        color=color, line=dict(color='black', width=0.5)),
            hovertemplate=(f'{name}<br>Lon: %{{x:.4f}}°<br>Lat: %{{y:.4f}}°'
                           '<br>Alt: %{z:.0f} m<extra></extra>'),
        ))

        # Vertical drop lines at key points
        for j in range(0, len(wp), max(1, len(wp) // 10)):
            terrain_z = dem.elevation(wp[j, 0], wp[j, 1])
            if not np.isnan(terrain_z):
                fig.add_trace(go.Scatter3d(
                    x=[wp[j, 1], wp[j, 1]],
                    y=[wp[j, 0], wp[j, 0]],
                    z=[terrain_z, wp[j, 2]],
                    mode='lines',
                    line=dict(color=color, width=1, dash='dot'),
                    showlegend=False, hoverinfo='skip',
                ))

    # Compute physically-proportional aspect ratio
    mid_lat = (lat_min + lat_max) / 2.0
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(np.radians(mid_lat))
    x_extent = (lon_max - lon_min) * m_per_deg_lon  # metres
    y_extent = (lat_max - lat_min) * m_per_deg_lat
    z_extent = float(np.nanmax(elev_sub) - np.nanmin(elev_sub))
    if z_extent < 1:
        z_extent = 1.0
    max_xy = max(x_extent, y_extent, 1.0)
    # Exaggerate vertical by 2x so terrain relief is visible
    z_exag = 2.0
    aspect = dict(
        x=x_extent / max_xy,
        y=y_extent / max_xy,
        z=(z_extent / max_xy) * z_exag,
    )

    fig.update_layout(
        **_layout(title),
        height=650,
        scene=dict(
            xaxis=dict(title=dict(text='Longitude [°]', font=_AXIS_FONT)),
            yaxis=dict(title=dict(text='Latitude [°]', font=_AXIS_FONT)),
            zaxis=dict(title=dict(text='Altitude [m]', font=_AXIS_FONT)),
            camera=dict(
                eye=dict(
                    x=np.cos(np.radians(azim_angle)) * np.cos(np.radians(elev_angle)) * 2,
                    y=np.sin(np.radians(azim_angle)) * np.cos(np.radians(elev_angle)) * 2,
                    z=np.sin(np.radians(elev_angle)) * 2,
                )
            ),
            aspectmode='manual',
            aspectratio=aspect,
        ),
    )
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 4. PATH EVOLUTION (SNAPSHOTS)
# ═══════════════════════════════════════════════════════════════════════════

def plot_path_evolution(
    dem: DEMInterface,
    initial_path: FlightPath,
    opt_result,
    ac: AircraftEnergyParams,
    title: str = "Path Evolution During Optimization",
    save_path: str = None,
    n_snapshots: int = 6,
) -> go.Figure:
    """Interactive path evolution snapshots during optimization."""
    snapshots = opt_result.parameter_history
    if len(snapshots) == 0:
        snapshots = [opt_result.parameter_vector]

    n = min(n_snapshots, len(snapshots))
    indices = np.linspace(0, len(snapshots) - 1, n, dtype=int)

    n_cols = (n + 1) // 2
    n_rows = 2
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f'Gen {int(i / max(len(snapshots)-1, 1) * opt_result.n_iterations)}'
                        for i in indices],
    )

    wp_init = initial_path.get_waypoints_array()
    terrain_elevs = dem.elevation_batch(wp_init[:, 0], wp_init[:, 1])

    from .optimizer import PathOptimizer
    cloner = PathOptimizer.__new__(PathOptimizer)

    for panel_idx, snap_idx in enumerate(indices):
        row = panel_idx // n_cols + 1
        col = panel_idx % n_cols + 1

        theta = snapshots[snap_idx]
        path_snap = cloner._clone_path(initial_path)
        try:
            path_snap.parameter_vector = theta
            wp = path_snap.get_waypoints_array()
            dist_km = wp[:, 4] / 1000.0
            terrain_snap = dem.elevation_batch(wp[:, 0], wp[:, 1])

            # Terrain fill
            x_fill = dist_km.tolist() + dist_km[::-1].tolist()
            y_fill = terrain_snap.tolist() + [float(np.nanmin(terrain_snap))] * len(terrain_snap)
            fig.add_trace(go.Scatter(
                x=x_fill, y=y_fill, fill='toself',
                fillcolor=_hex_to_rgba('#795548', 0.3),
                line=dict(color='#795548', width=0.5),
                showlegend=False, hoverinfo='skip',
            ), row=row, col=col)

            # Segment-colored path
            seg_starts = [0]
            for seg in path_snap.segments:
                seg_starts.append(seg_starts[-1] + len(seg.waypoints))

            for s_idx, seg in enumerate(path_snap.segments):
                s = seg_starts[s_idx]
                e = min(seg_starts[s_idx + 1], len(dist_km))
                color = _SEGMENT_COLORS.get(seg.segment_type.value, '#333333')
                fig.add_trace(go.Scatter(
                    x=dist_km[s:e], y=wp[s:e, 2],
                    mode='lines', line=dict(color=color, width=2),
                    showlegend=False, hoverinfo='skip',
                ), row=row, col=col)

        except Exception:
            pass

    fig.update_layout(**_layout(title), height=550)
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 5. ENERGY / SOC PROFILE
# ═══════════════════════════════════════════════════════════════════════════

def plot_energy_profile(
    energy_result: MissionEnergyResult,
    title: str = "Energy & SOC Profile",
    save_path: str = None,
) -> go.Figure:
    """Interactive power and SOC profiles along the optimized path."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Power Profile by Segment', 'Battery State of Charge'),
        shared_xaxes=True,
        vertical_spacing=0.12,
    )

    # Power profile
    cum_time = 0.0
    for seg in energy_result.segments:
        t_start = cum_time
        t_end = cum_time + seg.duration
        color = _SEGMENT_COLORS.get(seg.segment_type, '#333333')
        fill_color = _hex_to_rgba(color, 0.5)

        fig.add_trace(go.Scatter(
            x=[t_start, t_end, t_end, t_start, t_start],
            y=[0, 0, seg.P_elec, seg.P_elec, 0],
            fill='toself', fillcolor=fill_color,
            line=dict(color=color, width=1),
            name=seg.segment_type,
            hovertemplate=(f'Segment: {seg.segment_type}<br>'
                           f'Power: {seg.P_elec:.0f} W<br>'
                           f'Duration: {seg.duration:.0f} s<extra></extra>'),
        ), row=1, col=1)

        cum_time = t_end

    # SOC profile
    timeline = energy_result.battery_timeline
    times = [s.time for s in timeline]
    socs = [s.percent_remaining for s in timeline]

    fig.add_trace(go.Scatter(
        x=times, y=socs,
        mode='lines', name='SOC',
        line=dict(color='#2196F3', width=2.5),
        fill='tozeroy', fillcolor=_hex_to_rgba('#2196F3', 0.15),
        hovertemplate='Time: %{x:.0f} s<br>SOC: %{y:.1f}%<extra></extra>',
    ), row=2, col=1)
    fig.add_hline(y=15, line=dict(color='red', dash='dash', width=1.5),
                  annotation_text='Min SOC (15%)', annotation_position='right',
                  row=2, col=1)

    fig.update_yaxes(title_text='Electrical Power [W]', row=1, col=1)
    fig.update_yaxes(title_text='Battery SOC [%]', range=[0, 105], row=2, col=1)
    fig.update_xaxes(title_text='Mission Time [s]', row=2, col=1)
    fig.update_layout(**_layout(title), height=550)
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 6. PARETO FRONT
# ═══════════════════════════════════════════════════════════════════════════

def plot_pareto_front(
    results: List,
    title: str = "Pareto Front: Energy vs. Time",
    save_path: str = None,
) -> go.Figure:
    """Interactive Pareto front from a weight sweep."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Pareto Front', 'Trade-off by Weight'),
        specs=[[{'secondary_y': False}, {'secondary_y': True}]],
    )

    energies = [r.energy_wh for r in results]
    times = [r.flight_time_s / 60 for r in results]
    w_energies = [r.weights[0] for r in results]
    feasible = [r.fully_feasible for r in results]

    # Pareto scatter (color = w_energy)
    fig.add_trace(go.Scatter(
        x=times, y=energies,
        mode='markers+text+lines',
        marker=dict(
            color=w_energies, colorscale='RdBu', size=12,
            colorbar=dict(title=dict(text='w_energy', font=_FONT), x=0.47),
            symbol=['circle' if f else 'x' for f in feasible],
            line=dict(color='black', width=0.8),
        ),
        text=[f'w={w:.1f}' for w in w_energies],
        textposition='top right',
        textfont=dict(family=_FONT['family'], size=9),
        line=dict(color='gray', dash='dash', width=1),
        name='Pareto',
        hovertemplate='Time: %{x:.1f} min<br>Energy: %{y:.0f} Wh<extra></extra>',
    ), row=1, col=1)

    # Trade-off bars
    x_labels = [r.mode.upper() if hasattr(r, 'mode') else f'w=({r.weights[0]:.1f},{r.weights[1]:.1f})'
                for r in results]
    fig.add_trace(go.Bar(
        x=x_labels, y=energies, name='Energy [Wh]',
        marker_color='#2196F3', opacity=0.7, offsetgroup='e',
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        x=x_labels, y=times, name='Time [min]',
        marker_color='#FF5722', opacity=0.7, offsetgroup='t',
        yaxis='y3',
    ), row=1, col=2)

    fig.update_xaxes(title_text='Flight Time [min]', row=1, col=1)
    fig.update_yaxes(title_text='Energy [Wh]', row=1, col=1)
    fig.update_yaxes(title_text='Energy [Wh]', row=1, col=2)
    fig.update_layout(
        **_layout(title),
        yaxis3=dict(title='Time [min]', overlaying='y2', side='right'),
        barmode='group', height=500,
    )
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 7. SCENARIO COMPARISON DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

def plot_scenario_dashboard(
    scenario_results: Dict[str, List],
    title: str = "Scenario Optimization Dashboard",
    save_path: str = None,
) -> go.Figure:
    """Interactive dashboard comparing optimization results across legs."""
    n_legs = len(scenario_results)
    cols = max(n_legs, 2)
    col_titles = []
    for leg_key in scenario_results:
        col_titles.extend([leg_key.replace('_', ' '), ''])  # top + bottom
    col_titles = col_titles[:cols * 2]

    fig = make_subplots(
        rows=2, cols=cols,
        subplot_titles=col_titles[:cols] + [''] * cols,
        specs=[[{'secondary_y': False}] * cols,
               [{'secondary_y': False}] * cols],
    )

    for c_idx, (leg_key, results) in enumerate(scenario_results.items()):
        col = c_idx + 1
        modes = [r.mode.upper() for r in results]
        initial_e = [r.initial_energy_wh for r in results]
        opt_e = [r.energy_wh for r in results]
        initial_t = [r.initial_time_s / 60 for r in results]
        opt_t = [r.flight_time_s / 60 for r in results]

        show = c_idx == 0
        fig.add_trace(go.Bar(x=modes, y=initial_e, name='Initial' if show else None,
                             marker_color='#BBDEFB', showlegend=show, offsetgroup='i'), row=1, col=col)
        fig.add_trace(go.Bar(x=modes, y=opt_e, name='Optimized' if show else None,
                             marker_color='#2196F3', showlegend=show, offsetgroup='o'), row=1, col=col)
        fig.add_trace(go.Bar(x=modes, y=initial_t, name='Initial t' if show else None,
                             marker_color='#FFCCBC', showlegend=show, offsetgroup='it'), row=2, col=col)
        fig.add_trace(go.Bar(x=modes, y=opt_t, name='Optimized t' if show else None,
                             marker_color='#FF5722', showlegend=show, offsetgroup='ot'), row=2, col=col)

        fig.update_yaxes(title_text='Energy [Wh]', row=1, col=col)
        fig.update_yaxes(title_text='Time [min]', row=2, col=col)

    fig.update_layout(**_layout(title), barmode='group', height=650)
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 8. AIRSPACE MAP
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
) -> go.Figure:
    """Interactive airspace map with DEM, zones, and flight paths."""
    fig = go.Figure()

    # Determine extent
    if lat_range is None or lon_range is None:
        lat_min, lat_max = dem.metadata.lat_min, dem.metadata.lat_max
        lon_min, lon_max = dem.metadata.lon_min, dem.metadata.lon_max
        if paths:
            for p in paths.values():
                wp = p.get_waypoints_array()
                lat_min = min(lat_min, float(np.min(wp[:, 0])) - 0.02)
                lat_max = max(lat_max, float(np.max(wp[:, 0])) + 0.02)
                lon_min = min(lon_min, float(np.min(wp[:, 1])) - 0.02)
                lon_max = max(lon_max, float(np.max(wp[:, 1])) + 0.02)
    else:
        lat_min, lat_max = lat_range
        lon_min, lon_max = lon_range

    lat_sub, lon_sub, elev_sub = _crop_dem(dem, lat_min, lat_max, lon_min, lon_max)
    fig.add_trace(_dem_heatmap(lat_sub, lon_sub, elev_sub, showscale=True))

    # Contour lines
    levels = np.arange(np.nanmin(elev_sub) // 200 * 200,
                       np.nanmax(elev_sub) + 300, 200).tolist()
    fig.add_trace(go.Contour(
        z=elev_sub, x=lon_sub, y=lat_sub,
        contours=dict(start=levels[0], end=levels[-1], size=200,
                      showlabels=True, labelfont=dict(size=7)),
        line=dict(width=0.4, color='black'),
        contours_coloring='none',
        showscale=False, name='Contours', hoverinfo='skip',
    ))

    # Airspace zones
    for t in _zone_traces(airspace, lat_range=(lat_min, lat_max),
                          lon_range=(lon_min, lon_max)):
        fig.add_trace(t)

    # All paths
    all_paths_dict: Dict[str, FlightPath] = {}
    if paths:
        all_paths_dict.update(paths)
    if routed_paths:
        for name, rp in routed_paths.items():
            all_paths_dict[name] = rp.flight_path

    path_palette = ['#E91E63', '#00BCD4', '#CDDC39', '#7C4DFF', '#FF6E40']
    for idx, (name, path) in enumerate(all_paths_dict.items()):
        wp = path.get_waypoints_array()
        color = path_palette[idx % len(path_palette)]
        fig.add_trace(go.Scatter(
            x=wp[:, 1], y=wp[:, 0], mode='lines',
            name=name, line=dict(color=color, width=3),
            hovertemplate=f'{name}<br>Lon: %{{x:.4f}}°<br>Lat: %{{y:.4f}}°<extra></extra>',
        ))
        fig.add_trace(go.Scatter(
            x=[wp[0, 1], wp[-1, 1]], y=[wp[0, 0], wp[-1, 0]],
            mode='markers', marker=dict(symbol=['circle', 'square'],
                                        size=10, color=color,
                                        line=dict(color='black', width=1)),
            showlegend=False, hoverinfo='skip',
        ))

    # Waypoints for routed paths
    if routed_paths:
        for idx, (name, rp) in enumerate(routed_paths.items()):
            color = path_palette[idx % len(path_palette)]
            for i, (lat, lon) in enumerate(rp.waypoint_positions[1:-1], 1):
                fig.add_trace(go.Scatter(
                    x=[lon], y=[lat], mode='markers+text',
                    marker=dict(symbol='diamond', size=9, color=color,
                                line=dict(color='black', width=0.8)),
                    text=[f'WP{i}'], textposition='top right',
                    textfont=dict(family=_FONT['family'], size=8),
                    showlegend=False,
                ))

    # Facilities
    if facilities:
        for fac in facilities:
            if lat_min <= fac.lat <= lat_max and lon_min <= fac.lon <= lon_max:
                is_h = getattr(fac, 'is_hangar', False)
                fig.add_trace(go.Scatter(
                    x=[fac.lon], y=[fac.lat], mode='markers+text',
                    marker=dict(symbol='triangle-up' if is_h else 'triangle-down',
                                size=11, color='#E91E63' if is_h else 'white',
                                line=dict(color='black', width=1)),
                    text=[getattr(fac, 'short_name', fac.name)],
                    textposition='top right',
                    textfont=dict(family=_FONT['family'], size=9, color='black'),
                    name=getattr(fac, 'name', ''),
                    showlegend=False,
                ))

    fig.update_layout(
        **_layout(title),
        height=int(figsize[1] * 80),
        width=int(figsize[0] * 80),
        xaxis=dict(title='Longitude [°]', range=[lon_min, lon_max]),
        yaxis=dict(title='Latitude [°]', range=[lat_min, lat_max],
                   scaleanchor='x', scaleratio=1.0),
    )
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 9. PATH VS CEILING
# ═══════════════════════════════════════════════════════════════════════════

def plot_path_vs_ceiling(
    dem: DEMInterface,
    airspace: AirspaceManager,
    paths: Dict[str, FlightPath],
    title: str = "Flight Profile with Regulatory Ceiling",
    save_path: str = None,
) -> go.Figure:
    """Interactive altitude profile with terrain floor and AGL ceiling bands."""
    fig = go.Figure()

    ref_path = list(paths.values())[0]
    wp0 = ref_path.get_waypoints_array()
    dist_km = wp0[:, 4] / 1000.0
    lats, lons = wp0[:, 0], wp0[:, 1]
    terrain = dem.elevation_batch(lats, lons)

    agl_ceilings = terrain + 120.0
    urban_ceilings = np.full_like(terrain, np.nan)
    for i in range(len(lats)):
        if np.isnan(terrain[i]):
            continue
        for zone in airspace.active_zones():
            if zone.zone_type == ZoneType.POPULATED_AREA:
                if zone.horizontal_contains(lats[i], lons[i]):
                    if zone.is_agl:
                        urban_ceilings[i] = terrain[i] + zone.altitude_ceiling_m
                    break

    clearance_floor = terrain + 50.0
    valid = ~np.isnan(terrain)
    dk_v = dist_km[valid]

    # Terrain fill
    fig.add_trace(go.Scatter(
        x=np.concatenate([dist_km, dist_km[::-1]]).tolist(),
        y=np.concatenate([terrain, np.full(len(terrain), np.nanmin(terrain) - 100)]).tolist(),
        fill='toself', fillcolor=_hex_to_rgba('#8D6E63', 0.4),
        line=dict(color='#5D4037', width=1),
        name='Terrain', hoverinfo='skip',
    ))

    # Min clearance floor
    fig.add_trace(go.Scatter(
        x=dist_km.tolist(), y=clearance_floor.tolist(),
        mode='lines', name='Min clearance (50m AGL)',
        line=dict(color='#F44336', dash='dash', width=1),
    ))

    # AGL ceiling band (shaded above 120m AGL ceiling)
    y_top = float(np.nanmax(agl_ceilings[valid])) + 300
    fig.add_trace(go.Scatter(
        x=np.concatenate([dk_v, dk_v[::-1]]).tolist(),
        y=np.concatenate([np.full(len(dk_v), y_top),
                          agl_ceilings[valid][::-1]]).tolist(),
        fill='toself', fillcolor=_hex_to_rgba('#FFC107', 0.12),
        line=dict(color='#F57F17', width=1.5),
        name='Above 120m AGL (restricted)', hoverinfo='skip',
    ))

    # Urban ceiling
    urban_valid = ~np.isnan(urban_ceilings)
    if np.any(urban_valid):
        dk_u = dist_km[urban_valid]
        fig.add_trace(go.Scatter(
            x=dk_u.tolist(), y=urban_ceilings[urban_valid].tolist(),
            mode='lines', name='60m AGL urban ceiling',
            line=dict(color='#E65100', width=1.5),
        ))

    # Flight paths
    path_palette = ['#E91E63', '#00BCD4', '#CDDC39', '#7C4DFF']
    for idx, (name, path) in enumerate(paths.items()):
        wp = path.get_waypoints_array()
        color = path_palette[idx % len(path_palette)]
        fig.add_trace(go.Scatter(
            x=wp[:, 4] / 1000.0, y=wp[:, 2],
            mode='lines', name=name,
            line=dict(color=color, width=3),
            hovertemplate=f'{name}<br>Dist: %{{x:.1f}} km<br>Alt: %{{y:.0f}} m<extra></extra>',
        ))

    y_max = max(y_top,
                max(float(np.max(path.get_waypoints_array()[:, 2])) for path in paths.values()) + 200)
    fig.update_layout(
        **_layout(title),
        xaxis=dict(title='Ground Distance [km]'),
        yaxis=dict(title='Altitude [m AMSL]',
                   range=[float(np.nanmin(terrain)) - 100, y_max]),
        height=500,
    )
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 10. TOPOLOGY COMPARISON
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
) -> go.Figure:
    """Interactive side-by-side topology comparison (airspace ON vs OFF)."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Plan View — Airspace OFF (no airspace)',
            'Plan View — Airspace ON (RDAC 101)',
            'Altitude Profile — Airspace OFF',
            'Altitude Profile — Airspace ON',
        ],
        vertical_spacing=0.12, horizontal_spacing=0.08,
    )

    wp_off = path_off.get_waypoints_array()
    wp_on = path_on.get_waypoints_array()
    margin = 0.025
    lat_min = min(np.min(wp_off[:, 0]), np.min(wp_on[:, 0])) - margin
    lat_max = max(np.max(wp_off[:, 0]), np.max(wp_on[:, 0])) + margin
    lon_min = min(np.min(wp_off[:, 1]), np.min(wp_on[:, 1])) - margin
    lon_max = max(np.max(wp_off[:, 1]), np.max(wp_on[:, 1])) + margin

    lat_sub, lon_sub, elev_sub = _crop_dem(dem, lat_min, lat_max, lon_min, lon_max)

    configs = [
        ('Airspace OFF', path_off, wp_off, rp_off, '#2196F3', False, 1),
        ('Airspace ON',  path_on,  wp_on,  rp_on,  '#F44336', True,  2),
    ]

    for label, path, wp, rp, color, show_zones, col in configs:
        # Plan view (row 1)
        fig.add_trace(
            go.Heatmap(z=elev_sub, x=lon_sub, y=lat_sub,
                       colorscale='earth', showscale=False, opacity=0.75,
                       hoverinfo='skip', name='Terrain'),
            row=1, col=col,
        )

        if show_zones:
            for t in _zone_traces(airspace, lat_range=(lat_min, lat_max),
                                   lon_range=(lon_min, lon_max), show_labels=True):
                fig.add_trace(t, row=1, col=col)

        # Direct line
        fig.add_trace(go.Scatter(
            x=[float(wp[0, 1]), float(wp[-1, 1])],
            y=[float(wp[0, 0]), float(wp[-1, 0])],
            mode='lines', line=dict(color='gray', dash='dot', width=1),
            showlegend=False, hoverinfo='skip',
        ), row=1, col=col)

        # Flight path
        fig.add_trace(go.Scatter(
            x=wp[:, 1], y=wp[:, 0], mode='lines',
            name=label, line=dict(color=color, width=3),
            hovertemplate=f'{label}<br>Lon: %{{x:.4f}}°<br>Lat: %{{y:.4f}}°<extra></extra>',
        ), row=1, col=col)

        # Start/end markers
        fig.add_trace(go.Scatter(
            x=[float(wp[0, 1]), float(wp[-1, 1])],
            y=[float(wp[0, 0]), float(wp[-1, 0])],
            mode='markers',
            marker=dict(symbol=['circle', 'square'], size=12, color=color,
                        line=dict(color='black', width=1)),
            showlegend=False, hoverinfo='skip',
        ), row=1, col=col)

        # Intermediate waypoints
        if rp is not None:
            for lat_w, lon_w in rp.waypoint_positions[1:-1]:
                fig.add_trace(go.Scatter(
                    x=[lon_w], y=[lat_w], mode='markers',
                    marker=dict(symbol='diamond', size=9, color=color,
                                line=dict(color='black', width=0.8)),
                    showlegend=False, hoverinfo='skip',
                ), row=1, col=col)

        if facilities:
            for fac in facilities:
                if lat_min <= fac.lat <= lat_max and lon_min <= fac.lon <= lon_max:
                    fig.add_trace(go.Scatter(
                        x=[fac.lon], y=[fac.lat], mode='markers+text',
                        marker=dict(symbol='triangle-up', size=9, color='white',
                                    line=dict(color='black', width=0.8)),
                        text=[getattr(fac, 'short_name', fac.name)],
                        textposition='top right',
                        textfont=dict(family=_FONT['family'], size=8),
                        showlegend=False,
                    ), row=1, col=col)

        # Altitude profile (row 2)
        d_km = wp[:, 4] / 1000.0
        terrain = dem.elevation_batch(wp[:, 0], wp[:, 1])
        valid = ~np.isnan(terrain)

        fig.add_trace(go.Scatter(
            x=np.concatenate([d_km, d_km[::-1]]).tolist(),
            y=np.concatenate([terrain, np.full(len(terrain), float(np.nanmin(terrain)) - 50)]).tolist(),
            fill='toself', fillcolor=_hex_to_rgba('#8D6E63', 0.35),
            line=dict(color='#5D4037', width=0.8),
            showlegend=False, hoverinfo='skip',
        ), row=2, col=col)

        if show_zones:
            agl_ceil = terrain.copy()
            agl_ceil[valid] += 120.0
            fig.add_trace(go.Scatter(
                x=d_km[valid].tolist(), y=agl_ceil[valid].tolist(),
                mode='lines', name='120m AGL',
                line=dict(color='#F57F17', width=1.2),
                showlegend=(col == 2),
            ), row=2, col=col)

        fig.add_trace(go.Scatter(
            x=d_km.tolist(), y=wp[:, 2].tolist(),
            mode='lines', name=label + ' path',
            line=dict(color=color, width=3),
            showlegend=False,
            hovertemplate=f'{label}<br>Dist: %{{x:.1f}} km<br>Alt: %{{y:.0f}} m<extra></extra>',
        ), row=2, col=col)

        fig.update_xaxes(title_text='Longitude [°]', row=1, col=col)
        fig.update_yaxes(title_text='Latitude [°]',
                         scaleanchor=f'x{"" if col == 1 else col}',
                         scaleratio=1, row=1, col=col)
        fig.update_xaxes(title_text='Ground Distance [km]', row=2, col=col)
        fig.update_yaxes(title_text='Altitude [m AMSL]', row=2, col=col)

    fig.update_layout(**_layout(title), height=800)
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 11. CONSTRAINT PENALTY BUDGET
# ═══════════════════════════════════════════════════════════════════════════

def plot_constraint_budget(
    scenario_results: Dict[str, List[Dict]],
    title: str = "Constraint Penalty Budget — Energy Cost per Regulatory Layer",
    save_path: str = None,
) -> go.Figure:
    """Interactive stacked bar chart of constraint layer energy costs."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Energy by Constraint Configuration',
                        'Incremental Constraint Cost'),
    )

    scenarios = list(scenario_results.keys())
    config_labels = ['A: terrain', 'B: AGL', 'C: urban', 'D: full']
    config_colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336']

    # Grouped bars (absolute)
    for i, (clbl, ccol) in enumerate(zip(config_labels, config_colors)):
        vals = []
        for sname in scenarios:
            rows = scenario_results[sname]
            vals.append(rows[i]['E'] if i < len(rows) else 0)
        fig.add_trace(go.Bar(
            x=scenarios, y=vals, name=clbl,
            marker_color=ccol, opacity=0.8, offsetgroup=str(i),
        ), row=1, col=1)

    # Incremental stacked bars
    layer_labels = ['B−A: AGL ceiling', 'C−B: Urban zones', 'D−C: Airport+eco']
    layer_colors = ['#FF9800', '#4CAF50', '#F44336']

    for s_idx, sname in enumerate(scenarios):
        rows = scenario_results[sname]
        energies = [r['E'] for r in rows]
        bottom = 0.0
        for i in range(1, min(len(energies), 4)):
            delta = energies[i] - energies[i - 1]
            color = layer_colors[min(i - 1, len(layer_colors) - 1)]
            lbl = layer_labels[min(i - 1, len(layer_labels) - 1)] if s_idx == 0 else None
            fig.add_trace(go.Bar(
                x=[sname], y=[delta], base=bottom,
                name=lbl, marker_color=color, opacity=0.8 if delta >= 0 else 0.3,
                showlegend=(lbl is not None),
            ), row=1, col=2)
            bottom += delta

    fig.update_yaxes(title_text='Energy [Wh]', row=1, col=1)
    fig.update_yaxes(title_text='ΔEnergy [Wh] per layer', row=1, col=2)
    fig.update_layout(**_layout(title), barmode='group', height=480)
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 12. MISSION SOC PROFILE
# ═══════════════════════════════════════════════════════════════════════════

def plot_mission_soc(
    leg_results: List[Dict],
    title: str = "Multi-Point Mission — Battery SOC Profile",
    save_path: str = None,
) -> go.Figure:
    """Interactive multi-leg mission SOC and payload timeline."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Battery State of Charge', 'Payload & Distance'),
        row_heights=[0.67, 0.33],
        vertical_spacing=0.12,
        shared_xaxes=True,
    )

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
    cum_time = 0.0

    for i, r in enumerate(leg_results):
        t_start = cum_time
        t_end = cum_time + r['t_min']
        color = colors[i % len(colors)]

        # SOC segment
        fig.add_trace(go.Scatter(
            x=[t_start, t_end], y=[r['soc_i'] * 100, r['soc_f'] * 100],
            mode='lines', name=r['label'],
            line=dict(color=color, width=3),
            fill='tozeroy', fillcolor=_hex_to_rgba(color, 0.15),
            hovertemplate=f"{r['label']}<br>SOC: %{{y:.1f}}%<extra></extra>",
        ), row=1, col=1)

        # Leg boundary
        if i > 0:
            fig.add_vline(x=t_start, line=dict(color='gray', dash='dot', width=1),
                          row=1, col=1)

        # Payload bar
        fig.add_trace(go.Bar(
            x=[(t_start + t_end) / 2], y=[r['payload']],
            width=[t_end - t_start],
            name=f"Payload {r['label']}", marker_color=color,
            opacity=0.6, showlegend=False,
            hovertemplate=f"Payload: {r['payload']:.1f} kg<extra></extra>",
            yaxis='y3',
        ), row=2, col=1)

        cum_time = t_end

    # Reserve line
    fig.add_hline(y=15, line=dict(color='red', dash='dash', width=1.5),
                  annotation_text='15% SOC reserve',
                  row=1, col=1)
    fig.add_hrect(y0=0, y1=15, fillcolor=_hex_to_rgba('#F44336', 0.08),
                  line_width=0, row=1, col=1)

    fig.update_yaxes(title_text='Battery SOC [%]', range=[0, 105], row=1, col=1)
    fig.update_yaxes(title_text='Payload [kg]', row=2, col=1)
    fig.update_xaxes(title_text='Mission Time [min]', row=2, col=1)
    fig.update_layout(**_layout(title), height=580)
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 13. STALL ENVELOPE
# ═══════════════════════════════════════════════════════════════════════════

def plot_stall_envelope(
    ac: AircraftEnergyParams = None,
    gamma_deg: float = 8.0,
    title: str = "Aerodynamic Flight Envelope at High Altitude",
    save_path: str = None,
) -> go.Figure:
    """Interactive stall envelope and power vs altitude."""
    if ac is None:
        ac = AircraftEnergyParams()

    from .energy import power_fw_cruise, power_fw_climb
    from .atmosphere import isa_density

    altitudes = np.linspace(0, 5500, 200)
    alts_c = np.linspace(0, 5000, 100)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Stall Speed vs Altitude', 'Power Requirements vs Altitude'),
        specs=[[{'secondary_y': False}, {'secondary_y': True}]],
    )

    gamma_styles = [(0, 'solid', 'Level'), (8, 'dash', 'γ=8°'), (15, 'dot', 'γ=15°')]
    for gamma, ls, lbl in gamma_styles:
        v_stall = np.array([ac.stall_speed_at(alt, gamma) for alt in altitudes])
        v_safe = v_stall * 1.3
        fig.add_trace(go.Scatter(
            x=v_stall, y=altitudes, mode='lines',
            name=f'V_stall ({lbl})', line=dict(color='#F44336', dash=ls, width=1.5),
            showlegend=True,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=v_safe, y=altitudes, mode='lines',
            name=f'V_safe 1.3× ({lbl})', line=dict(color='#FF9800', dash=ls, width=1.5),
            showlegend=True,
        ), row=1, col=1)

    # Cruise speed
    fig.add_vline(x=25, line=dict(color='#2196F3', dash='dash', width=2),
                  annotation_text='V_cruise=25 m/s', row=1, col=1)

    # Infeasible fill
    v_safe_level = np.array([ac.stall_speed_at(alt) * 1.3 for alt in altitudes])
    fig.add_trace(go.Scatter(
        x=np.concatenate([v_safe_level, np.zeros(len(altitudes))]).tolist(),
        y=np.concatenate([altitudes, altitudes[::-1]]).tolist(),
        fill='toself', fillcolor=_hex_to_rgba('#F44336', 0.08),
        line=dict(width=0), showlegend=False, hoverinfo='skip',
    ), row=1, col=1)

    # Altitude bands
    for y0, y1, color, label in [
            (2700, 3000, '#2196F3', 'Quito base alt'),
            (3500, 4500, '#9C27B0', 'Mountain crossing')]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=_hex_to_rgba(color, 0.15),
                      line_width=0, annotation_text=label,
                      annotation_position='right',
                      row=1, col=1)

    # Power curves
    p_cruise = np.array([power_fw_cruise(ac, 25.0, alt) / ac.eta_prop for alt in alts_c])
    p_climb = np.array([power_fw_climb(ac, 22.0, 8.0, alt) / ac.eta_prop for alt in alts_c])
    rho_arr = np.array([isa_density(alt) for alt in alts_c])

    fig.add_trace(go.Scatter(
        x=p_cruise, y=alts_c, mode='lines', name='P_cruise (V=25)',
        line=dict(color='#2196F3', width=2),
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=p_climb, y=alts_c, mode='lines', name='P_climb (V=22, γ=8°)',
        line=dict(color='#F44336', width=2),
    ), row=1, col=2)
    # Air density on secondary y
    fig.add_trace(go.Scatter(
        x=rho_arr, y=alts_c, mode='lines', name='ρ [kg/m³]',
        line=dict(color='#9E9E9E', dash='dash', width=1.5),
        yaxis='y3',
    ), row=1, col=2)

    for y0, y1, color in [(2700, 3000, '#2196F3'), (3500, 4500, '#9C27B0')]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=_hex_to_rgba(color, 0.1),
                      line_width=0, row=1, col=2)

    fig.update_xaxes(title_text='Airspeed [m/s]', range=[10, 35], row=1, col=1)
    fig.update_yaxes(title_text='Altitude [m AMSL]', row=1, col=1)
    fig.update_xaxes(title_text='Electrical Power [W]', row=1, col=2)
    fig.update_yaxes(title_text='Altitude [m AMSL]', row=1, col=2)
    fig.update_layout(
        **_layout(title),
        yaxis3=dict(title=dict(text='Air density ρ [kg/m³]', font=dict(color='gray', **_AXIS_FONT)),
                    overlaying='y2', side='right'),
        height=520,
    )
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 14. THREE-PATH PROGRESSIVE COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

def plot_three_path_comparison(
    dem, airspace, orig, dest,
    fp_straight, fp_direct, fp_rdac,
    rp_direct=None, rp_rdac=None,
    e_straight=None, e_direct=None, e_rdac=None,
    tr_straight=None, tr_direct=None, tr_rdac=None,
    ar_straight=None, ar_direct=None, ar_rdac=None,
    title="Progressive Constraint Analysis",
    save_path=None,
) -> go.Figure:
    """Interactive three-column comparison: straight / direct / RDAC."""
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
        _ac = AEP()
        e_straight = analyze_path_energy(fp_straight, _ac)
        e_direct = analyze_path_energy(fp_direct, _ac)
        e_rdac = analyze_path_energy(fp_rdac, _ac)

    all_fps = [fp_straight, fp_direct, fp_rdac]
    all_lats = np.concatenate([f.get_waypoints_array()[:, 0] for f in all_fps])
    all_lons = np.concatenate([f.get_waypoints_array()[:, 1] for f in all_fps])
    lat_min, lat_max = float(np.min(all_lats)) - 0.025, float(np.max(all_lats)) + 0.025
    lon_min, lon_max = float(np.min(all_lons)) - 0.025, float(np.max(all_lons)) + 0.025

    lat_sub, lon_sub, elev_sub = _crop_dem(dem, lat_min, lat_max, lon_min, lon_max)

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            'Straight line (no optimization)', 'Optimized direct (no airspace)',
            'Optimized routed (RDAC 101)',
            'Altitude Profile — Straight', 'Altitude Profile — Direct',
            'Altitude Profile — RDAC',
        ],
        vertical_spacing=0.10, horizontal_spacing=0.06,
    )

    configs = [
        ('Straight', fp_straight, None, '#9E9E9E',
         e_straight, ar_straight, False, 1),
        ('Direct opt', fp_direct, rp_direct, '#2196F3',
         e_direct, ar_direct, False, 2),
        ('RDAC opt', fp_rdac, rp_rdac, '#F44336',
         e_rdac, ar_rdac, True, 3),
    ]

    for label, fp, rp, color, e_r, ar_r, show_zones, col in configs:
        wp = fp.get_waypoints_array()

        # DEM heatmap (plan view)
        fig.add_trace(
            go.Heatmap(z=elev_sub, x=lon_sub, y=lat_sub,
                       colorscale='earth', showscale=False, opacity=0.75,
                       hoverinfo='skip', name='Terrain'),
            row=1, col=col,
        )

        # Airspace zones (only for RDAC column)
        if show_zones:
            for t in _zone_traces(airspace, lat_range=(lat_min, lat_max),
                                   lon_range=(lon_min, lon_max), show_labels=True):
                fig.add_trace(t, row=1, col=col)

        # Direct line
        fig.add_trace(go.Scatter(
            x=[orig.lon, dest.lon], y=[orig.lat, dest.lat],
            mode='lines', line=dict(color='gray', dash='dot', width=1),
            showlegend=False, hoverinfo='skip',
        ), row=1, col=col)

        # Flight path
        fig.add_trace(go.Scatter(
            x=wp[:, 1], y=wp[:, 0], mode='lines',
            name=label, line=dict(color=color, width=3),
            hovertemplate=f'{label}<br>Lon: %{{x:.4f}}°<br>Lat: %{{y:.4f}}°<extra></extra>',
        ), row=1, col=col)

        # Origin/dest markers
        fig.add_trace(go.Scatter(
            x=[orig.lon, dest.lon], y=[orig.lat, dest.lat],
            mode='markers+text',
            marker=dict(symbol=['circle', 'square'], size=14,
                        color=['#43A047', '#E53935'],
                        line=dict(color='black', width=1.5)),
            text=[getattr(orig, 'name', 'O'), getattr(dest, 'name', 'D')],
            textposition='top right',
            textfont=dict(family=_FONT['family'], size=9),
            showlegend=False,
        ), row=1, col=col)

        # Intermediate WPs for routed paths
        if rp is not None and hasattr(rp, 'n_intermediate') and rp.n_intermediate > 0:
            for lat_w, lon_w in rp.waypoint_positions[1:-1]:
                fig.add_trace(go.Scatter(
                    x=[lon_w], y=[lat_w], mode='markers',
                    marker=dict(symbol='diamond', size=8, color=color,
                                line=dict(color='black', width=0.8)),
                    showlegend=False, hoverinfo='skip',
                ), row=1, col=col)

        # Info annotation as shape + annotation
        info_text = (f"E = {e_r.total_energy_wh:.0f} Wh<br>"
                     f"t = {e_r.total_time/60:.1f} min<br>"
                     f"AirV = {ar_r.n_violations}")
        fig.add_annotation(
            x=0.02, y=0.98,
            xref=f'x{"" if col == 1 else col} domain',
            yref=f'y{"" if col == 1 else col} domain',
            text=info_text, showarrow=False, align='left', valign='top',
            font=dict(family='Courier New, monospace', size=9),
            bgcolor='rgba(255,255,255,0.9)', bordercolor='gray', borderwidth=1,
        )

        fig.update_xaxes(title_text='Longitude [°]', row=1, col=col)
        fig.update_yaxes(title_text='Latitude [°]',
                         scaleanchor=f'x{"" if col == 1 else col}',
                         scaleratio=1, row=1, col=col)

        # Altitude profile (row 2)
        dk = wp[:, 4] / 1000.0
        terr = dem.elevation_batch(wp[:, 0], wp[:, 1])
        valid = ~np.isnan(terr)

        fig.add_trace(go.Scatter(
            x=np.concatenate([dk, dk[::-1]]).tolist(),
            y=np.concatenate([terr, np.full(len(terr), float(np.nanmin(terr)) - 50)]).tolist(),
            fill='toself', fillcolor=_hex_to_rgba('#8D6E63', 0.35),
            line=dict(color='#5D4037', width=0.8),
            showlegend=False, hoverinfo='skip',
        ), row=2, col=col)

        if show_zones:
            ceil = terr.copy()
            ceil[valid] += 120.0
            fig.add_trace(go.Scatter(
                x=dk[valid].tolist(), y=ceil[valid].tolist(),
                mode='lines', name='120m AGL',
                line=dict(color='#F57F17', width=1.2),
                showlegend=(col == 3),
            ), row=2, col=col)

        fig.add_trace(go.Scatter(
            x=dk.tolist(), y=wp[:, 2].tolist(),
            mode='lines', name=label + ' alt',
            line=dict(color=color, width=3),
            showlegend=False,
            hovertemplate=f'{label}<br>Dist: %{{x:.1f}} km<br>Alt: %{{y:.0f}} m<extra></extra>',
        ), row=2, col=col)

        fig.update_xaxes(title_text='Ground Distance [km]', row=2, col=col)
        fig.update_yaxes(title_text='Altitude [m AMSL]', row=2, col=col)

    fig.update_layout(**_layout(title), height=850)
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 15. VEHICLE COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

def plot_vehicle_comparison(
    altitude: float = 2850.0,
    airspeed: float = 25.0,
    title: str = "Parametric Vehicle Comparison at High Altitude",
    save_path: str = None,
) -> go.Figure:
    """Interactive vehicle comparison: stall envelope + performance bars."""
    from .vehicles import get_vehicle, compare_vehicles_at_altitude

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Safe Speed vs Altitude', f'Performance at {altitude:.0f} m ASL'),
        specs=[[{'secondary_y': False}, {'secondary_y': True}]],
    )

    alts = np.linspace(0, 5500, 100)
    v_colors = {
        'baseline': '#2196F3', 'heavy_cargo': '#F44336',
        'long_range': '#4CAF50', 'high_altitude': '#9C27B0',
    }

    for name, color in v_colors.items():
        ac_v = get_vehicle(name)
        vs_safe = [ac_v.stall_speed_at(a) * 1.3 for a in alts]
        fig.add_trace(go.Scatter(
            x=vs_safe, y=alts.tolist(), mode='lines',
            name=name, line=dict(color=color, width=2),
        ), row=1, col=1)

    fig.add_vline(x=airspeed, line=dict(color='black', dash='dash', width=1.5),
                  annotation_text=f'V_cruise={airspeed} m/s', row=1, col=1)
    fig.add_hrect(y0=2700, y1=3000, fillcolor=_hex_to_rgba('#2196F3', 0.1),
                  line_width=0, row=1, col=1)

    comp = compare_vehicles_at_altitude(altitude, airspeed)
    names = list(comp.keys())
    ranges = [comp[n]['range_cruise_km'] for n in names]
    hovers = [comp[n]['endurance_hover_min'] for n in names]
    bar_colors = [v_colors[n] for n in names]

    fig.add_trace(go.Bar(
        x=names, y=ranges, name='Cruise Range [km]',
        marker_color=bar_colors, opacity=0.8, offsetgroup='r',
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        x=names, y=hovers, name='Hover Endurance [min]',
        marker_color=bar_colors, opacity=0.35, offsetgroup='h',
        yaxis='y3',
    ), row=1, col=2)

    fig.update_xaxes(title_text='V_safe (1.3×V_stall) [m/s]', row=1, col=1)
    fig.update_yaxes(title_text='Altitude [m]', row=1, col=1)
    fig.update_yaxes(title_text='Cruise Range [km]', row=1, col=2)
    fig.update_layout(
        **_layout(title),
        barmode='group',
        yaxis3=dict(title='Hover Endurance [min]', overlaying='y2', side='right'),
        height=500,
    )
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 16. A* VS DE COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

def plot_astar_vs_de(
    labels, e_astar, e_de, t_astar, t_de,
    title: str = "A* Grid vs DE Continuous Optimization",
    save_path: str = None,
) -> go.Figure:
    """Interactive bar chart comparing A* grid vs DE continuous optimizer."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Energy Comparison', 'Time Comparison'),
    )

    for row_col, (metric, ya, yd, ylabel) in enumerate([
            ('Energy [Wh]', e_astar, e_de, 'Energy [Wh]'),
            ('Time [min]', t_astar, t_de, 'Time [min]')], 1):
        fig.add_trace(go.Bar(
            x=labels, y=ya, name='A* Grid' if row_col == 1 else None,
            marker_color='#FF9800', opacity=0.8,
            marker_line_color='black', marker_line_width=0.5,
            offsetgroup='a', showlegend=(row_col == 1),
        ), row=1, col=row_col)
        fig.add_trace(go.Bar(
            x=labels, y=yd, name='DE Continuous' if row_col == 1 else None,
            marker_color='#2196F3', opacity=0.8,
            marker_line_color='black', marker_line_width=0.5,
            offsetgroup='d', showlegend=(row_col == 1),
        ), row=1, col=row_col)
        fig.update_yaxes(title_text=ylabel, row=1, col=row_col)

    fig.update_layout(**_layout(title), barmode='group', height=480)
    _save(fig, save_path)
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
) -> Dict[str, go.Figure]:
    """
    Generate all applicable interactive Plotly figures based on available data.

    Returns a dict of name → go.Figure. Saves each figure as an .html file
    in save_dir (if provided). Only generates figures for which sufficient
    data was provided.

    Parameters are identical to visualization.plot_all().
    """
    figs = {}

    def _sp(name: str) -> Optional[str]:
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            return os.path.join(save_dir, name)
        return None

    def _t(t: str) -> str:
        return f"{title_prefix}{t}" if title_prefix else t

    # 1. Convergence
    if opt_results:
        figs['convergence'] = plot_convergence(
            opt_results, title=_t("Optimization Convergence"),
            save_path=_sp('convergence.html'))

    # 2. 2D path
    if dem is not None and paths:
        figs['path_2d'] = plot_path_2d(
            dem, paths, facilities=facilities,
            title=_t("Flight Paths over DEM"),
            save_path=_sp('path_2d.html'))

    # 3. 3D path
    if dem is not None and paths:
        figs['path_3d'] = plot_path_3d(
            dem, paths, title=_t("3D Flight Path over DEM"),
            save_path=_sp('path_3d.html'))

    # 4. Path evolution
    if (dem is not None and initial_path is not None
            and opt_results and ac is not None):
        try:
            figs['path_evolution'] = plot_path_evolution(
                dem, initial_path, opt_results[0], ac,
                title=_t("Path Evolution During Optimization"),
                save_path=_sp('path_evolution.html'))
        except Exception:
            pass

    # 5. Energy profiles
    if energy_results:
        for i, er in enumerate(energy_results):
            suffix = f"_{i}" if len(energy_results) > 1 else ""
            figs[f'energy_profile{suffix}'] = plot_energy_profile(
                er, title=_t(f"Energy & SOC Profile{suffix}"),
                save_path=_sp(f'energy_profile{suffix}.html'))

    # 6. Pareto front
    if opt_results and len(opt_results) >= 3:
        try:
            figs['pareto'] = plot_pareto_front(
                opt_results, title=_t("Pareto Front: Energy vs. Time"),
                save_path=_sp('pareto.html'))
        except Exception:
            pass

    # 7. Airspace map
    if dem is not None and airspace is not None:
        figs['airspace_map'] = plot_airspace_map(
            dem, airspace, paths=paths, facilities=facilities,
            title=_t("Airspace Restrictions and Flight Paths"),
            save_path=_sp('airspace_map.html'))

    # 8. Path vs ceiling
    if dem is not None and airspace is not None and paths:
        figs['path_vs_ceiling'] = plot_path_vs_ceiling(
            dem, airspace, paths,
            title=_t("Flight Profile with Regulatory Ceiling"),
            save_path=_sp('path_vs_ceiling.html'))

    # 9. Topology comparison
    if (dem is not None and airspace is not None
            and fp_direct is not None and fp_rdac is not None):
        figs['topology_comparison'] = plot_topology_comparison(
            dem, airspace, fp_direct, fp_rdac,
            rp_off=rp_direct, rp_on=rp_rdac,
            facilities=facilities,
            title=_t("Topology Change: Airspace OFF vs ON"),
            save_path=_sp('topology_comparison.html'))

    # 10. Three-path comparison
    if (dem is not None and airspace is not None
            and fp_straight is not None and fp_direct is not None
            and fp_rdac is not None and orig is not None and dest is not None):
        figs['three_path'] = plot_three_path_comparison(
            dem, airspace, orig, dest,
            fp_straight, fp_direct, fp_rdac,
            rp_direct=rp_direct, rp_rdac=rp_rdac,
            e_straight=e_straight, e_direct=e_direct, e_rdac=e_rdac,
            title=_t("Progressive Constraint Analysis"),
            save_path=_sp('three_path_comparison.html'))

    # 11. Stall envelope
    if ac is not None:
        figs['stall_envelope'] = plot_stall_envelope(
            ac, title=_t("Aerodynamic Flight Envelope"),
            save_path=_sp('stall_envelope.html'))

    # 12. A* vs DE
    if (astar_labels is not None and e_astar is not None
            and e_de is not None and t_astar is not None and t_de is not None):
        figs['astar_vs_de'] = plot_astar_vs_de(
            astar_labels, e_astar, e_de, t_astar, t_de,
            title=_t("A* Grid vs DE Continuous"),
            save_path=_sp('astar_vs_de.html'))

    # 13. Mission SOC
    if leg_data:
        figs['mission_soc'] = plot_mission_soc(
            leg_data, title=_t("Multi-Point Mission — Battery SOC Profile"),
            save_path=_sp('mission_soc.html'))

    return figs
