"""
Terrain Analyzer — Evaluate Paths Against the DEM
===================================================

Checks flight paths for terrain clearance violations, computes
AGL (above ground level) profiles, and provides constraint values
suitable for optimization (continuous, differentiable-ish penalties).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from .dem import DEMInterface
from .path import FlightPath, Waypoint
from .config import MissionConstraints


@dataclass
class TerrainReport:
    """
    Result of terrain analysis for a flight path.

    Attributes
    ----------
    is_feasible : bool
        True if all terrain clearance constraints are satisfied.
    min_agl : float
        Minimum altitude above ground level along the path [m].
    min_agl_location : Tuple[float, float, int]
        (lat, lon, segment_idx) where minimum AGL occurs.
    max_terrain_below : float
        Maximum terrain elevation directly below the flight path [m].
    n_violations : int
        Number of waypoints that violate terrain clearance.
    violation_fraction : float
        Fraction of waypoints in violation.
    agl_profile : np.ndarray
        AGL at each waypoint [m].
    terrain_profile : np.ndarray
        Terrain elevation at each waypoint [m].
    clearance_margin : float
        min_agl minus required clearance (negative = violation) [m].
    constraint_penalty : float
        Smooth penalty value for optimization (0 = feasible).
    """
    is_feasible: bool
    min_agl: float
    min_agl_location: Tuple[float, float, int]
    max_terrain_below: float
    n_violations: int
    violation_fraction: float
    agl_profile: np.ndarray
    terrain_profile: np.ndarray
    clearance_margin: float
    constraint_penalty: float


class TerrainAnalyzer:
    """
    Evaluates flight paths against terrain.

    Provides both binary feasibility checks and continuous penalty
    functions suitable for gradient-based or stochastic optimization.

    Parameters
    ----------
    dem : DEMInterface
        The terrain model.
    constraints : MissionConstraints
        Clearance requirements and limits.

    Example
    -------
    >>> analyzer = TerrainAnalyzer(dem, constraints)
    >>> report = analyzer.analyze(path)
    >>> print(f"Feasible: {report.is_feasible}, min AGL: {report.min_agl:.0f} m")
    """

    def __init__(self, dem: DEMInterface, constraints: MissionConstraints):
        self.dem = dem
        self.constraints = constraints

    def analyze(self, path: FlightPath) -> TerrainReport:
        """
        Full terrain analysis of a flight path.

        Queries terrain elevation at every waypoint, computes AGL,
        checks clearance constraints, and produces a TerrainReport.
        """
        waypoints = path.get_waypoints()

        if not waypoints:
            return TerrainReport(
                is_feasible=False, min_agl=0.0,
                min_agl_location=(0, 0, 0),
                max_terrain_below=0.0, n_violations=0,
                violation_fraction=1.0,
                agl_profile=np.array([]),
                terrain_profile=np.array([]),
                clearance_margin=-np.inf,
                constraint_penalty=np.inf,
            )

        # Extract coordinates
        lats = np.array([wp.lat for wp in waypoints])
        lons = np.array([wp.lon for wp in waypoints])
        alts = np.array([wp.alt for wp in waypoints])
        seg_indices = np.array([wp.segment_idx for wp in waypoints])

        # Query terrain
        terrain_elevs = self.dem.elevation_batch(lats, lons)

        # Compute AGL
        agl = alts - terrain_elevs

        # Determine required clearance per waypoint
        required_clearance = np.full_like(agl, self.constraints.min_terrain_clearance)
        for i, wp in enumerate(waypoints):
            seg = path.segments[wp.segment_idx]
            seg_type = seg.segment_type.value
            required_clearance[i] = self.constraints.terrain_clearance_for_segment(seg_type)

        # Find violations (excluding NaN terrain)
        valid_mask = ~np.isnan(terrain_elevs)
        violations = valid_mask & (agl < required_clearance)
        n_valid = valid_mask.sum()
        n_violations = violations.sum()
        violation_frac = n_violations / n_valid if n_valid > 0 else 0.0

        # Min AGL (over valid terrain)
        agl_valid = agl.copy()
        agl_valid[~valid_mask] = np.inf
        min_agl_idx = np.argmin(agl_valid)
        min_agl = float(agl_valid[min_agl_idx])
        min_agl_loc = (float(lats[min_agl_idx]),
                       float(lons[min_agl_idx]),
                       int(seg_indices[min_agl_idx]))

        max_terrain = float(np.nanmax(terrain_elevs))

        # Clearance margin: how much slack above minimum required
        clearance_margin = min_agl - self.constraints.min_terrain_clearance

        # Smooth penalty for optimization
        # penalty = sum of squared violations (0 when feasible)
        deficit = required_clearance - agl  # positive = violation
        deficit[~valid_mask] = 0.0
        deficit = np.maximum(deficit, 0.0)
        constraint_penalty = float(np.sum(deficit ** 2))

        # Fill terrain info back into waypoints
        for i, wp in enumerate(waypoints):
            wp.terrain_elev = terrain_elevs[i]
            wp.agl = agl[i]

        return TerrainReport(
            is_feasible=(n_violations == 0),
            min_agl=min_agl,
            min_agl_location=min_agl_loc,
            max_terrain_below=max_terrain,
            n_violations=int(n_violations),
            violation_fraction=violation_frac,
            agl_profile=agl,
            terrain_profile=terrain_elevs,
            clearance_margin=clearance_margin,
            constraint_penalty=constraint_penalty,
        )

    def clearance_at_point(self, lat: float, lon: float,
                           alt: float) -> float:
        """AGL at a single point."""
        terrain = self.dem.elevation(lat, lon)
        if np.isnan(terrain):
            return np.inf
        return alt - terrain

    def required_altitude(self, lat: float, lon: float,
                          segment_type: str = "FW_CRUISE") -> float:
        """
        Minimum safe altitude at a point given the terrain and constraints.

        Returns the altitude [m AMSL] needed to satisfy clearance.
        """
        terrain = self.dem.elevation(lat, lon)
        if np.isnan(terrain):
            return 0.0
        clearance = self.constraints.terrain_clearance_for_segment(segment_type)
        return terrain + clearance

    def required_altitude_profile(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        segment_type: str = "FW_CRUISE",
        n: int = 200
    ) -> dict:
        """
        Compute the minimum safe altitude along a straight-line profile.

        Returns
        -------
        dict with keys:
            'distances' : ndarray
            'required_altitudes' : ndarray — minimum safe alt [m AMSL]
            'terrain_elevations' : ndarray
            'max_required_altitude' : float
        """
        profile = self.dem.terrain_profile(start, end, n)
        clearance = self.constraints.terrain_clearance_for_segment(segment_type)
        req_alt = profile['elevations'] + clearance

        return {
            'distances': profile['distances'],
            'required_altitudes': req_alt,
            'terrain_elevations': profile['elevations'],
            'max_required_altitude': float(np.nanmax(req_alt)),
            'lats': profile['lats'],
            'lons': profile['lons'],
        }

    def compute_penalty_gradient(
        self,
        path: FlightPath,
        epsilon: float = 1.0
    ) -> np.ndarray:
        """
        Numerical gradient of the constraint penalty w.r.t. path parameters.

        Uses central finite differences. Useful for gradient-based optimizers.

        Parameters
        ----------
        path : FlightPath
        epsilon : float
            Finite difference step size.

        Returns
        -------
        gradient : ndarray, shape (n_parameters,)
        """
        theta = path.parameter_vector.copy()
        n = len(theta)
        grad = np.zeros(n)

        for i in range(n):
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus[i] += epsilon
            theta_minus[i] -= epsilon

            path.parameter_vector = theta_plus
            penalty_plus = self.analyze(path).constraint_penalty

            path.parameter_vector = theta_minus
            penalty_minus = self.analyze(path).constraint_penalty

            grad[i] = (penalty_plus - penalty_minus) / (2 * epsilon)

        # Restore original
        path.parameter_vector = theta
        return grad
