"""
Terrain Analyzer — Evaluate Paths Against the DEM
===================================================

Checks flight paths for terrain clearance violations, computes
AGL (above ground level) profiles, and provides constraint values
suitable for optimization (continuous, differentiable-ish penalties).
"""

from __future__ import annotations
from dataclasses import dataclass, field
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

    # ── Vertical corridor (floor *and* ceiling) ───────────────────────
    max_agl: float = np.nan
    #: Waypoints above the regulatory AGL ceiling (RDAC 101.185).
    n_ceiling_violations: int = 0
    ceiling_violation_fraction: float = 0.0
    #: Worst exceedance of the ceiling [m].
    max_ceiling_excess: float = 0.0
    ceiling_feasible: bool = True
    #: Required clearance applied at each waypoint [m AGL]; 0 in the
    #: take-off/landing funnels.
    required_clearance: np.ndarray = field(default_factory=lambda: np.array([]))
    #: Waypoints exempted from the clearance floor (terminal funnels).
    n_exempt: int = 0
    #: Lowest AGL among waypoints that actually violate the floor [m].
    #: ``min_agl`` is normally 0 at the pads and says nothing about
    #: whether anything is wrong; this is the number to quote.
    min_violating_agl: float = float('nan')
    #: Smooth measure of how far the path lies outside the corridor,
    #: in metre-fraction units: mean(|excursion|)/corridor_width.
    corridor_penalty: float = 0.0

    @property
    def corridor_feasible(self) -> bool:
        """True when the path stays inside the vertical corridor."""
        return self.is_feasible and self.ceiling_feasible


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

        # Take-off and landing funnels: the aircraft *must* reach AGL 0 at
        # its pads, so demanding clearance there marks every path — however
        # well routed — as infeasible at its own endpoints. Exempt waypoints
        # within terminal_exempt_radius_m of either terminus.
        exempt = self._terminal_mask(path, lats, lons)
        required_clearance = np.where(exempt, 0.0, required_clearance)

        # Find violations (excluding NaN terrain). The tolerance keeps
        # accumulated floating-point error in the position propagation from
        # deciding feasibility; it is orders of magnitude below DEM error.
        tol = getattr(self.constraints, 'clearance_tolerance_m', 0.0)
        valid_mask = ~np.isnan(terrain_elevs)
        violations = valid_mask & (agl < required_clearance - tol)
        n_valid = valid_mask.sum()
        n_violations = violations.sum()
        violation_frac = n_violations / n_valid if n_valid > 0 else 0.0

        # Regulatory ceiling (RDAC 101.185). Climb-out and approach are
        # exempt for the same reason as the floor: the funnels are where
        # the aircraft legitimately transits the whole band.
        max_agl = float(self.constraints.max_agl)
        ceiling_excess = np.where(valid_mask & ~exempt, agl - max_agl - tol, 0.0)
        ceiling_excess = np.maximum(ceiling_excess, 0.0)
        n_ceiling = int(np.sum(ceiling_excess > 0))
        ceiling_frac = n_ceiling / n_valid if n_valid > 0 else 0.0

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
        deficit = required_clearance - agl - tol  # positive = violation
        deficit[~valid_mask] = 0.0
        deficit = np.maximum(deficit, 0.0)
        constraint_penalty = float(np.sum(deficit ** 2))

        # Corridor penalty: one scale-free number the optimizer can weight
        # without retuning per region. Excursions below the floor and above
        # the ceiling are measured in units of the corridor's own height,
        # then averaged over the path — so it grows with both how far and
        # how often the path leaves the band.
        width = max(self.constraints.corridor_width("FW_CRUISE"), 1.0)
        excursion = deficit + ceiling_excess
        corridor_penalty = float(np.mean(excursion[valid_mask]) / width) \
            if n_valid > 0 else 0.0

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
            max_agl=max_agl,
            n_ceiling_violations=n_ceiling,
            ceiling_violation_fraction=ceiling_frac,
            max_ceiling_excess=float(np.max(ceiling_excess)) if len(ceiling_excess) else 0.0,
            ceiling_feasible=(n_ceiling == 0),
            required_clearance=required_clearance,
            n_exempt=int(np.sum(exempt)),
            min_violating_agl=(float(np.min(agl[violations]))
                               if n_violations > 0 else float('nan')),
            corridor_penalty=corridor_penalty,
        )

    def _terminal_mask(self, path: FlightPath,
                       lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """
        Waypoints inside the take-off or landing funnel.

        A funnel is a horizontal cylinder of radius
        ``constraints.terminal_exempt_radius_m`` around each terminus.
        Inside it, neither the clearance floor nor the regulatory ceiling
        is enforced: the aircraft is climbing out of, or descending onto,
        a pad, and must legitimately pass through every height in the band.
        """
        r = getattr(self.constraints, 'terminal_exempt_radius_m', 0.0)
        if r <= 0:
            return np.zeros(len(lats), dtype=bool)

        mask = np.zeros(len(lats), dtype=bool)
        for term in (path.origin, path.destination):
            d = _haversine_m(term[0], term[1], lats, lons)
            mask |= (d <= r)
        return mask

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


def _haversine_m(lat0: float, lon0: float,
                 lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Vectorised great-circle distance from one point to many [m]."""
    R = 6_371_000.0
    phi0 = np.radians(lat0)
    phi = np.radians(lats)
    dphi = phi - phi0
    dlam = np.radians(lons - lon0)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi0) * np.cos(phi) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
