"""
Path Builder — DEM-Aware Feasible Path Generation
===================================================

The PathBuilder creates physically reasonable flight paths between
two facilities by analyzing the terrain along the route and
constructing appropriate segment sequences.

Strategies:
    1. HIGH_OVERFLY  — climb above all terrain, cruise at safe altitude
    2. TERRAIN_FOLLOW — adapt altitude to terrain, multiple climb/descend
    3. MINIMAL_ENERGY — minimum altitude changes, single cruise level
    4. CUSTOM         — user-defined segment sequence

The builder examines the DEM terrain profile between origin and
destination to determine:
    - Maximum terrain elevation along the route
    - Required cruise altitude for terrain clearance
    - Whether intermediate climbs/descents are needed
    - Departure and arrival vertical profiles
"""

from __future__ import annotations
from enum import Enum
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import numpy as np

from .config import UAVConfig, MissionConstraints
from .dem import DEMInterface
from .segments import (
    SegmentType, FlightSegment,
    VTOLAscend, VTOLDescend, FWClimb, FWDescend, FWCruise, Transition,
    SegmentState
)
from .path import FlightPath
from .terrain import TerrainAnalyzer


class PathStrategy(Enum):
    """Path construction strategy."""
    HIGH_OVERFLY = "high_overfly"
    TERRAIN_FOLLOW = "terrain_follow"
    MINIMAL_ENERGY = "minimal_energy"
    CUSTOM = "custom"


@dataclass
class FacilityNode:
    """A facility with geographic and elevation data."""
    name: str
    lat: float
    lon: float
    ground_elev: float  # m AMSL
    is_hangar: bool = False  # True if UAV base/hangar

    @property
    def short_name(self) -> str:
        return self.name

    def coords(self) -> Tuple[float, float]:
        return (self.lat, self.lon)


class PathBuilder:
    """
    Builds feasible flight paths between facilities over real terrain.

    The builder uses the DEM to analyze terrain along the route and
    constructs segment sequences that satisfy clearance constraints.

    Parameters
    ----------
    dem : DEMInterface
        Terrain elevation model.
    uav : UAVConfig
        Aircraft performance limits.
    constraints : MissionConstraints
        Clearance and mission constraints.

    Example
    -------
    >>> builder = PathBuilder(dem, uav, constraints)
    >>> origin = FacilityNode("H. Enrique Garcés", -0.2444, -78.5411, 3009)
    >>> dest = FacilityNode("H. Metropolitano", -0.1844, -78.5037, 2838)
    >>> path = builder.build(origin, dest, strategy=PathStrategy.HIGH_OVERFLY)
    """

    def __init__(self, dem: DEMInterface, uav: UAVConfig,
                 constraints: MissionConstraints):
        self.dem = dem
        self.uav = uav
        self.constraints = constraints
        self.terrain_analyzer = TerrainAnalyzer(dem, constraints)

    # ── Main entry point ─────────────────────────────────────────────────

    def build(
        self,
        origin: FacilityNode,
        destination: FacilityNode,
        strategy: PathStrategy = PathStrategy.HIGH_OVERFLY,
        **kwargs
    ) -> FlightPath:
        """
        Build a flight path between two facilities.

        Parameters
        ----------
        origin : FacilityNode
        destination : FacilityNode
        strategy : PathStrategy
        **kwargs : strategy-specific options

        Returns
        -------
        FlightPath with populated segments.
        """
        if strategy == PathStrategy.HIGH_OVERFLY:
            return self._build_high_overfly(origin, destination, **kwargs)
        elif strategy == PathStrategy.TERRAIN_FOLLOW:
            return self._build_terrain_follow(origin, destination, **kwargs)
        elif strategy == PathStrategy.MINIMAL_ENERGY:
            return self._build_minimal_energy(origin, destination, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    # ── Terrain analysis helpers ─────────────────────────────────────────

    def _analyze_route_terrain(
        self,
        origin: FacilityNode,
        destination: FacilityNode,
        n_samples: int = 300
    ) -> dict:
        """
        Analyze the terrain along the direct route.

        Returns a dict with terrain statistics needed for path design.
        """
        profile = self.dem.terrain_profile(
            origin.coords(), destination.coords(), n_samples
        )

        # Find terrain peaks and valleys
        elevs = profile['elevations']
        dists = profile['distances']
        valid = ~np.isnan(elevs)

        if not np.any(valid):
            # No valid terrain data — use facility elevations
            return {
                'profile': profile,
                'max_terrain': max(origin.ground_elev, destination.ground_elev),
                'max_terrain_distance': 0,
                'min_terrain': min(origin.ground_elev, destination.ground_elev),
                'mean_terrain': np.mean([origin.ground_elev, destination.ground_elev]),
                'terrain_peaks': [],
                'total_distance': profile['total_distance'],
            }

        max_idx = np.nanargmax(elevs)
        max_terrain = float(elevs[max_idx])

        # Detect significant peaks (local maxima above mean + threshold)
        mean_elev = np.nanmean(elevs)
        threshold = mean_elev + 200  # peaks at least 200 m above mean

        peaks = []
        for i in range(1, len(elevs) - 1):
            if (valid[i] and elevs[i] > threshold and
                    elevs[i] > elevs[i-1] and elevs[i] > elevs[i+1]):
                peaks.append({
                    'distance': float(dists[i]),
                    'elevation': float(elevs[i]),
                    'lat': float(profile['lats'][i]),
                    'lon': float(profile['lons'][i]),
                })

        return {
            'profile': profile,
            'max_terrain': max_terrain,
            'max_terrain_distance': float(dists[max_idx]),
            'min_terrain': float(np.nanmin(elevs)),
            'mean_terrain': float(mean_elev),
            'terrain_peaks': peaks,
            'total_distance': profile['total_distance'],
        }

    # ── Strategy: HIGH OVERFLY ───────────────────────────────────────────

    def _build_high_overfly(
        self,
        origin: FacilityNode,
        destination: FacilityNode,
        cruise_margin: float = None,
        vtol_ascend_fraction: float = 0.3,
        cruise_airspeed: float = None,
    ) -> FlightPath:
        """
        Simple strategy: climb above all terrain, cruise at safe altitude.

        Segment sequence:
            VTOL_ASCEND → TRANSITION → FW_CLIMB → FW_CRUISE →
            FW_DESCEND → TRANSITION → VTOL_DESCEND

        Parameters
        ----------
        cruise_margin : float
            Extra altitude above max terrain [m]. Default: cruise clearance.
        vtol_ascend_fraction : float
            Fraction of total climb done in VTOL mode (0-1).
        cruise_airspeed : float
            Override cruise speed [m/s].
        """
        if cruise_margin is None:
            cruise_margin = self.constraints.min_cruise_terrain_clearance
        if cruise_airspeed is None:
            cruise_airspeed = self.uav.fw_cruise_airspeed

        # Analyze terrain
        terrain = self._analyze_route_terrain(origin, destination)
        total_dist = terrain['total_distance']

        # Determine cruise altitude: above all terrain + margin
        cruise_alt = terrain['max_terrain'] + cruise_margin

        # Cap at service ceiling
        cruise_alt = min(cruise_alt, self.uav.fw_service_ceiling)

        # Build path
        path = FlightPath(
            origin.lat, origin.lon, origin.ground_elev,
            destination.lat, destination.lon, destination.ground_elev,
        )

        # ── DEPARTURE PHASE ──
        total_climb = cruise_alt - origin.ground_elev
        vtol_climb = total_climb * vtol_ascend_fraction
        fw_climb = total_climb - vtol_climb

        # VTOL ascend
        path.add_segment(VTOLAscend(
            altitude_gain=max(vtol_climb, 20.0),
            climb_rate=self.uav.vtol_max_climb_rate * 0.8,
        ))

        # Transition VTOL → FW
        path.add_segment(Transition(
            duration=self.uav.vtol_transition_duration,
            altitude_change=self.uav.vtol_transition_alt_change,
            ground_distance=self.uav.vtol_transition_distance,
        ))

        # FW climb to cruise altitude
        if fw_climb > 10:
            path.add_segment(FWClimb(
                altitude_gain=fw_climb,
                climb_angle_deg=min(8.0, self.uav.fw_max_climb_angle * 0.6),
                airspeed=cruise_airspeed * 0.9,
            ))

        # ── CRUISE PHASE ──
        # Compute remaining horizontal distance
        departure_dist = sum(
            s.kinematics.ground_distance for s in path.segments
        )

        # Estimate arrival distance
        total_descent = cruise_alt - destination.ground_elev
        vtol_descent = total_descent * vtol_ascend_fraction
        fw_descent = total_descent - vtol_descent
        descent_angle = min(6.0, self.uav.fw_max_descent_angle * 0.5)
        arrival_dist = (
            fw_descent / np.tan(np.radians(descent_angle)) +
            self.uav.vtol_transition_distance
        )

        cruise_dist = max(total_dist - departure_dist - arrival_dist, 100.0)

        path.add_segment(FWCruise(
            ground_distance=cruise_dist,
            airspeed=cruise_airspeed,
        ))

        # ── ARRIVAL PHASE ──
        if fw_descent > 10:
            path.add_segment(FWDescend(
                altitude_loss=fw_descent,
                descent_angle_deg=descent_angle,
                airspeed=cruise_airspeed * 0.9,
            ))

        # Transition FW → VTOL
        path.add_segment(Transition(
            duration=self.uav.vtol_transition_duration,
            altitude_change=-self.uav.vtol_transition_alt_change,
            ground_distance=self.uav.vtol_transition_distance,
        ))

        # VTOL descend to landing
        path.add_segment(VTOLDescend(
            altitude_loss=max(vtol_descent, 20.0),
            descent_rate=self.uav.vtol_max_descent_rate * 0.8,
        ))

        return path

    # ── Strategy: TERRAIN FOLLOW ─────────────────────────────────────────

    def _build_terrain_follow(
        self,
        origin: FacilityNode,
        destination: FacilityNode,
        target_agl: float = None,
        n_waypoints: int = 8,
        cruise_airspeed: float = None,
    ) -> FlightPath:
        """
        Adaptive strategy: follow terrain contours with multiple segments.

        Divides the route into sections, determines the required altitude
        for each section based on local terrain, and creates climb/cruise/
        descend segments as needed.

        This produces more complex paths but potentially lower energy
        consumption on routes with varying terrain.

        Parameters
        ----------
        target_agl : float
            Target AGL to maintain [m]. Default: cruise clearance.
        n_waypoints : int
            Number of intermediate altitude decision points.
        cruise_airspeed : float
            Override cruise speed [m/s].
        """
        if target_agl is None:
            target_agl = self.constraints.min_cruise_terrain_clearance
        if cruise_airspeed is None:
            cruise_airspeed = self.uav.fw_cruise_airspeed

        terrain = self._analyze_route_terrain(origin, destination,
                                               n_samples=500)
        profile = terrain['profile']
        total_dist = terrain['total_distance']

        # Sample terrain at waypoint locations
        wp_dists = np.linspace(0, total_dist, n_waypoints + 2)
        wp_elevs = np.interp(wp_dists, profile['distances'],
                             profile['elevations'])
        wp_lats = np.interp(wp_dists, profile['distances'], profile['lats'])
        wp_lons = np.interp(wp_dists, profile['distances'], profile['lons'])

        # Required altitude at each waypoint: terrain + target AGL
        required_alts = wp_elevs + target_agl

        # Smooth out the altitude profile (avoid jitter)
        # Use a moving average with padding
        kernel = np.ones(3) / 3
        smoothed_alts = np.convolve(required_alts, kernel, mode='same')
        # Ensure we don't go below required
        target_alts = np.maximum(smoothed_alts, required_alts)

        # Force endpoints to ground level
        target_alts[0] = origin.ground_elev
        target_alts[-1] = destination.ground_elev

        # Build path
        path = FlightPath(
            origin.lat, origin.lon, origin.ground_elev,
            destination.lat, destination.lon, destination.ground_elev,
        )

        # ── DEPARTURE: VTOL ascend + transition ──
        initial_climb = target_alts[1] - origin.ground_elev
        vtol_portion = min(initial_climb * 0.4, 100.0)

        path.add_segment(VTOLAscend(
            altitude_gain=max(vtol_portion, 20.0),
            climb_rate=self.uav.vtol_max_climb_rate * 0.8,
        ))

        path.add_segment(Transition(
            duration=self.uav.vtol_transition_duration,
            altitude_change=self.uav.vtol_transition_alt_change,
            ground_distance=self.uav.vtol_transition_distance,
        ))

        # Current altitude after departure
        current_alt = origin.ground_elev + vtol_portion + self.uav.vtol_transition_alt_change

        # ── ENROUTE: adaptive segments between waypoints ──
        for i in range(1, len(target_alts) - 1):
            seg_dist = wp_dists[i + 1] - wp_dists[i] if i < len(wp_dists) - 2 else wp_dists[i] - wp_dists[i-1]
            seg_dist = max(seg_dist, 100.0)
            target = target_alts[i]

            delta_h = target - current_alt

            if delta_h > 20:
                # Need to climb
                climb_angle = min(
                    np.degrees(np.arctan2(delta_h, seg_dist)),
                    self.uav.fw_max_climb_angle * 0.7
                )
                climb_angle = max(climb_angle, 2.0)
                path.add_segment(FWClimb(
                    altitude_gain=delta_h,
                    climb_angle_deg=climb_angle,
                    airspeed=cruise_airspeed * 0.9,
                ))
            elif delta_h < -20:
                # Need to descend
                desc_angle = min(
                    np.degrees(np.arctan2(-delta_h, seg_dist)),
                    self.uav.fw_max_descent_angle * 0.7
                )
                desc_angle = max(desc_angle, 2.0)
                path.add_segment(FWDescend(
                    altitude_loss=-delta_h,
                    descent_angle_deg=desc_angle,
                    airspeed=cruise_airspeed,
                ))
            else:
                # Roughly level — cruise
                path.add_segment(FWCruise(
                    ground_distance=seg_dist,
                    airspeed=cruise_airspeed,
                ))

            current_alt = target

        # ── ARRIVAL: transition + VTOL descend ──
        # Compute remaining descent
        remaining_descent = current_alt - destination.ground_elev

        if remaining_descent > 80:
            fw_desc = remaining_descent * 0.6
            path.add_segment(FWDescend(
                altitude_loss=fw_desc,
                descent_angle_deg=min(6.0, self.uav.fw_max_descent_angle * 0.5),
                airspeed=cruise_airspeed * 0.9,
            ))
            remaining_descent -= fw_desc

        path.add_segment(Transition(
            duration=self.uav.vtol_transition_duration,
            altitude_change=-min(self.uav.vtol_transition_alt_change, remaining_descent * 0.2),
            ground_distance=self.uav.vtol_transition_distance,
        ))

        # Final VTOL descent
        final_descent = max(remaining_descent * 0.8, 10.0)
        path.add_segment(VTOLDescend(
            altitude_loss=final_descent,
            descent_rate=self.uav.vtol_max_descent_rate * 0.8,
        ))

        return path

    # ── Strategy: MINIMAL ENERGY ─────────────────────────────────────────

    def _build_minimal_energy(
        self,
        origin: FacilityNode,
        destination: FacilityNode,
        cruise_airspeed: float = None,
    ) -> FlightPath:
        """
        Energy-efficient strategy: minimize total altitude changes.

        Finds the lowest safe cruise altitude and uses gentle angles.
        Good for short routes or routes without major terrain obstacles.
        """
        if cruise_airspeed is None:
            cruise_airspeed = self.uav.fw_cruise_airspeed

        terrain = self._analyze_route_terrain(origin, destination)
        max_terrain = terrain['max_terrain']
        total_dist = terrain['total_distance']

        # Cruise just above the highest point
        cruise_alt = max_terrain + self.constraints.min_cruise_terrain_clearance

        # Use gentle angles for efficiency
        climb_angle = min(5.0, self.uav.fw_max_climb_angle * 0.35)
        descent_angle = min(4.0, self.uav.fw_max_descent_angle * 0.35)

        path = FlightPath(
            origin.lat, origin.lon, origin.ground_elev,
            destination.lat, destination.lon, destination.ground_elev,
        )

        total_climb = cruise_alt - origin.ground_elev
        total_descent = cruise_alt - destination.ground_elev

        # Minimal VTOL (just enough for safe transition)
        vtol_up = min(50.0, total_climb * 0.2)
        vtol_down = min(50.0, total_descent * 0.2)

        # Departure
        path.add_segment(VTOLAscend(
            altitude_gain=vtol_up,
            climb_rate=self.uav.vtol_max_climb_rate * 0.7,
        ))
        path.add_segment(Transition(
            duration=self.uav.vtol_transition_duration,
            altitude_change=self.uav.vtol_transition_alt_change,
            ground_distance=self.uav.vtol_transition_distance,
        ))

        fw_climb = total_climb - vtol_up - self.uav.vtol_transition_alt_change
        if fw_climb > 5:
            path.add_segment(FWClimb(
                altitude_gain=fw_climb,
                climb_angle_deg=climb_angle,
                airspeed=cruise_airspeed * 0.85,
            ))

        # Estimate distances used by departure/arrival
        dep_dist = sum(s.kinematics.ground_distance for s in path.segments)
        fw_descent = total_descent - vtol_down - self.uav.vtol_transition_alt_change
        arr_dist = (fw_descent / np.tan(np.radians(descent_angle)) +
                    self.uav.vtol_transition_distance) if fw_descent > 5 else 200

        cruise_dist = max(total_dist - dep_dist - arr_dist, 200.0)

        path.add_segment(FWCruise(
            ground_distance=cruise_dist,
            airspeed=cruise_airspeed,
        ))

        if fw_descent > 5:
            path.add_segment(FWDescend(
                altitude_loss=fw_descent,
                descent_angle_deg=descent_angle,
                airspeed=cruise_airspeed,
            ))

        path.add_segment(Transition(
            duration=self.uav.vtol_transition_duration,
            altitude_change=-self.uav.vtol_transition_alt_change,
            ground_distance=self.uav.vtol_transition_distance,
        ))

        path.add_segment(VTOLDescend(
            altitude_loss=vtol_down,
            descent_rate=self.uav.vtol_max_descent_rate * 0.7,
        ))

        return path

    # ── Utilities ────────────────────────────────────────────────────────

    def compare_strategies(
        self,
        origin: FacilityNode,
        destination: FacilityNode,
    ) -> Dict[str, dict]:
        """
        Build paths with all strategies and compare.

        Returns a dict of strategy_name → {path, metrics, terrain_report}.
        """
        results = {}
        for strategy in [PathStrategy.HIGH_OVERFLY,
                         PathStrategy.TERRAIN_FOLLOW,
                         PathStrategy.MINIMAL_ENERGY]:
            path = self.build(origin, destination, strategy)
            report = self.terrain_analyzer.analyze(path)
            metrics = path.metrics

            results[strategy.value] = {
                'path': path,
                'metrics': metrics,
                'terrain_report': report,
                'feasible': report.is_feasible,
                'total_time': metrics.total_time,
                'total_distance': metrics.total_ground_distance,
                'max_altitude': metrics.max_altitude,
                'n_segments': metrics.n_segments,
                'min_agl': report.min_agl,
                'penalty': report.constraint_penalty,
            }

        return results
