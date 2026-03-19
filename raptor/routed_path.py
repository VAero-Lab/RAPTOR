"""
Routed Path — Waypoint-Based Lateral Routing for Constrained Environments
===========================================================================

When regulatory zones or complex terrain prevent direct straight-line
flight from origin to destination, the path must deviate laterally.
RoutedPath parameterizes this deviation through intermediate waypoints
whose positions are design variables for the optimizer.

Architecture:
    RoutedPath manages a set of **routing parameters** (lateral offsets,
    cruise altitudes, cruise airspeeds) and deterministically builds a
    standard FlightPath from them. The optimizer sees only the routing
    parameters; the segment-level details are computed automatically.

Key property:
    When all lateral offsets = 0, the RoutedPath reduces to a standard
    straight-line path. The optimizer is free to discover that deviations
    from the direct route produce lower total cost when regulatory or
    terrain constraints are present.

Path structure (for N intermediate waypoints):
    VTOL_ASCEND → TRANSITION →
    [FW_CLIMB/DESCEND →] FW_CRUISE  (origin → WP_1)
    [FW_CLIMB/DESCEND →] FW_CRUISE  (WP_1 → WP_2)
    ...
    [FW_CLIMB/DESCEND →] FW_CRUISE  (WP_N → destination)
    → FW_DESCEND → TRANSITION → VTOL_DESCEND

Author: Victor (EPN / LUAS-EPN)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np

from .dem import DEMInterface
from .config import UAVConfig, MissionConstraints
from .segments import (
    SegmentType, VTOLAscend, VTOLDescend, FWClimb, FWDescend,
    FWCruise, Transition
)
from .path import FlightPath
from .builder import FacilityNode


# ═══════════════════════════════════════════════════════════════════════════
# ROUTED PATH
# ═══════════════════════════════════════════════════════════════════════════

class RoutedPath:
    """
    A multi-leg flight path with lateral routing through intermediate waypoints.

    Parameters
    ----------
    origin : FacilityNode
        Takeoff facility.
    destination : FacilityNode
        Landing facility.
    dem : DEMInterface
        Terrain model.
    uav : UAVConfig
        Aircraft performance limits.
    constraints : MissionConstraints
        Clearance and mission constraints.
    n_intermediate : int
        Number of intermediate waypoints (0 = direct, like v0.3).
        Each intermediate waypoint adds 1 lateral offset design variable.
    vtol_fraction : float
        Fraction of total departure/arrival climb done in VTOL mode.

    Design Variables
    ----------------
    The parameter vector θ is structured as:

        θ = [lateral_offsets (N),
             cruise_altitudes (N+1),
             cruise_airspeeds (N+1),
             vtol_ascend_rate,
             vtol_descend_rate]

    Where N = n_intermediate.

    - lateral_offset_i [m]: perpendicular displacement of waypoint i from
      the direct line (positive = right of bearing, negative = left).
    - cruise_altitude_i [m AMSL]: cruise altitude for sub-leg i.
    - cruise_airspeed_i [m/s]: airspeed for sub-leg i.
    - vtol_ascend_rate [m/s]: VTOL vertical climb rate.
    - vtol_descend_rate [m/s]: VTOL vertical descent rate.

    Total design variables: N + (N+1) + (N+1) + 2 = 3N + 4
      N=0 → 4 variables (direct path)
      N=1 → 7 variables
      N=2 → 10 variables
      N=3 → 13 variables
    """

    def __init__(
        self,
        origin: FacilityNode,
        destination: FacilityNode,
        dem: DEMInterface,
        uav: UAVConfig,
        constraints: MissionConstraints,
        n_intermediate: int = 2,
        vtol_fraction: float = 0.3,
    ):
        self.origin = origin
        self.destination = destination
        self.dem = dem
        self.uav = uav
        self.constraints = constraints
        self.vtol_fraction = vtol_fraction

        # Direct-line geometry (computed first, needed for validation)
        self._direct_bearing = DEMInterface.bearing(
            origin.lat, origin.lon, destination.lat, destination.lon
        )
        self._direct_distance = DEMInterface.haversine(
            origin.lat, origin.lon, destination.lat, destination.lon
        )

        # Validate n_intermediate: each sub-leg must be long enough
        # for departure/arrival phases (minimum ~2 km per sub-leg)
        min_leg_dist = 2000.0  # meters
        max_reasonable = max(0, int(self._direct_distance / min_leg_dist) - 1)
        if n_intermediate > max_reasonable:
            n_intermediate = max_reasonable
        self.n_intermediate = n_intermediate

        # Positions along the direct line where waypoints sit
        # Equally spaced between origin and destination
        self._wp_fractions = np.linspace(0, 1, n_intermediate + 2)[1:-1]
        # These are the distances along the direct line
        self._wp_along_dists = self._wp_fractions * self._direct_distance

        # Initialize design variables
        self._lateral_offsets = np.zeros(n_intermediate)            # m
        n_legs = n_intermediate + 1
        # Initial cruise altitude: overfly max terrain + clearance
        init_alt = self._estimate_cruise_altitude()
        self._cruise_altitudes = np.full(n_legs, init_alt)          # m AMSL
        self._cruise_airspeeds = np.full(n_legs, uav.fw_cruise_airspeed)  # m/s
        self._vtol_ascend_rate = uav.vtol_max_climb_rate * 0.8      # m/s
        self._vtol_descend_rate = uav.vtol_max_descent_rate * 0.8   # m/s

        # Built path (cached)
        self._flight_path: Optional[FlightPath] = None
        self._waypoint_coords: Optional[List[Tuple[float, float]]] = None
        self._dirty = True

        # Ensure initial cruise altitudes clear terrain along each sub-leg
        self._update_cruise_altitudes_for_terrain()

    # ── Geometry helpers ──────────────────────────────────────────────

    def _estimate_cruise_altitude(self) -> float:
        """Estimate a safe cruise altitude from direct-line terrain profile."""
        profile = self.dem.terrain_profile(
            self.origin.coords(), self.destination.coords(), n=200
        )
        elevs = profile['elevations']
        valid = ~np.isnan(elevs)
        if np.any(valid):
            max_terrain = float(np.nanmax(elevs))
        else:
            max_terrain = max(self.origin.ground_elev,
                              self.destination.ground_elev)
        return max_terrain + self.constraints.min_cruise_terrain_clearance

    def _update_cruise_altitudes_for_terrain(self):
        """
        After waypoint positions change, update cruise altitudes so each
        sub-leg clears the terrain along its actual route.

        This is called before building the flight path to ensure the
        initial configuration is terrain-feasible. The optimizer can
        then adjust altitudes further.
        """
        wp_coords = self._compute_waypoint_positions()
        n_legs = self.n_intermediate + 1

        for leg_idx in range(n_legs):
            lat_s, lon_s = wp_coords[leg_idx]
            lat_e, lon_e = wp_coords[leg_idx + 1]

            # Sample terrain along this sub-leg
            try:
                profile = self.dem.terrain_profile(
                    (lat_s, lon_s), (lat_e, lon_e), n=50
                )
                elevs = profile['elevations']
                valid = ~np.isnan(elevs)
                if np.any(valid):
                    max_terr = float(np.nanmax(elevs))
                    min_safe = max_terr + self.constraints.min_cruise_terrain_clearance
                    # Only raise altitude, never lower it below what optimizer set
                    if self._cruise_altitudes[leg_idx] < min_safe:
                        self._cruise_altitudes[leg_idx] = min_safe
            except Exception:
                pass  # Keep existing altitude if DEM query fails

    def _compute_waypoint_positions(self) -> List[Tuple[float, float]]:
        """
        Compute geographic positions of all waypoints (origin + intermediate + dest).

        Intermediate waypoints are displaced laterally from the direct line.
        A lateral offset > 0 displaces the waypoint to the right (relative
        to the direction of travel), < 0 to the left.

        Returns list of (lat, lon) for all N+2 points.
        """
        positions = [(self.origin.lat, self.origin.lon)]

        # Perpendicular bearing: direct_bearing + 90°
        perp_bearing = (self._direct_bearing + 90.0) % 360.0

        for i in range(self.n_intermediate):
            # Position along the direct line
            along_dist = self._wp_along_dists[i]
            base_lat, base_lon = DEMInterface.destination_point(
                self.origin.lat, self.origin.lon,
                self._direct_bearing, along_dist
            )
            # Displace laterally
            offset = self._lateral_offsets[i]
            if abs(offset) > 1.0:  # Don't bother for tiny offsets
                wp_lat, wp_lon = DEMInterface.destination_point(
                    base_lat, base_lon, perp_bearing, offset
                )
            else:
                wp_lat, wp_lon = base_lat, base_lon
            positions.append((wp_lat, wp_lon))

        positions.append((self.destination.lat, self.destination.lon))
        return positions

    # ── Path construction ─────────────────────────────────────────────

    def _build_flight_path(self) -> FlightPath:
        """
        Construct a FlightPath from the current routing parameters.

        Two-pass construction:
          Pass 1: Departure + sub-leg cruises (with adaptive inter-leg transitions)
          Pass 2: Arrival phases computed from ACTUAL altitude after cruises

        This avoids the bug where pre-computed arrival uses the design-variable
        altitude that inter-leg capping may have changed.
        """
        wp_coords = self._compute_waypoint_positions()
        self._waypoint_coords = wp_coords
        n_legs = self.n_intermediate + 1

        # Per-sub-leg distances
        leg_dists = [
            DEMInterface.haversine(
                wp_coords[j][0], wp_coords[j][1],
                wp_coords[j+1][0], wp_coords[j+1][1]
            )
            for j in range(n_legs)
        ]

        # ── Adaptive departure angles ─────────────────────────────────
        cruise_alt_first = self._cruise_altitudes[0]
        total_climb = max(cruise_alt_first - self.origin.ground_elev, 20)
        climb_angle = min(8.0, self.uav.fw_max_climb_angle * 0.6)
        vtol_frac = self.vtol_fraction

        for _ in range(8):
            vtol_climb = max(total_climb * vtol_frac, 20.0)
            fw_climb = max(total_climb - vtol_climb - self.uav.vtol_transition_alt_change, 0)
            dep_dist = self.uav.vtol_transition_distance
            if fw_climb > 10:
                dep_dist += fw_climb / np.tan(np.radians(climb_angle))
            if dep_dist < leg_dists[0] * 0.7:
                break
            if climb_angle < self.uav.fw_max_climb_angle * 0.95:
                climb_angle = min(climb_angle + 2.0, self.uav.fw_max_climb_angle)
            elif vtol_frac < 0.95:
                vtol_frac = min(vtol_frac + 0.1, 0.95)
            else:
                break

        vtol_climb = max(total_climb * vtol_frac, 20.0)
        fw_climb_needed = max(total_climb - vtol_climb - self.uav.vtol_transition_alt_change, 0)

        # ── Build path: departure ─────────────────────────────────────
        path = FlightPath(
            self.origin.lat, self.origin.lon, self.origin.ground_elev,
            self.destination.lat, self.destination.lon,
            self.destination.ground_elev,
        )

        path.add_segment(VTOLAscend(
            altitude_gain=vtol_climb,
            climb_rate=self._vtol_ascend_rate,
        ))
        path.add_segment(Transition(
            duration=self.uav.vtol_transition_duration,
            altitude_change=self.uav.vtol_transition_alt_change,
            ground_distance=self.uav.vtol_transition_distance,
        ))
        if fw_climb_needed > 10:
            path.add_segment(FWClimb(
                altitude_gain=fw_climb_needed,
                climb_angle_deg=climb_angle,
                airspeed=self._cruise_airspeeds[0] * 0.9,
            ))

        # ── Sub-legs (no arrival reservation on last leg yet) ─────────
        dep_overhead = sum(s.kinematics.ground_distance for s in path.segments)
        current_alt = cruise_alt_first  # Track actual altitude through capping
        arr_angle = min(6.0, self.uav.fw_max_descent_angle * 0.5)  # defaults
        arr_vtol_frac = vtol_frac

        for leg_idx in range(n_legs):
            cruise_speed = self._cruise_airspeeds[leg_idx]
            leg_dist = leg_dists[leg_idx]

            overhead = dep_overhead if leg_idx == 0 else 0.0
            budget = leg_dist - overhead

            # Inter-leg altitude transition with adaptive angle
            interleg_dist = 0.0
            if leg_idx > 0:
                target_alt = self._cruise_altitudes[leg_idx]
                alt_change = target_alt - current_alt

                if abs(alt_change) > 10:
                    is_climb = alt_change > 0
                    alt_abs = abs(alt_change)
                    if is_climb:
                        il_angle = min(8.0, self.uav.fw_max_climb_angle * 0.6)
                        max_angle = self.uav.fw_max_climb_angle
                    else:
                        il_angle = min(6.0, self.uav.fw_max_descent_angle * 0.5)
                        max_angle = self.uav.fw_max_descent_angle

                    il_dist = alt_abs / np.tan(np.radians(il_angle))

                    # Steepen if too long
                    while il_dist > budget * 0.6 and il_angle < max_angle * 0.95:
                        il_angle = min(il_angle + 2.0, max_angle)
                        il_dist = alt_abs / np.tan(np.radians(il_angle))

                    # Cap altitude change if still too long
                    if il_dist > budget * 0.8:
                        alt_abs = min(alt_abs, budget * 0.6 * np.tan(np.radians(il_angle)))
                        il_dist = alt_abs / np.tan(np.radians(il_angle))

                    interleg_dist = il_dist
                    if is_climb:
                        current_alt += alt_abs
                        path.add_segment(FWClimb(
                            altitude_gain=alt_abs,
                            climb_angle_deg=il_angle,
                            airspeed=cruise_speed * 0.9,
                        ))
                    else:
                        current_alt -= alt_abs
                        path.add_segment(FWDescend(
                            altitude_loss=alt_abs,
                            descent_angle_deg=il_angle,
                            airspeed=cruise_speed * 0.95,
                        ))

            cruise_dist = max(budget - interleg_dist, 50.0)

            # On last leg, reserve space for arrival based on ACTUAL altitude
            if leg_idx == n_legs - 1:
                arr_alt = current_alt
                arr_descent = max(arr_alt - self.destination.ground_elev, 0)
                remaining = budget - interleg_dist

                # Adaptive arrival: steepen descent and increase VTOL fraction
                # until arrival reservation fits within remaining distance
                arr_angle = min(6.0, self.uav.fw_max_descent_angle * 0.5)
                arr_vtol_frac = vtol_frac

                for _ in range(8):
                    arr_vd = max(arr_descent * arr_vtol_frac, 20.0) if arr_descent > 10 else max(arr_descent, 20.0)
                    arr_fw = max(arr_descent - arr_vd - self.uav.vtol_transition_alt_change, 0)
                    arr_reserve = self.uav.vtol_transition_distance
                    if arr_fw > 10:
                        arr_reserve += arr_fw / np.tan(np.radians(arr_angle))

                    if arr_reserve < remaining - 50:
                        break  # Fits

                    # Steepen
                    if arr_angle < self.uav.fw_max_descent_angle * 0.95:
                        arr_angle = min(arr_angle + 2.0, self.uav.fw_max_descent_angle)
                    elif arr_vtol_frac < 0.95:
                        arr_vtol_frac = min(arr_vtol_frac + 0.1, 0.95)
                    else:
                        break  # Can't fit, will rely on endpoint correction

                cruise_dist = max(remaining - arr_reserve, 50.0)

            path.add_segment(FWCruise(
                ground_distance=cruise_dist,
                airspeed=cruise_speed,
            ))

        # ── Arrival: computed from ACTUAL altitude ────────────────────
        actual_alt = current_alt
        total_descent = max(actual_alt - self.destination.ground_elev, 0)

        if total_descent > 10:
            # Use the adaptive arrival parameters from the reservation step
            # (arr_angle and arr_vtol_frac are set in the last-leg block above)
            vtol_desc = max(total_descent * arr_vtol_frac, 20.0)
            fw_desc = max(total_descent - vtol_desc - self.uav.vtol_transition_alt_change, 0)

            if fw_desc > 10:
                path.add_segment(FWDescend(
                    altitude_loss=fw_desc,
                    descent_angle_deg=arr_angle,
                    airspeed=self._cruise_airspeeds[-1] * 0.95,
                ))
        else:
            vtol_desc = max(total_descent, 20.0)

        path.add_segment(Transition(
            duration=self.uav.vtol_transition_duration,
            altitude_change=-self.uav.vtol_transition_alt_change,
            ground_distance=self.uav.vtol_transition_distance,
        ))
        path.add_segment(VTOLDescend(
            altitude_loss=max(vtol_desc, 20.0),
            descent_rate=self._vtol_descend_rate,
        ))

        # ── Fix bearings and correct endpoint ─────────────────────────
        self._fix_bearings(path, wp_coords)
        self._correct_endpoint(path)

        return path

    def _correct_endpoint(self, path: FlightPath):
        """
        Adjust the last cruise segment to ensure the path ends at
        the destination. This compensates for accumulated rounding
        errors from bearing changes and adaptive angle adjustments.
        """
        # Find the last cruise segment
        last_cruise_idx = None
        for i in range(len(path.segments) - 1, -1, -1):
            if path.segments[i].segment_type == SegmentType.FW_CRUISE:
                last_cruise_idx = i
                break

        if last_cruise_idx is None:
            return

        # Compute how much ground distance the arrival phase
        # (everything after the last cruise) consumes
        arrival_dist = sum(
            s.kinematics.ground_distance
            for s in path.segments[last_cruise_idx + 1:]
        )

        # Compute where the last cruise starts
        pre_cruise_end = path.segments[last_cruise_idx].start_state
        # Distance from that point to destination
        dist_to_dest = DEMInterface.haversine(
            pre_cruise_end.lat, pre_cruise_end.lon,
            self.destination.lat, self.destination.lon
        )

        # The last cruise should cover: dist_to_dest - arrival_dist
        needed_cruise = max(dist_to_dest - arrival_dist, 50.0)

        # Update the cruise segment distance
        seg = path.segments[last_cruise_idx]
        old_dist = seg.kinematics.ground_distance
        if abs(needed_cruise - old_dist) > 10:
            # Set via parameter_vector to invalidate caches
            pv = seg.parameter_vector.copy()
            pv[0] = needed_cruise  # ground_distance_param is first for FWCruise
            seg.parameter_vector = pv
            # Re-propagate from this segment onward
            state = seg.start_state
            for s in path.segments[last_cruise_idx:]:
                s.start_state = state
                state = s.end_state

    def _fix_bearings(self, path: FlightPath,
                       wp_coords: List[Tuple[float, float]]):
        """
        Re-propagate path states with correct per-sub-leg bearings.

        The approach: assign a target bearing to each segment based on
        which sub-leg it belongs to, then re-propagate from origin.
        """
        from .segments import SegmentState

        # Determine bearing for each sub-leg
        n_legs = len(wp_coords) - 1
        leg_bearings = []
        for j in range(n_legs):
            brg = DEMInterface.bearing(
                wp_coords[j][0], wp_coords[j][1],
                wp_coords[j+1][0], wp_coords[j+1][1]
            )
            leg_bearings.append(brg)

        # Map each segment to its sub-leg
        # Strategy: departure segments use leg 0 bearing,
        # cruise segments mark leg boundaries, arrival segments use last leg bearing
        cruise_indices = [
            i for i, s in enumerate(path.segments)
            if s.segment_type == SegmentType.FW_CRUISE
        ]

        # Build bearing assignment: segment_index → bearing
        seg_bearings = {}

        if not cruise_indices:
            return  # No cruise segments, nothing to fix

        # Everything before first cruise: use first leg bearing
        for i in range(cruise_indices[0] + 1):
            seg_bearings[i] = leg_bearings[0] if leg_bearings else path.nominal_bearing

        # Each cruise and segments between cruises
        for leg_idx in range(len(cruise_indices)):
            ci = cruise_indices[leg_idx]
            brg = leg_bearings[min(leg_idx, len(leg_bearings) - 1)]
            seg_bearings[ci] = brg

            # Segments between this cruise and the next (inter-leg climb/descend)
            if leg_idx + 1 < len(cruise_indices):
                next_ci = cruise_indices[leg_idx + 1]
                next_brg = leg_bearings[min(leg_idx + 1, len(leg_bearings) - 1)]
                for i in range(ci + 1, next_ci):
                    seg_bearings[i] = next_brg

        # Everything after last cruise: use last leg bearing
        last_ci = cruise_indices[-1]
        last_brg = leg_bearings[-1] if leg_bearings else path.nominal_bearing
        for i in range(last_ci + 1, len(path.segments)):
            seg_bearings[i] = last_brg

        # Re-propagate with corrected bearings
        state = SegmentState(
            lat=path.origin[0],
            lon=path.origin[1],
            alt=path.origin[2],
            bearing=seg_bearings.get(0, path.nominal_bearing),
            time=0.0,
            distance=0.0,
        )

        for i, seg in enumerate(path.segments):
            brg = seg_bearings.get(i, state.bearing)
            if brg != state.bearing:
                state = SegmentState(
                    lat=state.lat, lon=state.lon, alt=state.alt,
                    bearing=brg, time=state.time, distance=state.distance,
                )
            seg.start_state = state
            state = seg.end_state

    # ── Parameter vector interface ────────────────────────────────────

    @property
    def parameter_vector(self) -> np.ndarray:
        """
        Flat design variable vector for the optimizer.

        Structure: [lateral_offsets, cruise_altitudes, cruise_airspeeds,
                    vtol_ascend_rate, vtol_descend_rate]
        """
        parts = [
            self._lateral_offsets,            # N values
            self._cruise_altitudes,           # N+1 values
            self._cruise_airspeeds,           # N+1 values
            np.array([self._vtol_ascend_rate,
                       self._vtol_descend_rate]),
        ]
        return np.concatenate(parts)

    @parameter_vector.setter
    def parameter_vector(self, theta: np.ndarray):
        """Set all routing parameters from a flat vector and rebuild path."""
        N = self.n_intermediate
        n_legs = N + 1
        idx = 0

        self._lateral_offsets = theta[idx:idx + N].copy()
        idx += N

        self._cruise_altitudes = theta[idx:idx + n_legs].copy()
        idx += n_legs

        self._cruise_airspeeds = theta[idx:idx + n_legs].copy()
        idx += n_legs

        self._vtol_ascend_rate = float(theta[idx])
        self._vtol_descend_rate = float(theta[idx + 1])

        self._dirty = True

    @property
    def parameter_bounds(self) -> List[Tuple[float, float]]:
        """Bounds for each design variable."""
        N = self.n_intermediate
        n_legs = N + 1

        bounds = []

        # Lateral offsets: ±max_offset meters
        max_offset = min(self._direct_distance * 0.4, 8000.0)
        for _ in range(N):
            bounds.append((-max_offset, max_offset))

        # Cruise altitudes: above max terrain along any plausible route
        # Use the initial (terrain-safe) altitudes as lower bound reference
        # This prevents DE from selecting altitudes below terrain
        min_cruise_alt = float(np.min(self._cruise_altitudes)) - 50.0  # small margin below init
        min_cruise_alt = max(min_cruise_alt, 
                            min(self.origin.ground_elev, self.destination.ground_elev) + 50.0)
        for _ in range(n_legs):
            bounds.append((min_cruise_alt, self.uav.fw_service_ceiling))

        # Cruise airspeeds
        for _ in range(n_legs):
            bounds.append((self.uav.fw_min_airspeed,
                           self.uav.fw_max_airspeed))

        # VTOL rates
        bounds.append((0.5, self.uav.vtol_max_climb_rate))
        bounds.append((0.5, self.uav.vtol_max_descent_rate))

        return bounds

    @property
    def parameter_names(self) -> List[str]:
        """Human-readable names for each design variable."""
        N = self.n_intermediate
        n_legs = N + 1
        names = []

        for i in range(N):
            names.append(f"lateral_offset_{i}")
        for i in range(n_legs):
            names.append(f"cruise_altitude_{i}")
        for i in range(n_legs):
            names.append(f"cruise_airspeed_{i}")
        names.append("vtol_ascend_rate")
        names.append("vtol_descend_rate")

        return names

    @property
    def n_parameters(self) -> int:
        """Total number of design variables: N + (N+1) + (N+1) + 2 = 3N + 4."""
        return 3 * self.n_intermediate + 4

    # ── Flight path access ────────────────────────────────────────────

    @property
    def flight_path(self) -> FlightPath:
        """
        Get the current FlightPath (rebuilds if parameters changed).

        This is the path used for energy analysis, terrain checking,
        and airspace checking.
        """
        if self._dirty or self._flight_path is None:
            self._flight_path = self._build_flight_path()
            self._dirty = False
        return self._flight_path

    @property
    def waypoint_positions(self) -> List[Tuple[float, float]]:
        """
        Get the geographic positions of all waypoints.

        Returns [(lat, lon), ...] for origin, intermediates, destination.
        """
        if self._dirty or self._waypoint_coords is None:
            self._waypoint_coords = self._compute_waypoint_positions()
        return self._waypoint_coords

    # ── Convenience accessors ─────────────────────────────────────────

    @property
    def lateral_offsets(self) -> np.ndarray:
        return self._lateral_offsets.copy()

    @property
    def cruise_altitudes(self) -> np.ndarray:
        return self._cruise_altitudes.copy()

    @property
    def cruise_airspeeds(self) -> np.ndarray:
        return self._cruise_airspeeds.copy()

    @property
    def max_lateral_deviation(self) -> float:
        """Maximum absolute lateral offset [m]."""
        if self.n_intermediate == 0:
            return 0.0
        return float(np.max(np.abs(self._lateral_offsets)))

    @property
    def total_route_distance(self) -> float:
        """Total route distance through all waypoints [m]."""
        coords = self.waypoint_positions
        total = 0.0
        for i in range(len(coords) - 1):
            total += DEMInterface.haversine(
                coords[i][0], coords[i][1],
                coords[i+1][0], coords[i+1][1]
            )
        return total

    @property
    def route_stretch_factor(self) -> float:
        """Ratio of routed distance to direct distance."""
        return self.total_route_distance / max(self._direct_distance, 1.0)

    def topology_summary(self) -> dict:
        """
        Describe the path topology for analysis.

        Returns a dict with metrics about the path's geometric complexity.
        """
        fp = self.flight_path
        n_cruise = sum(1 for s in fp.segments
                       if s.segment_type == SegmentType.FW_CRUISE)
        n_climb = sum(1 for s in fp.segments
                      if s.segment_type == SegmentType.FW_CLIMB)
        n_descend = sum(1 for s in fp.segments
                        if s.segment_type == SegmentType.FW_DESCEND)

        return {
            'n_segments': fp.n_segments,
            'n_cruise_legs': n_cruise,
            'n_climbs': n_climb,
            'n_descents': n_descend,
            'n_intermediate_waypoints': self.n_intermediate,
            'max_lateral_deviation_m': self.max_lateral_deviation,
            'route_stretch_factor': self.route_stretch_factor,
            'direct_distance_m': self._direct_distance,
            'total_route_distance_m': self.total_route_distance,
            'altitude_range_m': (
                float(np.min(self._cruise_altitudes)),
                float(np.max(self._cruise_altitudes)),
            ),
            'is_direct': self.max_lateral_deviation < 50.0,  # ~50m threshold
        }

    def __repr__(self) -> str:
        return (f"RoutedPath({self.origin.name} → {self.destination.name}, "
                f"N_wp={self.n_intermediate}, "
                f"n_params={self.n_parameters}, "
                f"max_offset={self.max_lateral_deviation:.0f}m)")
