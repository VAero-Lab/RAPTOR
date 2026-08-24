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

Author: Victor (LUAS-EPN / KU Leuven)
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
from .corridor import build_corridor_profile, CorridorProfile


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
        corridor_mode: str = "agl",
        min_leg_distance_m: float = 1200.0,
        free_along_track: bool = True,
    ):
        self.origin = origin
        self.destination = destination
        self.dem = dem
        self.uav = uav
        self.constraints = constraints
        self.vtol_fraction = vtol_fraction

        corridor_mode = str(corridor_mode).lower()
        if corridor_mode not in ("agl", "amsl"):
            raise ValueError(
                f"corridor_mode must be 'agl' or 'amsl', got {corridor_mode!r}"
            )
        self.corridor_mode = corridor_mode
        #: Per-leg corridor profiles, populated in AGL mode on each rebuild.
        self.leg_profiles: List[CorridorProfile] = []

        # Direct-line geometry (computed first, needed for validation)
        self._direct_bearing = DEMInterface.bearing(
            origin.lat, origin.lon, destination.lat, destination.lon
        )
        self._direct_distance = DEMInterface.haversine(
            origin.lat, origin.lon, destination.lat, destination.lon
        )

        # Validate n_intermediate: each sub-leg must be long enough to
        # hold the departure or arrival phase it carries.
        #
        # The old floor of 2 km per leg capped a 5 km corridor at a single
        # waypoint, which is one lateral degree of freedom to route around
        # a mountain with. Only the first and last legs actually need room
        # for a climb-out or an approach; the ones between are pure cruise
        # and can be much shorter. 1.2 km is comfortably longer than the
        # ~450 m a departure consumes.
        self.min_leg_distance_m = float(min_leg_distance_m)
        max_reasonable = max(
            0, int(self._direct_distance / self.min_leg_distance_m) - 1)
        if n_intermediate > max_reasonable:
            n_intermediate = max_reasonable
        self.n_intermediate = n_intermediate

        # Where along the direct line each waypoint sits. Uniform to
        # start with, and — unless switched off — a design variable.
        #
        # With only a perpendicular offset per waypoint, the route can
        # bulge sideways but cannot slide the bulge to where the obstacle
        # is. Letting the along-track position move as well turns a fixed
        # comb of detour points into a real polyline, at the cost of one
        # variable per waypoint.
        self.free_along_track = bool(free_along_track) and n_intermediate > 0
        self._wp_fractions = np.linspace(0, 1, n_intermediate + 2)[1:-1]

        # Initialize design variables
        self._lateral_offsets = np.zeros(n_intermediate)            # m
        n_legs = n_intermediate + 1

        # In AGL mode the vertical design variable is a *height above
        # ground* the leg tracks; in AMSL mode it is a fixed altitude.
        floor, ceiling = constraints.corridor("FW_CRUISE")
        self._cruise_agls = np.full(n_legs, 0.5 * (floor + ceiling))  # m AGL
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
        if self.corridor_mode == "amsl":
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

        # Sorted so the route always advances toward the destination:
        # differential evolution has no notion of ordering between
        # variables, and an unsorted set would let it produce a path that
        # doubles back on itself and still scores well on distance.
        fracs = np.sort(np.clip(self._wp_fractions, 0.02, 0.98))

        for i in range(self.n_intermediate):
            # Position along the direct line
            along_dist = float(fracs[i]) * self._direct_distance
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
        """Dispatch to the constant-altitude or terrain-following builder."""
        if self.corridor_mode == "agl":
            return self._build_flight_path_agl()
        return self._build_flight_path_amsl()

    def _build_flight_path_amsl(self) -> FlightPath:
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

    # ── Terrain-following construction (corridor_mode="agl") ──────────

    def _build_flight_path_agl(self) -> FlightPath:
        """
        Construct a FlightPath that tracks the terrain at a chosen height.

        Each sub-leg gets a corridor profile (see :mod:`raptor.corridor`):
        the lowest reference the aircraft can actually fly while staying
        ``cruise_agl`` above the ground. That reference is then realised as
        a short chain of climb / cruise / descend segments, so the existing
        energy model applies unchanged.

        This is what makes an AGL ceiling obeyable. Holding one altitude
        across a leg with 400 m of relief puts the aircraft hundreds of
        metres above ground over the low ground no matter which altitude
        is chosen; following the terrain keeps the height above ground
        near the target everywhere the aircraft's descent rate allows.
        """
        wp_coords = self._compute_waypoint_positions()
        self._waypoint_coords = wp_coords
        n_legs = self.n_intermediate + 1
        uav = self.uav

        leg_dists = [
            DEMInterface.haversine(
                wp_coords[j][0], wp_coords[j][1],
                wp_coords[j + 1][0], wp_coords[j + 1][1]
            )
            for j in range(n_legs)
        ]

        # ── Corridor profile for every sub-leg ────────────────────────
        profiles: List[CorridorProfile] = []
        for j in range(n_legs):
            profiles.append(build_corridor_profile(
                self.dem,
                wp_coords[j], wp_coords[j + 1],
                target_agl=float(self._cruise_agls[j]),
                max_climb_angle_deg=uav.fw_max_climb_angle,
                max_descent_angle_deg=uav.fw_max_descent_angle,
                max_agl=self.constraints.max_agl,
            ))
        self.leg_profiles = profiles

        path = FlightPath(
            self.origin.lat, self.origin.lon, self.origin.ground_elev,
            self.destination.lat, self.destination.lon,
            self.destination.ground_elev,
        )
        # Which sub-leg each segment belongs to. Recorded as the path is
        # built because it cannot be recovered afterwards: the legacy
        # bearing pass infers leg boundaries by counting cruise segments,
        # which is only correct when each leg contains exactly one. A
        # terrain-following leg contains many, so that inference assigns
        # most segments the wrong heading and the route walks off its own
        # waypoints.
        seg_leg: List[int] = []

        def add(segment, leg: int):
            path.add_segment(segment)
            seg_leg.append(leg)

        # ── Departure: climb to the first leg's entry altitude ────────
        entry_alt = profiles[0].start_alt
        total_climb = max(entry_alt - self.origin.ground_elev, 20.0)
        dep_angle = min(8.0, uav.fw_max_climb_angle * 0.6)
        vtol_frac = self.vtol_fraction

        vtol_climb = max(total_climb * vtol_frac, 20.0)
        fw_climb = max(
            total_climb - vtol_climb - uav.vtol_transition_alt_change, 0.0
        )
        dep_dist = uav.vtol_transition_distance
        # Steepen, then shift work into the VTOL phase, until the climb-out
        # fits comfortably inside the first leg.
        for _ in range(8):
            dep_dist = uav.vtol_transition_distance
            if fw_climb > 10:
                dep_dist += fw_climb / np.tan(np.radians(dep_angle))
            if dep_dist < leg_dists[0] * 0.5:
                break
            if dep_angle < uav.fw_max_climb_angle * 0.95:
                dep_angle = min(dep_angle + 2.0, uav.fw_max_climb_angle)
            elif vtol_frac < 0.95:
                vtol_frac = min(vtol_frac + 0.1, 0.95)
            else:
                break
            vtol_climb = max(total_climb * vtol_frac, 20.0)
            fw_climb = max(
                total_climb - vtol_climb - uav.vtol_transition_alt_change, 0.0
            )

        add(VTOLAscend(altitude_gain=vtol_climb,
                       climb_rate=self._vtol_ascend_rate), 0)
        add(Transition(
            duration=uav.vtol_transition_duration,
            altitude_change=uav.vtol_transition_alt_change,
            ground_distance=uav.vtol_transition_distance,
        ), 0)
        if fw_climb > 10:
            add(FWClimb(
                altitude_gain=fw_climb,
                climb_angle_deg=dep_angle,
                airspeed=self._cruise_airspeeds[0] * 0.9,
            ), 0)

        current_alt = (self.origin.ground_elev + vtol_climb
                       + uav.vtol_transition_alt_change
                       + (fw_climb if fw_climb > 10 else 0.0))
        dep_overhead = sum(sg.kinematics.ground_distance for sg in path.segments)

        # ── Arrival reservation on the final leg ──────────────────────
        # The descent starts where the cruise profile hands over, not at
        # the leg's far end, and how far back that is depends on how much
        # altitude has to be lost — so solve the two together.
        last = profiles[-1]
        arr_angle = min(6.0, uav.fw_max_descent_angle * 0.5)
        arr_vtol_frac = self.vtol_fraction
        arr_reserve = uav.vtol_transition_distance

        for _ in range(8):
            handover = max(leg_dists[-1] - arr_reserve, 0.0)
            exit_alt = float(np.interp(handover, last.distances, last.altitudes))
            arr_descent = max(exit_alt - self.destination.ground_elev, 0.0)

            arr_vtol = max(arr_descent * arr_vtol_frac, 20.0)
            arr_fw = max(
                arr_descent - arr_vtol - uav.vtol_transition_alt_change, 0.0
            )
            new_reserve = uav.vtol_transition_distance
            if arr_fw > 10:
                new_reserve += arr_fw / np.tan(np.radians(arr_angle))

            if new_reserve < leg_dists[-1] * 0.5:
                arr_reserve = new_reserve
                break
            if arr_angle < uav.fw_max_descent_angle * 0.95:
                arr_angle = min(arr_angle + 2.0, uav.fw_max_descent_angle)
            elif arr_vtol_frac < 0.95:
                arr_vtol_frac = min(arr_vtol_frac + 0.1, 0.95)
            else:
                arr_reserve = min(new_reserve, leg_dists[-1] * 0.5)
                break
            arr_reserve = new_reserve

        # ── Walk each leg's profile ───────────────────────────────────
        for j in range(n_legs):
            prof = profiles[j]
            speed = float(self._cruise_airspeeds[j])
            leg_len = leg_dists[j]

            d0 = dep_overhead if j == 0 else 0.0
            d_end = leg_len - arr_reserve if j == n_legs - 1 else leg_len
            if d_end - d0 < 100.0:
                d_end = d0 + 100.0

            # Reconcile the altitude we are actually at with where this
            # leg's profile begins. The connector is paid for out of the
            # front of the leg — reserving its ground distance here is what
            # keeps the sum of segment lengths equal to the leg length, so
            # the path lands where the geometry says it should.
            a_at = float(np.interp(d0, prof.distances, prof.altitudes))
            delta = a_at - current_alt
            conn_dist = 0.0
            if abs(delta) > 5.0:
                lim = (uav.fw_max_climb_angle if delta > 0
                       else uav.fw_max_descent_angle)
                ang = min(8.0 if delta > 0 else 6.0, lim * 0.6)
                conn_dist = abs(delta) / np.tan(np.radians(ang))
                room = max((d_end - d0) * 0.4, 1.0)
                if conn_dist > room:
                    conn_dist = room
                    ang = float(np.clip(
                        np.degrees(np.arctan2(abs(delta), conn_dist)), 0.5, lim
                    ))

            d_start = min(d0 + conn_dist, d_end - 50.0)
            dd, aa = self._clip_profile(prof, d_start, d_end)

            if conn_dist > 0.0:
                delta = aa[0] - current_alt
                ang = float(np.clip(
                    np.degrees(np.arctan2(abs(delta), max(conn_dist, 1.0))),
                    0.5,
                    uav.fw_max_climb_angle if delta > 0 else uav.fw_max_descent_angle,
                ))
                if delta > 0:
                    add(FWClimb(altitude_gain=abs(delta),
                                climb_angle_deg=ang,
                                airspeed=speed * 0.9), j)
                else:
                    add(FWDescend(altitude_loss=abs(delta),
                                  descent_angle_deg=ang,
                                  airspeed=speed * 0.95), j)
                current_alt = aa[0]

            for k in range(len(dd) - 1):
                dx = float(dd[k + 1] - dd[k])
                dh = float(aa[k + 1] - aa[k])
                if dx < 20.0:
                    continue
                if abs(dh) < 3.0:
                    add(FWCruise(ground_distance=dx, airspeed=speed), j)
                elif dh > 0:
                    ang = float(np.degrees(np.arctan2(dh, dx)))
                    ang = float(np.clip(ang, 0.3, uav.fw_max_climb_angle))
                    add(FWClimb(altitude_gain=dh, climb_angle_deg=ang,
                                airspeed=speed * 0.95), j)
                else:
                    ang = float(np.degrees(np.arctan2(-dh, dx)))
                    ang = float(np.clip(ang, 0.3, uav.fw_max_descent_angle))
                    add(FWDescend(altitude_loss=-dh, descent_angle_deg=ang,
                                  airspeed=speed), j)
                current_alt += dh

        # ── Arrival ───────────────────────────────────────────────────
        # Sized from the altitude actually reached, and closed out so the
        # aircraft finishes on the pad rather than a fixed 20 m below it.
        # Read the altitude off the propagated path rather than the running
        # tally: sub-5 m connector steps are skipped as not worth a segment,
        # and those skipped metres are exactly what would otherwise leave
        # the aircraft a metre or two underground at touchdown.
        end = path.end_state
        if end is not None:
            current_alt = end.alt
        total_descent = max(current_alt - self.destination.ground_elev, 0.0)
        if total_descent > 10:
            vtol_target = max(total_descent * arr_vtol_frac, 20.0)
            fw_desc = max(
                total_descent - vtol_target - uav.vtol_transition_alt_change, 0.0
            )
            if fw_desc > 10:
                add(FWDescend(
                    altitude_loss=fw_desc,
                    descent_angle_deg=arr_angle,
                    airspeed=self._cruise_airspeeds[-1] * 0.95,
                ), n_legs - 1)
                current_alt -= fw_desc

        trans_drop = min(uav.vtol_transition_alt_change,
                         max(current_alt - self.destination.ground_elev, 0.0))
        add(Transition(
            duration=uav.vtol_transition_duration,
            altitude_change=-trans_drop,
            ground_distance=uav.vtol_transition_distance,
        ), n_legs - 1)
        current_alt -= trans_drop

        vtol_desc = max(current_alt - self.destination.ground_elev, 0.0)
        add(VTOLDescend(altitude_loss=vtol_desc,
                        descent_rate=self._vtol_descend_rate), n_legs - 1)

        self._aim_at_waypoints(path, wp_coords, seg_leg)
        self._correct_endpoint_agl(path)
        return path

    def _aim_at_waypoints(self, path: FlightPath,
                          wp_coords: List[Tuple[float, float]],
                          seg_leg: List[int]):
        """
        Re-propagate, heading each segment at the waypoint it is bound for.

        Assigning a fixed heading per leg makes any along-track error
        permanent: fall a hundred metres short on the first leg and every
        later segment flies parallel to the intended track rather than
        toward the next waypoint, so the route ends up beside its
        destination rather than on it. Recomputing the bearing from the
        position actually reached closes that error out continuously.
        """
        from .segments import SegmentState

        state = SegmentState(
            lat=path.origin[0], lon=path.origin[1], alt=path.origin[2],
            bearing=path.nominal_bearing, time=0.0, distance=0.0,
        )
        n_wp = len(wp_coords)

        for i, seg in enumerate(path.segments):
            leg = seg_leg[i] if i < len(seg_leg) else n_wp - 2
            target = wp_coords[min(leg + 1, n_wp - 1)]
            brg = DEMInterface.bearing(state.lat, state.lon, target[0], target[1])
            state = SegmentState(
                lat=state.lat, lon=state.lon, alt=state.alt,
                bearing=brg, time=state.time, distance=state.distance,
            )
            seg.start_state = state
            state = seg.end_state

    def _correct_endpoint_agl(self, path: FlightPath):
        """
        Close any residual gap to the destination without breaking the
        terrain-following profile.

        The constant-altitude builder absorbs endpoint error by stretching
        the last cruise segment. Here that would be destructive: cruise
        segments are level stretches scattered through the profile, and
        lengthening one drags it away from the ground it was matched to.

        The gap itself comes from the arrival phase. Its length is
        reserved before the cruise is flown, from the altitude the profile
        is *expected* to reach; clipping, breakpoint simplification and
        the inter-leg connector all move the altitude actually reached, so
        the descent ends up slightly longer or shorter than reserved. That
        residual is absorbed here, in the arrival phase where it belongs —
        first by re-angling the last fixed-wing segment, and if the flight
        envelope will not stretch that far, by a level trim segment.

        Landing short matters more than it sounds: the altitude schedule
        still ends at pad elevation, so a path that stops 100 m short
        finishes at the elevation of the pad over ground that is not the
        pad — which reads as flying underground.
        """
        for _ in range(3):
            idx = self._last_fw_index(path)
            if idx is None:
                return

            residual = self._endpoint_residual(path, idx)
            seg = path.segments[idx]
            current = seg.kinematics.ground_distance
            if residual <= 5.0 or abs(residual - current) < 1.0:
                return

            pv = seg.parameter_vector.copy()
            if seg.segment_type == SegmentType.FW_CRUISE:
                pv[0] = residual
            else:
                limit = (self.uav.fw_max_climb_angle
                         if seg.segment_type == SegmentType.FW_CLIMB
                         else self.uav.fw_max_descent_angle)
                pv[1] = float(np.clip(
                    np.degrees(np.arctan2(float(pv[0]), residual)), 0.3, limit
                ))
            seg.parameter_vector = pv
            self._repropagate(path, idx)

            # Re-angling is bounded by the flight envelope; when the
            # segment cannot be made long enough, add level distance ahead
            # of it rather than leaving the path short of the pad.
            residual = self._endpoint_residual(path, idx)
            gap = residual - path.segments[idx].kinematics.ground_distance
            if gap > 25.0:
                path.insert_segment(idx, FWCruise(
                    ground_distance=gap,
                    airspeed=float(self._cruise_airspeeds[-1]),
                ))
                continue
            return

    @staticmethod
    def _last_fw_index(path: FlightPath) -> Optional[int]:
        """
        Index of the last fixed-wing segment before the terminal transition.

        It is not always a descent: when the cruise profile already ends
        near pad height the arrival descent is omitted entirely, and a
        search that only looked for FW_DESCEND would correct nothing.
        """
        for i in range(len(path.segments) - 1, -1, -1):
            t = path.segments[i].segment_type
            if t in (SegmentType.FW_DESCEND, SegmentType.FW_CRUISE,
                     SegmentType.FW_CLIMB):
                return i
            if t not in (SegmentType.TRANSITION, SegmentType.VTOL_DESCEND):
                return None
        return None

    def _endpoint_residual(self, path: FlightPath, idx: int) -> float:
        """Ground distance segment ``idx`` must cover to reach the pad."""
        tail = sum(sg.kinematics.ground_distance
                   for sg in path.segments[idx + 1:])
        start = path.segments[idx].start_state
        to_dest = DEMInterface.haversine(
            start.lat, start.lon,
            self.destination.lat, self.destination.lon,
        )
        return to_dest - tail

    @staticmethod
    def _repropagate(path: FlightPath, idx: int):
        """Re-walk the chain from ``idx`` onward."""
        state = path.segments[idx].start_state
        for sg in path.segments[idx:]:
            sg.start_state = state
            state = sg.end_state

    @staticmethod
    def _clip_profile(prof: CorridorProfile, d_start: float, d_end: float
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Restrict a corridor profile to ``[d_start, d_end]`` along-track.

        The window edges are interpolated so the clipped profile starts and
        ends exactly where the departure and arrival phases hand over.
        """
        d, a = prof.distances, prof.altitudes
        d_start = float(np.clip(d_start, d[0], d[-1]))
        d_end = float(np.clip(d_end, d_start, d[-1]))

        inner = (d > d_start) & (d < d_end)
        dd = np.concatenate(([d_start], d[inner], [d_end]))
        aa = np.concatenate((
            [np.interp(d_start, d, a)],
            a[inner],
            [np.interp(d_end, d, a)],
        ))
        return dd, aa

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

    # ── Seeding ───────────────────────────────────────────────────────

    def project_onto_route(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Express a point as (along-track distance, lateral offset) [m].

        Lateral is positive to the right of the direct bearing, matching
        the sign convention of the offset design variables.
        """
        m_lat = 111_320.0
        m_lon = 111_320.0 * np.cos(np.radians(self.origin.lat))
        dx = (lon - self.origin.lon) * m_lon
        dy = (lat - self.origin.lat) * m_lat

        brg = np.radians(self._direct_bearing)
        # Unit vector along the route, and the one 90° right of it.
        ax, ay = np.sin(brg), np.cos(brg)
        rx, ry = np.cos(brg), -np.sin(brg)
        return float(dx * ax + dy * ay), float(dx * rx + dy * ry)

    def seed_from_waypoints(self, coords: List[Tuple[float, float]],
                            set_along: bool = True) -> "RoutedPath":
        """
        Set the routing variables to follow an existing polyline.

        Used to hand differential evolution a starting point from the A*
        baseline, which already knows how to get round a prohibited zone.
        DE spends its early generations discovering that a detour is
        needed at all; starting from a route that has one lets it spend
        them on making the detour cheap instead.

        Points are projected onto the direct line and resampled at this
        path's waypoint positions, so a polyline of any length maps onto
        the fixed number of design variables.

        Parameters
        ----------
        coords : list of (lat, lon)
            The polyline to follow. Endpoints are ignored — they are the
            origin and destination by construction.
        set_along : bool
            Also move the waypoints along the route to match, when this
            path has along-track freedom.

        Returns
        -------
        self, so it chains.
        """
        if self.n_intermediate == 0 or len(coords) < 2:
            return self

        projected = [self.project_onto_route(la, lo) for la, lo in coords]
        along = np.array([p[0] for p in projected])
        lateral = np.array([p[1] for p in projected])

        # Keep only the monotone advancing part; A* grids can emit points
        # that step back slightly, and np.interp needs increasing x.
        order = np.argsort(along)
        along, lateral = along[order], lateral[order]
        keep = np.concatenate(([True], np.diff(along) > 1.0))
        along, lateral = along[keep], lateral[keep]
        if len(along) < 2:
            return self

        if set_along and self.free_along_track:
            # Place waypoints where the route bends most: the turning
            # points carry the information, and spacing them uniformly
            # would average the corner away.
            curvature = np.abs(np.gradient(np.gradient(lateral)))
            weight = np.cumsum(curvature + 1e-6)
            # Normalise to start at 0 and end at 1 over the same index
            # range as `along`. A bare cumsum starts at the first
            # element's weight, which shifts every target back by one
            # interval and pins the first waypoint to its lower bound —
            # exactly where it has the least room to be useful.
            weight = (weight - weight[0]) / max(weight[-1] - weight[0], 1e-12)
            targets = np.interp(
                np.linspace(0, 1, self.n_intermediate + 2)[1:-1],
                weight, along,
            )
            self._wp_fractions = np.clip(
                targets / max(self._direct_distance, 1.0), 0.02, 0.98)

        sample_at = self.along_fractions * self._direct_distance
        max_offset = min(self._direct_distance * 0.4, 8000.0)
        self._lateral_offsets = np.clip(
            np.interp(sample_at, along, lateral), -max_offset, max_offset)

        self._dirty = True
        return self

    def seed_from_astar(self, planner, payload_kg: float = 0.0,
                        verbose: bool = False) -> bool:
        """
        Seed the routing variables from an A* grid search.

        Returns True if a route was found and applied. A failed A* is not
        an error — the direct route stays as the starting point — but it
        is worth knowing about, because it usually means the corridor is
        blocked in a way lateral routing alone will not fix.
        """
        try:
            result = planner.plan(
                (self.origin.lat, self.origin.lon, self.origin.ground_elev),
                (self.destination.lat, self.destination.lon,
                 self.destination.ground_elev),
            )
        except Exception as exc:
            if verbose:
                print(f"  A* seeding failed: {type(exc).__name__}: {exc}")
            return False

        # AStarResult reports `path_found`; other planners may say
        # `success`. Accept either, and treat a missing flag as failure
        # rather than as consent — a seeding step that silently does
        # nothing is worse than one that says it could not.
        found = getattr(result, "path_found", None)
        if found is None:
            found = getattr(result, "success", None)
        if not found or not getattr(result, "waypoints", None):
            if verbose:
                print("  A* found no route; keeping the direct seed.")
            return False

        coords = [(w[0], w[1]) for w in result.waypoints]
        self.seed_from_waypoints(coords)
        if verbose:
            print(f"  Seeded from A*: {len(coords)} points, "
                  f"lateral offsets {np.round(self._lateral_offsets, 0)}")
        return True

    # ── Parameter vector interface ────────────────────────────────────

    @property
    def parameter_vector(self) -> np.ndarray:
        """
        Flat design variable vector for the optimizer.

        Structure: [lateral_offsets, cruise_altitudes, cruise_airspeeds,
                    vtol_ascend_rate, vtol_descend_rate]
        """
        vertical = (self._cruise_agls if self.corridor_mode == "agl"
                    else self._cruise_altitudes)
        parts = [
            self._lateral_offsets,            # N values
            vertical,                         # N+1 values
            self._cruise_airspeeds,           # N+1 values
            np.array([self._vtol_ascend_rate,
                       self._vtol_descend_rate]),
        ]
        if self.free_along_track:
            parts.append(self._wp_fractions)  # N values
        return np.concatenate(parts)

    @parameter_vector.setter
    def parameter_vector(self, theta: np.ndarray):
        """Set all routing parameters from a flat vector and rebuild path."""
        N = self.n_intermediate
        n_legs = N + 1
        idx = 0

        self._lateral_offsets = theta[idx:idx + N].copy()
        idx += N

        if self.corridor_mode == "agl":
            self._cruise_agls = theta[idx:idx + n_legs].copy()
        else:
            self._cruise_altitudes = theta[idx:idx + n_legs].copy()
        idx += n_legs

        self._cruise_airspeeds = theta[idx:idx + n_legs].copy()
        idx += n_legs

        self._vtol_ascend_rate = float(theta[idx])
        self._vtol_descend_rate = float(theta[idx + 1])
        idx += 2

        if self.free_along_track:
            self._wp_fractions = theta[idx:idx + N].copy()

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

        # Vertical design variables.
        if self.corridor_mode == "agl":
            # Height above ground, bounded by the regulatory corridor itself.
            # Bounding the *search* by the rule — rather than only penalising
            # violations after the fact — is what keeps the optimizer inside
            # legal airspace instead of exploring altitudes it can never use.
            floor, ceiling = self.constraints.corridor("FW_CRUISE")
            hi = max(ceiling * 0.95, floor + 1.0)
            for _ in range(n_legs):
                bounds.append((floor, hi))
        else:
            # Constant-altitude legacy mode: keep the optimizer above the
            # terrain-safe initial altitude.
            min_cruise_alt = float(np.min(self._cruise_altitudes)) - 50.0
            min_cruise_alt = max(
                min_cruise_alt,
                min(self.origin.ground_elev, self.destination.ground_elev) + 50.0,
            )
            for _ in range(n_legs):
                bounds.append((min_cruise_alt, self.uav.fw_service_ceiling))

        # Cruise airspeeds
        for _ in range(n_legs):
            bounds.append((self.uav.fw_min_airspeed,
                           self.uav.fw_max_airspeed))

        # VTOL rates
        bounds.append((0.5, self.uav.vtol_max_climb_rate))
        bounds.append((0.5, self.uav.vtol_max_descent_rate))

        # Along-track positions, kept clear of the endpoints so the
        # departure and arrival phases always have room.
        if self.free_along_track:
            lo = max(self.min_leg_distance_m / self._direct_distance, 0.05)
            for _ in range(N):
                bounds.append((lo, 1.0 - lo))

        return bounds

    @property
    def parameter_names(self) -> List[str]:
        """Human-readable names for each design variable."""
        N = self.n_intermediate
        n_legs = N + 1
        names = []

        vert = "cruise_agl" if self.corridor_mode == "agl" else "cruise_altitude"
        for i in range(N):
            names.append(f"lateral_offset_{i}")
        for i in range(n_legs):
            names.append(f"{vert}_{i}")
        for i in range(n_legs):
            names.append(f"cruise_airspeed_{i}")
        names.append("vtol_ascend_rate")
        names.append("vtol_descend_rate")
        if self.free_along_track:
            for i in range(N):
                names.append(f"along_fraction_{i}")

        return names

    @property
    def n_parameters(self) -> int:
        """
        Number of design variables.

        ``3N + 4`` with lateral offsets only, ``4N + 4`` when the
        waypoints may also slide along the route.
        """
        n = 3 * self.n_intermediate + 4
        if self.free_along_track:
            n += self.n_intermediate
        return n

    @property
    def along_fractions(self) -> np.ndarray:
        """Where each waypoint sits along the direct line, 0 to 1."""
        return np.sort(np.clip(self._wp_fractions, 0.02, 0.98))

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
    def cruise_agls(self) -> np.ndarray:
        """Target height above ground for each sub-leg [m AGL]."""
        return self._cruise_agls.copy()

    @property
    def cruise_altitudes(self) -> np.ndarray:
        """
        Representative cruise altitude of each sub-leg [m AMSL].

        In terrain-following mode the cruise altitude is not a design
        variable — it varies along the leg — so this reports the mean of
        the flown reference profile, for plotting and comparison with
        constant-altitude runs.
        """
        if self.corridor_mode == "agl":
            if self._dirty or self._flight_path is None:
                self.flight_path  # force a rebuild so profiles exist
            if self.leg_profiles:
                return np.array([float(np.mean(p.sample_altitudes))
                                 for p in self.leg_profiles])
        return self._cruise_altitudes.copy()

    @property
    def corridor_ceiling_excess_m(self) -> float:
        """
        Worst exceedance of the AGL ceiling the relief forces [m].

        Non-zero means the terrain along this route falls away faster than
        the aircraft can descend, so no legal profile exists on this line.
        """
        if self.corridor_mode != "agl":
            return float('nan')
        if self._dirty or self._flight_path is None:
            self.flight_path
        if not self.leg_profiles:
            return 0.0
        return max(p.ceiling_excess_m for p in self.leg_profiles)

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
                float(np.min(self.cruise_altitudes)),
                float(np.max(self.cruise_altitudes)),
            ),
            'corridor_mode': self.corridor_mode,
            'cruise_agl_m': (
                [float(x) for x in self._cruise_agls]
                if self.corridor_mode == "agl" else None
            ),
            'corridor_ceiling_excess_m': self.corridor_ceiling_excess_m,
            'is_direct': self.max_lateral_deviation < 50.0,  # ~50m threshold
        }

    def __repr__(self) -> str:
        return (f"RoutedPath({self.origin.name} → {self.destination.name}, "
                f"N_wp={self.n_intermediate}, "
                f"n_params={self.n_parameters}, "
                f"max_offset={self.max_lateral_deviation:.0f}m)")
