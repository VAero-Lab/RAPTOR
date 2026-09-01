"""
UAV Configuration — Performance Envelope & Mission Constraints
===============================================================

Defines the physical and operational limits of the eVTOL aircraft.
These constraints bound the feasible parameter space for path planning.

Typical eVTOL for medical delivery at high altitude (Quito, ~2800 m AMSL):
- Reduced air density → higher power required, lower max payload
- Reduced rotor efficiency in VTOL mode
- Reduced fixed-wing L/D at lower Reynolds numbers
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import warnings

import numpy as np


@dataclass
class UAVConfig:
    """
    eVTOL aircraft performance parameters.

    All speeds in m/s, altitudes in m (AMSL), angles in degrees,
    rates in m/s, distances in m.

    Attributes
    ----------
    name : str
        Aircraft identifier.

    VTOL Performance
    ----------------
    vtol_max_climb_rate : float
        Maximum vertical climb rate in VTOL/hover mode [m/s].
    vtol_max_descent_rate : float
        Maximum vertical descent rate in VTOL/hover mode [m/s].
    vtol_hover_ceiling : float
        Maximum altitude for sustained hover [m AMSL].
    vtol_transition_duration : float
        Time to transition between VTOL and FW modes [s].
    vtol_transition_distance : float
        Horizontal distance covered during transition [m].
    vtol_transition_alt_change : float
        Altitude gained/lost during transition [m].

    Fixed-Wing Performance
    ----------------------
    fw_min_airspeed : float
        Stall speed in fixed-wing mode [m/s].
    fw_cruise_airspeed : float
        Design cruise airspeed [m/s].
    fw_max_airspeed : float
        Maximum (never exceed) airspeed [m/s].
    fw_max_climb_angle : float
        Maximum sustainable climb angle [deg].
    fw_max_descent_angle : float
        Maximum descent angle (positive value) [deg].
    fw_max_climb_rate : float
        Maximum rate of climb in FW mode [m/s].
    fw_max_descent_rate : float
        Maximum rate of descent in FW mode [m/s].
    fw_service_ceiling : float
        Maximum operating altitude [m AMSL].

    General
    -------
    max_bank_angle : float
        Maximum bank angle for turns [deg].
    min_turn_radius : float
        Minimum turn radius at cruise speed [m].
    """

    name: str = "eVTOL-MedDelivery"

    # --- VTOL mode ---
    vtol_max_climb_rate: float = 3.0       # m/s
    # Held below the vortex-ring threshold, which is 0.25x the hover
    # induced velocity — about 2.2 m/s on the baseline airframe. Note the
    # direction of that constraint: *larger* rotors have a lower induced
    # velocity and therefore a lower safe descent rate, so correcting the
    # disk loading upward in efficiency tightened this limit rather than
    # relaxing it.
    vtol_max_descent_rate: float = 2.0     # m/s (positive = downward)
    vtol_hover_ceiling: float = 5500.0     # m AMSL
    vtol_transition_duration: float = 15.0 # s
    vtol_transition_distance: float = 150.0  # m horizontal
    vtol_transition_alt_change: float = 20.0 # m gained during transition
    fw_min_transition_cruise_m: float = 50.0 # m horizontal straight flight for stabilization

    # --- Fixed-wing mode ---
    # These used to be typed in independently of the aerodynamics, and
    # drifted: with the wing's real C_L,max the old 15 m/s minimum was
    # *below stall* at Quito altitude, and the 25 m/s cruise sat under the
    # 1.3x margin. `UAVConfig.for_vehicle()` derives them instead.
    fw_min_airspeed: float = 26.1          # m/s (~94 km/h), 1.3x stall at 2800 m
    fw_cruise_airspeed: float = 30.0       # m/s (~108 km/h)
    fw_max_airspeed: float = 42.0          # m/s (~151 km/h)
    fw_max_climb_angle: float = 15.0       # deg
    fw_max_descent_angle: float = 12.0     # deg
    fw_max_climb_rate: float = 5.0         # m/s
    fw_max_descent_rate: float = 4.0       # m/s
    fw_service_ceiling: float = 6000.0     # m AMSL

    # --- General ---
    max_bank_angle: float = 30.0           # deg
    min_turn_radius: float = 100.0         # m

    # --- Aerodynamic safety ---
    stall_safety_margin: float = 1.3       # V_min = margin × V_stall (FAR 23 standard)

    @classmethod
    def for_vehicle(cls, ac, altitude: float = 2800.0,
                    cruise_margin: float = 1.5,
                    max_margin: float = 2.1, **overrides) -> "UAVConfig":
        """
        Build a speed envelope from an aircraft's actual aerodynamics.

        Stall speed depends on wing area, weight, air density and the
        wing's real maximum lift — none of which a hand-typed minimum
        airspeed knows about. Deriving the envelope means it cannot
        silently contradict the aerodynamic model, and it re-derives
        correctly when the wing, the mass or the operating altitude
        changes.

        Parameters
        ----------
        ac : AircraftEnergyParams
            The vehicle. If a computed polar is attached its ``C_L,max``
            is used; otherwise the declared one, which is usually
            optimistic.
        altitude : float
            Altitude the envelope must be valid at [m AMSL]. Use the
            highest terrain the aircraft will cross, not sea level:
            stall speed rises as density falls, so an envelope sized at
            sea level is unsafe in the Andes.
        cruise_margin, max_margin : float
            Multiples of stall speed for cruise and never-exceed. The
            minimum airspeed is set at the UAV's own
            ``stall_safety_margin`` (1.3, the FAR 23 convention).

        Returns
        -------
        UAVConfig whose speeds are consistent with the aerodynamics.
        """
        margin = overrides.pop("stall_safety_margin",
                               cls.__dataclass_fields__[
                                   "stall_safety_margin"].default)
        v_stall = ac.stall_speed_at(altitude)

        # Every limit below is derived from the aircraft's drag, so an
        # aircraft with only half an aerodynamic model produces a
        # complete and wrong envelope rather than an error.
        if not getattr(ac, "aero_is_complete", True):
            warnings.warn(
                f"{ac.name or 'vehicle'}: deriving a flight envelope from "
                f"an incomplete aerodynamic model ({ac.aero_source}). The "
                f"wing alone has roughly twice the real lift-to-drag "
                f"ratio, and the climb angle derived from it will be far "
                f"too shallow. Call build_aero() first.",
                stacklevel=2,
            )

        # Round *up*, never to nearest. Rounding 26.03 down to 26.0 leaves
        # an envelope that fails the very margin it was built to satisfy —
        # a small thing, but a derived envelope that does not pass its own
        # check is worse than no derivation at all.
        def up(x: float) -> float:
            return float(np.ceil(x * 10.0) / 10.0)

        v_min = up(margin * v_stall)
        v_cruise = up(cruise_margin * v_stall)
        v_max = up(max_margin * v_stall)

        # ── Manoeuvre limits, derived rather than typed ────────────────
        # These used to be constants sitting beside the speeds, and they
        # contradicted them. A 100 m minimum turn radius is not
        # achievable at 30 m/s with a 30 degree bank — the geometry gives
        # 159 m — and a 15 degree climb angle at 30 m/s is 7.8 m/s of
        # climb, not the 5.0 m/s written next to it. Deriving each from
        # the quantity it actually depends on removes the possibility.
        bank = overrides.pop(
            "max_bank_angle",
            cls.__dataclass_fields__["max_bank_angle"].default)

        # Level turn: the horizontal component of lift supplies the
        # centripetal force, so R = V^2 / (g * tan(phi)). Evaluated at
        # cruise speed because that is the speed a route is flown at;
        # the aircraft can turn tighter if it slows down, and a planner
        # that assumed otherwise would cut corners it cannot fly.
        turn_radius = float(np.ceil(
            v_cruise ** 2 / (9.80665 * np.tan(np.radians(bank)))))

        # ── VTOL descent, held clear of the vortex-ring band ──────────
        # A descending rotor re-ingests its own wake between roughly 0.25
        # and 1.5 times the hover induced velocity, and inside that band
        # thrust can collapse. Where the band starts depends on disk
        # loading, so it moves with the aircraft: a constant 2.0 m/s was
        # outside it for a 12 kg machine on 0.82 m^2 of disk and inside
        # it for a 10 kg machine on 0.98 m^2 — the safer aircraft, by the
        # wrong number.
        from .energy import vortex_ring_margin
        vrs = vortex_ring_margin(ac, 0.0, altitude)
        vtol_descent = overrides.pop(
            "vtol_max_descent_rate",
            round(float(np.floor(0.8 * vrs["vrs_lower"] * 10.0) / 10.0), 1))
        vtol_descent = max(vtol_descent, 0.3)

        climb_angle = overrides.pop("fw_max_climb_angle", None)
        if climb_angle is None:
            climb_angle = round(ac.max_climb_angle_at(altitude, v_cruise), 1)
        descent_angle = overrides.pop("fw_max_descent_angle", None)
        if descent_angle is None:
            descent_angle = round(ac.max_descent_angle_at(altitude, v_max), 1)

        # Rate follows from angle and speed; it is not independent.
        climb_rate = overrides.pop(
            "fw_max_climb_rate",
            round(v_cruise * np.sin(np.radians(climb_angle)), 2))
        descent_rate = overrides.pop(
            "fw_max_descent_rate",
            round(v_max * np.sin(np.radians(descent_angle)), 2))

        return cls(
            fw_min_airspeed=v_min,
            fw_cruise_airspeed=v_cruise,
            fw_max_airspeed=v_max,
            fw_max_climb_angle=float(climb_angle),
            fw_max_descent_angle=float(descent_angle),
            fw_max_climb_rate=float(climb_rate),
            fw_max_descent_rate=float(descent_rate),
            max_bank_angle=float(bank),
            min_turn_radius=turn_radius,
            vtol_max_descent_rate=float(vtol_descent),
            stall_safety_margin=margin,
            **overrides,
        )

    def manoeuvre_problems(self) -> list:
        """
        Manoeuvre limits that contradict each other.

        Turn radius, bank angle and cruise speed are three names for two
        independent facts; so are climb angle, climb rate and speed. Set
        by hand they drift apart, and nothing downstream notices, because
        each is used in a different place.
        """
        out = []
        implied_r = (self.fw_cruise_airspeed ** 2
                     / (9.80665 * np.tan(np.radians(self.max_bank_angle))))
        if self.min_turn_radius < implied_r * 0.98:
            out.append(
                f"min_turn_radius {self.min_turn_radius:.0f} m is tighter "
                f"than {self.max_bank_angle:.0f} deg of bank allows at "
                f"{self.fw_cruise_airspeed:.1f} m/s, which needs "
                f"{implied_r:.0f} m. The planner will lay out turns the "
                f"aircraft cannot fly."
            )
        implied_climb = (self.fw_cruise_airspeed
                         * np.sin(np.radians(self.fw_max_climb_angle)))
        if self.fw_max_climb_rate < implied_climb * 0.95:
            out.append(
                f"fw_max_climb_rate {self.fw_max_climb_rate:.1f} m/s "
                f"contradicts fw_max_climb_angle "
                f"{self.fw_max_climb_angle:.1f} deg at "
                f"{self.fw_cruise_airspeed:.1f} m/s, which is "
                f"{implied_climb:.1f} m/s. The corridor envelope uses the "
                f"angle and the segment builder uses the rate, so the two "
                f"describe different aircraft."
            )
        implied_desc = (self.fw_max_airspeed
                        * np.sin(np.radians(self.fw_max_descent_angle)))
        if self.fw_max_descent_rate < implied_desc * 0.95:
            out.append(
                f"fw_max_descent_rate {self.fw_max_descent_rate:.1f} m/s "
                f"contradicts fw_max_descent_angle "
                f"{self.fw_max_descent_angle:.1f} deg at "
                f"{self.fw_max_airspeed:.1f} m/s ({implied_desc:.1f} m/s)."
            )
        if self.fw_min_airspeed >= self.fw_cruise_airspeed:
            out.append("fw_min_airspeed is at or above cruise.")
        if self.fw_cruise_airspeed >= self.fw_max_airspeed:
            out.append("fw_cruise_airspeed is at or above maximum.")
        return out

    def envelope_summary(self, ac=None, altitude: float = 2800.0) -> str:
        """The speed band, and how much of it is usable."""
        lines = [
            f"Speed envelope at {altitude:.0f} m",
            f"  minimum   {self.fw_min_airspeed:5.1f} m/s",
            f"  cruise    {self.fw_cruise_airspeed:5.1f} m/s",
            f"  maximum   {self.fw_max_airspeed:5.1f} m/s",
        ]
        if ac is not None:
            v_s = ac.stall_speed_at(altitude)
            lines += [
                f"  stall     {v_s:5.1f} m/s  "
                f"(C_L,max {ac.C_L_max_at(self.fw_cruise_airspeed, altitude):.3f})",
                f"  margin at minimum: {self.fw_min_airspeed / v_s:.2f}x stall",
                f"  margin at cruise:  {self.fw_cruise_airspeed / v_s:.2f}x stall",
            ]
        return "\n".join(lines)

    def validate_vtol_climb(self, climb_rate: float) -> bool:
        """Check if a VTOL climb rate is within limits."""
        return 0 < climb_rate <= self.vtol_max_climb_rate

    def validate_vtol_descent(self, descent_rate: float) -> bool:
        """Check if a VTOL descent rate is within limits."""
        return 0 < descent_rate <= self.vtol_max_descent_rate

    def validate_fw_airspeed(self, airspeed: float) -> bool:
        """Check if an airspeed is within the FW flight envelope."""
        return self.fw_min_airspeed <= airspeed <= self.fw_max_airspeed

    def validate_fw_climb_angle(self, angle_deg: float) -> bool:
        """Check if a FW climb angle is achievable."""
        return 0 < angle_deg <= self.fw_max_climb_angle

    def validate_fw_descent_angle(self, angle_deg: float) -> bool:
        """Check if a FW descent angle is within limits."""
        return 0 < angle_deg <= self.fw_max_descent_angle

    def validate_altitude(self, altitude: float) -> bool:
        """Check if altitude is within service ceiling."""
        return altitude <= self.fw_service_ceiling

    def fw_climb_rate(self, airspeed: float, climb_angle_deg: float) -> float:
        """Compute vertical speed from airspeed and climb angle."""
        return airspeed * np.sin(np.radians(climb_angle_deg))

    def fw_ground_speed(self, airspeed: float, climb_angle_deg: float) -> float:
        """Compute ground speed from airspeed and flight path angle."""
        return airspeed * np.cos(np.radians(climb_angle_deg))


@dataclass
class MissionConstraints:
    """
    Mission-level constraints for path planning.

    The vertical corridor
    ---------------------
    Terrain clearance and the regulatory ceiling are two walls of the
    same corridor, and they must be set together. RDAC 101.185 caps
    flight at 122 m AGL; if the cruise clearance floor is set to 100 m,
    the corridor is 22 m tall, and no route over terrain with more than
    22 m of relief between two points can stay inside it. Over Quito,
    where corridors between hospitals cross 180–470 m of relief, that
    makes every plan infeasible for reasons that have nothing to do with
    the route chosen.

    ``validate()`` checks the corridor is wide enough to fly in, and
    ``corridor()`` reports the band for a given segment type. Call
    ``validate()`` before optimizing — a silently empty corridor is the
    single easiest way to make this package produce nonsense.

    Attributes
    ----------
    clearance_tolerance_m : float
        Numerical slack on the clearance floor and ceiling [m].
    min_terrain_clearance : float
        Minimum height above ground level (AGL) at all points [m].
        RDAC 101.160(a)(2) requires 30 m from any building.
    min_cruise_terrain_clearance : float
        Minimum AGL during cruise segments [m].
    max_agl : float
        Regulatory ceiling [m AGL]. RDAC 101.185: 122 m (400 ft).
        Kept here as well as in the airspace zones so that the path
        *builder* can respect it, not merely be judged against it.
    terminal_exempt_radius_m : float
        Horizontal radius around the origin and destination within which
        the clearance floor does not apply [m]. Take-off and landing end
        at AGL 0 by definition; without this exemption every path ever
        built is reported infeasible at its own pads.
    max_flight_time : float
        Maximum total flight time [s].
    max_range : float
        Maximum total ground distance [m].
    vtol_clearance_radius : float
        Required obstacle-free radius around takeoff/landing [m].
    emergency_landing_alt_margin : float
        Extra altitude margin above terrain for emergency scenarios [m].
    max_path_segments : int
        Soft ceiling on the number of flight segments in a path. This is
        a cost bound, not a physical one: every segment is a power
        evaluation, and the optimizer runs thousands of paths. It was 20
        when a path was a climb, a cruise and a descent; a
        terrain-following leg over 30 m terrain is forty. Reported by
        :func:`raptor.terrain.check_path_envelope`, not enforced —
        exceeding it makes the search slow, not wrong.
        Maximum number of segments in a path (bounds complexity).
    """

    min_terrain_clearance: float = 30.0          # m AGL — RDAC 101.160(a)(2)
    min_cruise_terrain_clearance: float = 60.0   # m AGL — during cruise
    max_agl: float = 122.0                       # m AGL — RDAC 101.185
    terminal_exempt_radius_m: float = 400.0      # m — take-off/landing funnel
    #: Slack applied to the clearance floor and the AGL ceiling [m].
    #: Segment endpoints are computed by accumulating great-circle steps,
    #: so a path that lands exactly on its pad can read as 1e-9 m below
    #: it. Without slack that rounding noise flips a feasibility verdict,
    #: which is both wrong and impossible to debug from the outside. The
    #: default is far below DEM vertical error, so it cannot mask a real
    #: violation.
    clearance_tolerance_m: float = 0.5
    max_flight_time: float = 3600.0              # s (1 hour)
    max_range: float = 60000.0                   # m (60 km)
    vtol_clearance_radius: float = 30.0          # m
    emergency_landing_alt_margin: float = 30.0   # m
    max_path_segments: int = 80
    max_maneuvers_per_km: float = 0.3            # maneuvers per km of leg distance

    def terrain_clearance_for_segment(self, segment_type: str) -> float:
        """Return the applicable terrain clearance for a segment type."""
        if segment_type in ("FW_CRUISE",):
            return self.min_cruise_terrain_clearance
        return self.min_terrain_clearance

    def corridor(self, segment_type: str = "FW_CRUISE") -> Tuple[float, float]:
        """
        The usable AGL band ``(floor, ceiling)`` for a segment type [m].

        A non-positive width means no path can be flown legally, whatever
        the route.
        """
        return (self.terrain_clearance_for_segment(segment_type), self.max_agl)

    def corridor_width(self, segment_type: str = "FW_CRUISE") -> float:
        """Height of the usable AGL band [m]. Negative means impossible."""
        floor, ceiling = self.corridor(segment_type)
        return ceiling - floor

    def validate(self, min_width_m: float = 30.0) -> List[str]:
        """
        Check the constraint set is self-consistent and flyable.

        Parameters
        ----------
        min_width_m : float
            Corridor height below which terrain-following is considered
            impractical, given DEM error and altitude-hold accuracy.

        Returns
        -------
        List of human-readable problems. Empty means the set is usable.
        """
        problems: List[str] = []

        for seg in ("FW_CRUISE", "FW_CLIMB"):
            w = self.corridor_width(seg)
            if w <= 0:
                problems.append(
                    f"{seg}: clearance floor "
                    f"{self.terrain_clearance_for_segment(seg):.0f} m AGL is at "
                    f"or above the {self.max_agl:.0f} m AGL regulatory ceiling — "
                    f"the corridor is empty, so no path can ever be feasible."
                )
            elif w < min_width_m:
                problems.append(
                    f"{seg}: corridor is only {w:.0f} m tall "
                    f"({self.terrain_clearance_for_segment(seg):.0f}–"
                    f"{self.max_agl:.0f} m AGL). Terrain-following needs room "
                    f"for DEM error and altitude-hold tolerance; expect "
                    f"frequent infeasibility."
                )

        if self.min_cruise_terrain_clearance < self.min_terrain_clearance:
            problems.append(
                "min_cruise_terrain_clearance is below min_terrain_clearance; "
                "cruise should not be the least-cleared phase of flight."
            )

        if self.terminal_exempt_radius_m <= 0:
            problems.append(
                "terminal_exempt_radius_m is 0: take-off and landing waypoints "
                "sit at AGL 0 and will be reported as clearance violations."
            )

        return problems

    def describe_corridor(self) -> str:
        """One-line description of the vertical corridor, for logs."""
        f, c = self.corridor("FW_CRUISE")
        return (f"Cruise corridor: {f:.0f}–{c:.0f} m AGL "
                f"({c - f:.0f} m tall); manoeuvre floor "
                f"{self.min_terrain_clearance:.0f} m AGL")
