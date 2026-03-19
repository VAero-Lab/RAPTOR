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
from typing import Optional
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
    vtol_max_descent_rate: float = 2.5     # m/s (positive = downward)
    vtol_hover_ceiling: float = 5500.0     # m AMSL
    vtol_transition_duration: float = 15.0 # s
    vtol_transition_distance: float = 150.0  # m horizontal
    vtol_transition_alt_change: float = 20.0 # m gained during transition

    # --- Fixed-wing mode ---
    fw_min_airspeed: float = 15.0          # m/s (~54 km/h)
    fw_cruise_airspeed: float = 25.0       # m/s (~90 km/h)
    fw_max_airspeed: float = 35.0          # m/s (~126 km/h)
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

    Attributes
    ----------
    min_terrain_clearance : float
        Minimum height above ground level (AGL) at all points [m].
    min_cruise_terrain_clearance : float
        Minimum AGL during cruise segments [m].
    max_flight_time : float
        Maximum total flight time [s].
    max_range : float
        Maximum total ground distance [m].
    vtol_clearance_radius : float
        Required obstacle-free radius around takeoff/landing [m].
    emergency_landing_alt_margin : float
        Extra altitude margin above terrain for emergency scenarios [m].
    max_path_segments : int
        Maximum number of segments in a path (bounds complexity).
    """

    min_terrain_clearance: float = 50.0         # m AGL — absolute minimum
    min_cruise_terrain_clearance: float = 100.0  # m AGL — during cruise
    max_flight_time: float = 3600.0              # s (1 hour)
    max_range: float = 60000.0                   # m (60 km)
    vtol_clearance_radius: float = 30.0          # m
    emergency_landing_alt_margin: float = 30.0   # m
    max_path_segments: int = 20

    def terrain_clearance_for_segment(self, segment_type: str) -> float:
        """Return the applicable terrain clearance for a segment type."""
        if segment_type in ("FW_CRUISE",):
            return self.min_cruise_terrain_clearance
        return self.min_terrain_clearance
