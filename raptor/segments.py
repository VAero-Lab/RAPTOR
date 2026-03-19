"""
Flight Segments — Parameterized Building Blocks for Flight Paths
=================================================================

Each segment type represents a distinct flight phase with its own
physics, constraints, and free parameters for optimization.

Segment taxonomy:
    VTOL_ASCEND   — Vertical takeoff / hover climb (no horizontal displacement)
    VTOL_DESCEND  — Vertical descent / hover landing (no horizontal displacement)
    FW_CLIMB      — Fixed-wing climb at a flight-path angle γ > 0
    FW_DESCEND    — Fixed-wing descent at a flight-path angle γ < 0
    FW_CRUISE     — Level fixed-wing flight at constant altitude
    TRANSITION    — Mode change (VTOL ↔ FW) with partial climb/distance

Each segment stores:
    - Its type-specific parameters (the optimization variables)
    - Derived kinematics (computed from parameters)
    - Connection points (start/end in 3D)

The key design principle: given a start state (lat, lon, alt, bearing),
a segment's parameters fully determine the end state. This enables
chaining segments into complete paths.
"""

from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List
import numpy as np


class SegmentType(Enum):
    """Flight segment types."""
    VTOL_ASCEND = "VTOL_ASCEND"
    VTOL_DESCEND = "VTOL_DESCEND"
    FW_CLIMB = "FW_CLIMB"
    FW_DESCEND = "FW_DESCEND"
    FW_CRUISE = "FW_CRUISE"
    TRANSITION = "TRANSITION"


@dataclass
class SegmentState:
    """
    State at a point along the flight path.

    This is what connects one segment to the next.
    """
    lat: float           # degrees
    lon: float           # degrees
    alt: float           # m AMSL
    bearing: float       # degrees (0=N, 90=E) — direction of travel
    time: float = 0.0    # cumulative mission time [s]
    distance: float = 0.0  # cumulative ground distance [m]


@dataclass
class SegmentKinematics:
    """Derived kinematic quantities for a segment."""
    duration: float          # s
    ground_distance: float   # m (horizontal)
    altitude_change: float   # m (positive = climb)
    vertical_speed: float    # m/s (positive = climb)
    ground_speed: float      # m/s (horizontal)
    airspeed: float          # m/s (along flight path)
    flight_path_angle: float # deg (positive = climb)


class FlightSegment:
    """
    Base class for all flight segments.

    A segment is defined by:
        1. Its type (SegmentType enum)
        2. A set of free parameters (the optimization variables)
        3. A start state (set when the segment is placed in a path)

    From (1) + (2) + (3), all kinematics and the end state are computed.

    Subclasses must implement:
        - _compute_kinematics() → SegmentKinematics
        - _compute_end_state() → SegmentState
        - parameter_vector → np.ndarray  (for optimizer interface)
        - parameter_names → list of str
        - parameter_bounds → list of (min, max)
    """

    def __init__(self, segment_type: SegmentType):
        self.segment_type = segment_type
        self._start_state: Optional[SegmentState] = None
        self._kinematics: Optional[SegmentKinematics] = None
        self._end_state: Optional[SegmentState] = None
        self._waypoints: Optional[np.ndarray] = None

    @property
    def start_state(self) -> Optional[SegmentState]:
        return self._start_state

    @start_state.setter
    def start_state(self, state: SegmentState):
        self._start_state = state
        # Recompute derived quantities when start changes
        self._kinematics = self._compute_kinematics()
        self._end_state = self._compute_end_state()
        self._waypoints = self._compute_waypoints()

    @property
    def kinematics(self) -> SegmentKinematics:
        if self._kinematics is None:
            self._kinematics = self._compute_kinematics()
        return self._kinematics

    @property
    def end_state(self) -> SegmentState:
        if self._end_state is None:
            if self._start_state is None:
                raise ValueError("Start state must be set before accessing end state.")
            self._end_state = self._compute_end_state()
        return self._end_state

    @property
    def waypoints(self) -> np.ndarray:
        """
        Dense waypoints along the segment for terrain checking.

        Returns ndarray of shape (N, 5): [lat, lon, alt, time, distance]
        """
        if self._waypoints is None:
            self._waypoints = self._compute_waypoints()
        return self._waypoints

    def _compute_kinematics(self) -> SegmentKinematics:
        raise NotImplementedError

    def _compute_end_state(self) -> SegmentState:
        raise NotImplementedError

    def _compute_waypoints(self, n: int = 20) -> np.ndarray:
        """Default waypoint generation: linear interpolation."""
        if self._start_state is None:
            raise ValueError("Start state not set.")

        s = self._start_state
        e = self._compute_end_state()
        k = self._compute_kinematics()

        t = np.linspace(0, 1, n)
        lats = s.lat + t * (e.lat - s.lat)
        lons = s.lon + t * (e.lon - s.lon)
        alts = s.alt + t * k.altitude_change
        times = s.time + t * k.duration
        dists = s.distance + t * k.ground_distance

        return np.column_stack([lats, lons, alts, times, dists])

    @property
    def parameter_vector(self) -> np.ndarray:
        """Current parameter values as a flat array (for optimizer)."""
        raise NotImplementedError

    @parameter_vector.setter
    def parameter_vector(self, values: np.ndarray):
        """Set parameters from a flat array (from optimizer)."""
        raise NotImplementedError

    @property
    def parameter_names(self) -> List[str]:
        raise NotImplementedError

    @property
    def parameter_bounds(self) -> List[Tuple[float, float]]:
        raise NotImplementedError

    @property
    def n_parameters(self) -> int:
        return len(self.parameter_names)

    def __repr__(self):
        k = self.kinematics
        return (f"{self.segment_type.value}("
                f"Δh={k.altitude_change:+.0f}m, "
                f"d={k.ground_distance:.0f}m, "
                f"t={k.duration:.1f}s)")


# ═════════════════════════════════════════════════════════════════════════════
# CONCRETE SEGMENT TYPES
# ═════════════════════════════════════════════════════════════════════════════


class VTOLAscend(FlightSegment):
    """
    Vertical takeoff / hover climb.

    No horizontal displacement — pure vertical ascent.

    Free parameters:
        altitude_gain [m]     — how high to climb
        climb_rate    [m/s]   — vertical speed
    """

    def __init__(self, altitude_gain: float = 50.0, climb_rate: float = 2.0):
        super().__init__(SegmentType.VTOL_ASCEND)
        self.altitude_gain = altitude_gain
        self.climb_rate = climb_rate

    def _compute_kinematics(self) -> SegmentKinematics:
        duration = self.altitude_gain / self.climb_rate
        return SegmentKinematics(
            duration=duration,
            ground_distance=0.0,
            altitude_change=self.altitude_gain,
            vertical_speed=self.climb_rate,
            ground_speed=0.0,
            airspeed=self.climb_rate,  # in hover, airspeed ≈ climb rate
            flight_path_angle=90.0,
        )

    def _compute_end_state(self) -> SegmentState:
        s = self._start_state
        k = self._compute_kinematics()
        return SegmentState(
            lat=s.lat,
            lon=s.lon,
            alt=s.alt + k.altitude_change,
            bearing=s.bearing,
            time=s.time + k.duration,
            distance=s.distance,
        )

    def _compute_waypoints(self, n: int = 15) -> np.ndarray:
        s = self._start_state
        k = self._compute_kinematics()
        t = np.linspace(0, 1, n)
        return np.column_stack([
            np.full(n, s.lat),
            np.full(n, s.lon),
            s.alt + t * k.altitude_change,
            s.time + t * k.duration,
            np.full(n, s.distance),
        ])

    @property
    def parameter_vector(self) -> np.ndarray:
        return np.array([self.altitude_gain, self.climb_rate])

    @parameter_vector.setter
    def parameter_vector(self, values: np.ndarray):
        self.altitude_gain, self.climb_rate = values
        self._kinematics = None
        self._end_state = None
        self._waypoints = None

    @property
    def parameter_names(self) -> List[str]:
        return ["altitude_gain", "climb_rate"]

    @property
    def parameter_bounds(self) -> List[Tuple[float, float]]:
        return [(5.0, 500.0), (0.5, 5.0)]


class VTOLDescend(FlightSegment):
    """
    Vertical descent / hover landing.

    No horizontal displacement — pure vertical descent.

    Free parameters:
        altitude_loss  [m]    — how far to descend (positive value)
        descent_rate   [m/s]  — vertical descent speed (positive value)
    """

    def __init__(self, altitude_loss: float = 50.0, descent_rate: float = 2.0):
        super().__init__(SegmentType.VTOL_DESCEND)
        self.altitude_loss = altitude_loss
        self.descent_rate = descent_rate

    def _compute_kinematics(self) -> SegmentKinematics:
        duration = self.altitude_loss / self.descent_rate
        return SegmentKinematics(
            duration=duration,
            ground_distance=0.0,
            altitude_change=-self.altitude_loss,
            vertical_speed=-self.descent_rate,
            ground_speed=0.0,
            airspeed=self.descent_rate,
            flight_path_angle=-90.0,
        )

    def _compute_end_state(self) -> SegmentState:
        s = self._start_state
        k = self._compute_kinematics()
        return SegmentState(
            lat=s.lat,
            lon=s.lon,
            alt=s.alt + k.altitude_change,
            bearing=s.bearing,
            time=s.time + k.duration,
            distance=s.distance,
        )

    def _compute_waypoints(self, n: int = 15) -> np.ndarray:
        s = self._start_state
        k = self._compute_kinematics()
        t = np.linspace(0, 1, n)
        return np.column_stack([
            np.full(n, s.lat),
            np.full(n, s.lon),
            s.alt + t * k.altitude_change,
            s.time + t * k.duration,
            np.full(n, s.distance),
        ])

    @property
    def parameter_vector(self) -> np.ndarray:
        return np.array([self.altitude_loss, self.descent_rate])

    @parameter_vector.setter
    def parameter_vector(self, values: np.ndarray):
        self.altitude_loss, self.descent_rate = values
        self._kinematics = None
        self._end_state = None
        self._waypoints = None

    @property
    def parameter_names(self) -> List[str]:
        return ["altitude_loss", "descent_rate"]

    @property
    def parameter_bounds(self) -> List[Tuple[float, float]]:
        return [(5.0, 500.0), (0.5, 4.0)]


class Transition(FlightSegment):
    """
    Mode transition segment (VTOL → FW or FW → VTOL).

    Covers a fixed duration/distance during which the aircraft
    transitions between flight modes. Slight altitude change is possible.

    Free parameters:
        duration       [s]   — transition duration
        altitude_change [m]  — net altitude gained (+) or lost (-)
        ground_distance [m]  — horizontal distance covered
    """

    def __init__(self, duration: float = 15.0,
                 altitude_change: float = 20.0,
                 ground_distance: float = 150.0):
        super().__init__(SegmentType.TRANSITION)
        self.duration = duration
        self.altitude_change_param = altitude_change
        self.ground_distance_param = ground_distance

    def _compute_kinematics(self) -> SegmentKinematics:
        gs = self.ground_distance_param / self.duration if self.duration > 0 else 0
        vs = self.altitude_change_param / self.duration if self.duration > 0 else 0
        airspeed = np.sqrt(gs**2 + vs**2)
        fpa = np.degrees(np.arctan2(vs, gs)) if gs > 0 else (90.0 if vs > 0 else -90.0)
        return SegmentKinematics(
            duration=self.duration,
            ground_distance=self.ground_distance_param,
            altitude_change=self.altitude_change_param,
            vertical_speed=vs,
            ground_speed=gs,
            airspeed=airspeed,
            flight_path_angle=fpa,
        )

    def _compute_end_state(self) -> SegmentState:
        s = self._start_state
        k = self._compute_kinematics()
        from .dem import DEMInterface
        new_lat, new_lon = DEMInterface.destination_point(
            s.lat, s.lon, s.bearing, k.ground_distance
        )
        return SegmentState(
            lat=new_lat, lon=new_lon,
            alt=s.alt + k.altitude_change,
            bearing=s.bearing,
            time=s.time + k.duration,
            distance=s.distance + k.ground_distance,
        )

    @property
    def parameter_vector(self) -> np.ndarray:
        return np.array([self.duration, self.altitude_change_param,
                         self.ground_distance_param])

    @parameter_vector.setter
    def parameter_vector(self, values: np.ndarray):
        self.duration = values[0]
        self.altitude_change_param = values[1]
        self.ground_distance_param = values[2]
        self._kinematics = None
        self._end_state = None
        self._waypoints = None

    @property
    def parameter_names(self) -> List[str]:
        return ["duration", "altitude_change", "ground_distance"]

    @property
    def parameter_bounds(self) -> List[Tuple[float, float]]:
        return [(5.0, 30.0), (-10.0, 50.0), (50.0, 400.0)]


class FWClimb(FlightSegment):
    """
    Fixed-wing climb segment.

    The aircraft climbs at a constant flight-path angle γ and airspeed V.

    Free parameters:
        altitude_gain    [m]    — total altitude to gain
        climb_angle_deg  [deg]  — flight-path angle (0 < γ ≤ max)
        airspeed         [m/s]  — true airspeed along flight path
    """

    def __init__(self, altitude_gain: float = 200.0,
                 climb_angle_deg: float = 8.0,
                 airspeed: float = 22.0):
        super().__init__(SegmentType.FW_CLIMB)
        self.altitude_gain = altitude_gain
        self.climb_angle_deg = climb_angle_deg
        self.airspeed = airspeed

    def _compute_kinematics(self) -> SegmentKinematics:
        gamma_rad = np.radians(self.climb_angle_deg)
        ground_dist = self.altitude_gain / np.tan(gamma_rad)
        gs = self.airspeed * np.cos(gamma_rad)
        vs = self.airspeed * np.sin(gamma_rad)
        duration = self.altitude_gain / vs if vs > 0 else 0
        return SegmentKinematics(
            duration=duration,
            ground_distance=ground_dist,
            altitude_change=self.altitude_gain,
            vertical_speed=vs,
            ground_speed=gs,
            airspeed=self.airspeed,
            flight_path_angle=self.climb_angle_deg,
        )

    def _compute_end_state(self) -> SegmentState:
        s = self._start_state
        k = self._compute_kinematics()
        from .dem import DEMInterface
        new_lat, new_lon = DEMInterface.destination_point(
            s.lat, s.lon, s.bearing, k.ground_distance
        )
        return SegmentState(
            lat=new_lat, lon=new_lon,
            alt=s.alt + k.altitude_change,
            bearing=s.bearing,
            time=s.time + k.duration,
            distance=s.distance + k.ground_distance,
        )

    @property
    def parameter_vector(self) -> np.ndarray:
        return np.array([self.altitude_gain, self.climb_angle_deg, self.airspeed])

    @parameter_vector.setter
    def parameter_vector(self, values: np.ndarray):
        self.altitude_gain, self.climb_angle_deg, self.airspeed = values
        self._kinematics = None
        self._end_state = None
        self._waypoints = None

    @property
    def parameter_names(self) -> List[str]:
        return ["altitude_gain", "climb_angle_deg", "airspeed"]

    @property
    def parameter_bounds(self) -> List[Tuple[float, float]]:
        return [(10.0, 2000.0), (1.0, 15.0), (15.0, 35.0)]


class FWDescend(FlightSegment):
    """
    Fixed-wing descent segment.

    The aircraft descends at a constant flight-path angle γ and airspeed V.

    Free parameters:
        altitude_loss      [m]    — total altitude to lose (positive value)
        descent_angle_deg  [deg]  — flight-path angle magnitude (positive)
        airspeed           [m/s]  — true airspeed
    """

    def __init__(self, altitude_loss: float = 200.0,
                 descent_angle_deg: float = 6.0,
                 airspeed: float = 25.0):
        super().__init__(SegmentType.FW_DESCEND)
        self.altitude_loss = altitude_loss
        self.descent_angle_deg = descent_angle_deg
        self.airspeed = airspeed

    def _compute_kinematics(self) -> SegmentKinematics:
        gamma_rad = np.radians(self.descent_angle_deg)
        ground_dist = self.altitude_loss / np.tan(gamma_rad)
        gs = self.airspeed * np.cos(gamma_rad)
        vs = self.airspeed * np.sin(gamma_rad)
        duration = self.altitude_loss / vs if vs > 0 else 0
        return SegmentKinematics(
            duration=duration,
            ground_distance=ground_dist,
            altitude_change=-self.altitude_loss,
            vertical_speed=-vs,
            ground_speed=gs,
            airspeed=self.airspeed,
            flight_path_angle=-self.descent_angle_deg,
        )

    def _compute_end_state(self) -> SegmentState:
        s = self._start_state
        k = self._compute_kinematics()
        from .dem import DEMInterface
        new_lat, new_lon = DEMInterface.destination_point(
            s.lat, s.lon, s.bearing, k.ground_distance
        )
        return SegmentState(
            lat=new_lat, lon=new_lon,
            alt=s.alt + k.altitude_change,
            bearing=s.bearing,
            time=s.time + k.duration,
            distance=s.distance + k.ground_distance,
        )

    @property
    def parameter_vector(self) -> np.ndarray:
        return np.array([self.altitude_loss, self.descent_angle_deg, self.airspeed])

    @parameter_vector.setter
    def parameter_vector(self, values: np.ndarray):
        self.altitude_loss, self.descent_angle_deg, self.airspeed = values
        self._kinematics = None
        self._end_state = None
        self._waypoints = None

    @property
    def parameter_names(self) -> List[str]:
        return ["altitude_loss", "descent_angle_deg", "airspeed"]

    @property
    def parameter_bounds(self) -> List[Tuple[float, float]]:
        return [(10.0, 2000.0), (1.0, 12.0), (15.0, 35.0)]


class FWCruise(FlightSegment):
    """
    Level fixed-wing cruise at constant altitude.

    Free parameters:
        ground_distance  [m]    — horizontal distance to cover
        airspeed         [m/s]  — cruise true airspeed
    """

    def __init__(self, ground_distance: float = 5000.0,
                 airspeed: float = 25.0):
        super().__init__(SegmentType.FW_CRUISE)
        self.ground_distance_param = ground_distance
        self.airspeed = airspeed

    def _compute_kinematics(self) -> SegmentKinematics:
        duration = self.ground_distance_param / self.airspeed
        return SegmentKinematics(
            duration=duration,
            ground_distance=self.ground_distance_param,
            altitude_change=0.0,
            vertical_speed=0.0,
            ground_speed=self.airspeed,
            airspeed=self.airspeed,
            flight_path_angle=0.0,
        )

    def _compute_end_state(self) -> SegmentState:
        s = self._start_state
        k = self._compute_kinematics()
        from .dem import DEMInterface
        new_lat, new_lon = DEMInterface.destination_point(
            s.lat, s.lon, s.bearing, k.ground_distance
        )
        return SegmentState(
            lat=new_lat, lon=new_lon,
            alt=s.alt,
            bearing=s.bearing,
            time=s.time + k.duration,
            distance=s.distance + k.ground_distance,
        )

    @property
    def parameter_vector(self) -> np.ndarray:
        return np.array([self.ground_distance_param, self.airspeed])

    @parameter_vector.setter
    def parameter_vector(self, values: np.ndarray):
        self.ground_distance_param, self.airspeed = values
        self._kinematics = None
        self._end_state = None
        self._waypoints = None

    @property
    def parameter_names(self) -> List[str]:
        return ["ground_distance", "airspeed"]

    @property
    def parameter_bounds(self) -> List[Tuple[float, float]]:
        return [(100.0, 60000.0), (15.0, 35.0)]


# ═════════════════════════════════════════════════════════════════════════════
# SEGMENT FACTORY
# ═════════════════════════════════════════════════════════════════════════════

SEGMENT_CLASSES = {
    SegmentType.VTOL_ASCEND: VTOLAscend,
    SegmentType.VTOL_DESCEND: VTOLDescend,
    SegmentType.FW_CLIMB: FWClimb,
    SegmentType.FW_DESCEND: FWDescend,
    SegmentType.FW_CRUISE: FWCruise,
    SegmentType.TRANSITION: Transition,
}


def create_segment(seg_type: SegmentType, **kwargs) -> FlightSegment:
    """Factory function to create a segment by type."""
    cls = SEGMENT_CLASSES[seg_type]
    return cls(**kwargs)
