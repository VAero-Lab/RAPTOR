"""
Flight Path — Ordered Sequence of Segments
============================================

A FlightPath chains multiple FlightSegments into a complete trajectory
from origin to destination. It provides:

- Automatic state propagation (each segment's end → next segment's start)
- Unified parameter vector for the entire path (optimizer interface)
- Dense waypoint generation for terrain checking
- Path-level metrics (total time, distance, energy proxy)
- Structural validation (legal segment sequences)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np

from .segments import (
    FlightSegment, SegmentState, SegmentType,
    VTOLAscend, VTOLDescend, FWClimb, FWDescend, FWCruise, Transition
)


@dataclass
class Waypoint:
    """A single point along the flight path."""
    lat: float
    lon: float
    alt: float           # m AMSL
    time: float          # s from mission start
    distance: float      # cumulative ground distance [m]
    segment_idx: int     # which segment this belongs to
    terrain_elev: float = np.nan   # terrain elevation below (filled by TerrainAnalyzer)
    agl: float = np.nan            # altitude above ground level (filled by TerrainAnalyzer)


@dataclass
class PathMetrics:
    """Summary metrics for a complete flight path."""
    total_time: float           # s
    total_ground_distance: float  # m
    total_altitude_gain: float  # m (sum of all positive Δh)
    total_altitude_loss: float  # m (sum of all negative Δh, as positive)
    max_altitude: float         # m AMSL
    min_altitude: float         # m AMSL (during flight, not on ground)
    n_segments: int
    n_vtol_segments: int
    n_fw_segments: int
    segment_summary: List[str]


# Legal segment transitions: what can follow what
_LEGAL_TRANSITIONS = {
    SegmentType.VTOL_ASCEND: {SegmentType.VTOL_ASCEND, SegmentType.TRANSITION,
                               SegmentType.FW_CLIMB, SegmentType.FW_CRUISE},
    SegmentType.TRANSITION:  {SegmentType.FW_CLIMB, SegmentType.FW_CRUISE,
                               SegmentType.FW_DESCEND, SegmentType.VTOL_ASCEND,
                               SegmentType.VTOL_DESCEND},
    SegmentType.FW_CLIMB:    {SegmentType.FW_CLIMB, SegmentType.FW_CRUISE,
                               SegmentType.FW_DESCEND, SegmentType.TRANSITION},
    SegmentType.FW_CRUISE:   {SegmentType.FW_CRUISE, SegmentType.FW_CLIMB,
                               SegmentType.FW_DESCEND, SegmentType.TRANSITION},
    SegmentType.FW_DESCEND:  {SegmentType.FW_DESCEND, SegmentType.FW_CRUISE,
                               SegmentType.FW_CLIMB, SegmentType.TRANSITION},
    SegmentType.VTOL_DESCEND: {SegmentType.VTOL_DESCEND},  # terminal
}

# Required first segment
_VALID_FIRST = {SegmentType.VTOL_ASCEND}

# Required last segment
_VALID_LAST = {SegmentType.VTOL_DESCEND}


class FlightPath:
    """
    A complete flight path from origin to destination.

    The path is an ordered list of FlightSegments. When segments are
    added or parameters updated, the chain is automatically propagated
    so each segment's start state matches the previous segment's end.

    Parameters
    ----------
    origin_lat, origin_lon : float
        Takeoff coordinates [degrees].
    origin_elev : float
        Ground elevation at origin [m AMSL].
    destination_lat, destination_lon : float
        Landing coordinates [degrees].
    destination_elev : float
        Ground elevation at destination [m AMSL].

    Example
    -------
    >>> path = FlightPath(-0.2444, -78.5411, 3009,
    ...                   -0.1844, -78.5037, 2838)
    >>> path.add_segment(VTOLAscend(altitude_gain=100, climb_rate=2.5))
    >>> path.add_segment(Transition())
    >>> path.add_segment(FWClimb(altitude_gain=300, climb_angle_deg=8))
    >>> path.add_segment(FWCruise(ground_distance=5000, airspeed=25))
    >>> path.add_segment(FWDescend(altitude_loss=350, descent_angle_deg=6))
    >>> path.add_segment(Transition())
    >>> path.add_segment(VTOLDescend(altitude_loss=80, descent_rate=2))
    >>> print(path.metrics)
    """

    def __init__(
        self,
        origin_lat: float, origin_lon: float, origin_elev: float,
        destination_lat: float, destination_lon: float, destination_elev: float,
    ):
        self.origin = (origin_lat, origin_lon, origin_elev)
        self.destination = (destination_lat, destination_lon, destination_elev)
        self._segments: List[FlightSegment] = []

        # Compute bearing from origin to destination
        from .dem import DEMInterface
        self._nominal_bearing = DEMInterface.bearing(
            origin_lat, origin_lon, destination_lat, destination_lon
        )
        self._nominal_distance = DEMInterface.haversine(
            origin_lat, origin_lon, destination_lat, destination_lon
        )

    @property
    def nominal_bearing(self) -> float:
        """Direct bearing from origin to destination [degrees]."""
        return self._nominal_bearing

    @property
    def nominal_distance(self) -> float:
        """Direct ground distance from origin to destination [m]."""
        return self._nominal_distance

    @property
    def segments(self) -> List[FlightSegment]:
        return self._segments

    @property
    def n_segments(self) -> int:
        return len(self._segments)

    # ── Segment management ───────────────────────────────────────────────

    def add_segment(self, segment: FlightSegment) -> 'FlightPath':
        """
        Append a segment to the path and propagate states.

        Returns self for method chaining.
        """
        self._segments.append(segment)
        self._propagate()
        return self

    def insert_segment(self, index: int, segment: FlightSegment) -> 'FlightPath':
        """Insert a segment at a specific position."""
        self._segments.insert(index, segment)
        self._propagate()
        return self

    def remove_segment(self, index: int) -> FlightSegment:
        """Remove and return a segment by index."""
        seg = self._segments.pop(index)
        self._propagate()
        return seg

    def replace_segment(self, index: int, segment: FlightSegment) -> 'FlightPath':
        """Replace a segment at a given index."""
        self._segments[index] = segment
        self._propagate()
        return self

    def clear(self):
        """Remove all segments."""
        self._segments.clear()

    # ── State propagation ────────────────────────────────────────────────

    def _propagate(self):
        """
        Propagate states through the segment chain.

        The first segment starts at the origin; each subsequent segment
        starts where the previous one ended.
        """
        if not self._segments:
            return

        # Initial state: on the ground at origin
        state = SegmentState(
            lat=self.origin[0],
            lon=self.origin[1],
            alt=self.origin[2],  # ground level at origin
            bearing=self._nominal_bearing,
            time=0.0,
            distance=0.0,
        )

        for seg in self._segments:
            seg.start_state = state
            state = seg.end_state

    # ── Unified parameter vector (optimizer interface) ───────────────────

    @property
    def parameter_vector(self) -> np.ndarray:
        """
        Concatenated parameter vector for ALL segments.

        This is the interface for numerical optimizers: a single flat
        array θ that fully describes the path geometry.
        """
        vectors = [seg.parameter_vector for seg in self._segments]
        return np.concatenate(vectors) if vectors else np.array([])

    @parameter_vector.setter
    def parameter_vector(self, values: np.ndarray):
        """
        Set all segment parameters from a flat array and re-propagate.

        The array is split according to each segment's n_parameters.
        """
        idx = 0
        for seg in self._segments:
            n = seg.n_parameters
            seg.parameter_vector = values[idx:idx + n]
            idx += n
        self._propagate()

    @property
    def parameter_names(self) -> List[str]:
        """Full list of parameter names with segment prefix."""
        names = []
        for i, seg in enumerate(self._segments):
            prefix = f"seg{i}_{seg.segment_type.value}"
            for pname in seg.parameter_names:
                names.append(f"{prefix}.{pname}")
        return names

    @property
    def parameter_bounds(self) -> List[Tuple[float, float]]:
        """Concatenated bounds for the full parameter vector."""
        bounds = []
        for seg in self._segments:
            bounds.extend(seg.parameter_bounds)
        return bounds

    @property
    def n_parameters(self) -> int:
        return sum(seg.n_parameters for seg in self._segments)

    # ── Waypoint extraction ──────────────────────────────────────────────

    def get_waypoints(self, points_per_segment: int = 20) -> List[Waypoint]:
        """
        Generate dense waypoints along the entire path.

        Returns a list of Waypoint objects for terrain analysis.
        """
        waypoints = []
        for i, seg in enumerate(self._segments):
            wp_array = seg.waypoints  # (N, 5): lat, lon, alt, time, dist
            for row in wp_array:
                waypoints.append(Waypoint(
                    lat=row[0], lon=row[1], alt=row[2],
                    time=row[3], distance=row[4],
                    segment_idx=i,
                ))
        return waypoints

    def get_waypoints_array(self) -> np.ndarray:
        """
        Get all waypoints as a single (N, 5) array.

        Columns: [lat, lon, alt, time, distance]
        """
        arrays = [seg.waypoints for seg in self._segments]
        return np.vstack(arrays) if arrays else np.empty((0, 5))

    # ── Path metrics ─────────────────────────────────────────────────────

    @property
    def metrics(self) -> PathMetrics:
        """Compute summary metrics for the path."""
        if not self._segments:
            return PathMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, [])

        total_time = sum(s.kinematics.duration for s in self._segments)
        total_dist = sum(s.kinematics.ground_distance for s in self._segments)
        total_gain = sum(s.kinematics.altitude_change
                         for s in self._segments
                         if s.kinematics.altitude_change > 0)
        total_loss = sum(-s.kinematics.altitude_change
                         for s in self._segments
                         if s.kinematics.altitude_change < 0)

        wp = self.get_waypoints_array()
        max_alt = float(wp[:, 2].max()) if len(wp) > 0 else 0
        min_alt = float(wp[:, 2].min()) if len(wp) > 0 else 0

        vtol_types = {SegmentType.VTOL_ASCEND, SegmentType.VTOL_DESCEND}
        n_vtol = sum(1 for s in self._segments if s.segment_type in vtol_types)
        n_fw = sum(1 for s in self._segments
                   if s.segment_type in {SegmentType.FW_CLIMB,
                                         SegmentType.FW_CRUISE,
                                         SegmentType.FW_DESCEND})

        summary = [repr(s) for s in self._segments]

        return PathMetrics(
            total_time=total_time,
            total_ground_distance=total_dist,
            total_altitude_gain=total_gain,
            total_altitude_loss=total_loss,
            max_altitude=max_alt,
            min_altitude=min_alt,
            n_segments=len(self._segments),
            n_vtol_segments=n_vtol,
            n_fw_segments=n_fw,
            segment_summary=summary,
        )

    @property
    def end_state(self) -> Optional[SegmentState]:
        """State at the end of the last segment."""
        if self._segments:
            return self._segments[-1].end_state
        return None

    # ── Validation ───────────────────────────────────────────────────────

    def validate_structure(self) -> Tuple[bool, List[str]]:
        """
        Check that the segment sequence is physically valid.

        Rules:
        1. Path must start with VTOL_ASCEND
        2. Path must end with VTOL_DESCEND
        3. Each segment transition must be legal
        4. Altitudes must remain positive
        5. No altitude below ground level

        Returns (is_valid, list_of_issues)
        """
        issues = []

        if not self._segments:
            return False, ["Path has no segments."]

        # Rule 1: first segment
        if self._segments[0].segment_type not in _VALID_FIRST:
            issues.append(
                f"Path must start with VTOL_ASCEND, "
                f"got {self._segments[0].segment_type.value}")

        # Rule 2: last segment
        if self._segments[-1].segment_type not in _VALID_LAST:
            issues.append(
                f"Path must end with VTOL_DESCEND, "
                f"got {self._segments[-1].segment_type.value}")

        # Rule 3: legal transitions
        for i in range(len(self._segments) - 1):
            curr = self._segments[i].segment_type
            next_ = self._segments[i + 1].segment_type
            if next_ not in _LEGAL_TRANSITIONS.get(curr, set()):
                issues.append(
                    f"Illegal transition: {curr.value} → {next_.value} "
                    f"(segments {i} → {i+1})")

        # Rule 4: altitude check
        wp = self.get_waypoints_array()
        if len(wp) > 0 and np.any(wp[:, 2] < 0):
            issues.append("Altitude goes below 0 m AMSL.")

        return len(issues) == 0, issues

    # ── Display ──────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable path summary."""
        m = self.metrics
        valid, issues = self.validate_structure()
        lines = [
            "=" * 60,
            "FLIGHT PATH SUMMARY",
            "=" * 60,
            f"Origin:      ({self.origin[0]:.4f}, {self.origin[1]:.4f}) "
            f"@ {self.origin[2]:.0f} m",
            f"Destination: ({self.destination[0]:.4f}, {self.destination[1]:.4f}) "
            f"@ {self.destination[2]:.0f} m",
            f"Direct dist: {self._nominal_distance:.0f} m "
            f"({self._nominal_distance/1000:.1f} km)",
            f"Bearing:     {self._nominal_bearing:.1f}°",
            "",
            f"Segments:    {m.n_segments} "
            f"({m.n_vtol_segments} VTOL, {m.n_fw_segments} FW, "
            f"{m.n_segments - m.n_vtol_segments - m.n_fw_segments} transition)",
            f"Total time:  {m.total_time:.1f} s ({m.total_time/60:.1f} min)",
            f"Ground dist: {m.total_ground_distance:.0f} m "
            f"({m.total_ground_distance/1000:.1f} km)",
            f"Alt gain:    {m.total_altitude_gain:.0f} m",
            f"Alt loss:    {m.total_altitude_loss:.0f} m",
            f"Alt range:   [{m.min_altitude:.0f}, {m.max_altitude:.0f}] m AMSL",
            f"Valid:       {'YES' if valid else 'NO'}",
        ]
        if not valid:
            lines.append("Issues:")
            for iss in issues:
                lines.append(f"  ⚠ {iss}")

        lines.append("")
        lines.append("Segment details:")
        for i, seg in enumerate(self._segments):
            k = seg.kinematics
            s = seg.start_state
            e = seg.end_state
            lines.append(
                f"  [{i}] {seg.segment_type.value:15s}  "
                f"alt {s.alt:.0f}→{e.alt:.0f} m  "
                f"Δh={k.altitude_change:+.0f}m  "
                f"d={k.ground_distance:.0f}m  "
                f"t={k.duration:.1f}s  "
                f"V={k.airspeed:.1f}m/s  "
                f"γ={k.flight_path_angle:+.1f}°"
            )

        lines.append("")
        lines.append(f"Parameter vector: {self.n_parameters} dimensions")
        return "\n".join(lines)

    def __repr__(self):
        m = self.metrics
        return (f"FlightPath({m.n_segments} segs, "
                f"{m.total_ground_distance/1000:.1f} km, "
                f"{m.total_time:.0f} s)")
