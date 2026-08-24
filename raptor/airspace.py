"""
Airspace Restrictions — 3D Geofence and Regulatory Zone Management
====================================================================

Defines, stores, and queries 3D airspace restriction volumes for
regulatory-aware eVTOL path planning.

The AirspaceManager integrates with the DEM to resolve AGL constraints
(which depend on terrain elevation) and provides efficient point-in-zone
and path-through-zone checking for the optimizer.

Every zone carries a :class:`~raptor.regulations.PermissionClass`, which
is what the optimizer actually reasons about:

    FREE        fly through at no cost
    PERMIT      legal with prior authorisation — a *soft* cost, so the
                optimizer may still route through when the detour is
                worse, and reports which permits the plan needs
    PROHIBITED  hard barrier, never overflown

Regulatory reference: RDAC 101 (DGAC Ecuador, 2024). The numeric limits
live in :mod:`raptor.regulations`; this module only says *where* they apply.

Author: Victor (LUAS-EPN / KU Leuven)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Union
from enum import Enum
import numpy as np

from .regulations import (
    PermissionClass, RegulatoryProfile, OperationalContext, RDAC_101,
)


# ═══════════════════════════════════════════════════════════════════════════
# ZONE TYPES
# ═══════════════════════════════════════════════════════════════════════════

class ZoneType(Enum):
    """
    What kind of thing a zone is.

    The type drives cartography and the *default* permission class; the
    permission class drives the optimizer. Several types map to the same
    permission, and any zone may override its default.
    """
    PROHIBITED = "prohibited"                 # 101.190(b) — never overflown
    RESTRICTED = "restricted"                 # 101.190(a) — needs authorisation
    AERODROME_CTR = "aerodrome_ctr"           # 101.195(a)(1) — prohibited surfaces
    AERODROME_RING = "aerodrome_ring"         # 101.195(a)(3) — 5 km @ 40 m AGL
    HELIPORT = "heliport"                     # 101.195(c) — 0.9 km
    CONTROLLED_AIRSPACE = "controlled_airspace"  # 101.190(a)(2) — Class B/C/D/E
    ALTITUDE_LIMIT = "altitude_limit"         # 101.185 — AGL ceiling
    POPULATED_AREA = "populated_area"         # 101.160 — urban operations
    ECOLOGICAL = "ecological"                 # 101.190(a)(6) — MAATE/SNAP
    MILITARY = "military"                     # 101.190(a)(3)/(b)(6)
    GOVERNMENT = "government"                 # 101.190(b)(3)
    SECURITY = "security"                     # 101.190(b)(1) — state security
    HOSPITAL = "hospital"                     # 101.190(a)(3)
    STRATEGIC = "strategic"                   # 101.190(a)(3) — strategic infra
    INDUSTRIAL = "industrial"                 # industrial installations
    CROWD = "crowd"                           # 101.190(a)(5) — gatherings
    TEMPORAL = "temporal"                     # 101.190(b)(7) — NOTAM/TFR


#: Default permission class for each zone type. A zone may override it.
DEFAULT_PERMISSION: Dict[ZoneType, PermissionClass] = {
    ZoneType.PROHIBITED:          PermissionClass.PROHIBITED,
    ZoneType.SECURITY:            PermissionClass.PROHIBITED,
    ZoneType.GOVERNMENT:          PermissionClass.PROHIBITED,
    ZoneType.AERODROME_CTR:       PermissionClass.PROHIBITED,
    ZoneType.RESTRICTED:          PermissionClass.PERMIT,
    ZoneType.AERODROME_RING:      PermissionClass.PERMIT,
    ZoneType.HELIPORT:            PermissionClass.PERMIT,
    ZoneType.CONTROLLED_AIRSPACE: PermissionClass.PERMIT,
    ZoneType.MILITARY:            PermissionClass.PERMIT,
    ZoneType.HOSPITAL:            PermissionClass.PERMIT,
    ZoneType.STRATEGIC:           PermissionClass.PERMIT,
    ZoneType.INDUSTRIAL:          PermissionClass.PERMIT,
    ZoneType.CROWD:               PermissionClass.PERMIT,
    ZoneType.ECOLOGICAL:          PermissionClass.PERMIT,
    ZoneType.TEMPORAL:            PermissionClass.PROHIBITED,
    ZoneType.POPULATED_AREA:      PermissionClass.FREE,   # a ceiling, not a barrier
    ZoneType.ALTITUDE_LIMIT:      PermissionClass.FREE,   # a ceiling, not a barrier
}

#: Which permit family each zone type belongs to (see regulations.DEFAULT_PERMIT_COST).
DEFAULT_PERMIT_FAMILY: Dict[ZoneType, str] = {
    ZoneType.AERODROME_CTR:       "aerodrome",
    ZoneType.AERODROME_RING:      "aerodrome",
    ZoneType.HELIPORT:            "heliport",
    ZoneType.CONTROLLED_AIRSPACE: "controlled_airspace",
    ZoneType.MILITARY:            "military",
    ZoneType.GOVERNMENT:          "government",
    ZoneType.SECURITY:            "government",
    ZoneType.HOSPITAL:            "hospital",
    ZoneType.STRATEGIC:           "strategic_infrastructure",
    ZoneType.INDUSTRIAL:          "industrial",
    ZoneType.CROWD:               "crowd",
    ZoneType.ECOLOGICAL:          "ecological",
    ZoneType.POPULATED_AREA:      "populated",
    ZoneType.RESTRICTED:          "generic",
    ZoneType.TEMPORAL:            "generic",
    ZoneType.PROHIBITED:          "generic",
    ZoneType.ALTITUDE_LIMIT:      "generic",
}

#: Zone types that express a *ceiling* rather than a lateral barrier.
CEILING_ZONE_TYPES = (
    ZoneType.ALTITUDE_LIMIT,
    ZoneType.POPULATED_AREA,
    ZoneType.AERODROME_RING,
)

# Retained for backward compatibility with figures that weight zone types.
ZONE_PENALTY_WEIGHTS = {
    ZoneType.PROHIBITED: 1000.0,
    ZoneType.RESTRICTED: 500.0,
    ZoneType.AERODROME_CTR: 800.0,
    ZoneType.ALTITUDE_LIMIT: 200.0,
    ZoneType.POPULATED_AREA: 300.0,
    ZoneType.ECOLOGICAL: 600.0,
    ZoneType.TEMPORAL: 400.0,
}


# ═══════════════════════════════════════════════════════════════════════════
# ZONE GEOMETRY
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CircularZone:
    """
    Circular horizontal footprint.

    Defined by a center point and radius. Used for aerodrome zones,
    point-like ecological reserves, and facility exclusion areas.
    """
    center_lat: float        # degrees
    center_lon: float        # degrees
    radius_m: float          # meters

    def contains_point(self, lat: float, lon: float) -> bool:
        """Check if a point is inside this circular zone."""
        d = _haversine_distance(self.center_lat, self.center_lon, lat, lon)
        return d <= self.radius_m

    def contains_points(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """Vectorised containment test over many points."""
        d = _haversine_distance(self.center_lat, self.center_lon, lats, lons)
        return d <= self.radius_m

    def distance_to_boundary(self, lat: float, lon: float) -> float:
        """
        Signed distance to boundary [m].
        Negative = inside, Positive = outside.
        """
        d = _haversine_distance(self.center_lat, self.center_lon, lat, lon)
        return d - self.radius_m


@dataclass
class PolygonalZone:
    """
    Polygonal horizontal footprint.

    Defined by an ordered list of vertices (lat, lon). The polygon is
    closed automatically (last vertex connects to first). Used for
    urban areas, irregular ecological reserves, and complex shapes.
    """
    vertices: List[Tuple[float, float]]  # [(lat, lon), ...]

    def __post_init__(self):
        """Precompute polygon properties."""
        self._n = len(self.vertices)
        if self._n < 3:
            raise ValueError("Polygon must have at least 3 vertices")
        # Precompute arrays for ray-casting
        self._lats = np.array([v[0] for v in self.vertices])
        self._lons = np.array([v[1] for v in self.vertices])
        # Bounding box, so the great majority of points are rejected with
        # four comparisons instead of a walk around the boundary. With
        # hundreds of waypoints checked against every zone on every one of
        # tens of thousands of objective evaluations, this is the
        # difference between an interactive run and an overnight one.
        self._bbox = (self._lats.min(), self._lats.max(),
                      self._lons.min(), self._lons.max())

    def contains_points(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """Vectorised ray-casting containment test over many points."""
        la0, la1, lo0, lo1 = self._bbox
        inside = np.zeros(len(lats), dtype=bool)
        cand = (lats >= la0) & (lats <= la1) & (lons >= lo0) & (lons <= lo1)
        if not cand.any():
            return inside

        y, x = lats[cand], lons[cand]
        yi, xi = self._lats, self._lons
        yj = np.roll(yi, 1)
        xj = np.roll(xi, 1)

        acc = np.zeros(len(y), dtype=bool)
        for k in range(self._n):
            cond = (yi[k] > y) != (yj[k] > y)
            if not cond.any():
                continue
            denom = (yj[k] - yi[k]) + 1e-15
            xint = (xj[k] - xi[k]) * (y - yi[k]) / denom + xi[k]
            acc ^= cond & (x < xint)

        inside[cand] = acc
        return inside

    def contains_point(self, lat: float, lon: float) -> bool:
        """
        Ray-casting algorithm for point-in-polygon test.
        Works in geographic coordinates for small regions (< 100 km).
        """
        n = self._n
        inside = False
        j = n - 1
        for i in range(n):
            yi, xi = self._lats[i], self._lons[i]
            yj, xj = self._lats[j], self._lons[j]
            if ((yi > lat) != (yj > lat)) and \
               (lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-15) + xi):
                inside = not inside
            j = i
        return inside

    def distance_to_boundary(self, lat: float, lon: float) -> float:
        """
        Approximate signed distance to polygon boundary [m].
        Negative = inside, Positive = outside.
        Uses minimum distance to all edges.
        """
        min_dist = float('inf')
        n = self._n
        for i in range(n):
            j = (i + 1) % n
            d = _point_to_segment_distance(
                lat, lon,
                self._lats[i], self._lons[i],
                self._lats[j], self._lons[j]
            )
            min_dist = min(min_dist, d)

        if self.contains_point(lat, lon):
            return -min_dist  # Inside: negative distance
        return min_dist


# ═══════════════════════════════════════════════════════════════════════════
# AIRSPACE ZONE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AirspaceZone:
    """
    A 3D airspace restriction volume.

    The zone is defined by:
    - A horizontal footprint (circular or polygonal)
    - Vertical extent (floor and ceiling)
    - Whether vertical limits are AGL (terrain-relative) or AMSL (fixed)
    - A buffer distance for soft-constraint gradient

    For AGL zones, the actual altitude limits depend on the terrain
    elevation at each point, requiring DEM lookup during checking.

    Examples
    --------
    Airport CTR (9 km radius, surface to 3700 m AMSL):
        AirspaceZone("SEQM", ZoneType.AERODROME_CTR,
                     CircularZone(-0.1292, -78.3588, 9000.0),
                     altitude_floor_m=0, altitude_ceiling_m=3700,
                     is_agl=False)

    General AGL ceiling (120 m above terrain everywhere):
        AirspaceZone("AGL_120", ZoneType.ALTITUDE_LIMIT,
                     None,  # global — no horizontal boundary
                     altitude_floor_m=0, altitude_ceiling_m=120,
                     is_agl=True)
    """
    zone_id: str
    zone_type: ZoneType
    geometry: Optional[Union[CircularZone, PolygonalZone]]
    altitude_floor_m: float = 0.0        # Lower bound [m] (AMSL or AGL)
    altitude_ceiling_m: float = 99999.0  # Upper bound [m] (AMSL or AGL)
    is_agl: bool = False                 # If True, altitudes are relative to DEM
    buffer_m: float = 200.0              # Horizontal safety buffer [m]
    active: bool = True                  # Toggle for sensitivity studies
    description: str = ""

    #: Operational permission. Defaults from ``DEFAULT_PERMISSION[zone_type]``.
    #: Set explicitly to override — e.g. a hospital helipad the operator has
    #: a standing agreement with can be declared FREE.
    permission: Optional[PermissionClass] = None
    #: Permit family, keying into ``RegulatoryProfile.permit_cost``.
    permit_family: Optional[str] = None
    #: Authority responsible for granting the permit (for the plan's annex).
    authority: str = ""
    #: Regulatory citation, e.g. "RDAC 101.190(a)(3)".
    citation: str = ""

    def __post_init__(self):
        if self.permission is None:
            self.permission = DEFAULT_PERMISSION.get(
                self.zone_type, PermissionClass.PERMIT
            )
        if self.permit_family is None:
            self.permit_family = DEFAULT_PERMIT_FAMILY.get(
                self.zone_type, "generic"
            )

    @property
    def is_ceiling_zone(self) -> bool:
        """True if this zone caps altitude rather than blocking laterally."""
        return self.zone_type in CEILING_ZONE_TYPES

    @property
    def is_global(self) -> bool:
        """True if this zone has no horizontal boundary (applies everywhere)."""
        return self.geometry is None

    def horizontal_contains(self, lat: float, lon: float) -> bool:
        """Check if a point is within the horizontal footprint."""
        if self.is_global:
            return True
        return self.geometry.contains_point(lat, lon)

    def horizontal_contains_many(self, lats: np.ndarray,
                                 lons: np.ndarray) -> np.ndarray:
        """Vectorised footprint test over many points."""
        if self.is_global:
            return np.ones(len(lats), dtype=bool)
        return self.geometry.contains_points(lats, lons)

    def horizontal_distance(self, lat: float, lon: float) -> float:
        """Signed distance to horizontal boundary. Negative = inside."""
        if self.is_global:
            return -float('inf')  # Always inside
        return self.geometry.distance_to_boundary(lat, lon)

    def in_buffer(self, lat: float, lon: float) -> bool:
        """Check if point is within the buffer zone (outside but near boundary)."""
        if self.is_global:
            return False
        d = self.geometry.distance_to_boundary(lat, lon)
        return 0 < d <= self.buffer_m

    def check_point(self, lat: float, lon: float, alt_amsl: float,
                    terrain_elev: float = np.nan,
                    with_penetration: bool = False) -> 'ZoneViolation':
        """
        Check if a 3D point violates this zone.

        Parameters
        ----------
        lat, lon : float
            Geographic coordinates [degrees].
        alt_amsl : float
            Altitude above mean sea level [m].
        terrain_elev : float
            Terrain elevation at this point [m AMSL]. Required for AGL zones.

        Returns
        -------
        ZoneViolation or None if no violation.
        """
        if not self.active:
            return None

        # Check horizontal containment
        if not self.horizontal_contains(lat, lon):
            return None

        # Resolve vertical limits
        if self.is_agl:
            if terrain_elev is None or np.isnan(terrain_elev):
                # No ground reference here — an AGL rule cannot be evaluated,
                # and guessing one would fabricate a violation.
                return None
            floor = terrain_elev + self.altitude_floor_m
            ceiling = terrain_elev + self.altitude_ceiling_m
        else:
            floor = self.altitude_floor_m
            ceiling = self.altitude_ceiling_m

        # Ceiling zones cap altitude; the violation is flying ABOVE them.
        if self.is_ceiling_zone:
            if alt_amsl > ceiling:
                return ZoneViolation(
                    zone_id=self.zone_id,
                    zone_type=self.zone_type,
                    permission=self.permission,
                    permit_family=self.permit_family,
                    violation_type='above_ceiling',
                    excess_m=alt_amsl - ceiling,
                    lat=lat, lon=lon, alt=alt_amsl,
                    limit_m=ceiling,
                )
            return None

        # Lateral zones: entering the volume between floor and ceiling is an
        # entry. Whether that entry is fatal or merely costly is decided by
        # the permission class, not here.
        if floor <= alt_amsl <= ceiling:
            # How deep inside the zone a point lies is used for reporting
            # and for the legacy zone-weighted penalty, but not by the
            # constraint model, which grades a prohibited entry by how
            # much of the route is inside it. Computing it walks every
            # polygon edge for every waypoint in every zone, which on
            # real 24-vertex footprints was the single largest cost in
            # the objective — so it is opt-in.
            penetration = 0.0
            if with_penetration and self.permission == PermissionClass.PROHIBITED:
                penetration = abs(self.horizontal_distance(lat, lon))
            return ZoneViolation(
                zone_id=self.zone_id,
                zone_type=self.zone_type,
                permission=self.permission,
                permit_family=self.permit_family,
                violation_type='inside_zone',
                excess_m=0.0,
                lat=lat, lon=lon, alt=alt_amsl,
                limit_m=0.0,
                penetration_m=penetration,
            )

        return None


@dataclass
class ZoneViolation:
    """
    One waypoint's interaction with one zone.

    Not necessarily illegal: a point inside a PERMIT zone is recorded
    here so the planner can price the detour and list the authorisation,
    but the route remains flyable once that permit is held.
    """
    zone_id: str
    zone_type: ZoneType
    violation_type: str     # 'inside_zone' | 'above_ceiling'
    excess_m: float         # Altitude excess above ceiling [m]
    lat: float
    lon: float
    alt: float
    limit_m: float          # The limit that was violated [m AMSL]
    penetration_m: float = 0.0  # Horizontal penetration depth [m]
    permission: PermissionClass = PermissionClass.PROHIBITED
    permit_family: str = "generic"

    @property
    def is_hard(self) -> bool:
        """True if this makes the route illegal no matter what."""
        if self.violation_type == 'above_ceiling':
            return True     # exceeding the 101.185 ceiling is never permitted
        return self.permission == PermissionClass.PROHIBITED


@dataclass
class AirspaceReport:
    """
    A whole path checked against the airspace.

    Separates the three outcomes the planner must not confuse:

    ``hard_violations``
        Prohibited-zone entries and ceiling exceedances. These make the
        route illegal; no paperwork fixes them.
    ``permit_violations``
        Entries into zones that are legal with prior authorisation.
        The route stays flyable; ``permits_required`` lists what to ask
        for and ``permit_cost`` prices it.
    ``feasible``
        True when there are no *hard* violations. Permits do not make a
        route infeasible — that distinction is the whole point of the
        permission model.
    """
    violations: List[ZoneViolation]
    n_violations: int = 0
    max_excess_m: float = 0.0           # Worst ceiling violation
    max_penetration_m: float = 0.0      # Deepest horizontal penetration
    total_penalty: float = 0.0          # Weighted penalty sum (hard violations)
    violated_zones: List[str] = field(default_factory=list)
    feasible: bool = True

    # ── Permission-aware breakdown ────────────────────────────────────
    hard_violations: List[ZoneViolation] = field(default_factory=list)
    permit_violations: List[ZoneViolation] = field(default_factory=list)
    prohibited_zones: List[str] = field(default_factory=list)
    ceiling_zones: List[str] = field(default_factory=list)
    #: zone_id → permit family, for every PERMIT zone the route enters.
    permits_required: Dict[str, str] = field(default_factory=dict)
    #: Summed permit effort, using RegulatoryProfile.permit_cost.
    permit_cost: float = 0.0
    #: Fraction of sampled waypoints lying inside each PERMIT zone.
    permit_exposure: Dict[str, float] = field(default_factory=dict)
    #: Number of waypoints sampled (denominator for exposure).
    n_samples: int = 0

    def __post_init__(self):
        self.n_violations = len(self.violations)
        if not self.violations:
            self.feasible = True
            return

        self.hard_violations = [v for v in self.violations if v.is_hard]
        self.permit_violations = [v for v in self.violations if not v.is_hard]

        self.feasible = len(self.hard_violations) == 0
        self.max_excess_m = max((v.excess_m for v in self.violations), default=0.0)
        self.max_penetration_m = max(
            (v.penetration_m for v in self.violations), default=0.0
        )
        self.violated_zones = sorted({v.zone_id for v in self.violations})
        self.prohibited_zones = sorted({
            v.zone_id for v in self.hard_violations
            if v.violation_type == 'inside_zone'
        })
        self.ceiling_zones = sorted({
            v.zone_id for v in self.hard_violations
            if v.violation_type == 'above_ceiling'
        })

        for v in self.permit_violations:
            self.permits_required[v.zone_id] = v.permit_family
        counts: Dict[str, int] = {}
        for v in self.permit_violations:
            counts[v.zone_id] = counts.get(v.zone_id, 0) + 1
        if self.n_samples > 0:
            self.permit_exposure = {
                k: n / self.n_samples for k, n in counts.items()
            }

        # Penalty covers hard violations only — permits are priced separately
        # by the optimizer, which knows the regulatory profile's cost table.
        for v in self.hard_violations:
            w = ZONE_PENALTY_WEIGHTS.get(v.zone_type, 100.0)
            if v.violation_type == 'above_ceiling':
                self.total_penalty += w * (v.excess_m / 100.0)
            else:
                self.total_penalty += w * (1.0 + v.penetration_m / 500.0)

    def permit_cost_with(self, profile: RegulatoryProfile = RDAC_101) -> float:
        """Total permit effort under a given regulatory profile."""
        return sum(
            profile.permit_cost.get(fam, profile.permit_cost["generic"])
            for fam in self.permits_required.values()
        )

    def summary(self) -> str:
        if self.feasible and not self.permits_required:
            return "Airspace: CLEAR — no violations, no permits required"

        lines = []
        if self.feasible:
            lines.append("Airspace: LEGAL WITH AUTHORISATION")
        else:
            lines.append(
                f"Airspace: INFEASIBLE — {len(self.hard_violations)} hard "
                f"violation(s)"
            )
            if self.prohibited_zones:
                lines.append(f"  Prohibited zones entered: "
                             f"{', '.join(self.prohibited_zones)}")
            if self.ceiling_zones:
                lines.append(f"  Ceiling exceeded in:      "
                             f"{', '.join(self.ceiling_zones)}")
                lines.append(f"  Max ceiling excess:       {self.max_excess_m:.1f} m")

        if self.permits_required:
            lines.append(f"  Permits required ({len(self.permits_required)}):")
            for zid, fam in sorted(self.permits_required.items()):
                exp = self.permit_exposure.get(zid, 0.0) * 100
                lines.append(f"    - {zid:<28s} [{fam}]  {exp:.1f}% of route")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# AIRSPACE MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class AirspaceManager:
    """
    Central manager for all airspace restriction zones.

    Provides efficient checking of individual points and complete flight
    paths against all active zones. Integrates with the DEM for AGL
    altitude resolution.

    Parameters
    ----------
    zones : List[AirspaceZone]
        All defined zones.
    dem : optional
        DEMInterface for AGL zone checking. If None, AGL zones use
        terrain_elev=0 (conservative assumption).
    """

    def __init__(self, zones: List[AirspaceZone] = None, dem=None):
        self.zones = zones or []
        self.dem = dem

    def add_zone(self, zone: AirspaceZone):
        """Add a zone to the manager."""
        self.zones.append(zone)

    def remove_zone(self, zone_id: str):
        """Remove a zone by ID."""
        self.zones = [z for z in self.zones if z.zone_id != zone_id]

    def get_zone(self, zone_id: str) -> Optional[AirspaceZone]:
        """Retrieve a zone by ID."""
        for z in self.zones:
            if z.zone_id == zone_id:
                return z
        return None

    def set_active(self, zone_id: str, active: bool):
        """Enable or disable a zone (for sensitivity analysis)."""
        z = self.get_zone(zone_id)
        if z:
            z.active = active

    def active_zones(self) -> List[AirspaceZone]:
        """Return only currently active zones."""
        return [z for z in self.zones if z.active]

    # ── Point checking ────────────────────────────────────────────────

    def check_point(self, lat: float, lon: float,
                    alt_amsl: float) -> List[ZoneViolation]:
        """
        Check a single 3D point against all active zones.

        Automatically resolves terrain elevation for AGL zones via the DEM.

        Returns
        -------
        List of ZoneViolation (empty if no violations).
        """
        # Terrain elevation, needed for AGL zones. NaN (outside the DEM)
        # is propagated rather than replaced with 0 — see _terrain_batch.
        terrain_elev = np.nan
        if self.dem is not None:
            try:
                terrain_elev = float(self.dem.elevation(lat, lon))
            except Exception:
                terrain_elev = np.nan

        violations = []
        for zone in self.active_zones():
            v = zone.check_point(lat, lon, alt_amsl, terrain_elev)
            if v is not None:
                violations.append(v)

        return violations

    # ── Path checking ─────────────────────────────────────────────────

    def check_path(self, path, with_penetration: bool = False) -> AirspaceReport:
        """
        Check a complete FlightPath against all active zones.

        Evaluates all waypoints in the path. For efficiency, uses
        the terrain elevations already computed by the TerrainAnalyzer
        if available.

        Parameters
        ----------
        path : FlightPath
            The path to check (must have waypoints populated).

        Parameters
        ----------
        path : FlightPath
            The path to check (must have waypoints populated).
        with_penetration : bool
            Also compute how deep inside each prohibited zone the route
            goes. Off by default: it walks every polygon edge per
            waypoint per zone and nothing in the constraint model needs
            it. Turn it on for figures and reports.

        Returns
        -------
        AirspaceReport with aggregate violation metrics.
        """
        waypoints = path.get_waypoints()
        if not waypoints:
            return AirspaceReport(violations=[])

        lats = np.array([wp.lat for wp in waypoints])
        lons = np.array([wp.lon for wp in waypoints])
        alts = np.array([wp.alt for wp in waypoints])

        terrain = self._terrain_batch(lats, lons, waypoints)

        all_violations = []
        for zone in self.active_zones():
            inside = zone.horizontal_contains_many(lats, lons)
            if not inside.any():
                continue
            for i in np.flatnonzero(inside):
                v = zone.check_point(lats[i], lons[i], alts[i], terrain[i],
                                     with_penetration=with_penetration)
                if v is not None:
                    all_violations.append(v)

        return AirspaceReport(violations=all_violations,
                              n_samples=len(waypoints))

    def _terrain_batch(self, lats: np.ndarray, lons: np.ndarray,
                       waypoints=None) -> np.ndarray:
        """
        Terrain elevation under each sample point [m AMSL].

        AGL-referenced zones are meaningless without this. Resolving it
        wrongly does not merely degrade the answer, it inverts it: with
        terrain taken as 0, a 122 m AGL ceiling becomes a 122 m AMSL
        ceiling, and every point of a Quito route sits ~2 800 m "above"
        it. So the DEM is queried directly here rather than trusting
        cached per-waypoint values, which callers may never have filled.

        Points outside the DEM keep their cached elevation if one exists,
        and are otherwise excluded (NaN) so that no zone check fires on
        an invented ground level.
        """
        if self.dem is not None:
            try:
                terrain = np.asarray(
                    self.dem.elevation_batch(lats, lons), dtype=float
                )
            except Exception:
                terrain = np.full(len(lats), np.nan)
        else:
            terrain = np.full(len(lats), np.nan)

        # Fall back to any elevation a previous analysis already attached.
        if waypoints is not None:
            missing = np.isnan(terrain)
            if missing.any():
                cached = np.array(
                    [getattr(wp, 'terrain_elev', np.nan) for wp in waypoints],
                    dtype=float,
                )
                terrain = np.where(missing, cached, terrain)

        return terrain

    # ── Segment checking ──────────────────────────────────────────────

    def check_segment(self, start_lat: float, start_lon: float,
                      end_lat: float, end_lon: float,
                      altitude: float, n_samples: int = 20
                      ) -> List[ZoneViolation]:
        """
        Check a straight-line segment at constant altitude.

        Samples n_samples points along the great circle from start to end
        and checks each against all zones.

        Useful for quick feasibility checking before full path construction.
        """
        violations = []
        lats = np.linspace(start_lat, end_lat, n_samples)
        lons = np.linspace(start_lon, end_lon, n_samples)

        for lat, lon in zip(lats, lons):
            vs = self.check_point(lat, lon, altitude)
            violations.extend(vs)

        return violations

    # ── Query helpers ─────────────────────────────────────────────────

    def get_altitude_ceiling(self, lat: float, lon: float) -> float:
        """
        Get the effective maximum altitude [m AMSL] at a given point.

        Considers all active ALTITUDE_LIMIT and POPULATED_AREA zones.
        Returns the most restrictive (lowest) ceiling.
        """
        min_ceiling = float('inf')

        terrain_elev = np.nan
        if self.dem is not None:
            try:
                terrain_elev = float(self.dem.elevation(lat, lon))
            except Exception:
                terrain_elev = np.nan

        for zone in self.active_zones():
            if not zone.is_ceiling_zone:
                continue
            if not zone.horizontal_contains(lat, lon):
                continue

            if zone.is_agl:
                if np.isnan(terrain_elev):
                    continue
                ceiling = terrain_elev + zone.altitude_ceiling_m
            else:
                ceiling = zone.altitude_ceiling_m

            min_ceiling = min(min_ceiling, ceiling)

        return min_ceiling

    def is_prohibited(self, lat: float, lon: float) -> bool:
        """Check if a horizontal position is inside any prohibited zone."""
        for zone in self.active_zones():
            if zone.permission != PermissionClass.PROHIBITED:
                continue
            if zone.horizontal_contains(lat, lon):
                return True
        return False

    def get_clearance_to_zones(self, lat: float, lon: float
                               ) -> Dict[str, float]:
        """
        Get horizontal distance to all zone boundaries [m].

        Returns dict: zone_id → signed distance (negative = inside).
        """
        result = {}
        for zone in self.active_zones():
            if zone.geometry is not None:
                d = zone.horizontal_distance(lat, lon)
                result[zone.zone_id] = d
        return result


# ═══════════════════════════════════════════════════════════════════════════
# GENERIC AIRSPACE LOADER — Load zones from JSON data file
# ═══════════════════════════════════════════════════════════════════════════

def load_airspace_from_file(filepath: str, dem=None) -> AirspaceManager:
    """
    Load airspace zones from a JSON file.

    This is the recommended way to configure airspace for any region.
    The JSON file should contain a list of zones, each with:
        - zone_id, zone_type, geometry_type
        - For circles: center_lat, center_lon, radius_m
        - For polygons: vertices (list of [lat, lon] pairs)
        - For global zones: geometry_type = "global"
        - altitude_floor_m, altitude_ceiling_m, is_agl, buffer_m
        - description (optional)

    Parameters
    ----------
    filepath : str
        Path to the airspace JSON file.
    dem : DEMInterface, optional
        DEM for AGL-to-AMSL conversion. Required if any zone uses is_agl=True.

    Returns
    -------
    AirspaceManager with all zones loaded.

    Example
    -------
    >>> airspace = load_airspace_from_file('data/ecuador_uas_zones.json', dem=dem)
    >>> report = airspace.check_path(flight_path)
    """
    import json

    with open(filepath, 'r') as f:
        data = json.load(f)

    zone_list = data.get('zones', data) if isinstance(data, dict) else data
    if isinstance(data, dict) and 'zones' in data:
        zone_list = data['zones']

    type_map = {v.value: v for v in ZoneType}

    zones = []
    for zd in zone_list:
        zone_type = type_map.get(zd['zone_type'])
        if zone_type is None:
            raise ValueError(f"Unknown zone_type '{zd['zone_type']}'. "
                             f"Valid: {list(type_map.keys())}")

        geo_type = zd.get('geometry_type', 'circle')
        if geo_type == 'circle':
            geometry = CircularZone(
                center_lat=zd['center_lat'],
                center_lon=zd['center_lon'],
                radius_m=zd['radius_m'],
            )
        elif geo_type == 'polygon':
            vertices = [(v[0], v[1]) for v in zd['vertices']]
            geometry = PolygonalZone(vertices=vertices)
        elif geo_type == 'global':
            geometry = None  # Global zone — applies everywhere
        else:
            raise ValueError(f"Unknown geometry_type '{geo_type}' "
                             f"for zone '{zd['zone_id']}'")

        # Permission may be stated explicitly; otherwise it follows from
        # the zone type. Stating it matters when the source data already
        # classifies a zone (Prohibida / Restringida), because that
        # classification is the regulator's, not ours to infer.
        permission = zd.get('permission')
        if permission is not None:
            try:
                permission = PermissionClass(str(permission).lower())
            except ValueError:
                raise ValueError(
                    f"Unknown permission '{zd['permission']}' for zone "
                    f"'{zd['zone_id']}'. Valid: "
                    f"{[p.value for p in PermissionClass]}"
                )

        zones.append(AirspaceZone(
            zone_id=zd['zone_id'],
            zone_type=zone_type,
            geometry=geometry,
            altitude_floor_m=zd.get('altitude_floor_m', 0.0),
            altitude_ceiling_m=zd.get('altitude_ceiling_m', 10000.0),
            is_agl=zd.get('is_agl', False),
            buffer_m=zd.get('buffer_m', 0.0),
            description=zd.get('description', ''),
            permission=permission,
            permit_family=zd.get('permit_family'),
            authority=zd.get('authority', ''),
            citation=zd.get('citation', ''),
        ))

    return AirspaceManager(zones=zones, dem=dem)


def build_airspace(dem=None) -> AirspaceManager:
    """
    Load the bundled Ecuadorian zone set, for the DMQ case study.

    The file is plain data — a snapshot of the published DGAC and MAATE
    footprints, generated once by ``scripts/import_zones_ecuador.py``.
    Nothing in the package imports anything Ecuador-specific, and no
    country-specific package is a dependency: to plan somewhere else,
    write a zone file for that place and load it with
    :func:`load_airspace_from_file`. The regulation itself lives in
    :mod:`raptor.regulations` as a swappable
    :class:`~raptor.regulations.RegulatoryProfile`.
    """
    import os
    here = os.path.dirname(__file__)
    names = ['ecuador_uas_zones.json']
    roots = [os.path.join(here, '..', 'data'), 'data', '../data', '.']
    for name in names:
        for root in roots:
            p = os.path.join(root, name)
            if os.path.exists(p):
                return load_airspace_from_file(p, dem=dem)

    raise FileNotFoundError(
        "Cannot find an airspace file. Expected one of "
        f"{names} under data/, or pass your own to load_airspace_from_file()."
    )


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _haversine_distance(lat1: float, lon1: float,
                        lat2: float, lon2: float) -> float:
    """
    Haversine great-circle distance between two points [m].
    """
    R = 6_371_000.0  # Earth radius [m]
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _point_to_segment_distance(plat: float, plon: float,
                                alat: float, alon: float,
                                blat: float, blon: float) -> float:
    """
    Approximate minimum distance [m] from point P to line segment AB.
    Uses flat-Earth approximation (valid for < 100 km).
    """
    # Convert to local meters (flat Earth at mid-latitude)
    mid_lat = (alat + blat) / 2
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(np.radians(mid_lat))

    px = (plon - alon) * m_per_deg_lon
    py = (plat - alat) * m_per_deg_lat
    ax, ay = 0.0, 0.0
    bx = (blon - alon) * m_per_deg_lon
    by = (blat - alat) * m_per_deg_lat

    # Project P onto AB
    abx, aby = bx - ax, by - ay
    ab_sq = abx**2 + aby**2
    if ab_sq < 1e-10:
        return np.sqrt(px**2 + py**2)

    t = max(0.0, min(1.0, (px * abx + py * aby) / ab_sq))
    projx = ax + t * abx
    projy = ay + t * aby

    return np.sqrt((px - projx)**2 + (py - projy)**2)
