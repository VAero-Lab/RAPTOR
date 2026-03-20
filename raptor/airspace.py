"""
Airspace Restrictions — 3D Geofence and Regulatory Zone Management
====================================================================

Defines, stores, and queries 3D airspace restriction volumes for
regulatory-aware eVTOL path planning.

Zone types:
    PROHIBITED       — No flight allowed (military, security)
    RESTRICTED       — Authorized operations only
    AERODROME_CTR    — Airport control zone (cylindrical)
    ALTITUDE_LIMIT   — AGL ceiling (terrain-following)
    POPULATED_AREA   — Reduced altitude ceiling over urban areas
    ECOLOGICAL       — Protected natural areas
    TEMPORAL         — Temporary flight restriction (TFR)

The AirspaceManager integrates with the DEM to resolve AGL constraints
(which depend on terrain elevation) and provides efficient point-in-zone
and path-through-zone checking for the optimizer.

Regulatory reference: RDAC 101 (DGAC Ecuador, 2024)
    - Max altitude: 120 m AGL (400 ft) general
    - Aerodrome exclusion: 9 km radius
    - VLOS requirement (waivable for specific operations)
    - Populated area restrictions

Author: Victor (LUAS-EPN / KU Leuven)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Union
from enum import Enum
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# ZONE TYPES
# ═══════════════════════════════════════════════════════════════════════════

class ZoneType(Enum):
    """Classification of airspace restriction zones."""
    PROHIBITED = "prohibited"           # No flight under any condition
    RESTRICTED = "restricted"           # Flight requires special authorization
    AERODROME_CTR = "aerodrome_ctr"     # Airport control zone
    ALTITUDE_LIMIT = "altitude_limit"   # AGL ceiling (terrain-following)
    POPULATED_AREA = "populated_area"   # Urban area with reduced ceiling
    ECOLOGICAL = "ecological"           # Protected natural area
    TEMPORAL = "temporal"               # Temporary flight restriction


# Penalty weights by zone type (used by the optimizer)
# Higher weight = harder constraint
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

    @property
    def is_global(self) -> bool:
        """True if this zone has no horizontal boundary (applies everywhere)."""
        return self.geometry is None

    def horizontal_contains(self, lat: float, lon: float) -> bool:
        """Check if a point is within the horizontal footprint."""
        if self.is_global:
            return True
        return self.geometry.contains_point(lat, lon)

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
                    terrain_elev: float = 0.0) -> 'ZoneViolation':
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
            floor = terrain_elev + self.altitude_floor_m
            ceiling = terrain_elev + self.altitude_ceiling_m
        else:
            floor = self.altitude_floor_m
            ceiling = self.altitude_ceiling_m

        # For ALTITUDE_LIMIT type: violation is being ABOVE the ceiling
        if self.zone_type == ZoneType.ALTITUDE_LIMIT:
            if alt_amsl > ceiling:
                return ZoneViolation(
                    zone_id=self.zone_id,
                    zone_type=self.zone_type,
                    violation_type='above_ceiling',
                    excess_m=alt_amsl - ceiling,
                    lat=lat, lon=lon, alt=alt_amsl,
                    limit_m=ceiling,
                )
            return None

        # For POPULATED_AREA: same as altitude limit but with different penalty
        if self.zone_type == ZoneType.POPULATED_AREA:
            if alt_amsl > ceiling:
                return ZoneViolation(
                    zone_id=self.zone_id,
                    zone_type=self.zone_type,
                    violation_type='above_ceiling',
                    excess_m=alt_amsl - ceiling,
                    lat=lat, lon=lon, alt=alt_amsl,
                    limit_m=ceiling,
                )
            return None

        # For PROHIBITED, RESTRICTED, AERODROME_CTR, ECOLOGICAL:
        # Any point inside the volume (between floor and ceiling) is a violation
        if floor <= alt_amsl <= ceiling:
            return ZoneViolation(
                zone_id=self.zone_id,
                zone_type=self.zone_type,
                violation_type='inside_prohibited',
                excess_m=0.0,
                lat=lat, lon=lon, alt=alt_amsl,
                limit_m=0.0,
                penetration_m=abs(self.horizontal_distance(lat, lon)),
            )

        return None


@dataclass
class ZoneViolation:
    """A single airspace violation at a specific point."""
    zone_id: str
    zone_type: ZoneType
    violation_type: str     # 'inside_prohibited', 'above_ceiling'
    excess_m: float         # Altitude excess above ceiling [m]
    lat: float
    lon: float
    alt: float
    limit_m: float          # The limit that was violated [m AMSL]
    penetration_m: float = 0.0  # Horizontal penetration depth [m]


@dataclass
class AirspaceReport:
    """
    Result of checking a complete flight path against airspace zones.

    Provides aggregate metrics for the optimizer penalty function.
    """
    violations: List[ZoneViolation]
    n_violations: int = 0
    max_excess_m: float = 0.0           # Worst ceiling violation
    max_penetration_m: float = 0.0      # Deepest horizontal penetration
    total_penalty: float = 0.0          # Weighted penalty sum
    violated_zones: List[str] = field(default_factory=list)
    feasible: bool = True

    def __post_init__(self):
        self.n_violations = len(self.violations)
        self.feasible = self.n_violations == 0
        if self.violations:
            self.max_excess_m = max(
                v.excess_m for v in self.violations
            )
            self.max_penetration_m = max(
                v.penetration_m for v in self.violations
            )
            self.violated_zones = list(set(
                v.zone_id for v in self.violations
            ))
            # Compute weighted penalty
            for v in self.violations:
                w = ZONE_PENALTY_WEIGHTS.get(v.zone_type, 100.0)
                if v.violation_type == 'above_ceiling':
                    self.total_penalty += w * (v.excess_m / 100.0)
                else:  # inside_prohibited
                    self.total_penalty += w * (1.0 + v.penetration_m / 500.0)

    def summary(self) -> str:
        if self.feasible:
            return "Airspace: CLEAR — no violations"
        lines = [
            f"Airspace: {self.n_violations} VIOLATIONS across "
            f"{len(self.violated_zones)} zone(s)",
            f"  Max ceiling excess:      {self.max_excess_m:.1f} m",
            f"  Max penetration depth:   {self.max_penetration_m:.1f} m",
            f"  Total penalty:           {self.total_penalty:.1f}",
            f"  Violated zones:          {', '.join(self.violated_zones)}",
        ]
        return '\n'.join(lines)


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
        # Get terrain elevation (needed for AGL zones)
        terrain_elev = 0.0
        if self.dem is not None:
            try:
                terrain_elev = self.dem.elevation(lat, lon)
                if np.isnan(terrain_elev):
                    terrain_elev = 0.0
            except Exception:
                terrain_elev = 0.0

        violations = []
        for zone in self.active_zones():
            v = zone.check_point(lat, lon, alt_amsl, terrain_elev)
            if v is not None:
                violations.append(v)

        return violations

    # ── Path checking ─────────────────────────────────────────────────

    def check_path(self, path) -> AirspaceReport:
        """
        Check a complete FlightPath against all active zones.

        Evaluates all waypoints in the path. For efficiency, uses
        the terrain elevations already computed by the TerrainAnalyzer
        if available.

        Parameters
        ----------
        path : FlightPath
            The path to check (must have waypoints populated).

        Returns
        -------
        AirspaceReport with aggregate violation metrics.
        """
        all_violations = []

        waypoints = path.get_waypoints()
        if not waypoints:
            return AirspaceReport(violations=[])

        for wp in waypoints:
            terrain_elev = wp.terrain_elev if not np.isnan(wp.terrain_elev) else 0.0
            if np.isnan(terrain_elev) and self.dem is not None:
                try:
                    terrain_elev = self.dem.elevation(wp.lat, wp.lon)
                except Exception:
                    terrain_elev = 0.0

            for zone in self.active_zones():
                v = zone.check_point(wp.lat, wp.lon, wp.alt, terrain_elev)
                if v is not None:
                    all_violations.append(v)

        return AirspaceReport(violations=all_violations)

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

        terrain_elev = 0.0
        if self.dem is not None:
            try:
                terrain_elev = self.dem.elevation(lat, lon)
                if np.isnan(terrain_elev):
                    terrain_elev = 0.0
            except Exception:
                terrain_elev = 0.0

        for zone in self.active_zones():
            if zone.zone_type not in (ZoneType.ALTITUDE_LIMIT,
                                       ZoneType.POPULATED_AREA):
                continue
            if not zone.horizontal_contains(lat, lon):
                continue

            if zone.is_agl:
                ceiling = terrain_elev + zone.altitude_ceiling_m
            else:
                ceiling = zone.altitude_ceiling_m

            min_ceiling = min(min_ceiling, ceiling)

        return min_ceiling

    def is_prohibited(self, lat: float, lon: float) -> bool:
        """Check if a horizontal position is inside any prohibited zone."""
        for zone in self.active_zones():
            if zone.zone_type in (ZoneType.PROHIBITED, ZoneType.ECOLOGICAL):
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
    >>> airspace = load_airspace_from_file('data/quito_airspace.json', dem=dem)
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

        zones.append(AirspaceZone(
            zone_id=zd['zone_id'],
            zone_type=zone_type,
            geometry=geometry,
            altitude_floor_m=zd.get('altitude_floor_m', 0.0),
            altitude_ceiling_m=zd.get('altitude_ceiling_m', 10000.0),
            is_agl=zd.get('is_agl', False),
            buffer_m=zd.get('buffer_m', 0.0),
            description=zd.get('description', ''),
        ))

    return AirspaceManager(zones=zones, dem=dem)


def build_airspace(dem=None) -> AirspaceManager:
    """
    Convenience function: load the Quito DMQ airspace from the bundled data file.

    Searches for 'quito_airspace.json' in common locations:
        data/quito_airspace.json
        ../data/quito_airspace.json
        quito_airspace.json

    For other regions, use load_airspace_from_file() directly with your own JSON.
    """
    import os
    search_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'data', 'quito_airspace.json'),
        'data/quito_airspace.json',
        '../data/quito_airspace.json',
        'quito_airspace.json',
    ]
    for p in search_paths:
        if os.path.exists(p):
            return load_airspace_from_file(p, dem=dem)

    raise FileNotFoundError(
        "Cannot find quito_airspace.json. Place it in data/ or provide "
        "a custom airspace file via load_airspace_from_file()."
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
