"""
Regression tests for the regulatory and corridor machinery.

Each test here pins a specific defect that made every optimized path
report as infeasible. They are cheap — no optimization runs — so they can
guard the invariants on every commit.

    pytest tests/ -q
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from raptor.airspace import (
    AirspaceManager, AirspaceZone, CircularZone, PolygonalZone, ZoneType,
    build_airspace, load_airspace_from_file,
)
from raptor.builder import FacilityNode
from raptor.config import MissionConstraints, UAVConfig
from raptor.corridor import build_corridor_profile, climb_limited_envelope
from raptor.dem import DEMInterface
from raptor.regulations import (
    PermissionClass, RDAC_101, OperationalContext,
)
from raptor.routed_path import RoutedPath
from raptor.terrain import TerrainAnalyzer

DEM_PATH = "data/dmq_dem.npz"
pytestmark = pytest.mark.skipif(
    not os.path.exists(DEM_PATH), reason="bundled DMQ DEM not available"
)


@pytest.fixture(scope="module")
def dem():
    return DEMInterface(DEM_PATH)


@pytest.fixture(scope="module")
def uav():
    return UAVConfig()


@pytest.fixture(scope="module")
def con():
    return MissionConstraints()


# ── The vertical corridor must have room to fly in ────────────────────────

def test_default_corridor_is_flyable(con):
    assert con.corridor_width("FW_CRUISE") > 0
    assert con.validate() == []


def test_empty_corridor_is_reported_not_silent():
    """A floor at or above the ceiling used to fail silently, every time."""
    bad = MissionConstraints(min_cruise_terrain_clearance=130.0, max_agl=122.0)
    problems = bad.validate()
    assert problems, "an impossible corridor must be reported"
    assert "empty" in problems[0]


# ── Terrain analysis ──────────────────────────────────────────────────────

def test_takeoff_and_landing_are_not_clearance_violations(dem, uav, con):
    """
    Pads sit at AGL 0. Demanding clearance there made every path — however
    well routed — infeasible at its own endpoints.
    """
    o = FacilityNode("O", -0.2000, -78.4930, dem.elevation(-0.2000, -78.4930))
    d = FacilityNode("D", -0.3242, -78.5493, dem.elevation(-0.3242, -78.5493))
    fp = RoutedPath(o, d, dem, uav, con, n_intermediate=1,
                    corridor_mode="agl").flight_path

    report = TerrainAnalyzer(dem, con).analyze(fp)
    assert report.n_exempt > 0, "terminal funnels must be exempted"
    assert report.is_feasible, "a terrain-following path should clear terrain"


def test_terrain_report_tracks_the_ceiling_too(dem, uav, con):
    o = FacilityNode("O", -0.2000, -78.4930, dem.elevation(-0.2000, -78.4930))
    d = FacilityNode("D", -0.3242, -78.5493, dem.elevation(-0.3242, -78.5493))

    agl = RoutedPath(o, d, dem, uav, con, n_intermediate=1,
                     corridor_mode="agl").flight_path
    amsl = RoutedPath(o, d, dem, uav, con, n_intermediate=1,
                      corridor_mode="amsl").flight_path

    ta = TerrainAnalyzer(dem, con)
    r_agl, r_amsl = ta.analyze(agl), ta.analyze(amsl)

    # The whole point: a constant-altitude cruise cannot hold an AGL band
    # over this relief, and a terrain-following one can.
    assert r_amsl.n_ceiling_violations > 0
    assert r_agl.n_ceiling_violations == 0
    assert r_agl.corridor_penalty < r_amsl.corridor_penalty


# ── AGL zones need real ground, not sea level ─────────────────────────────

def test_agl_zone_uses_dem_not_zero(dem):
    """
    With terrain taken as 0, a 122 m AGL ceiling becomes 122 m AMSL, and
    every point of a 2 800 m Quito route reads as ~2 700 m 'above' it.
    """
    zone = AirspaceZone(
        zone_id="AGL122", zone_type=ZoneType.ALTITUDE_LIMIT, geometry=None,
        altitude_ceiling_m=122.0, is_agl=True,
    )
    mgr = AirspaceManager([zone], dem=dem)

    lat, lon = -0.2000, -78.4930
    ground = dem.elevation(lat, lon)
    assert ground > 2000, "sanity: Quito is high"

    assert mgr.check_point(lat, lon, ground + 100.0) == []
    hits = mgr.check_point(lat, lon, ground + 200.0)
    assert len(hits) == 1
    assert hits[0].excess_m == pytest.approx(78.0, abs=1.0)


def test_agl_zone_is_skipped_where_terrain_is_unknown():
    """Outside the DEM there is no ground reference, so no verdict."""
    zone = AirspaceZone(
        zone_id="AGL122", zone_type=ZoneType.ALTITUDE_LIMIT, geometry=None,
        altitude_ceiling_m=122.0, is_agl=True,
    )
    assert zone.check_point(0.0, 0.0, 3000.0, terrain_elev=np.nan) is None


# ── Permission classes ────────────────────────────────────────────────────

def test_permit_zone_does_not_make_a_route_infeasible(dem):
    """RDAC 101.190 distinguishes 'needs authorisation' from 'never'."""
    from raptor.path import FlightPath
    from raptor.segments import FWCruise

    ground = dem.elevation(-0.2000, -78.4930)
    zone = AirspaceZone(
        zone_id="HOSP", zone_type=ZoneType.HOSPITAL,
        geometry=CircularZone(-0.2000, -78.4930, 5000.0),
        altitude_ceiling_m=9000.0,
    )
    assert zone.permission == PermissionClass.PERMIT

    fp = FlightPath(-0.2000, -78.4930, ground,
                    -0.1900, -78.4930, ground)
    fp.add_segment(FWCruise(ground_distance=1000.0, airspeed=25.0))
    for seg in fp.segments:                       # fly at a legal height
        st = seg.start_state
        seg.start_state = type(st)(
            lat=st.lat, lon=st.lon, alt=ground + 90.0,
            bearing=st.bearing, time=st.time, distance=st.distance,
        )

    report = AirspaceManager([zone], dem=dem).check_path(fp)
    assert report.permits_required == {"HOSP": "hospital"}
    assert report.feasible, "a permit zone is legal with authorisation"


def test_prohibited_zone_does_make_a_route_infeasible(dem):
    zone = AirspaceZone(
        zone_id="GOV", zone_type=ZoneType.GOVERNMENT,
        geometry=CircularZone(-0.2000, -78.4930, 1000.0),
        altitude_ceiling_m=9000.0,
    )
    assert zone.permission == PermissionClass.PROHIBITED
    hits = AirspaceManager([zone], dem=dem).check_point(-0.2000, -78.4930, 2900.0)
    assert hits and hits[0].is_hard


def test_authorized_zones_are_free_for_that_operation():
    zone = AirspaceZone(
        zone_id="HOSP", zone_type=ZoneType.HOSPITAL, geometry=None,
    )
    ctx = OperationalContext(authorized_zone_ids=["HOSP"])
    assert RDAC_101.resolve_permission(zone, ctx) == PermissionClass.FREE


def test_special_authorization_does_not_unlock_prohibited_airspace():
    """101.190(b) admits no exception."""
    zone = AirspaceZone(
        zone_id="SEC", zone_type=ZoneType.SECURITY, geometry=None,
    )
    ctx = OperationalContext(has_special_authorization=True)
    assert RDAC_101.resolve_permission(zone, ctx) == PermissionClass.PROHIBITED


# ── Corridor envelope ─────────────────────────────────────────────────────

def test_envelope_clears_terrain_and_respects_the_flight_envelope():
    terrain = np.array([100.0, 100, 300, 300, 100, 100, 250, 100])
    step = 100.0
    tan_up, tan_down = np.tan(np.radians(15.0)), np.tan(np.radians(12.0))

    env = climb_limited_envelope(terrain, step, tan_up, tan_down)

    assert np.all(env >= terrain - 1e-9)
    d = np.diff(env)
    assert d.max() <= step * tan_up + 1e-6
    assert (-d).max() <= step * tan_down + 1e-6


def test_corridor_sampling_follows_the_dem_grid(dem, uav, con):
    """
    Sampling at or above the DEM cell size steps over ridges between
    samples: the envelope then clears only the points it looked at, and
    the flown profile dips below its target height in between.
    """
    a, b = (-0.2000, -78.4930), (-0.2158, -78.4085)   # 597 m of relief
    prof = build_corridor_profile(
        dem, a, b, target_agl=60.0,
        max_climb_angle_deg=uav.fw_max_climb_angle,
        max_descent_angle_deg=uav.fw_max_descent_angle,
        max_agl=con.max_agl,
    )
    # Re-check the reference against terrain sampled far more finely
    # than the profile itself used.
    dense = dem.terrain_profile(a, b, n=800)
    d = np.asarray(dense["distances"], float)
    t = np.asarray(dense["elevations"], float)
    good = ~np.isnan(t)
    t = np.interp(d, d[good], t[good])
    ref = np.interp(d, prof.sample_distances, prof.sample_altitudes)

    assert (ref - t).min() > 55.0, (
        "reference dips between samples — corridor sampling is too coarse "
        "for the DEM grid"
    )


def test_terrain_following_holds_the_target_height(dem, uav, con):
    prof = build_corridor_profile(
        dem, (-0.2000, -78.4930), (-0.2641, -78.5504),
        target_agl=90.0,
        max_climb_angle_deg=uav.fw_max_climb_angle,
        max_descent_angle_deg=uav.fw_max_descent_angle,
        max_agl=con.max_agl,
    )
    assert prof.agl.min() >= 90.0 - 1e-6, "never below the target"
    assert prof.n_breakpoints <= 12, "profile must stay cheap to fly"


# ── Path construction ─────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["agl", "amsl"])
@pytest.mark.parametrize("n_wp", [0, 1, 2])
def test_path_ends_at_the_destination(dem, uav, con, mode, n_wp):
    a, b = (-0.2641, -78.5504), (-0.3833, -78.3889)
    o = FacilityNode("O", a[0], a[1], dem.elevation(*a))
    d = FacilityNode("D", b[0], b[1], dem.elevation(*b))

    fp = RoutedPath(o, d, dem, uav, con, n_intermediate=n_wp,
                    corridor_mode=mode).flight_path
    end = fp.end_state

    assert DEMInterface.haversine(end.lat, end.lon, d.lat, d.lon) < 25.0
    assert abs(end.alt - d.ground_elev) < 5.0


def test_vectorised_containment_matches_scalar(dem):
    """The fast path must agree with the slow one it replaced."""
    airspace = build_airspace(dem=dem)
    rng = np.random.default_rng(0)
    lats = rng.uniform(-0.40, 0.05, 500)
    lons = rng.uniform(-78.70, -78.30, 500)

    for zone in airspace.zones:
        if zone.is_global:
            continue
        fast = zone.geometry.contains_points(lats, lons)
        slow = np.array([zone.geometry.contains_point(a, b)
                         for a, b in zip(lats, lons)])
        assert np.array_equal(fast, slow), zone.zone_id
