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

DEM_PATH = "data/dmq_dem_30m.npz"
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
    assert prof.n_breakpoints <= 32, "profile must stay cheap to fly"


def test_flown_profile_never_dips_below_the_reference(dem, uav, con):
    """
    The breakpoint polyline is what the aircraft flies; the dense
    envelope is only what it was simplified from. A chord that cuts
    below the envelope flies lower than the corridor floor allows, and
    for a while the profile reported that as clean because the report
    was computed from the envelope rather than from the chord.
    """
    for a, b in [((-0.2000, -78.4930), (-0.3242, -78.5493)),
                 ((-0.2528, -78.5203), (-0.1440, -78.4230)),
                 ((-0.2100, -78.5100), (-0.2600, -78.6200))]:
        prof = build_corridor_profile(
            dem, a, b, target_agl=90.0,
            max_climb_angle_deg=uav.fw_max_climb_angle,
            max_descent_angle_deg=uav.fw_max_descent_angle,
            max_agl=con.max_agl,
        )
        dip = float(np.max(prof.envelope_altitudes - prof.sample_altitudes))
        assert dip <= 0.5, f"flown profile dips {dip:.1f} m below the envelope"
        assert prof.agl.min() >= 90.0 - 1e-6


def test_ceiling_excess_is_measured_on_the_flown_profile(dem, uav, con):
    """
    Simplification lifts the profile to preserve clearance, and that
    lift eats ceiling headroom. Reporting the envelope's exceedance
    instead of the flown one hides exactly the cost that was just
    incurred.
    """
    prof = build_corridor_profile(
        dem, (-0.2000, -78.4930), (-0.3242, -78.5493), target_agl=90.0,
        max_climb_angle_deg=uav.fw_max_climb_angle,
        max_descent_angle_deg=uav.fw_max_descent_angle,
        max_agl=con.max_agl, max_breakpoints=6, simplify_tol_m=20.0,
    )
    flown_excess = float(np.max(np.maximum(prof.agl - con.max_agl, 0.0)))
    assert prof.ceiling_excess_m == pytest.approx(flown_excess, abs=1e-6)
    # Starved of breakpoints, the flown profile must be worse than the
    # envelope — if these agree, the report is reading the wrong array.
    env_excess = float(np.max(np.maximum(
        prof.envelope_altitudes - prof.sample_terrain - con.max_agl, 0.0)))
    assert prof.ceiling_excess_m > env_excess


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


# ── Public API ────────────────────────────────────────────────────────────

def test_everything_in_all_is_importable():
    """
    A name in ``__all__`` that is not actually exported turns every README
    snippet using it into a broken example — which is how the docs came to
    document an import of MissionPlanner that failed.
    """
    import raptor
    missing = [n for n in raptor.__all__ if not hasattr(raptor, n)]
    assert missing == []


def test_readme_python_blocks_reference_real_api():
    """
    Parse every python block in the README and check the names it imports
    from raptor exist. Cheap, and it catches documentation drifting away
    from the package.
    """
    import ast
    import os
    import re

    if not os.path.exists("README.md"):
        pytest.skip("README not present")

    blocks = re.findall(r"```python\n(.*?)```", open("README.md").read(), re.S)
    assert blocks, "README should contain python examples"

    missing = []
    for block in blocks:
        tree = ast.parse(block)          # also asserts the snippet parses
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if not (node.module or "").startswith("raptor"):
                continue
            mod = __import__(node.module, fromlist=["x"])
            for alias in node.names:
                if alias.name != "*" and not hasattr(mod, alias.name):
                    missing.append(f"{node.module}.{alias.name}")

    assert missing == [], f"README references names that do not exist: {missing}"


def test_airspace_feasible_is_not_the_whole_verdict(dem, uav, con):
    """
    `AirspaceReport.feasible` knows nothing about terrain clearance.
    Reading it alone as "is this route legal" prints LEGAL for a path
    that flies into a hill — which is how a report came to contradict
    itself two lines apart.
    """
    from raptor.airspace import AirspaceManager
    from raptor.path import FlightPath
    from raptor.segments import FWCruise
    from raptor.terrain import TerrainAnalyzer

    a = (-0.2641, -78.5504)
    ground = float(dem.elevation(*a))

    # Straight and level at pad height across rising ground.
    fp = FlightPath(a[0], a[1], ground, -0.2358, -78.5886, ground)
    fp.add_segment(FWCruise(ground_distance=4000.0, airspeed=30.0))

    empty = AirspaceManager([], dem=dem)
    report = empty.check_path(fp)
    terrain = TerrainAnalyzer(dem, con).analyze(fp)

    assert report.feasible            # no zones, so airspace is clear
    assert not terrain.is_feasible    # but it is flying into the terrain


def test_facility_elevations_are_checked_against_the_terrain(dem):
    """
    Pad elevations are typed in by hand and nothing forces them to agree
    with the DEM. When they disagree the departure and arrival phases are
    built against the wrong ground — CS.Conocoto carried a 23 m error,
    a fifth of the whole legal corridor.
    """
    from raptor.builder import FacilityNode, check_facility_elevations

    lat, lon = -0.3080, -78.4710
    truth = float(dem.elevation(lat, lon))

    good = FacilityNode("right", lat, lon, truth)
    assert check_facility_elevations([good], dem) == []

    wrong = FacilityNode("CS.Conocoto", lat, lon, truth + 23.0)
    findings = check_facility_elevations([wrong], dem)
    assert findings and "CS.Conocoto" in findings[0]
    assert "+23" in findings[0]


def test_elevation_check_reports_worst_first(dem):
    from raptor.builder import FacilityNode, check_facility_elevations
    lat, lon = -0.2000, -78.4930
    truth = float(dem.elevation(lat, lon))
    findings = check_facility_elevations([
        FacilityNode("small", lat, lon, truth + 15.0),
        FacilityNode("large", lat, lon, truth + 60.0),
    ], dem)
    assert findings[0].startswith("large")


def test_catalogue_takes_elevations_from_the_terrain(dem):
    """
    The elevations typed into scenarios.py are mostly wrong — thirteen of
    fifteen disagree with the DEM by more than 10 m, one by 499 m — and
    departure and arrival are built from them. `update_facility_elevations`
    existed but nothing called it, so the catalogue always used the typed
    values.
    """
    from raptor.builder import check_facility_elevations
    from raptor.scenarios import ALL_FACILITIES, build_scenario_catalog

    build_scenario_catalog(dem)          # must correct them
    assert check_facility_elevations(ALL_FACILITIES, dem, tolerance_m=1.0) == []


def test_update_reports_what_it_moved(dem):
    from raptor.scenarios import ALL_FACILITIES, update_facility_elevations
    ALL_FACILITIES[0].ground_elev = 1.0          # deliberately absurd
    changed = update_facility_elevations(dem)
    assert any(name == ALL_FACILITIES[0].name for name, _, _ in changed)
    assert ALL_FACILITIES[0].ground_elev > 1000.0


def test_documented_commands_name_modules_that_exist():
    """
    Every ``python -m scripts.X`` or ``python -m examples.X`` in the
    README or the docs must name a module that is actually there.

    Cheap, and it catches the specific way documentation rots: a script
    is renamed or deleted and the command that invoked it lives on,
    looking authoritative.
    """
    import glob
    import os
    import re

    docs = ["README.md"] + sorted(glob.glob("docs/*.md")) + ["CHANGELOG.md"]
    docs = [d for d in docs if os.path.exists(d)]
    assert docs, "no documentation found"

    missing = []
    for doc in docs:
        text = open(doc, encoding="utf-8").read()
        for mod in re.findall(r"python -m ((?:scripts|examples)\.[\w.]+)", text):
            path = mod.replace(".", "/") + ".py"
            if not os.path.exists(path):
                missing.append(f"{doc}: {mod} -> {path}")
    assert missing == [], "documented commands with no module:\n" + "\n".join(missing)


def test_documented_figures_are_ones_make_figures_can_produce():
    """
    Figure filenames quoted in the docs have to be ones the figure
    script actually writes, and ``--only`` names have to be real keys.
    """
    import glob
    import os
    import re

    from scripts.make_figures import FIGURES

    docs = ["README.md"] + sorted(glob.glob("docs/*.md"))
    docs = [d for d in docs if os.path.exists(d)]

    source = open("scripts/make_figures.py", encoding="utf-8").read()
    produced = set(re.findall(r'save\(fig, out, "([^"]+)"', source))

    bad_files, bad_keys = [], []
    for doc in docs:
        text = open(doc, encoding="utf-8").read()
        for name in re.findall(r"figures/current/([\w./-]+\.png)", text):
            if name not in produced:
                bad_files.append(f"{doc}: {name}")
        for run in re.findall(r"make_figures[^\n]*--only ([\w ]+)", text):
            for key in run.split():
                if key not in FIGURES:
                    bad_keys.append(f"{doc}: --only {key}")

    assert bad_files == [], "figures named in docs that nothing produces:\n" \
        + "\n".join(bad_files)
    assert bad_keys == [], "unknown --only keys in docs:\n" + "\n".join(bad_keys)


# ── Scope of a compliance verdict ─────────────────────────────────────────

def test_a_verdict_declares_the_rules_it_did_not_test():
    """
    ``min_separation_persons_m``, ``min_separation_buildings_m`` and
    ``min_visibility_m`` sat in the regulatory profile, read by nothing,
    for the life of the package. They cannot be checked — doing so needs
    building footprints, crowd positions and a weather feed, none of
    which RAPTOR has.

    An unenforceable rule silently absent from a report is
    indistinguishable from a rule that passed. So the profile now names
    them and the assessment carries them, which turns a limitation into
    a declared one: "compliant" from this package means compliant with
    the rules it can see.
    """
    from raptor.regulations import RDAC_101

    rules = RDAC_101.unenforceable_rules()
    assert rules
    joined = " ".join(rules)
    assert "101.160" in joined and "101.215" in joined
    assert "not modelled" in joined


def test_the_unchecked_rules_reach_the_printed_report(dem):
    from raptor.airspace import build_airspace
    from raptor.compliance import assess_compliance
    from raptor.config import MissionConstraints, UAVConfig
    from raptor.builder import FacilityNode
    from raptor.routed_path import RoutedPath
    from raptor.terrain import TerrainAnalyzer
    from raptor.vehicles import get_vehicle

    con = MissionConstraints()
    ac = get_vehicle("va23").build_aero(verbose=False)
    uav = UAVConfig.for_vehicle(ac.with_payload(1.5), altitude=2800.0)
    a, b = (-0.2000, -78.4930), (-0.1273, -78.4977)
    o = FacilityNode("O", a[0], a[1], dem.elevation(*a))
    d = FacilityNode("D", b[0], b[1], dem.elevation(*b))
    fp = RoutedPath(o, d, dem, uav, con, n_intermediate=1,
                    corridor_mode="agl").flight_path

    assessment = assess_compliance(
        fp, TerrainAnalyzer(dem, con).analyze(fp),
        build_airspace(dem=dem).check_path(fp), constraints=con)

    # And the mistake that led here must not be silent.
    with pytest.raises(TypeError, match="RegulatoryProfile"):
        assess_compliance(fp, TerrainAnalyzer(dem, con).analyze(fp),
                          build_airspace(dem=dem).check_path(fp), con)

    assert assessment.unchecked_rules
    report = assessment.filing_summary()
    assert "did NOT test" in report
    assert "101.160" in report


def test_range_limit_is_enforced_not_merely_declared():
    """
    ``MissionConstraints.max_range`` was declared, documented and read by
    nothing, so a route could be twice the aircraft's stated range with
    no part of the model saying so.
    """
    from raptor.config import MissionConstraints, UAVConfig
    from raptor.path import FlightPath
    from raptor.segments import FWCruise
    from raptor.terrain import check_path_envelope
    from raptor.vehicles import get_vehicle

    ac = get_vehicle("va23").build_aero(verbose=False)
    uav = UAVConfig.for_vehicle(ac, altitude=2800.0)

    path = FlightPath(-0.20, -78.49, 2800.0, -0.21, -78.50, 2800.0)
    for _ in range(4):
        path.add_segment(FWCruise(ground_distance=10_000.0,
                                  airspeed=uav.fw_cruise_airspeed))

    within = MissionConstraints(max_range=60_000.0)
    beyond = MissionConstraints(max_range=25_000.0)
    assert not any("range limit" in p
                   for p in check_path_envelope(path, uav, within))
    assert any("range limit" in p
               for p in check_path_envelope(path, uav, beyond))
