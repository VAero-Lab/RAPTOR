"""
Regression tests for wind, vortex-ring margin and per-stretch waivers.
"""

from __future__ import annotations

import numpy as np
import pytest

from raptor.compliance import Excursion, HeightWaiver
from raptor.config import MissionConstraints, UAVConfig
from raptor.vehicles import get_vehicle
from raptor.energy import check_descent_rates, vortex_ring_margin
from raptor.vehicles import AircraftEnergyParams
from raptor.wind import CALM, WindField

ALT = 2800.0


# ── Wind ──────────────────────────────────────────────────────────────────

def test_wind_is_stronger_at_corridor_height_than_at_the_anemometer():
    """
    A 10 m reading understates what a 90 m corridor flies in. Planning
    on the reported figure understates every headwind.
    """
    # Same surface at the mast and along the corridor: the profile is a
    # single logarithm and speed_ref is recovered exactly at 10 m.
    w = WindField(speed_ref=6.0, reference_height_m=10.0,
                  roughness_length_m=0.3, reference_roughness_m=0.3)
    assert w.speed_at(10.0) == pytest.approx(6.0, rel=1e-6)
    assert w.speed_at(90.0) > 1.3 * w.speed_at(10.0)
    assert w.speed_at(120.0) > w.speed_at(60.0)


def test_a_measurement_over_one_surface_is_converted_to_another():
    """
    An airport anemometer stands on mown grass; the corridor crosses a
    city. Reading the measurement straight up with the *corridor's*
    roughness gives 6.0 m/s at 90 m where the correct conversion gives
    4.0 — a 47 % error in every crab angle and round-trip penalty
    computed from it.
    """
    naive = WindField(speed_ref=3.08, reference_height_m=10.0,
                      roughness_length_m=1.0, reference_roughness_m=1.0)
    proper = WindField(speed_ref=3.08, reference_height_m=10.0,
                       roughness_length_m=1.0, reference_roughness_m=0.03)

    assert naive.speed_at(90.0) > 1.4 * proper.speed_at(90.0)

    # Rougher ground below means slower wind at corridor height for the
    # same mast reading — the city eats the momentum.
    smooth = WindField(speed_ref=3.08, roughness_length_m=0.03,
                       reference_roughness_m=0.03)
    assert proper.speed_at(90.0) < smooth.speed_at(90.0)


def test_quito_defaults_carry_their_source():
    """
    A wind default with no attribution is a guess wearing a number. The
    two sources for Quito disagree by a factor of 1.9 in speed and by
    250 degrees in direction, so which one a result rests on has to be
    recoverable from the result.
    """
    from raptor.wind import (
        QUITO_ANNUAL_MEAN, QUITO_CALMEST_MONTH, QUITO_REANALYSIS,
        QUITO_WINDIEST_MONTH,
    )

    for w in (QUITO_ANNUAL_MEAN, QUITO_WINDIEST_MONTH,
              QUITO_CALMEST_MONTH, QUITO_REANALYSIS):
        assert w.source and w.source != "unspecified"

    assert (QUITO_CALMEST_MONTH.speed_ref
            < QUITO_ANNUAL_MEAN.speed_ref
            < QUITO_WINDIEST_MONTH.speed_ref)
    # The disagreement is the point: keep both, do not average them.
    assert QUITO_ANNUAL_MEAN.speed_ref > 1.5 * QUITO_REANALYSIS.speed_ref


def test_beaufort_ratings_are_read_at_the_bottom_of_the_band():
    """
    "Wind resistance level 5" is 8.0-10.7 m/s. An aircraft rated to
    force 5 has been shown to handle 8.0, not 10.7, and quoting the top
    of the band turns a rating into a claim nobody made.
    """
    from raptor.wind import beaufort_limit_ms
    from raptor.vehicles import get_vehicle

    assert beaufort_limit_ms(5) == pytest.approx(8.0)
    assert beaufort_limit_ms(5, conservative=False) == pytest.approx(10.7)
    assert beaufort_limit_ms(6) > beaufort_limit_ms(5)

    assert get_vehicle("va25").wind_limit_ms > get_vehicle("va23").wind_limit_ms


def test_meteorological_direction_sign():
    """
    Wind *from* the north pushes an aircraft south. Getting this backwards
    silently turns every headwind into a tailwind.
    """
    w = WindField(speed_ref=10.0, direction_deg=0.0, reference_height_m=10.0)
    north, east = w.components(10.0)
    assert north < 0 and abs(east) < 1e-6

    # Flying north into a northerly is a headwind.
    along, _ = w.resolve(bearing_deg=0.0, agl_m=10.0)
    assert along < 0


def test_crosswind_costs_ground_speed_through_the_crab_angle():
    w = WindField(speed_ref=8.0, direction_deg=0.0, reference_height_m=90.0)
    gs = w.ground_speed(airspeed=30.0, bearing_deg=90.0, agl_m=90.0)
    assert gs < 30.0                         # pure crosswind still costs
    assert abs(w.crab_angle_deg(30.0, 90.0, 90.0)) > 5.0


def test_wind_stronger_than_airspeed_is_untrackable():
    w = WindField(speed_ref=40.0, direction_deg=0.0, reference_height_m=90.0)
    assert w.ground_speed(airspeed=20.0, bearing_deg=90.0, agl_m=90.0) == 0.0


def test_round_trip_loses_to_wind_both_ways():
    """
    The slow leg lasts longer than the fast leg saves, so a round trip
    always costs time — a symmetry a per-leg average would hide.
    """
    w = WindField(speed_ref=8.0, direction_deg=90.0, reference_height_m=90.0)
    out = w.ground_speed(30.0, 90.0, 90.0)
    back = w.ground_speed(30.0, 270.0, 90.0)
    d = 10_000.0
    assert d / out + d / back > 2 * d / 30.0


def test_calm_wind_changes_nothing():
    assert CALM.speed_at(90.0) == 0.0
    assert CALM.ground_speed(25.0, 45.0, 90.0) == pytest.approx(25.0)


# ── Vortex ring state ─────────────────────────────────────────────────────

def test_vortex_ring_band_is_where_momentum_theory_fails():
    ac = AircraftEnergyParams()
    v_h = vortex_ring_margin(ac, 0.0, ALT)["hover_induced_velocity"]
    assert vortex_ring_margin(ac, 0.20 * v_h, ALT)["safe"]
    assert not vortex_ring_margin(ac, 0.8 * v_h, ALT)["safe"]
    assert "windmill" in vortex_ring_margin(ac, 2.0 * v_h, ALT)["state"]


def test_derived_descent_rate_stays_out_of_the_band():
    """
    Larger rotors have a *lower* induced velocity and therefore a lower
    safe descent rate. Which means the limit cannot be a constant: the
    hand-typed 2.0 m/s was outside the band for a 12 kg aircraft on
    0.82 m² of disk and inside it for the 10 kg VA23 on 0.98 m² — the
    safer aircraft, judged by the wrong number.

    ``UAVConfig.for_vehicle`` derives it, so it moves with the rotors.
    """
    for name in ("va17", "va23", "va25"):
        ac = get_vehicle(name).build_aero(verbose=False)
        uav = UAVConfig.for_vehicle(ac, altitude=ALT)
        margin = vortex_ring_margin(ac, uav.vtol_max_descent_rate, ALT)
        assert margin["safe"], (
            f"{name}: {uav.vtol_max_descent_rate:.2f} m/s is inside the "
            f"vortex-ring band, which starts at {margin['vrs_lower']:.2f} m/s"
        )
        assert uav.vtol_max_descent_rate > 0.3


def test_hand_typed_manoeuvre_limits_are_flagged_as_contradictory():
    """
    Turn radius, bank angle and cruise speed are three names for two
    independent facts; so are climb angle, climb rate and speed. The
    class defaults still carry the historical hand-typed values, and
    the point of ``manoeuvre_problems`` is that they do not survive
    being checked against each other.
    """
    problems = UAVConfig().manoeuvre_problems()
    assert any("turn_radius" in p for p in problems)
    assert any("climb_rate" in p for p in problems)

    # Derived from an aircraft, they agree by construction.
    derived = UAVConfig.for_vehicle(
        get_vehicle("va23").build_aero(verbose=False), altitude=ALT)
    assert derived.manoeuvre_problems() == []


def test_bigger_rotors_lower_the_safe_descent_rate():
    small = AircraftEnergyParams(rotor_diameter_m=0.35)
    big = AircraftEnergyParams(rotor_diameter_m=0.70)
    assert (vortex_ring_margin(big, 0.0, ALT)["vrs_lower"]
            < vortex_ring_margin(small, 0.0, ALT)["vrs_lower"])


def test_descent_check_flags_an_unsafe_path():
    from raptor.path import FlightPath
    from raptor.segments import VTOLDescend
    ac = AircraftEnergyParams()
    fp = FlightPath(-0.2, -78.5, 2800.0, -0.2, -78.5, 2700.0)
    fp.add_segment(VTOLDescend(altitude_loss=100.0, descent_rate=6.0))
    assert check_descent_rates(fp, ac)


# ── Per-stretch waivers ───────────────────────────────────────────────────

def _waiver():
    return HeightWaiver(
        limit_agl_m=122.0, requested_agl_m=139.0, max_excess_m=17.0,
        total_length_m=490.0, route_length_m=5310.0,
        excursions=[
            Excursion(1670, 1880, 139.0, 17.0, -0.2535, -78.5620),
            Excursion(2310, 2450, 136.0, 14.0, -0.2497, -78.5662),
            Excursion(3230, 3370, 131.0, 9.0, -0.2449, -78.5732),
        ],
    )


def test_per_stretch_requests_only_what_each_stretch_needs():
    w = _waiver()
    stretches = w.per_stretch()
    assert len(stretches) == 3
    # The last stretch needs less relief than the first.
    assert stretches[-1]["requested_agl_m"] < stretches[0]["requested_agl_m"]
    # Every request covers its own peak.
    for s, e in zip(stretches, w.excursions):
        assert s["requested_agl_m"] >= e.max_agl_m


def test_per_stretch_is_a_smaller_ask_than_a_blanket_ceiling():
    """
    A route-wide ceiling requests relief over the 90 % of track that never
    needed any — easy to ask for, hard to be granted.
    """
    w = _waiver()
    assert w.relief_saved_by_stretching() > 0
    blanket = w.max_excess_m * w.route_length_m / 1000.0
    assert w.relief_saved_by_stretching() > 0.5 * blanket


def test_requested_heights_are_rounded_up_not_to_nearest():
    """A request must cover its excursion, never sit a metre under it."""
    w = _waiver()
    for s, e in zip(w.per_stretch(granularity_m=10.0), w.excursions):
        assert s["requested_agl_m"] >= e.max_agl_m
        assert s["requested_agl_m"] % 10.0 == pytest.approx(0.0)


# ── Unavoidable permits ───────────────────────────────────────────────────

def test_permits_containing_a_terminal_are_flagged_unavoidable():
    """
    A medical network is built around hospitals, and hospitals are permit
    zones. No route from a pad inside one can leave it, so a
    permit-avoidance study run from there measures nothing.
    """
    import os
    if not os.path.exists("data/dmq_dem_30m.npz"):
        pytest.skip("30 m DEM not built")

    from raptor.airspace import build_airspace
    from raptor.builder import FacilityNode
    from raptor.compliance import assess_compliance
    from raptor.config import MissionConstraints, UAVConfig
    from raptor.dem import DEMInterface
    from raptor.routed_path import RoutedPath
    from raptor.terrain import TerrainAnalyzer

    dem = DEMInterface("data/dmq_dem_30m.npz")
    uav, con = UAVConfig(), MissionConstraints()
    airspace = build_airspace(dem=dem)

    # Hospital Espejo sits inside the Baca Ortiz hospital buffer.
    espejo = (-0.2000, -78.4930)
    zone = airspace.get_zone("HOSPITAL_Hospital_de_Ninos_Baca_Ortiz")
    assert zone is not None
    assert zone.horizontal_contains(*espejo)

    o = FacilityNode("O", *espejo, dem.elevation(*espejo))
    d = FacilityNode("D", -0.2158, -78.4085,
                     dem.elevation(-0.2158, -78.4085))
    fp = RoutedPath(o, d, dem, uav, con, n_intermediate=2,
                    corridor_mode="agl").flight_path

    a = assess_compliance(fp, TerrainAnalyzer(dem, con).analyze(fp),
                          airspace.check_path(fp), constraints=con,
                          airspace=airspace)
    assert "HOSPITAL_Hospital_de_Ninos_Baca_Ortiz" in a.unavoidable_permits


def test_unavoidable_needs_the_airspace_to_be_passed():
    """Without the zone geometry the question cannot be answered."""
    from raptor.compliance import ComplianceAssessment, ComplianceLevel
    a = ComplianceAssessment(level=ComplianceLevel.AUTHORIZATION_REQUIRED,
                             permits_required={"Z": "hospital"})
    assert a.unavoidable_permits == {}


# ── Wind applied to a real path ───────────────────────────────────────────

def test_wind_penalty_report_on_a_built_path():
    import os
    if not os.path.exists("data/dmq_dem_30m.npz"):
        pytest.skip("30 m DEM not built")

    from raptor.builder import FacilityNode
    from raptor.config import MissionConstraints
    from raptor.dem import DEMInterface
    from raptor.routed_path import RoutedPath
    from raptor.wind import CALM, WindField, wind_penalty_report

    dem = DEMInterface("data/dmq_dem_30m.npz")
    uav, con = UAVConfig(), MissionConstraints()
    o = FacilityNode("O", -0.2000, -78.4930, dem.elevation(-0.2000, -78.4930))
    d = FacilityNode("D", -0.2158, -78.4085, dem.elevation(-0.2158, -78.4085))
    fp = RoutedPath(o, d, dem, uav, con, n_intermediate=2,
                    corridor_mode="agl").flight_path

    calm = wind_penalty_report(fp, CALM, uav, dem)
    assert calm["time_penalty_s"] == 0.0

    windy = wind_penalty_report(fp, WindField(speed_ref=8.0,
                                              direction_deg=45.0), uav, dem)
    assert windy["feasible"], "an ordinary breezy day is not infeasible"
    assert windy["time_penalty_s"] > 0        # wind always costs something
    assert windy["max_crab_deg"] > 0
    # Rotor-borne phases drift rather than failing, and that is reported
    # separately — a transition at 10 m/s is not judged by the fixed-wing
    # trackability criterion.
    assert windy["vtol_drift_m"] > 0


def test_a_wind_stronger_than_the_aircraft_is_reported_infeasible():
    import os
    if not os.path.exists("data/dmq_dem_30m.npz"):
        pytest.skip("30 m DEM not built")

    from raptor.builder import FacilityNode
    from raptor.config import MissionConstraints
    from raptor.dem import DEMInterface
    from raptor.routed_path import RoutedPath
    from raptor.wind import WindField, wind_penalty_report

    dem = DEMInterface("data/dmq_dem_30m.npz")
    uav, con = UAVConfig(), MissionConstraints()
    o = FacilityNode("O", -0.2000, -78.4930, dem.elevation(-0.2000, -78.4930))
    d = FacilityNode("D", -0.2158, -78.4085, dem.elevation(-0.2158, -78.4085))
    fp = RoutedPath(o, d, dem, uav, con, n_intermediate=1,
                    corridor_mode="agl").flight_path

    # A gale the aircraft cannot out-fly: untrackable, not merely slow.
    r = wind_penalty_report(fp, WindField(speed_ref=60.0, direction_deg=0.0),
                            uav, dem)
    assert not r["feasible"]
    assert r["untrackable_segments"] > 0


def test_verdict_never_hides_a_terrain_bust():
    """
    ComplianceLevel is purely regulatory — clearance is a safety
    constraint and no filing covers flying into a hill. But the one-line
    verdict is what tables print, and reporting "AUTHORISED" for a route
    with a waypoint below the ground is worse than reporting it illegal.
    """
    from raptor.compliance import ComplianceAssessment, ComplianceLevel

    unsafe = ComplianceAssessment(
        level=ComplianceLevel.AUTHORIZATION_REQUIRED,
        permits_required={"Z": "hospital"},
        clearance_violations=1, min_agl_m=-1.0,
    )
    assert not unsafe.safe
    assert not unsafe.flyable                    # legal but not flyable
    assert "UNSAFE" in unsafe.verdict()
    assert "SAFETY" in unsafe.filing_summary().splitlines()[0]

    safe = ComplianceAssessment(
        level=ComplianceLevel.AUTHORIZATION_REQUIRED,
        permits_required={"Z": "hospital"},
        clearance_violations=0,
    )
    assert safe.safe and safe.flyable
    assert "AUTHORISED" in safe.verdict()


def test_prohibited_airspace_is_still_not_flyable_when_safe():
    from raptor.compliance import ComplianceAssessment, ComplianceLevel
    a = ComplianceAssessment(level=ComplianceLevel.NON_COMPLIANT,
                             blocking_zones=["GOV"], clearance_violations=0)
    assert a.safe                    # clears the terrain
    assert not a.flyable             # but enters prohibited airspace


def test_verdict_never_raises_on_an_inconsistent_assessment():
    """
    A one-line summary is the last place that should throw. A level of
    WAIVER_REQUIRED without an attached waiver is an inconsistent state,
    and the right response is to say so, not to crash whatever was
    printing a table.
    """
    from raptor.compliance import ComplianceAssessment, ComplianceLevel
    a = ComplianceAssessment(level=ComplianceLevel.WAIVER_REQUIRED)
    assert "not quantified" in a.verdict()
    assert a.filing_summary()          # and the long form survives too


@pytest.mark.parametrize("level", list(__import__(
    "raptor.compliance", fromlist=["ComplianceLevel"]).ComplianceLevel))
@pytest.mark.parametrize("violations", [0, 3])
def test_verdict_and_summary_survive_every_combination(level, violations):
    from raptor.compliance import ComplianceAssessment
    a = ComplianceAssessment(level=level, clearance_violations=violations,
                             min_agl_m=-2.0 if violations else float("nan"),
                             blocking_zones=["GOV"], permits_required={"Z": "x"})
    assert isinstance(a.verdict(), str) and a.verdict()
    assert isinstance(a.filing_summary(), str) and a.filing_summary()


# ── Envelope checks that nothing used to call ─────────────────────────────

def test_path_segments_stay_inside_the_aircraft_envelope():
    """
    ``UAVConfig`` carried six validators — airspeed, climb and descent
    angle, VTOL climb and descent rate, altitude — and nothing called any
    of them. The builder happened to produce segments inside the
    envelope, but that was a property of the builder rather than a
    checked fact, and it would have gone on being true silently right up
    until it stopped.
    """
    from raptor.builder import FacilityNode
    from raptor.dem import DEMInterface, find_dem
    from raptor.routed_path import RoutedPath
    from raptor.terrain import check_path_envelope

    dem = DEMInterface(find_dem())
    con = MissionConstraints()
    ac = get_vehicle("va23").build_aero(verbose=False)
    uav = UAVConfig.for_vehicle(ac.with_payload(1.5), altitude=2800.0)

    for a, b in [((-0.2641, -78.5504), (-0.3833, -78.3889)),
                 ((-0.2000, -78.4930), (-0.3242, -78.5493))]:
        o = FacilityNode("O", a[0], a[1], dem.elevation(*a))
        d = FacilityNode("D", b[0], b[1], dem.elevation(*b))
        fp = RoutedPath(o, d, dem, uav, con, n_intermediate=1,
                        corridor_mode="agl").flight_path
        assert check_path_envelope(fp, uav) == []


def test_envelope_check_catches_a_segment_that_is_out():
    """A check that cannot fail is not a check."""
    from raptor.path import FlightPath
    from raptor.segments import FWClimb, FWCruise
    from raptor.terrain import check_path_envelope

    ac = get_vehicle("va23").build_aero(verbose=False)
    uav = UAVConfig.for_vehicle(ac, altitude=ALT)

    path = FlightPath(-0.20, -78.49, 2800.0, -0.21, -78.50, 2800.0)
    path.add_segment(FWCruise(ground_distance=2000.0,
                              airspeed=uav.fw_max_airspeed + 12.0))
    path.add_segment(FWClimb(altitude_gain=300.0,
                             climb_angle_deg=uav.fw_max_climb_angle + 8.0,
                             airspeed=uav.fw_cruise_airspeed))

    problems = check_path_envelope(path, uav)
    assert any("airspeed" in p for p in problems)
    assert any("climb angle" in p for p in problems)


def test_segment_budget_is_reported_not_enforced():
    """
    ``max_path_segments`` was 20 and nothing read it. A
    terrain-following leg over 30 m terrain is forty segments, so the
    constraint was both unenforced and wrong. It is a cost bound now,
    reported rather than imposed — too many segments makes the search
    slow, not the route illegal.
    """
    from raptor.path import FlightPath
    from raptor.segments import FWCruise
    from raptor.terrain import check_path_envelope

    ac = get_vehicle("va23").build_aero(verbose=False)
    uav = UAVConfig.for_vehicle(ac, altitude=ALT)
    con = MissionConstraints(max_path_segments=5)

    path = FlightPath(-0.20, -78.49, 2800.0, -0.21, -78.50, 2800.0)
    for _ in range(9):
        path.add_segment(FWCruise(ground_distance=500.0,
                                  airspeed=uav.fw_cruise_airspeed))

    assert check_path_envelope(path, uav) == []          # nothing is illegal
    problems = check_path_envelope(path, uav, con)
    assert any("segment budget" in p or "budget" in p for p in problems)


# ── Mission planner accounting ────────────────────────────────────────────

def _one_leg_mission():
    from raptor.airspace import build_airspace
    from raptor.dem import DEMInterface, find_dem
    from raptor.mission_planner import MissionPlanner
    from raptor.missions import one_way
    from raptor.scenarios import (
        HOSPITAL_EUGENIO_ESPEJO, HOSPITAL_PABLO_ARTURO_SUAREZ,
    )

    dem = DEMInterface(find_dem())
    con = MissionConstraints()
    ac = get_vehicle("va23").build_aero(verbose=False)
    uav = UAVConfig.for_vehicle(ac.with_payload(1.5), altitude=ALT)
    planner = MissionPlanner(dem, uav, con, ac, airspace=build_airspace(dem=dem))
    mission = one_way(HOSPITAL_EUGENIO_ESPEJO, HOSPITAL_PABLO_ARTURO_SUAREZ,
                      payload_kg=2.0)
    result = planner.plan(mission, n_intermediate=1, maxiter=5, popsize=5,
                          seed=42, verbose=False)
    return ac, planner, result


def test_planner_uses_the_aircraft_pack_not_a_constant():
    """
    ``battery_wh`` was hard-coded to 600 Wh, so every mission's state of
    charge was tracked against a pack that had nothing to do with the
    aircraft flying it — on the VA23's 867 Wh of usable energy that
    overstated depletion by 45 %.
    """
    ac, planner, result = _one_leg_mission()
    assert planner.battery_wh == pytest.approx(ac.battery_usable_wh)
    assert result.battery_capacity_wh == pytest.approx(ac.battery_usable_wh)


def test_planner_carries_the_payload_into_its_own_energy_analysis():
    """
    The optimizer was given the payload and the planner's re-analysis
    was not, so the two disagreed by 15 % on a 2 kg delivery — and the
    number the mission reported was the one computed for an empty
    aircraft.
    """
    _, _, result = _one_leg_mission()
    leg = result.legs[0]
    assert leg.energy_result.total_energy_wh == pytest.approx(
        leg.opt_result.energy_wh, rel=1e-6)


def test_planner_soc_matches_the_energy_it_drew():
    """
    SOC advanced on energy *delivered* to the drivetrain. The I2R heat
    comes out of the same pack and never arrives, so charging the
    mission only for what arrived left the pack fuller than it was.
    """
    _, planner, result = _one_leg_mission()
    leg = result.legs[0]
    drawn = leg.energy_result.energy_from_cells_wh
    assert drawn > leg.energy_result.total_energy_wh     # there is a loss
    assert leg.soc_end == pytest.approx(
        leg.soc_start - drawn / planner.battery_wh, rel=1e-9)
