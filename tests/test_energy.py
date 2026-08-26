"""
Regression tests for the energy, battery and aerodynamic models.

Each test pins a defect that silently produced plausible-looking but
wrong numbers — the worst kind, because nothing crashes.

    pytest tests/test_energy.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from raptor.atmosphere import isa_density
from raptor.energy import (
    BatteryModel, analyze_path_energy,
    power_hover, power_vertical_ascent, power_vertical_descent,
    power_fw_cruise, power_fw_climb, power_fw_descent, power_transition,
)
from raptor.vehicles import AircraftEnergyParams

ALT = 2800.0


@pytest.fixture
def ac():
    return AircraftEnergyParams()


# ── Payload actually weighs something ─────────────────────────────────────

def test_payload_changes_weight(ac):
    """
    Derived quantities used to be frozen at construction, so raising
    m_tow — which is exactly how the optimizer models payload — changed
    nothing. Every medical payload the package ever flew weighed zero.
    """
    heavy = ac.with_payload(2.5)
    assert heavy.m_tow == pytest.approx(ac.m_tow + 2.5)
    assert heavy.W > ac.W
    assert heavy.W == pytest.approx(heavy.m_tow * heavy.g)


def test_payload_changes_power(ac):
    heavy = ac.with_payload(2.5)
    assert power_hover(heavy, ALT) > power_hover(ac, ALT) * 1.2
    assert power_fw_cruise(heavy, 25.0, ALT) > power_fw_cruise(ac, 25.0, ALT)


def test_with_payload_does_not_mutate_the_original(ac):
    before = ac.m_tow
    ac.with_payload(1.0)
    assert ac.m_tow == before


def test_overloading_past_mtow_is_not_silent(ac):
    """
    Exceeding MTOW does not make the physics wrong, it makes the flight
    illegal — and the energy numbers that come back look perfectly
    reasonable, which is exactly why it has to be said out loud.
    """
    over = ac.payload_headroom_kg + 1.0
    with pytest.warns(UserWarning, match="maximum"):
        loaded = ac.with_payload(over)
    assert loaded.m_tow > ac.m_tow_max


# ── Rotor momentum theory ─────────────────────────────────────────────────

def test_climbing_costs_more_than_hovering(ac):
    """
    Momentum theory: induced velocity *falls* in climb. Writing the
    descent sign in the climb branch overstated ascent induced power by
    18 % at 3 m/s — on the phase where a VTOL draws its worst power.
    """
    hover = power_hover(ac, ALT)
    for v in (0.5, 1.0, 2.0, 3.0):
        p = power_vertical_ascent(ac, v, ALT)
        assert p > hover, f"climb at {v} m/s should cost more than hover"
    # Monotone in climb rate
    rates = [0.5, 1.0, 2.0, 3.0]
    powers = [power_vertical_ascent(ac, v, ALT) for v in rates]
    assert all(b > a for a, b in zip(powers, powers[1:]))


def test_descending_costs_less_than_hovering(ac):
    hover = power_hover(ac, ALT)
    for v in (0.5, 1.0, 2.0, 2.5):
        assert power_vertical_descent(ac, v, ALT) < hover


def test_ascent_tends_to_hover_at_zero_rate(ac):
    assert power_vertical_ascent(ac, 0.0, ALT) == pytest.approx(
        power_hover(ac, ALT), rel=1e-9
    )


def test_induced_velocity_matches_momentum_theory(ac):
    """Check the closed form directly, not just its monotonicity."""
    rho = isa_density(ALT)
    T = ac.W
    A = ac.rotor_area_total          # derived from diameter, not the stale field
    v_h_sq = T / (2 * rho * A)
    P_o = rho * A * ac.V_tip ** 3 * (ac.sigma_rotor * ac.C_d_blade / 8)

    for V_y in (1.0, 2.4, 3.0):
        v_i = -V_y / 2 + np.sqrt((V_y / 2) ** 2 + v_h_sq)
        expected = T * V_y + ac.k_i * T * v_i + P_o
        assert power_vertical_ascent(ac, V_y, ALT) == pytest.approx(expected, rel=1e-9)


# ── Fixed-wing ────────────────────────────────────────────────────────────

def test_climb_costs_more_than_cruise_costs_more_than_descent(ac):
    V = 25.0
    climb = power_fw_climb(ac, V, 8.0, ALT)
    cruise = power_fw_cruise(ac, V, ALT)
    descent = power_fw_descent(ac, V, 6.0, ALT)
    assert climb > cruise > descent


def test_transition_uses_parasite_drag_not_the_cruise_polar(ac):
    """
    During transition the rotors carry the weight, so the wing is barely
    loaded. Charging it the full cruise C_D bills the aircraft for
    induced drag from lift it is not producing.
    """
    lo = power_transition(ac, 10.0, ALT)
    inflated = AircraftEnergyParams(**{
        **{k: v for k, v in vars(ac).items()
           if k in AircraftEnergyParams.__dataclass_fields__},
        'C_L': ac.C_L * 1.5,     # much higher cruise C_L → much higher C_D
    })
    assert power_transition(inflated, 10.0, ALT) == pytest.approx(lo, rel=1e-9)


def test_stall_is_penalised_not_silently_accepted(ac):
    slow = ac.stall_speed_at(ALT) * 0.7
    fast = ac.stall_speed_at(ALT) * 1.5
    assert power_fw_cruise(ac, slow, ALT) > power_fw_cruise(ac, fast, ALT)


# ── Battery ───────────────────────────────────────────────────────────────

def test_open_circuit_voltage_spans_the_real_range(ac):
    """
    The old linear model was ``V_nom × (0.85 + 0.15·SOC)``, which topped
    out at the *nominal* voltage — so full-charge currents came out high
    and empty-pack currents low, exactly backwards.
    """
    from raptor.battery import CELL_CHEMISTRY

    from raptor.battery import cell_voltage_window

    spec = CELL_CHEMISTRY[ac.cell_chemistry]
    n = ac.battery_cells_series
    v_empty, v_full = cell_voltage_window(ac.cell_chemistry)
    b = BatteryModel(ac)
    assert b.ocv == pytest.approx(v_full * n, rel=1e-6)

    # An empty cell rests *above* the cut-off it was discharged to: the
    # cut-off is a voltage under load, the window is open-circuit.
    assert v_empty > spec["v_cutoff"] - 0.6
    assert v_full > spec["v_nominal"] > v_empty

    # A flight never reaches either end: SOC 0 is the reserve, not empty.
    b.SOC = 0.0
    assert spec["v_cutoff"] * n < b.ocv < ac.battery_voltage


def test_each_chemistry_gets_its_own_curve(ac):
    """
    A LiPo falls steadily from 4.2 V; a high-nickel Li-ion holds a long
    plateau and then drops in the last fifth. Serving one curve for both
    puts the pack's mean voltage — and therefore its energy — several
    percent out.
    """
    from raptor.battery import (
        cell_ocv_curve, cell_voltage_window, mean_cell_voltage,
    )

    # They differ in shape, not merely in level. The distinguishing
    # feature is the bottom: a LiPo is done at 3.3 V/cell while a
    # high-nickel Li-ion still has usable charge down at 2.6 V, and that
    # tail is what decides how much of the pack sits above a cut-off.
    lipo_empty, _ = cell_voltage_window("lipo")
    liion_empty, _ = cell_voltage_window("li-ion")
    assert lipo_empty > liion_empty + 0.5

    assert mean_cell_voltage("lipo") != pytest.approx(
        mean_cell_voltage("li-ion"), abs=0.02)

    for chem in ("lipo", "li-ion", "li-ion-semisolid"):
        soc, v = cell_ocv_curve(chem)
        assert np.all(np.diff(v) > 0), f"{chem}: OCV must rise with SOC"

    with pytest.raises(KeyError, match="unknown cell chemistry"):
        cell_ocv_curve("lead-acid")


def test_curve_reproduces_the_cell_datasheet(ac):
    """
    The only check available without a bench: a datasheet gives capacity
    in amp-hours and energy in watt-hours, and their ratio is a mean
    voltage the curve has to hit. Molicel INR21700-P42A: 4.2 Ah typical,
    15.12 Wh typical.
    """
    from raptor.battery import check_pack_against_datasheet

    r = check_pack_against_datasheet("li-ion", 4.2, 15.12)
    assert r["within_tolerance"], r["note"]

    # And it has to be a real check — the LiPo curve must fail it.
    wrong = check_pack_against_datasheet("lipo", 4.2, 15.12)
    assert not wrong["within_tolerance"], (
        "a curve 0.2 V too high passes the datasheet check; the check is "
        "not testing anything"
    )


def test_resistance_climbs_as_the_pack_empties(ac):
    """
    A constant resistance understates voltage sag exactly where a
    mission has least margin. Sag near empty is what turns "enough
    energy left" into "cannot supply the power to land".
    """
    from raptor.battery import chen_rincon_mora_resistance_factor

    # The published shape: flat above 20 % charge, steep below it.
    assert chen_rincon_mora_resistance_factor(0.5) == pytest.approx(1.0, rel=1e-9)
    assert chen_rincon_mora_resistance_factor(0.2) < 1.05
    assert chen_rincon_mora_resistance_factor(0.02) > 1.5

    b = BatteryModel(ac)
    b.SOC = 0.9
    r_full = b.internal_resistance
    b.SOC = 0.0
    r_reserve = b.internal_resistance
    assert r_reserve > r_full

    # And it must reach the pack model, not merely be available on it.
    b.SOC = 0.9
    full_power = b.max_power_w
    b.SOC = 0.0
    assert b.max_power_w < full_power


def test_ohmic_loss_is_accounted(ac):
    b = BatteryModel(ac)
    ocv_before = b.ocv          # OCV falls as SOC drops, so capture it first
    r = b.discharge(4000.0, 60.0)
    assert r['ohmic_loss_wh'] > 0
    # More leaves the cells than reaches the drivetrain.
    assert r['energy_from_cells_wh'] > r['energy_wh']
    assert r['voltage_V'] < ocv_before


def test_higher_power_costs_disproportionately_more(ac):
    """I²R means doubling the load more than doubles the pack drain."""
    a, b = BatteryModel(ac), BatteryModel(ac)
    lo = a.discharge(1000.0, 60.0)['energy_from_cells_wh']
    hi = b.discharge(2000.0, 60.0)['energy_from_cells_wh']
    assert hi > 2.0 * lo


def test_soc_uses_usable_energy_not_nameplate(ac):
    assert ac.battery_usable_wh < ac.battery_energy_wh
    b = BatteryModel(ac)
    b.discharge(ac.battery_usable_wh * 3600.0 / 60.0 * 0.5, 60.0)
    # Half the usable energy should be roughly half the SOC, not less.
    assert 0.3 < b.SOC < 0.6


def test_unmeetable_power_is_flagged_not_imaginary(ac):
    b = BatteryModel(ac)
    r = b.discharge(b.max_power_w * 10.0, 1.0)
    assert r['power_limited']
    assert np.isfinite(r['current_A'])


def test_c_rate_is_reported(ac):
    b = BatteryModel(ac)
    r = b.discharge(4000.0, 60.0)
    assert r['c_rate'] > 0
    assert b.max_c_rate == pytest.approx(r['c_rate'])


# ── Efficiency split ──────────────────────────────────────────────────────

def test_vtol_and_cruise_efficiencies_can_differ(ac):
    ac.eta_vtol = 0.55
    ac.eta_cruise = 0.80
    assert ac.eta_vtol_effective == 0.55
    assert ac.eta_cruise_effective == 0.80


def test_efficiency_falls_back_to_eta_prop():
    """A vehicle that names only one efficiency uses it for both."""
    single = AircraftEnergyParams(eta_prop=0.60, eta_vtol=None, eta_cruise=None)
    assert single.eta_vtol_effective == 0.60
    assert single.eta_cruise_effective == 0.60


def test_shipped_vehicle_separates_the_two_efficiencies(ac):
    """Lift rotors and a cruise propulsor are different machines."""
    assert ac.eta_vtol_effective != ac.eta_cruise_effective


# ── Vehicle validation ────────────────────────────────────────────────────

def test_validate_flags_implausible_disk_loading():
    """
    A 12 kg aircraft on 0.20 m² of disk is 60 kg/m² — four times any real
    multirotor, and hover power scales with its square root.
    """
    bad = AircraftEnergyParams(rotor_diameter_m=0.0, A_rotor=0.20)
    assert any("Disk loading" in p for p in bad.validate())


def test_validate_is_quiet_for_the_shipped_vehicle(ac):
    assert ac.validate() == []


def test_rotor_area_derives_from_diameter_not_a_typed_area():
    """
    A total area and a rotor count can disagree with each other; a
    diameter and a count cannot.
    """
    v = AircraftEnergyParams(n_rotors=4, rotor_diameter_m=0.51, A_rotor=0.20)
    assert v.rotor_area_total == pytest.approx(4 * np.pi * 0.255 ** 2, rel=1e-6)
    assert v.rotor_area_total > v.A_rotor       # the stale field is ignored


def test_dataclass_defaults_match_the_default_vehicle():
    """
    Two sources of truth for "the default aircraft" is how a test ends
    up passing against a machine nobody flies. The dataclass literals
    have to stay equal to data/vehicles/va23.json, which is generated
    from the manufacturer's published figures by
    ``python -m scripts.build_vehicles``.
    """
    from raptor.vehicles import get_vehicle
    a, b = AircraftEnergyParams(), get_vehicle("default")
    skip = {"name", "description", "geometry", "provenance"}
    differing = []
    for f in a.__dataclass_fields__:
        if f in skip:
            continue
        x, y = getattr(a, f), getattr(b, f)
        if isinstance(x, float) and isinstance(y, float):
            # Aerodynamic coefficients come from a lifting-line solve and
            # are rounded when written; requiring bit equality would make
            # this test fail on rounding rather than on drift.
            same = x == pytest.approx(y, rel=1e-5, abs=1e-12)
        else:
            same = x == y
        if not same:
            differing.append((f, x, y))
    assert differing == []


# ── Aerodynamics ──────────────────────────────────────────────────────────

def test_polar_falls_back_cleanly_and_says_so():
    from raptor.aero import WingGeometry, parabolic_polar
    p = parabolic_polar(WingGeometry())
    assert p.source == "parabolic"
    assert p.C_D(0.5, 3e5) > p.C_D(0.0, 3e5)


def test_polar_cache_key_covers_the_grid():
    """
    Keying on geometry alone would serve a polar computed over a narrow
    incidence range to a caller that needs it out to stall.
    """
    from raptor.aero import WingGeometry, _cache_key
    g = WingGeometry()
    a = _cache_key(g, {"alpha_deg": [0, 5, 10]})
    b = _cache_key(g, {"alpha_deg": [0, 5, 10, 15, 20]})
    assert a != b
    assert a == _cache_key(g, {"alpha_deg": [0, 5, 10]})


def test_attached_polar_changes_drag(ac):
    from raptor.aero import WingGeometry, parabolic_polar
    plain = ac.drag_coefficient(0.6, 25.0, ALT)
    ac.attach_polar(parabolic_polar(WingGeometry(), C_D0=0.060))
    assert ac.drag_coefficient(0.6, 25.0, ALT) > plain
    assert "parabolic" in ac.aero_source


def test_envelope_defaults_are_flyable():
    """
    The speed envelope and the aerodynamics used to be typed in
    separately, and drifted until the declared minimum airspeed sat
    below the wing's stall speed at operating altitude.
    """
    from raptor.aero import check_flight_envelope
    from raptor.config import UAVConfig
    ac = AircraftEnergyParams()
    v_stall = ac.stall_speed_at(2800.0)
    assert UAVConfig().fw_min_airspeed >= v_stall


@pytest.mark.parametrize("altitude", [0.0, 1500.0, 2800.0, 3200.0, 4200.0])
def test_envelope_derives_from_the_vehicle(altitude, built_vehicle):
    """A derived envelope must always pass the check it was derived from."""
    from raptor.aero import check_flight_envelope
    from raptor.config import UAVConfig
    uav = UAVConfig.for_vehicle(built_vehicle, altitude=altitude)
    assert check_flight_envelope(uav, built_vehicle, altitude=altitude) == []


def test_envelope_rises_with_altitude(built_vehicle):
    from raptor.config import UAVConfig
    uav = UAVConfig.for_vehicle(built_vehicle, altitude=3200.0)
    # Thinner air at altitude must push the whole band up.
    low = UAVConfig.for_vehicle(built_vehicle, altitude=0.0)
    assert uav.fw_min_airspeed > low.fw_min_airspeed


def test_payload_keeps_the_polar_without_copying_it(ac):
    from raptor.aero import WingGeometry, parabolic_polar
    polar = parabolic_polar(WingGeometry())
    ac.attach_polar(polar)
    heavy = ac.with_payload(2.0)
    assert heavy.aero_polar is polar       # shared, not duplicated


def test_envelope_check_catches_speeds_below_stall():
    from raptor.aero import check_flight_envelope
    from raptor.config import UAVConfig
    ac = AircraftEnergyParams()
    reckless = UAVConfig(fw_min_airspeed=15.0, fw_cruise_airspeed=25.0)
    problems = check_flight_envelope(reckless, ac, altitude=2800.0)
    assert any("fw_min_airspeed" in p for p in problems)
    assert any("below the stall speed" in p.lower() or "BELOW the stall" in p
               for p in problems)


def test_envelope_check_is_quiet_for_a_sound_envelope(built_vehicle):
    from raptor.aero import check_flight_envelope
    from raptor.config import UAVConfig
    uav = UAVConfig.for_vehicle(built_vehicle, altitude=2800.0)
    assert check_flight_envelope(uav, built_vehicle, altitude=2800.0) == []


# ── Drag build-up ─────────────────────────────────────────────────────────

def test_buildup_dominated_by_stopped_rotors(ac):
    """
    The term most often omitted from a lift+cruise drag estimate, and
    usually the largest single non-wing contributor.
    """
    from raptor.drag import quadplane_buildup
    b = quadplane_buildup()
    parts = b.breakdown(25.0, ALT)
    assert next(iter(parts)).startswith("stopped rotors")


def test_rotor_parking_moves_the_answer():
    from raptor.drag import quadplane_buildup
    folded = quadplane_buildup(rotor_parking="folded").C_D0(25.0, ALT)
    aligned = quadplane_buildup(rotor_parking="aligned").C_D0(25.0, ALT)
    unaligned = quadplane_buildup(rotor_parking="unaligned").C_D0(25.0, ALT)
    assert folded < aligned < unaligned
    assert unaligned > 2 * folded


def test_buildup_varies_with_reynolds_number():
    """A constant C_D0 cannot; skin friction falls as Reynolds rises."""
    from raptor.drag import quadplane_buildup
    b = quadplane_buildup()
    assert b.C_D0(18.0, ALT) > b.C_D0(32.0, ALT)


def test_buildup_and_polar_sum_without_double_counting(ac):
    """
    The lifting-line polar already carries the wing's profile and induced
    drag; the build-up carries no wing. Total is the sum of the two.
    """
    from raptor.aero import WingGeometry, parabolic_polar
    from raptor.drag import quadplane_buildup
    polar = parabolic_polar(WingGeometry())
    ac.attach_polar(polar)
    wing_only = ac.drag_coefficient(0.6, 25.0, ALT)

    buildup = quadplane_buildup()
    ac.attach_drag_buildup(buildup)
    total = ac.drag_coefficient(0.6, 25.0, ALT)

    assert total == pytest.approx(wing_only + buildup.C_D0(25.0, ALT), rel=1e-9)


def test_buildup_validate_catches_a_missing_component():
    from raptor.drag import DragBuildup, fuselage
    thin = DragBuildup(S_ref=0.5)
    thin.add(fuselage(1.1, 0.18))
    assert thin.validate()          # fuselage alone is implausibly clean


# ── Compliance grading ────────────────────────────────────────────────────

def test_excursions_group_into_stretches_not_waypoints():
    """A waiver is granted over a segment of route, not per sample."""
    from raptor.compliance import find_excursions
    d = np.arange(0, 1000, 100.0)
    agl = np.array([60, 60, 140, 150, 60, 60, 130, 60, 60, 60], float)
    lats = np.full(10, -0.25)
    lons = np.full(10, -78.55)
    ex = find_excursions(d, agl, lats, lons, limit_m=122.0)
    assert len(ex) == 2
    assert ex[0].max_agl_m == 150.0
    assert ex[1].max_agl_m == 130.0


def test_prohibited_airspace_is_never_waivable():
    from raptor.compliance import ComplianceLevel
    assert not ComplianceLevel.NON_COMPLIANT.flyable
    assert ComplianceLevel.WAIVER_REQUIRED.flyable
    assert ComplianceLevel.AUTHORIZATION_REQUIRED.flyable


def test_ceiling_waivable_changes_feasibility_not_the_floor():
    from raptor.penalties import evaluate_penalties

    class Report:
        corridor_penalty = 0.02
        corridor_feasible = False
        is_feasible = True            # clearance floor is fine
        ceiling_feasible = False      # ceiling is not
        max_ceiling_excess = 103.0
        ceiling_violation_fraction = 0.3

    strict = evaluate_penalties(Report(), None, 0.6, 0.2, 600, 1800)
    waived = evaluate_penalties(Report(), None, 0.6, 0.2, 600, 1800,
                                ceiling_waivable=True)
    assert not strict.feasible
    assert waived.feasible
    assert waived.waiver > 0

    # A violated clearance floor stays fatal in both modes.
    class FloorBust(Report):
        is_feasible = False

    assert not evaluate_penalties(FloorBust(), None, 0.6, 0.2, 600, 1800,
                                  ceiling_waivable=True).feasible


def test_mission_compliance_takes_the_worst_leg():
    from raptor.compliance import (
        ComplianceAssessment, ComplianceLevel, mission_compliance,
    )
    legs = [
        ComplianceAssessment(level=ComplianceLevel.COMPLIANT),
        ComplianceAssessment(level=ComplianceLevel.AUTHORIZATION_REQUIRED,
                             permits_required={"A": "hospital"}),
        ComplianceAssessment(level=ComplianceLevel.WAIVER_REQUIRED),
    ]
    rolled = mission_compliance(legs)
    assert rolled.level is ComplianceLevel.WAIVER_REQUIRED
    assert rolled.permits_required == {"A": "hospital"}


# ── Mission profiles ──────────────────────────────────────────────────────

def _facilities():
    from raptor.scenarios import (
        HOSPITAL_EUGENIO_ESPEJO, CS_LLOA, CS_PINTAG,
    )
    return HOSPITAL_EUGENIO_ESPEJO, [CS_LLOA, CS_PINTAG]


def test_collection_round_is_heaviest_at_the_end():
    """
    Mass accumulates, so the last leg — flown on the least charge — is
    also the heaviest. A per-leg analysis never surfaces that coupling.
    """
    from raptor.missions import sample_collection
    base, centres = _facilities()
    m = sample_collection(base, centres, per_stop_kg=0.5)
    sched = m.payload_schedule()
    assert sched == sorted(sched), "collection payload should only grow"
    assert m.heaviest_leg == len(sched) - 1


def test_supply_tour_is_heaviest_at_the_start():
    from raptor.missions import supply_tour
    base, centres = _facilities()
    m = supply_tour(base, centres, per_stop_kg=0.7)
    sched = m.payload_schedule()
    assert sched == sorted(sched, reverse=True)
    assert m.heaviest_leg == 0


def test_out_and_back_returns_to_base():
    from raptor.missions import out_and_back
    base, centres = _facilities()
    m = out_and_back(base, centres[0], payload_out_kg=2.0)
    legs = m.to_legs()
    assert legs[0].origin is base
    assert legs[-1].destination is base
    assert legs[0].payload_kg == 2.0
    assert legs[-1].payload_kg == 0.0


def test_hub_and_spoke_visits_base_between_deliveries():
    from raptor.missions import hub_and_spoke
    base, centres = _facilities()
    m = hub_and_spoke(base, centres, payload_kg=2.0)
    legs = m.to_legs()
    assert len(legs) == 2 * len(centres)
    assert all(legs[i].origin is base for i in range(0, len(legs), 2))


def test_battery_swap_restores_charge_and_none_does_not():
    from raptor.missions import BatteryAction, Stop
    base, _ = _facilities()
    assert BatteryAction.SWAP.restores_charge
    assert BatteryAction.RECHARGE.restores_charge
    assert not BatteryAction.NONE.restores_charge

    # Recharging in place takes real time; swapping does not.
    recharge = Stop(base, battery_action=BatteryAction.RECHARGE,
                    service_time_s=300.0, recharge_rate_c=1.0)
    swap = Stop(base, battery_action=BatteryAction.SWAP, service_time_s=300.0)
    assert recharge.recharge_time_s(0.3) > swap.recharge_time_s(0.3)
    assert recharge.recharge_time_s(0.95) == pytest.approx(300.0)


def test_stops_must_match_leg_count():
    from raptor.missions import out_and_back
    base, centres = _facilities()
    m = out_and_back(base, centres[0])
    assert len(m.stops) == m.n_legs + 1


def test_mission_planning_is_reproducible_by_default():
    """
    Mission results defaulted to an unseeded optimizer while single-leg
    results were seeded, so the same comparison could rank two missions
    in opposite orders on consecutive runs — and the difference looked
    like a finding rather than noise.
    """
    import inspect
    from raptor.mission_planner import MissionPlanner
    sig = inspect.signature(MissionPlanner.plan_mission)
    assert sig.parameters["seed"].default is not None


# ── Battery characterisation ──────────────────────────────────────────────

def test_fit_recovers_a_known_resistance():
    """A stepped load separates open-circuit voltage from resistance."""
    from raptor.battery import fit_battery, synthetic_log
    log = synthetic_log(current_profile="stepped", resistance_ohm=0.0044)
    fit = fit_battery(log)
    assert fit.separable and fit.resistance_fitted
    assert fit.resistance_ohm == pytest.approx(0.0044, rel=0.15)


def test_constant_load_cannot_identify_resistance():
    """
    At constant current any resistance can be traded against a shifted
    voltage curve. Least squares still returns a number — the fit has to
    say it is not one to believe.
    """
    from raptor.battery import fit_battery, synthetic_log
    log = synthetic_log(current_profile="constant", resistance_ohm=0.0044)
    fit = fit_battery(log, resistance_prior_ohm=0.0044)
    assert not fit.separable
    assert not fit.resistance_fitted
    assert fit.resistance_ohm == pytest.approx(0.0044)   # held at the prior


def test_fit_grid_covers_only_the_measured_range():
    """
    Grid nodes outside the SOC the log visited are unconstrained, and
    least squares leaves them at zero — a pack reading 0 V when full.
    """
    from raptor.battery import fit_battery, synthetic_log
    log = synthetic_log(current_profile="stepped", duration_s=600.0)
    fit = fit_battery(log)
    lo, hi = fit.soc_range
    assert lo > 0.0                       # the log never emptied the pack
    assert fit.ocv_cell_v.min() > 3.0     # no zero-volt nodes
    assert np.all(np.diff(fit.ocv_cell_v) >= -1e-9)   # monotone


def test_fit_refuses_a_log_that_barely_discharges():
    from raptor.battery import fit_battery, synthetic_log
    log = synthetic_log(current_profile="stepped", duration_s=20.0, n=50)
    with pytest.raises(ValueError, match="state of charge"):
        fit_battery(log)


def test_provenance_distinguishes_measured_from_assumed(ac):
    from raptor.battery import synthetic_log
    assert "ASSUMED" in ac.battery_provenance()
    ac.fit_battery_from_log(synthetic_log(current_profile="stepped"))
    text = ac.battery_provenance()
    assert "MEASURED" in text
    assert "NOT MODELLED" in text          # honest about what is still missing


def test_fitted_curve_is_used_by_the_battery_model(ac):
    from raptor.battery import synthetic_log
    from raptor.energy import BatteryModel
    before = BatteryModel(ac).ocv
    ac.fit_battery_from_log(synthetic_log(current_profile="stepped"))
    after = BatteryModel(ac).ocv
    assert after != before or ac.battery_fit is not None


def test_polar_buildup_and_fit_all_survive_with_payload(ac):
    from raptor.aero import WingGeometry, parabolic_polar
    from raptor.battery import synthetic_log
    from raptor.drag import quadplane_buildup
    ac.attach_polar(parabolic_polar(WingGeometry()))
    ac.attach_drag_buildup(quadplane_buildup())
    ac.fit_battery_from_log(synthetic_log(current_profile="stepped"))
    heavy = ac.with_payload(2.0)
    assert heavy.aero_polar is ac.aero_polar
    assert heavy.drag_buildup is ac.drag_buildup
    assert heavy.battery_fit is ac.battery_fit
    assert heavy.m_tow > ac.m_tow


def test_soc_and_charge_fraction_are_different_things(ac):
    """
    ``SOC`` runs from 1 to 0 over the *usable* energy — that is what a
    reserve is expressed against. The OCV curve is a property of the
    whole cell. Reading the curve at the mission SOC put the pack at a
    flat 2.5 V/cell the moment the reserve was reached, which is a
    voltage no flight ever sees.
    """
    from raptor.battery import CELL_CHEMISTRY

    b = BatteryModel(ac)
    assert b.charge_fraction == pytest.approx(1.0)

    b.SOC = 0.0
    assert b.charge_fraction == pytest.approx(
        1.0 - ac.battery_usable_fraction, abs=1e-9)

    cutoff = CELL_CHEMISTRY[ac.cell_chemistry]["v_cutoff"]
    cell_v = b.ocv / ac.battery_cells_series
    assert cell_v > cutoff, (
        f"at the reserve limit the pack reads {cell_v:.2f} V/cell, at or "
        f"below the {cutoff:.2f} V cut-off — the mission would have ended "
        f"before its own reserve"
    )


def test_pack_energy_agrees_with_its_own_voltage_curve():
    """
    A pack states its energy twice: mass times specific energy, and the
    OCV curve integrated over the charge it holds. Nothing forces them
    to agree, and when they disagree every endurance figure is wrong by
    the difference.
    """
    from raptor.vehicles import get_vehicle

    for name in ("va17", "va23", "va25"):
        v = get_vehicle(name)
        problems = [p for p in v.validate() if "disagrees with itself" in p]
        assert problems == [], f"{name}: {problems}"


def test_cell_capacity_is_a_cell_not_a_pack():
    """
    ``cell_capacity_ah`` divided into the pack capacity is the parallel
    count, and the parallel count is what divides the pack's internal
    resistance. Putting the pack figure in the cell field collapses the
    count to one and inflates the modelled resistance by roughly the
    real parallel count.
    """
    from raptor.vehicles import get_vehicle

    for name in ("va17", "va23", "va25"):
        v = get_vehicle(name)
        assert v.battery_strings_parallel > 1.5, (
            f"{name}: {v.battery_strings_parallel:.1f} parallel strings for "
            f"a {v.battery_capacity_ah:.1f} Ah pack"
        )
        assert 0.005 < v.battery_resistance < 0.15, (
            f"{name}: pack resistance {v.battery_resistance:.3f} ohm is "
            f"outside anything plausible for this class"
        )

    bad = get_vehicle("va23")
    bad.cell_capacity_ah = bad.battery_capacity_ah      # the mistake
    assert any("looks like" in p for p in bad.validate())


def test_pack_power_is_sized_from_the_pack_not_a_cell():
    """
    C-rate multiplies the *pack's* capacity. Reading the cell field here
    sized a 30 Ah pack as a single 4.2 Ah string — a seventh of the real
    power — which then became the binding constraint on the derived
    climb angle and roughly halved it.
    """
    from raptor.vehicles import get_vehicle

    ac = get_vehicle("va23")
    expected = ac.battery_max_c_rate * ac.battery_capacity_ah * ac.battery_voltage
    assert ac.electrical_power_max_w == pytest.approx(expected)
    # Sanity: a 1.3 kWh pack at 10 C is kilowatts, not hundreds of watts.
    assert ac.electrical_power_max_w > 5000.0


def test_deriving_an_envelope_without_the_airframe_warns():
    """
    Every derived limit comes from drag, and the wing alone has about
    twice the real lift-to-drag ratio. Because climb angle goes as
    1/(L/D), a forgotten build_aero() does not fail — it quietly makes
    the aircraft look worse at climbing and better at cruising.
    """
    from raptor.config import UAVConfig
    from raptor.vehicles import get_vehicle

    bare = get_vehicle("va23")
    assert not bare.aero_is_complete
    with pytest.warns(UserWarning, match="incomplete aerodynamic model"):
        UAVConfig.for_vehicle(bare, altitude=2800.0)
