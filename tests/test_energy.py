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
    ac.with_payload(5.0)
    assert ac.m_tow == before


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
    v_h_sq = T / (2 * rho * ac.A_rotor)
    P_o = rho * ac.A_rotor * ac.V_tip ** 3 * (ac.sigma_rotor * ac.C_d_blade / 8)

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
    b = BatteryModel(ac)
    full = b.ocv
    b.SOC = 0.0
    empty = b.ocv
    # 6S LiPo: 25.2 V full, ~19.8 V empty. The old linear model topped
    # out at the 22.2 V *nominal*, making full-charge currents too high.
    assert full == pytest.approx(4.20 * ac.battery_cells_series, rel=1e-6)
    assert empty == pytest.approx(3.30 * ac.battery_cells_series, rel=1e-6)
    assert full > ac.battery_voltage > empty


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


def test_efficiency_falls_back_to_eta_prop(ac):
    assert ac.eta_vtol_effective == ac.eta_prop
    assert ac.eta_cruise_effective == ac.eta_prop


# ── Vehicle validation ────────────────────────────────────────────────────

def test_validate_flags_implausible_disk_loading(ac):
    problems = ac.validate()
    assert any("Disk loading" in p for p in problems)


def test_validate_is_quiet_for_a_sane_vehicle(ac):
    ac.A_rotor = 0.80          # 15 kg/m², typical
    assert ac.validate() == []


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
    ac.attach_polar(parabolic_polar(WingGeometry(), C_D0=0.030))
    assert ac.drag_coefficient(0.6, 25.0, ALT) > plain
    assert ac.aero_source == "parabolic"


def test_payload_keeps_the_polar_without_copying_it(ac):
    from raptor.aero import WingGeometry, parabolic_polar
    polar = parabolic_polar(WingGeometry())
    ac.attach_polar(polar)
    heavy = ac.with_payload(2.0)
    assert heavy.aero_polar is polar       # shared, not duplicated


def test_envelope_check_catches_speeds_below_stall():
    from raptor.aero import check_flight_envelope
    from raptor.config import UAVConfig
    uav, ac = UAVConfig(), AircraftEnergyParams()
    problems = check_flight_envelope(uav, ac, altitude=2800.0)
    assert any("fw_min_airspeed" in p for p in problems)


def test_envelope_check_is_quiet_for_a_sound_envelope():
    from raptor.aero import check_flight_envelope
    from raptor.config import UAVConfig
    uav = UAVConfig(fw_min_airspeed=22.0, fw_cruise_airspeed=30.0,
                    fw_max_airspeed=40.0)
    assert check_flight_envelope(uav, AircraftEnergyParams(),
                                 altitude=2800.0) == []


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
