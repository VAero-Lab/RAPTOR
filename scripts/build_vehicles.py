#!/usr/bin/env python3
"""
Build the shipped vehicle definitions from published aircraft data.

Why this is a script and not three hand-written JSON files
-----------------------------------------------------------
A vehicle definition has about forty numbers in it. Perhaps a dozen are
published by the manufacturer; the rest have to be worked out. Written
by hand, the two kinds become indistinguishable a week later, and a
reader has no way to tell which figures they can check against a
datasheet and which are somebody's judgement.

So every parameter is emitted here with a tag:

``referenced``  published on the manufacturer's page, quoted as given
``derived``     computed from referenced values by a relation stated
                alongside it
``assumed``     engineering judgement; the basis is given, and it is the
                first thing to revisit if the model disagrees with
                reality

The tags travel with the vehicle into its JSON file and are readable at
runtime as ``ac.provenance``.

The three aircraft
------------------
Chosen because they bracket the class that actually flies medical
payloads, and because all three publish enough to work from:

* **T-Drones VA17** — 4.3 kg tilt-rotor, 0.9 kg payload. The small end:
  one blood cooler, short hops.
  https://www.3dxr.co.uk/products/t-drones-va17-tilt-rotor-vtol-pnp
* **T-Drones VA23** — 12.5 kg lift+cruise quadplane, 2.5 kg payload.
  The workhorse: the largest payload of the three.
  https://www.3dxr.co.uk/products/t-drones-va23
* **Alafija VA25** — 13 kg lift+cruise quadplane, 2 kg payload, faster
  cruise and a higher wind rating. The range end.
  https://www.hldstore.com/vtol-alafija/va25-fixed-wing-vtol-drone/

Usage
-----
    python -m scripts.build_vehicles                 # write + report
    python -m scripts.build_vehicles --check         # report only
    python -m scripts.build_vehicles --no-aero       # skip AeroSandbox
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

from raptor.atmosphere import isa_density
from raptor.wind import BEAUFORT_MS
from raptor.vehicles import AircraftEnergyParams

OUT_DIR = "data/vehicles"

#: Planning altitude for every derived airspeed. Quito's basin floor is
#: near 2 800 m and the corridors run 60-122 m above it, so this is
#: where the aircraft actually works — not sea level, where the
#: manufacturers quote their cruise speeds.
PLANNING_ALT_M = 2800.0

#: Sea level, where published cruise and stall figures are quoted.
REFERENCE_ALT_M = 0.0

#: The cell the packs are assumed to be built from. None of the three
#: manufacturers says which cell they use, so one has to be chosen, and
#: this is the high-power 21700 that long-endurance VTOL packs are
#: usually built from. Its capacity fixes the parallel count and its
#: resistance the pack's voltage sag.
#: Molicel, "Product Data Sheet INR-21700-P42A", rev. V4, doc. 80092.
CELL_MODEL = dict(
    name="Molicel INR21700-P42A",
    capacity_ah=4.2,
    dc_resistance_ohm=0.025,
    typical_wh=15.12,
)

def propulsion_mass_kg(m_tow: float, n_lift_rotors: int,
                       tilt: bool = False) -> float:
    """
    Motors, ESCs, propellers and mounts [kg].

    Scaled from take-off mass, because thrust has to scale with weight
    and motor mass scales roughly with thrust. 12 % of MTOW is typical
    for a lift+cruise quadplane carrying four lift motors and a pusher;
    a tilt-rotor needs fewer motors but adds tilt servos and mechanisms,
    which roughly cancels.
    """
    return 0.12 * m_tow


def avionics_mass_kg(m_tow: float) -> float:
    """Autopilot, GNSS, radios, servos, wiring and connectors [kg]."""
    return max(0.25, 0.045 * m_tow)


def choose_pack(r: dict, spec_e: float, v_mean: float):
    """
    The largest listed pack this airframe can actually carry.

    Manufacturers publish an empty mass *and* a list of pack options,
    and the two are not always compatible: a pack has wiring, a BMS,
    interconnects and a case, and counting those can push the largest
    option past the mass the aircraft is said to weigh empty. Nothing
    catches this, because every figure is plausible on its own.

    Returns ``(capacity_ah, basis)``.
    """
    options = r.get("pack_options_ah") or (r["pack_capacity_ah"],)
    for cap in sorted(options, reverse=True):
        if mass_budget(r, v_mean * r["cells_series"] * cap / spec_e)["residual"] >= 0:
            return float(cap), ("referenced" if len(options) == 1
                                else f"referenced — largest of "
                                     f"{'/'.join(f'{o:g}' for o in options)} Ah "
                                     f"that fits the mass budget")

    # None of them fits. Derive the capacity the budget allows, and say so.
    smallest = min(options)
    fitted = mass_budget(
        r, v_mean * r["cells_series"] * smallest / spec_e)
    allowed_kg = max(fitted["battery"] + fitted["residual"], 0.0)
    cap = allowed_kg * spec_e / (v_mean * r["cells_series"])
    return float(cap), (
        f"derived — none of the listed options "
        f"({'/'.join(f'{o:g}' for o in options)} Ah) fits inside the "
        f"published {r['m_tow_kg'] - r['payload_max_kg']:.2f} kg empty mass "
        f"once pack overhead is counted; {cap:.1f} Ah is what the budget "
        f"allows")


def mass_budget(r: dict, m_batt: float) -> dict:
    """
    Does the empty mass account for the parts that make it up?

    A vehicle definition can be internally consistent and still describe
    an aircraft that cannot be built: the manufacturer states an empty
    mass and a pack, and if the pack plus the airframe plus the
    propulsion exceeds the empty mass then one of the published numbers
    is wrong. Nothing else in the model notices, because every component
    is plausible on its own.

    Returns the parts and the residual. A negative residual means the
    aircraft as specified does not close.
    """
    m_max = r["m_tow_kg"]
    m_empty = m_max - r["payload_max_kg"]
    m_frame = r.get("m_frame_kg")
    if m_frame is None:
        # Not published for the VA17. Carbon airframes of this class run
        # about a quarter of empty mass.
        m_frame = 0.25 * m_empty
        frame_source = "assumed at 25 % of empty mass"
    else:
        frame_source = "referenced"

    m_prop = propulsion_mass_kg(m_max, r.get("n_lift_rotors", 4))
    m_avio = avionics_mass_kg(m_max)
    accounted = m_batt + m_frame + m_prop + m_avio
    residual = m_empty - accounted

    # If it does not close, the pack is the term to question: frame mass
    # is published for two of the three, propulsion and avionics scale
    # with a weight that is also published, and the pack is the one the
    # manufacturer lists several options for.
    m_batt_fits = max(m_batt + residual, 0.0)
    return dict(
        empty=m_empty, battery=m_batt, frame=m_frame,
        frame_source=frame_source, propulsion=m_prop, avionics=m_avio,
        accounted=accounted, residual=residual,
        battery_mass_that_fits=m_batt_fits,
    )


#: Avionics, autopilot, radios, servos and payload conditioning [W].
#: Small in absolute terms and easy to leave out, but on a 4 kg aircraft
#: whose cruise needs under 100 W it is a fifth of the budget.
HOTEL_LOAD_W = 15.0


# ═══════════════════════════════════════════════════════════════════════════
# PUBLISHED DATA
# ═══════════════════════════════════════════════════════════════════════════
#
# Only what the manufacturer states. Nothing inferred, nothing rounded
# to look tidy. Where a page gives a range it is kept as a range.

REFERENCES = {
    "va17": dict(
        name="T-Drones VA17",
        url="https://www.3dxr.co.uk/products/t-drones-va17-tilt-rotor-vtol-pnp",
        configuration="tilt-rotor, 3 rotors",
        n_lift_rotors=3,
        wingspan_m=1.680,
        length_m=1.178,
        m_tow_kg=4.30,
        payload_max_kg=0.90,
        cruise_speed_ms=(15.0, 18.0),
        max_speed_ms=25.0,
        endurance_min=120.0,
        endurance_payload_kg=0.80,
        range_km=125.0,
        ceiling_m=4500.0,
        cells_series=8,
        pack_options_ah=(14.5,),
        pack_capacity_ah=14.5,
        pack_chemistry="li-ion",
        pack_cutoff_v=24.0,
        wind_beaufort=5,
    ),
    "va23": dict(
        name="T-Drones VA23",
        url="https://www.3dxr.co.uk/products/t-drones-va23",
        configuration="lift+cruise quadplane, 4 lift rotors + pusher",
        n_lift_rotors=4,
        wingspan_m=2.300,
        length_m=1.250,
        m_tow_kg=12.50,
        m_frame_kg=2.00,
        payload_max_kg=2.50,
        cruise_speed_ms=(18.0, 20.0),
        max_speed_ms=35.0,
        endurance_min=180.0,
        endurance_payload_kg=1.50,
        range_km=240.0,
        ceiling_m=4500.0,
        cells_series=12,
        pack_options_ah=(22.0, 27.0, 30.0),
        pack_capacity_ah=30.0,
        pack_chemistry="li-ion",
        wind_beaufort=5,
    ),
    "va25": dict(
        name="Alafija VA25",
        url="https://www.hldstore.com/vtol-alafija/va25-fixed-wing-vtol-drone/",
        configuration="lift+cruise quadplane, 4 lift rotors + pusher",
        n_lift_rotors=4,
        wingspan_m=2.500,
        length_m=1.560,
        m_tow_kg=13.00,
        m_frame_kg=3.50,
        payload_max_kg=2.00,
        cruise_speed_ms=(22.2, 25.0),      # 80–90 km/h
        max_speed_ms=33.3,                 # 120 km/h
        endurance_min=210.0,
        endurance_payload_kg=1.00,
        ceiling_m=4500.0,
        cells_series=12,                   # 2 × 6S in series
        pack_options_ah=(30.0,),
        pack_capacity_ah=30.0,
        pack_chemistry="li-ion-semisolid",
        wind_beaufort=6,
    ),
}




# ═══════════════════════════════════════════════════════════════════════════
# DERIVATION
# ═══════════════════════════════════════════════════════════════════════════

def nominal_cell_voltage(chemistry: str) -> float:
    """
    Nominal per-cell voltage for the pack chemistry.

    LiPo is 3.7 V; cylindrical Li-ion of the kind used for long-endurance
    packs sits at 3.6 V and holds a flatter curve. The distinction is
    worth keeping because pack voltage sets current for a given power,
    and current is what the resistive loss is quadratic in.
    """
    return {"lipo": 3.7, "li-ion": 3.6, "li-ion-semisolid": 3.65}.get(
        chemistry, 3.7)


#: Mass a real pack carries beyond its cells: interconnects, BMS,
#: balance leads, connectors, potting and case. 15 % is the usual figure
#: for a well-built UAV pack and is an assumption here, not a
#: measurement — but leaving it out entirely is not neutral, it is an
#: optimistic choice dressed as an omission.
PACK_OVERHEAD_FRACTION = 0.15


def cell_specific_energy(chemistry: str) -> float:
    """
    **Cell**-level specific energy [Wh/kg], on the chemistry's own
    open-circuit energy.

    Anchored on the Molicel P42A: 4.2 Ah at a 3.73 V mean open-circuit
    voltage is 15.7 Wh in a 70 g cell, or 224 Wh/kg. Datasheet
    watt-hours are quoted at the terminals under load and come out
    5-6 % lower; the model takes that loss out separately, so using the
    loaded figure here would count it twice.
    """
    return {"lipo": 190.0, "li-ion": 224.0,
            "li-ion-semisolid": 260.0}.get(chemistry, 190.0)


def pack_specific_energy(chemistry: str) -> float:
    """Pack-level specific energy [Wh/kg], cells plus their overhead."""
    return cell_specific_energy(chemistry) / (1.0 + PACK_OVERHEAD_FRACTION)


def derive(key: str, aspect_ratio: float, rotor_diameter_m: float,
           n_rotors: int, rotor_parking: str) -> tuple:
    """
    Turn one reference aircraft into a full parameter set.

    Returns ``(AircraftEnergyParams, provenance dict)``.
    """
    r = REFERENCES[key]
    prov = {"_source_url": r["url"],
            "_configuration": r["configuration"]}

    def ref(field, note):
        prov[field] = f"referenced — {note}"

    def der(field, note):
        prov[field] = f"derived — {note}"

    def asm(field, note):
        prov[field] = f"assumed — {note}"

    b = r["wingspan_m"]
    # The vehicle file carries the aircraft *empty*; RAPTOR adds payload
    # on top via with_payload(). Storing the datasheet MTOW here would
    # make every loaded mission fly over MTOW.
    m_max = r["m_tow_kg"]
    m_payload = r["payload_max_kg"]
    m = m_max - m_payload

    # ── Wing ──────────────────────────────────────────────────────────
    # Span is published; aspect ratio is not, and neither is wing area.
    # Fixing AR from the airframe's visible proportions and deriving the
    # area is the honest direction: AR is a shape you can read off a
    # photograph, whereas area is not, and S = b²/AR then follows with
    # no further assumption.
    S_ref = b ** 2 / aspect_ratio
    ref("m_tow_max", f"{m_max} kg maximum take-off mass")
    ref("payload_capacity_kg", f"{m_payload} kg maximum payload")
    ref("wind_rating_beaufort",
        f"level {r['wind_beaufort']} wind resistance. Beaufort "
        f"{r['wind_beaufort']} is a band; the limit is read at its lower "
        f"edge")
    der("m_tow", f"{m_max} kg MTOW minus {m_payload} kg payload = {m:.2f} kg "
                 f"empty. RAPTOR adds payload on top of this, so the field "
                 f"holds the empty mass, not the MTOW")
    der("S_ref", f"b^2/AR with b = {b} m (referenced) and AR = "
                 f"{aspect_ratio} (assumed), giving {S_ref:.3f} m^2 and a "
                 f"mean chord of {S_ref/b:.3f} m")
    asm("AR", f"{aspect_ratio}, read from the airframe's proportions; not "
              f"published. Wing loading follows at "
              f"{m/S_ref:.1f} kg/m^2, which is the number to sanity-check")

    # ── Rotors ────────────────────────────────────────────────────────
    A_total = n_rotors * math.pi * (rotor_diameter_m / 2) ** 2
    ref("n_rotors", f"{n_rotors}, from the published configuration "
                    f"({r['configuration']})")
    asm("rotor_diameter_m",
        f"{rotor_diameter_m:.3f} m ({rotor_diameter_m/0.0254:.0f} in), the "
        f"usual size for this mass class; not published. Gives a disk "
        f"loading of {m/A_total:.1f} kg/m^2 against the 5-15 kg/m^2 that "
        f"multirotors of this class run")
    ref("rotor_parking", f"{rotor_parking} — follows from the configuration, "
                         f"not a free choice")

    # ── Battery ───────────────────────────────────────────────────────
    v_cell = nominal_cell_voltage(r["pack_chemistry"])
    v_nom = r["cells_series"] * v_cell
    spec_e = pack_specific_energy(r["pack_chemistry"])

    # Energy from the measured OCV curve, not from the nominal voltage.
    # The nominal is a rounded convention; the curve is what the cells do.
    from raptor.battery import mean_cell_voltage
    v_mean = mean_cell_voltage(r["pack_chemistry"])

    # Which pack the aircraft can actually carry. The manufacturers list
    # options, and not all of them fit inside the empty mass they also
    # publish once the pack's own wiring, BMS and case are counted. Take
    # the largest listed option that closes the budget; if none does,
    # the capacity is derived from the budget and flagged as such.
    cap_ah, cap_basis = choose_pack(r, spec_e, v_mean)
    energy_wh = v_mean * r["cells_series"] * cap_ah
    m_batt = energy_wh / spec_e

    ref("battery_cells_series", f"{r['cells_series']}S pack")
    ref("cell_chemistry", f"{r['pack_chemistry']} — from the pack the "
                          f"manufacturer specifies")
    asm("cell_capacity_ah",
        f"{CELL_MODEL['capacity_ah']} Ah — the manufacturer does not say "
        f"which cell the pack is built from, so a {CELL_MODEL['name']} is "
        f"assumed. It implies {cap_ah/CELL_MODEL['capacity_ah']:.1f} "
        f"parallel strings, which is what divides the pack's internal "
        f"resistance")
    der("battery_voltage",
        f"{r['cells_series']}S x {v_cell} V nominal ({r['pack_chemistry']}) "
        f"= {v_nom:.1f} V")
    der("battery_mass",
        f"{energy_wh:.0f} Wh at {spec_e:.0f} Wh/kg pack level = "
        f"{m_batt:.2f} kg, which is {100*m_batt/m_max:.0f}% of the "
        f"{m_max} kg MTOW. That is "
        f"{energy_wh/cell_specific_energy(r['pack_chemistry']):.2f} kg of "
        f"cells plus {100*PACK_OVERHEAD_FRACTION:.0f}% for wiring, BMS "
        f"and case")
    der("specific_energy",
        f"{spec_e:.0f} Wh/kg at pack level = "
        f"{cell_specific_energy(r['pack_chemistry']):.0f} Wh/kg of cells "
        f"divided by 1 + {PACK_OVERHEAD_FRACTION:.2f} of overhead")
    prov["battery_capacity_mah"] = (
        f"{cap_basis} — {cap_ah:.1f} Ah, giving {energy_wh:.0f} Wh at a "
        f"{v_mean:.3f} V measured mean cell open-circuit voltage")

    # Internal resistance. Not published by anyone, and it is what sets
    # voltage sag at the hover current, so it gets an explicit basis.
    r_cell = CELL_MODEL["dc_resistance_ohm"]
    n_parallel = cap_ah / CELL_MODEL["capacity_ah"]
    r_pack = r["cells_series"] * r_cell / n_parallel
    asm("cell_internal_resistance_ohm",
        f"{r_cell*1000:.0f} mOhm DC for one {CELL_MODEL['name']}. Over "
        f"{r['cells_series']}S{n_parallel:.1f}P that is {r_pack*1000:.0f} "
        f"mOhm at the pack, which sets voltage sag at hover current. "
        f"Measure it if the pack is in hand")

    # ── Efficiencies ──────────────────────────────────────────────────
    if rotor_parking == "tilted":
        eta_vtol, eta_cruise = 0.60, 0.55
        asm("eta_cruise",
            f"{eta_cruise}, well below a lift+cruise machine's. A "
            f"tilt-rotor's cruise propellers are the hover rotors: large "
            f"diameter, low pitch, sized for static thrust. Run at cruise "
            f"advance ratio they are badly off-design. The tilt-rotor pays "
            f"here what a quadplane pays in parked-rotor drag")
    else:
        eta_vtol, eta_cruise = 0.62, 0.72
        asm("eta_cruise",
            f"{eta_cruise}, a dedicated pusher propeller sized for cruise")
    asm("eta_vtol",
        f"{eta_vtol}, motor x ESC x propeller in the hover working point")

    # ── Airframe geometry for the drag build-up ───────────────────────
    # Fuselage length is published; diameter is not, but it is bounded
    # below by the payload it has to swallow. A 2.5 kg cold-chain box is
    # roughly a 0.20 m cube, so the fuselage cannot be slimmer than that.
    fus_len = r["length_m"]
    fus_dia = max(0.16, 0.62 * r["payload_max_kg"] ** (1 / 3) * 0.30)
    geometry = dict(
        fuselage_length=round(fus_len, 3),
        fuselage_diameter=round(fus_dia, 3),
        boom_length=round(0.55 * b, 3),
        boom_diameter=round(0.020 + 0.0015 * m, 4),
        n_booms=2 if n_rotors == 4 else 1,
        tail_area=round(0.11 * S_ref, 4),
        tail_chord=round(0.55 * (S_ref / b), 3),
    )
    ref("geometry.fuselage_length", f"{fus_len} m overall length")
    der("geometry.fuselage_diameter",
        f"{fus_dia:.3f} m, from the {r['payload_max_kg']} kg payload's bulk "
        f"at a cold-chain packing density; not published")
    asm("geometry.boom_length",
        f"{geometry['boom_length']} m, 55% of span — the rotors sit "
        f"outboard of the fuselage but well inboard of the tips")
    asm("geometry.tail_area",
        f"{geometry['tail_area']} m^2, 11% of wing area (horizontal plus "
        f"vertical), a conventional tail volume for this layout")

    ac = AircraftEnergyParams(
        name=r["name"],
        description=(
            f"{r['configuration']}, {m} kg MTOW, {r['payload_max_kg']} kg "
            f"max payload. Parameters derived from published data; see "
            f"provenance."
        ),
        m_tow=round(m, 3),
        m_tow_max=m_max,
        payload_capacity_kg=m_payload,
        wind_rating_beaufort=r["wind_beaufort"],
        S_ref=round(S_ref, 4),
        AR=aspect_ratio,
        n_rotors=n_rotors,
        rotor_diameter_m=rotor_diameter_m,
        rotor_parking=rotor_parking,
        A_rotor=round(A_total, 4),
        eta_vtol=eta_vtol,
        eta_cruise=eta_cruise,
        eta_prop=round(0.5 * (eta_vtol + eta_cruise), 3),
        battery_mass=round(m_batt, 3),
        specific_energy=round(spec_e, 2),
        battery_voltage=round(v_nom, 2),
        battery_cells_series=r["cells_series"],
        cell_chemistry=r["pack_chemistry"],
        cell_capacity_ah=CELL_MODEL["capacity_ah"],
        battery_capacity_mah=cap_ah * 1000.0,
        cell_internal_resistance_ohm=r_cell,
        geometry=geometry,
        provenance=prov,
    )
    return ac, r


#: Aspect ratio, rotor diameter, rotor count and parking, per aircraft.
#: These are the four judgement calls the derivation rests on; they are
#: gathered here so a reader can change one and re-run.
DESIGN_CHOICES = {
    "va17": dict(aspect_ratio=7.5, rotor_diameter_m=0.406,   # 16 in
                 n_rotors=3, rotor_parking="tilted"),
    "va23": dict(aspect_ratio=8.0, rotor_diameter_m=0.559,   # 22 in
                 n_rotors=4, rotor_parking="aligned"),
    "va25": dict(aspect_ratio=8.5, rotor_diameter_m=0.559,   # 22 in
                 n_rotors=4, rotor_parking="aligned"),
}


# ═══════════════════════════════════════════════════════════════════════════
# CHECKS
# ═══════════════════════════════════════════════════════════════════════════

def cross_check(ac: AircraftEnergyParams, r: dict) -> list:
    """
    Compare the derived aircraft against what the manufacturer claims.

    These are not tests the model can pass by construction — the vehicle
    was built from the hardware specification, and the endurance and
    cruise-speed claims were held back precisely so they could be used
    as independent checks on it.
    """
    out = []
    v_stall_sl = ac.stall_speed_at(REFERENCE_ALT_M)
    v_stall_alt = ac.stall_speed_at(PLANNING_ALT_M)
    lo, hi = r["cruise_speed_ms"]

    out.append(
        f"stall {v_stall_sl:.1f} m/s at sea level, {v_stall_alt:.1f} m/s at "
        f"{PLANNING_ALT_M:.0f} m ({isa_density(PLANNING_ALT_M):.3f} kg/m^3)"
    )
    out.append(
        f"published cruise {lo:.0f}-{hi:.0f} m/s sits at "
        f"{lo/v_stall_sl:.2f}-{hi/v_stall_sl:.2f} x stall at sea level, "
        f"{lo/v_stall_alt:.2f}-{hi/v_stall_alt:.2f} x at "
        f"{PLANNING_ALT_M:.0f} m"
    )
    if hi < v_stall_sl * 1.15:
        out.append(
            "  ! published cruise is under 1.15 x stall even at sea level; "
            "the wing area or C_L_max is probably wrong"
        )

    # Endurance. The published claim states a payload and an economy
    # cruise, so the comparison is made at those, not at MTOW and not at
    # the fast end of the band — otherwise the model is being asked a
    # different question than the manufacturer answered.
    usable = ac.battery_energy_wh * ac.battery_usable_fraction
    p_avg_claim = usable / (r["endurance_min"] / 60.0)
    out.append(
        f"published endurance {r['endurance_min']:.0f} min at "
        f"{r['endurance_payload_kg']} kg implies {p_avg_claim:.0f} W average "
        f"electrical from {usable:.0f} Wh usable"
    )

    from raptor.energy import BatteryModel, power_fw_cruise
    loaded = ac.with_payload(r["endurance_payload_kg"])
    p_shaft = power_fw_cruise(loaded, lo, REFERENCE_ALT_M)
    p_elec = p_shaft / loaded.eta_cruise_effective + HOTEL_LOAD_W
    out.append(
        f"model at {lo:.1f} m/s and {loaded.m_tow:.2f} kg all-up: "
        f"{p_shaft:.0f} W shaft / {loaded.eta_cruise_effective:.2f} + "
        f"{HOTEL_LOAD_W:.0f} W hotel = {p_elec:.0f} W electrical "
        f"({p_elec/p_avg_claim:.2f} x the implied average)"
    )

    # And the same comparison through the pack model itself, which is
    # what the mission planner will use: OCV curve, internal resistance
    # and the usable-energy derating, rather than a flat watt-hour
    # division.
    pack = BatteryModel(ac)
    t_s = 0.0
    while pack.SOC > 0.0 and t_s < 8 * 3600:
        pack.discharge(p_elec, 60.0)
        t_s += 60.0
    ratio = (t_s / 60.0) / r["endurance_min"]
    out.append(
        f"pack model holds {p_elec:.0f} W for {t_s/60:.0f} min against a "
        f"claimed {r['endurance_min']:.0f} min ({ratio:.2f} x)"
    )
    if ratio < 1.0:
        out.append(
            "  ! the model runs out before the manufacturer says it "
            "should, on cruise power alone — a real mission also pays for "
            "hover, so the gap is larger than it looks"
        )
    elif ratio > 1.6:
        out.append(
            "  ! the model outlasts the claim by more than the VTOL "
            "phases can account for; the pack is probably smaller than "
            "the listing suggests, or the claim is conservative"
        )

    # The one battery check available without a bench.
    from raptor.battery import check_pack_against_datasheet
    chk = check_pack_against_datasheet(
        ac.cell_chemistry, CELL_MODEL["capacity_ah"], CELL_MODEL["typical_wh"])
    out.append(
        f"cell curve vs {CELL_MODEL['name']} datasheet: {chk['note']}"
        + ("" if chk["within_tolerance"] else "   ! outside tolerance")
    )

    # ── What the published endurance implies about efficiency ──────
    # The model and the claim disagree; rather than leaving that as an
    # unexplained gap, invert it. Hold everything else and solve for the
    # cruise propulsive efficiency that would make the model reproduce
    # the manufacturer's own endurance figure. The answer is a
    # *measurement of the claim*, not a fitted parameter — and the
    # ordering across the three aircraft is a physical prediction: a
    # tilt-rotor cruising on hover-sized rotors should come out worst.
    implied = p_shaft / max(p_avg_claim - HOTEL_LOAD_W, 1e-9)
    out.append(
        f"implied cruise efficiency, if the claim is exact: "
        f"{implied:.2f} (assumed {ac.eta_cruise_effective:.2f})"
    )
    if implied > 0.85:
        out.append(
            "  ! over 0.85 — no propeller does that, so the claim cannot "
            "be met at this speed however efficient the aircraft is"
        )

    # Before calling a claim unachievable, check the obvious alternative:
    # an endurance figure is often quoted at the minimum-power speed, not
    # at cruise. Those are different speeds and a fixed wing loiters well
    # below the speed it cruises at.
    v_grid = np.linspace(1.02 * loaded.stall_speed_at(REFERENCE_ALT_M),
                         1.8 * loaded.stall_speed_at(REFERENCE_ALT_M), 60)
    p_grid = np.array([
        power_fw_cruise(loaded, float(v), REFERENCE_ALT_M)
        / loaded.eta_cruise_effective + HOTEL_LOAD_W for v in v_grid])
    v_loiter = float(v_grid[int(np.argmin(p_grid))])
    p_loiter = float(np.min(p_grid))
    t_loiter = usable / p_loiter * 60.0
    t_cruise = t_s / 60.0
    out.append(
        f"minimum-power speed {v_loiter:.1f} m/s at {p_loiter:.0f} W gives "
        f"{t_loiter:.0f} min — the most this aircraft can stay up"
    )

    # The verdict. An endurance figure is a number without a condition
    # unless the manufacturer states the speed, and none of these do. So
    # the honest test is not "does the model match the claim" but "does
    # the claim fall between the model's cruise and loiter endurances" —
    # the band inside which any real flight profile must sit.
    lo_t, hi_t = min(t_cruise, t_loiter), max(t_cruise, t_loiter)
    claim = r["endurance_min"]
    if lo_t <= claim <= hi_t:
        out.append(
            f"VERDICT: the {claim:.0f} min claim falls between the model's "
            f"cruise ({t_cruise:.0f} min) and loiter ({t_loiter:.0f} min) "
            f"endurances — consistent, at an unstated speed between the two"
        )
    elif claim < lo_t:
        out.append(
            f"VERDICT: the {claim:.0f} min claim is below the model's whole "
            f"band ({lo_t:.0f}-{hi_t:.0f} min) — conservative, or the pack "
            f"is smaller than the airframe's mass budget allows for"
        )
    else:
        out.append(
            f"VERDICT: the {claim:.0f} min claim exceeds what this pack can "
            f"do at any speed ({lo_t:.0f}-{hi_t:.0f} min). Not reachable."
        )

    lo_w, hi_w = BEAUFORT_MS[r["wind_beaufort"]]
    out.append(
        f"wind rating Beaufort {r['wind_beaufort']} = {lo_w:.1f}-{hi_w:.1f} m/s"
    )
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="report only; do not write JSON")
    ap.add_argument("--no-aero", action="store_true",
                    help="skip the AeroSandbox polar (fast, less accurate)")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args(argv)

    if not args.check:
        os.makedirs(args.out, exist_ok=True)

    for key in ("va17", "va23", "va25"):
        ac, r = derive(key, **DESIGN_CHOICES[key])
        print("=" * 74)
        print(f"  {ac.name}   ({r['configuration']})")
        print("=" * 74)
        print(f"  empty {ac.m_tow:.2f} kg + {ac.payload_capacity_kg:.1f} kg payload "
              f"= {ac.m_tow_max:.1f} kg MTOW   span {r['wingspan_m']:.2f} m   "
              f"S {ac.S_ref:.3f} m^2   AR {ac.AR:.1f}   "
              f"chord {ac.wing_chord:.3f} m")
        print(f"  at MTOW: wing loading {ac.m_tow_max/ac.S_ref:.1f} kg/m^2   "
              f"disk loading {ac.m_tow_max/ac.rotor_area_total:.1f} kg/m^2   "
              f"{ac.n_rotors} x {ac.rotor_diameter_m*100:.0f} cm rotors "
              f"({ac.rotor_parking})")
        print(f"  battery {ac.battery_cells_series}S"
              f"{ac.battery_strings_parallel:.1f}P "
              f"{ac.battery_capacity_ah:.1f} Ah = {ac.battery_energy_wh:.0f} Wh, "
              f"{ac.battery_mass:.2f} kg "
              f"({100*ac.battery_mass/ac.m_tow_max:.0f}% of MTOW), "
              f"pack R {1000*ac.battery_resistance:.0f} mOhm, "
              f"{ac.cell_chemistry}")

        if not args.no_aero:
            print("  building aero model ...", end="", flush=True)
            ac = ac.build_aero()
            print(f" C_L_max {ac.C_L_max:.3f}, wing C_D0 {ac.C_D0:.4f}, "
                  f"e {ac.e_oswald:.3f}")
            bu = ac.drag_buildup
            if bu is not None:
                v_ref = 1.3 * ac.stall_speed_at(PLANNING_ALT_M)
                cd0_air = bu.C_D0(v_ref, PLANNING_ALT_M)
                print(f"  airframe C_D0 {cd0_air:.4f} at {v_ref:.1f} m/s "
                      f"on top of the wing (total {ac.C_D0 + cd0_air:.4f})")

        print("  --- cross-checks against published performance ---")
        for line in cross_check(ac, r):
            print(f"  {line}")

        cap_ah = ac.battery_capacity_ah
        budget = mass_budget(r, ac.battery_mass)
        print("  --- mass budget ---")
        print(f"  battery {budget['battery']:5.2f} + frame {budget['frame']:5.2f} "
              f"({budget['frame_source']}) + propulsion "
              f"{budget['propulsion']:5.2f} + avionics "
              f"{budget['avionics']:5.2f}")
        print(f"  = {budget['accounted']:5.2f} kg against "
              f"{budget['empty']:5.2f} kg empty  ->  "
              f"{budget['residual']:+5.2f} kg"
              + ("" if budget["residual"] >= 0 else
                 "   ! DOES NOT CLOSE"))
        if budget["residual"] < 0:
            spec_e = pack_specific_energy(r["pack_chemistry"])
            from raptor.battery import mean_cell_voltage
            v_mean = mean_cell_voltage(r["pack_chemistry"])
            ah_fits = (budget["battery_mass_that_fits"] * spec_e
                       / (v_mean * r["cells_series"]))
            print(f"  the pack, airframe and propulsion together exceed the "
                  f"published empty mass by {-budget['residual']:.2f} kg.")
            print(f"  the largest pack that fits the budget is "
                  f"{ah_fits:.1f} Ah, against the {cap_ah:.0f} Ah assumed "
                  f"here. Endurance scales with it.")

        problems = ac.validate()
        if problems:
            print("  --- validate() ---")
            for p in problems:
                print(f"  ! {p}")
        else:
            print("  validate(): no warnings")

        if not args.check:
            path = os.path.join(args.out, f"{key}.json")
            ac.to_json(path)
            print(f"  wrote {path}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
