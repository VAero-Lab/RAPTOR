"""
Medical Delivery Scenarios — Flight Mission Definitions
=========================================================

Defines the real-world medical delivery flight scenarios for the
Quito Metropolitan District (DMQ) eVTOL path planning optimization.

Facilities:
    - Major public hospitals (hangars / UAV bases)
    - Major private hospitals and clinics
    - Rural health sub-centers (delivery/pickup points)

Scenarios:
    1. Outbound loaded, return empty   (supply delivery)
    2. Outbound empty, return loaded    (sample collection)
    3. Both ways loaded                 (supply + sample exchange)
    4. Multi-point tour (3–4 stops)     (multi-facility logistics)

Each scenario includes:
    - Origin/destination facilities
    - Payload weight for each leg
    - Optimization priority (energy, time, or balanced)
    - Medical urgency level
    - DEM challenge description

Author: Victor (LUAS-EPN / KU Leuven)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# FACILITY DATABASE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Facility:
    """A medical facility with geographic data and classification."""
    name: str
    short_name: str
    lat: float
    lon: float
    ground_elev: float       # m AMSL (from DEM or known)
    facility_type: str       # 'hospital_public', 'hospital_private', 'clinic', 'sub_center'
    is_hangar: bool = False  # True if UAV base/hangar
    helipad: bool = True     # VTOL landing capability

    def coords(self) -> Tuple[float, float]:
        return (self.lat, self.lon)



# ── A note on these coordinates ───────────────────────────────────────────
#
# Every position below was replaced in August 2026 after a check against
# the DEM found six facilities sitting more than a corridor height (62 m)
# from the ground beneath them. The elevations were mostly right; the
# *coordinates* were not, and on the western slope a coordinate a
# kilometre out is half a kilometre of vertical error.
#
# The worst was Nanegalito, whose latitude carried the wrong sign: it sits
# at +0.063, north of the equator, and was listed at -0.067. That is
# **14.4 km** of error, and it put the facility 505 m above where it is.
#
# Positions now come from named OpenStreetMap features — the hospital
# itself where OSM has one, the parish centre otherwise — and each
# elevation is read from the shipped DEM at that point, so position and
# height cannot disagree. The hand-collected elevations are kept in the
# comments as a cross-check: after the correction, the worst residual
# between them is 89 m and most are inside 50 m, against 505 m before.
#
# Re-check with:  python -m scripts.check_facilities
#
# ── Major Public Hospitals (UAV Hangars) ──────────────────────────────────

HOSPITAL_ENRIQUE_GARCES = Facility(
    name="Hospital General Enrique Garcés",
    short_name="H.Garcés",
    lat=-0.24444, lon=-78.54119,
    ground_elev=2907.0,
    facility_type="hospital_public",
    is_hangar=True,
    # OSM node: Hospital del Sur (Enrique Garcés); elevation read from the shipped DEM at that point.
)

HOSPITAL_EUGENIO_ESPEJO = Facility(
    name="Hospital de Especialidades Eugenio Espejo",
    short_name="H.Espejo",
    lat=-0.21531, lon=-78.49848,
    ground_elev=2835.0,
    facility_type="hospital_public",
    is_hangar=True,
    # OSM node: Hospital Eugenio Espejo; elevation read from the shipped DEM at that point.
)

HOSPITAL_PABLO_ARTURO_SUAREZ = Facility(
    name="Hospital General Pablo Arturo Suárez",
    short_name="H.P.A.Suárez",
    lat=-0.12781, lon=-78.49734,
    ground_elev=2831.0,
    facility_type="hospital_public",
    is_hangar=True,
    # OSM node: Hospital Pablo Arturo Suárez; elevation read from the shipped DEM at that point.
)

HOSPITAL_DOCENTE_CALDERON = Facility(
    name="Hospital Docente de Calderón",
    short_name="H.Calderón",
    lat=-0.09251, lon=-78.43681,
    ground_elev=2705.0,
    facility_type="hospital_public",
    is_hangar=True,
    # OSM node: Hospital Docente Calderón; elevation read from the shipped DEM at that point.
)

HOSPITAL_METROPOLITANO = Facility(
    name="Hospital Metropolitano",
    short_name="H.Metropolitano",
    lat=-0.18459, lon=-78.50356,
    ground_elev=2927.0,
    facility_type="hospital_private",
    is_hangar=True,
    # OSM node: Hospital Metropolitano; elevation read from the shipped DEM at that point.
)

# ── Private Hospitals / Clinics ───────────────────────────────────────────

CLINICA_PASTEUR = Facility(
    name="Clínica Pasteur",
    short_name="Cl.Pasteur",
    lat=-0.19384, lon=-78.49099,
    ground_elev=2789.0,
    facility_type="clinic",
    is_hangar=False,
    # OSM node: Clínica Pasteur; elevation read from the shipped DEM at that point.
)

HOSPITAL_VOZANDES = Facility(
    name="Hospital Voz Andes",
    short_name="H.VozAndes",
    lat=-0.1728, lon=-78.48941,
    ground_elev=2803.0,
    facility_type="hospital_private",
    is_hangar=False,
    # OSM node: Hospital Vozandes; elevation read from the shipped DEM at that point.
)

# ── Rural Health Sub-Centers ──────────────────────────────────────────────

CS_LLOA = Facility(
    name="Centro de Salud Lloa",
    short_name="CS.Lloa",
    lat=-0.24813, lon=-78.58313,
    ground_elev=3065.0,
    facility_type="sub_center",
    is_hangar=False,
    # OSM: Lloa parish centre; elevation read from the shipped DEM at that point.
)

CS_PINTAG = Facility(
    name="Centro de Salud Píntag",
    short_name="CS.Píntag",
    lat=-0.37143, lon=-78.37536,
    ground_elev=2866.0,
    facility_type="sub_center",
    is_hangar=False,
    # OSM: Píntag parish centre; elevation read from the shipped DEM at that point.
)

CS_NANEGALITO = Facility(
    name="Centro de Salud Nanegalito",
    short_name="CS.Nanegalito",
    lat=0.06301, lon=-78.68144,
    ground_elev=1586.0,
    facility_type="sub_center",
    is_hangar=False,
    # OSM: Nanegalito parish centre; elevation read from the shipped DEM at that point.
)

CS_NONO = Facility(
    name="Centro de Salud Nono",
    short_name="CS.Nono",
    lat=-0.06544, lon=-78.57695,
    ground_elev=2733.0,
    facility_type="sub_center",
    is_hangar=False,
    # OSM: Nono parish centre; elevation read from the shipped DEM at that point.
)

CS_GUANGOPOLO = Facility(
    name="Centro de Salud Guangopolo",
    short_name="CS.Guangopolo",
    lat=-0.25712, lon=-78.45091,
    ground_elev=2465.0,
    facility_type="sub_center",
    is_hangar=False,
    # OSM node: Centro de Salud Guangopolo; elevation read from the shipped DEM at that point.
)

CS_CALACALI = Facility(
    name="Centro de Salud Calacalí",
    short_name="CS.Calacalí",
    lat=-0.00082, lon=-78.51438,
    ground_elev=2820.0,
    facility_type="sub_center",
    is_hangar=False,
    # OSM: Calacalí parish centre; elevation read from the shipped DEM at that point.
)

CS_AMAGUANIA = Facility(
    name="Centro de Salud Amaguaña",
    short_name="CS.Amaguaña",
    lat=-0.37665, lon=-78.50564,
    ground_elev=2605.0,
    facility_type="sub_center",
    is_hangar=False,
    # OSM: Amaguaña parish centre; elevation read from the shipped DEM at that point.
)

CS_CONOCOTO = Facility(
    name="Centro de Salud Conocoto",
    short_name="CS.Conocoto",
    lat=-0.29203, lon=-78.4778,
    ground_elev=2539.0,
    facility_type="sub_center",
    is_hangar=False,
    # OSM: Conocoto parish centre; elevation read from the shipped DEM at that point.
)

# Master facility list
ALL_FACILITIES = [
    HOSPITAL_ENRIQUE_GARCES, HOSPITAL_EUGENIO_ESPEJO,
    HOSPITAL_PABLO_ARTURO_SUAREZ, HOSPITAL_DOCENTE_CALDERON,
    HOSPITAL_METROPOLITANO,
    CLINICA_PASTEUR, HOSPITAL_VOZANDES,
    CS_LLOA, CS_PINTAG, CS_NANEGALITO, CS_NONO,
    CS_GUANGOPOLO, CS_CALACALI, CS_AMAGUANIA, CS_CONOCOTO,
]

HANGARS = [f for f in ALL_FACILITIES if f.is_hangar]
SUB_CENTERS = [f for f in ALL_FACILITIES if f.facility_type == "sub_center"]


def update_facility_elevations(dem, verbose: bool = False) -> list:
    """
    Replace declared facility elevations with the terrain model's.

    The elevations in this file were entered by hand and most of them are
    wrong — thirteen of fifteen disagree with the DEM by more than 10 m,
    one by 499 m. That is not a cosmetic error: departure and arrival
    phases are built from the declared value, so the whole altitude
    schedule of a mission starts from the wrong datum, and a 60 m error
    is most of the legal corridor.

    This function existed before but nothing called it, so the catalogue
    always used the typed-in numbers. :func:`build_scenario_catalog` now
    calls it whenever it is given a DEM.

    Parameters
    ----------
    dem : DEMInterface
    verbose : bool
        Print each correction.

    Returns
    -------
    List of ``(name, declared, actual)`` for every facility that moved.
    """
    changed = []
    for fac in ALL_FACILITIES:
        elev = dem.elevation(fac.lat, fac.lon)
        if np.isnan(elev):
            continue
        before = float(fac.ground_elev)
        after = float(elev)
        if abs(after - before) > 0.5:
            changed.append((fac.name, before, after))
            if verbose:
                print(f"  {fac.name}: {before:.0f} → {after:.0f} m "
                      f"({after - before:+.0f})")
        fac.ground_elev = after
    return changed


# ═══════════════════════════════════════════════════════════════════════════
# OPTIMIZATION PRIORITY
# ═══════════════════════════════════════════════════════════════════════════

class OptPriority(Enum):
    """Optimization priority flag for a scenario."""
    ENERGY = "energy"     # Minimize battery consumption (time constrained)
    TIME = "time"         # Minimize flight time (energy constrained)
    BALANCED = "balanced" # Multi-objective weighted sum

    def default_weights(self) -> Tuple[float, float]:
        """Default (w_energy, w_time) weights for multi-objective."""
        if self == OptPriority.ENERGY:
            return (0.9, 0.1)
        elif self == OptPriority.TIME:
            return (0.1, 0.9)
        else:
            return (0.5, 0.5)


class MedicalUrgency(Enum):
    """Medical urgency level influencing time constraints."""
    ROUTINE = "routine"           # Scheduled resupply, flexible time
    URGENT = "urgent"             # Needed within hours
    EMERGENCY = "emergency"       # Life-threatening, minimize time

    def max_flight_time_s(self) -> float:
        """Maximum allowed flight time per leg [s]."""
        if self == MedicalUrgency.ROUTINE:
            return 3600.0   # 1 hour
        elif self == MedicalUrgency.URGENT:
            return 1800.0   # 30 min
        else:
            return 900.0    # 15 min

    def min_soc_percent(self) -> float:
        """Minimum battery SOC remaining [fraction]."""
        if self == MedicalUrgency.ROUTINE:
            return 0.15
        elif self == MedicalUrgency.URGENT:
            return 0.15
        else:
            return 0.20     # Extra margin for emergencies


# ═══════════════════════════════════════════════════════════════════════════
# MISSION LEG
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MissionLeg:
    """A single flight leg between two facilities."""
    origin: Facility
    destination: Facility
    payload_kg: float = 0.0       # Cargo weight for this leg
    description: str = ""         # What's being carried

    @property
    def is_loaded(self) -> bool:
        return self.payload_kg > 0

    def __repr__(self):
        load = f"{self.payload_kg:.1f}kg" if self.is_loaded else "empty"
        return f"{self.origin.short_name} → {self.destination.short_name} ({load})"


# ═══════════════════════════════════════════════════════════════════════════
# FLIGHT SCENARIO
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FlightScenario:
    """
    A complete medical delivery flight scenario.

    A scenario consists of one or more legs, each with its own
    payload configuration. The optimization priority and medical
    urgency determine the objective weights and constraints.

    Attributes
    ----------
    name : str
        Human-readable scenario name.
    scenario_id : str
        Short identifier (e.g. "S1a", "S2b").
    legs : list of MissionLeg
        Ordered flight legs.
    priority : OptPriority
        What to optimize for.
    urgency : MedicalUrgency
        Medical urgency level.
    description : str
        Detailed description.
    dem_challenge : str
        Description of the terrain challenge.
    base_payload_kg : float
        Nominal payload mass (cargo, not aircraft structure).
    """
    name: str
    scenario_id: str
    legs: List[MissionLeg]
    priority: OptPriority
    urgency: MedicalUrgency
    description: str = ""
    dem_challenge: str = ""
    base_payload_kg: float = 2.0    # kg of medical cargo

    @property
    def n_legs(self) -> int:
        return len(self.legs)

    @property
    def is_round_trip(self) -> bool:
        return (self.n_legs >= 2 and
                self.legs[0].origin == self.legs[-1].destination)

    @property
    def is_multi_stop(self) -> bool:
        return self.n_legs >= 3

    @property
    def total_distance_estimate(self) -> float:
        """Rough total distance [m] (straight-line sum of legs)."""
        from .dem import DEMInterface
        return sum(
            DEMInterface.haversine(
                leg.origin.lat, leg.origin.lon,
                leg.destination.lat, leg.destination.lon
            )
            for leg in self.legs
        )

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"{'='*60}",
            f"Scenario: {self.name} [{self.scenario_id}]",
            f"{'='*60}",
            f"Priority:  {self.priority.value}",
            f"Urgency:   {self.urgency.value}",
            f"Legs:      {self.n_legs}",
        ]
        for i, leg in enumerate(self.legs):
            lines.append(f"  Leg {i+1}: {leg}")
        lines.append(f"DEM challenge: {self.dem_challenge}")
        lines.append(f"Description: {self.description}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# SCENARIO DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

def build_scenario_catalog(dem=None) -> Dict[str, FlightScenario]:
    """
    Build the complete catalog of medical delivery scenarios.

    Parameters
    ----------
    dem : DEMInterface, optional
        When given, facility elevations are taken from the terrain model
        instead of the values typed into this file — most of which are
        wrong, several by tens of metres. Pass it unless you have a
        reason not to.

    Returns a dictionary of scenario_id → FlightScenario.

    Scenarios are chosen to present diverse DEM challenges:
    - Urban valley flights
    - Mountain crossing (Pichincha foothills)
    - Deep valley descents (to Guangopolo, Conocoto)
    - Long-range with multiple altitude levels
    - Multi-stop logistics tours
    """
    if dem is not None:
        update_facility_elevations(dem)

    catalog = {}

    # ══════════════════════════════════════════════════════════════════════
    # SCENARIO 1: Outbound loaded → Return empty (supply delivery)
    # ══════════════════════════════════════════════════════════════════════

    # S1a: Garcés → Lloa (mountain crossing west of Quito)
    catalog["S1a"] = FlightScenario(
        name="Medical Supply Delivery to Lloa",
        scenario_id="S1a",
        legs=[
            MissionLeg(HOSPITAL_ENRIQUE_GARCES, CS_LLOA,
                       payload_kg=2.0, description="Medicines and vaccines"),
            MissionLeg(CS_LLOA, HOSPITAL_ENRIQUE_GARCES,
                       payload_kg=0.0, description="Return empty"),
        ],
        priority=OptPriority.ENERGY,
        urgency=MedicalUrgency.ROUTINE,
        description="Routine resupply of vaccines to remote Lloa community. "
                    "Energy priority since non-urgent.",
        dem_challenge="Westward flight into Pichincha foothills. Terrain rises "
                      "from 2950m to 3200m+ with ridges requiring cruise at "
                      "two levels. Short distance (4.4 km) but significant "
                      "altitude variation.",
    )

    # S1b: P.A.Suárez → Nono (mountain pass crossing)
    catalog["S1b"] = FlightScenario(
        name="Emergency Supplies to Nono",
        scenario_id="S1b",
        legs=[
            MissionLeg(HOSPITAL_PABLO_ARTURO_SUAREZ, CS_NONO,
                       payload_kg=2.5, description="Emergency medical kit"),
            MissionLeg(CS_NONO, HOSPITAL_PABLO_ARTURO_SUAREZ,
                       payload_kg=0.0, description="Return empty"),
        ],
        priority=OptPriority.TIME,
        urgency=MedicalUrgency.URGENT,
        description="Urgent delivery of emergency medical supplies to Nono. "
                    "Time-critical: prioritize speed.",
        dem_challenge="Must cross Pichincha mountain ridges (~3600m+ terrain). "
                      "Flight needs multiple cruise levels: climb over ridge, "
                      "descend into Nono valley (2710m). Major DEM obstacle.",
    )

    # ══════════════════════════════════════════════════════════════════════
    # SCENARIO 2: Outbound empty → Return loaded (sample collection)
    # ══════════════════════════════════════════════════════════════════════

    # S2a: Espejo → Guangopolo (valley descent/ascent)
    catalog["S2a"] = FlightScenario(
        name="Blood Sample Collection from Guangopolo",
        scenario_id="S2a",
        legs=[
            MissionLeg(HOSPITAL_EUGENIO_ESPEJO, CS_GUANGOPOLO,
                       payload_kg=0.0, description="Empty outbound"),
            MissionLeg(CS_GUANGOPOLO, HOSPITAL_EUGENIO_ESPEJO,
                       payload_kg=1.5, description="Blood/tissue samples"),
        ],
        priority=OptPriority.TIME,
        urgency=MedicalUrgency.URGENT,
        description="Collect blood samples from Guangopolo for lab analysis. "
                    "Samples are time-sensitive (4h viability window).",
        dem_challenge="Significant elevation drop: 2816m → 2520m. Must cross "
                      "the Ilaló hill (3169m) or route around it. Return "
                      "leg loaded uphill is energy-intensive.",
    )

    # S2b: Calderón → Conocoto (long urban traverse + valley)
    catalog["S2b"] = FlightScenario(
        name="Diagnostic Sample Collection from Conocoto",
        scenario_id="S2b",
        legs=[
            MissionLeg(HOSPITAL_DOCENTE_CALDERON, CS_CONOCOTO,
                       payload_kg=0.0, description="Empty outbound"),
            MissionLeg(CS_CONOCOTO, HOSPITAL_DOCENTE_CALDERON,
                       payload_kg=1.0, description="Diagnostic samples"),
        ],
        priority=OptPriority.BALANCED,
        urgency=MedicalUrgency.ROUTINE,
        description="Collect routine diagnostic samples from Conocoto. "
                    "Balance energy and time for regular service.",
        dem_challenge="Long traverse (~22 km) across Quito valley. Multiple "
                      "terrain undulations between 2500m and 2800m. Passes "
                      "over urban terrain with varying elevation.",
    )

    # ══════════════════════════════════════════════════════════════════════
    # SCENARIO 3: Both ways loaded (supply + sample exchange)
    # ══════════════════════════════════════════════════════════════════════

    # S3a: Metropolitano → Amaguaña (south valley, both loaded)
    catalog["S3a"] = FlightScenario(
        name="Supply Exchange with Amaguaña",
        scenario_id="S3a",
        legs=[
            MissionLeg(HOSPITAL_METROPOLITANO, CS_AMAGUANIA,
                       payload_kg=2.0, description="Medicines + supplies"),
            MissionLeg(CS_AMAGUANIA, HOSPITAL_METROPOLITANO,
                       payload_kg=1.5, description="Lab samples + reports"),
        ],
        priority=OptPriority.ENERGY,
        urgency=MedicalUrgency.ROUTINE,
        description="Bi-directional supply exchange with Amaguaña. Both legs "
                    "loaded, energy is the primary concern for sustainability.",
        dem_challenge="Southbound flight (~22 km) with descent from 2838m "
                      "to 2640m. Terrain includes the Pasochoa volcanic ridge "
                      "(3400m+) that must be avoided or overflown.",
    )

    # S3b: Garcés → Píntag (longest route, mountain terrain)
    catalog["S3b"] = FlightScenario(
        name="Emergency Exchange with Píntag",
        scenario_id="S3b",
        legs=[
            MissionLeg(HOSPITAL_ENRIQUE_GARCES, CS_PINTAG,
                       payload_kg=2.5, description="Emergency blood products"),
            MissionLeg(CS_PINTAG, HOSPITAL_ENRIQUE_GARCES,
                       payload_kg=2.0, description="Patient samples"),
        ],
        priority=OptPriority.TIME,
        urgency=MedicalUrgency.EMERGENCY,
        description="Emergency blood product delivery to Píntag + sample return. "
                    "Time-critical emergency. Longest route in catalog.",
        dem_challenge="Very long route (~32 km). Must traverse the entire "
                      "Quito valley diagonally SE. Passes near Cotopaxi foothills. "
                      "Multiple altitude changes, possibly 3 cruise levels.",
    )

    # ══════════════════════════════════════════════════════════════════════
    # SCENARIO 4: Multi-point flights (3–4 stops)
    # ══════════════════════════════════════════════════════════════════════

    # S4a: Hospital tour — 2 hospitals + 2 sub-centers
    catalog["S4a"] = FlightScenario(
        name="Multi-Point Medical Logistics Tour (North)",
        scenario_id="S4a",
        legs=[
            MissionLeg(HOSPITAL_PABLO_ARTURO_SUAREZ, CS_CALACALI,
                       payload_kg=2.0, description="Vaccines to Calacalí"),
            MissionLeg(CS_CALACALI, HOSPITAL_DOCENTE_CALDERON,
                       payload_kg=0.5, description="Samples to Calderón"),
            MissionLeg(HOSPITAL_DOCENTE_CALDERON, HOSPITAL_PABLO_ARTURO_SUAREZ,
                       payload_kg=1.0, description="Supplies return to P.A.Suárez"),
        ],
        priority=OptPriority.BALANCED,
        urgency=MedicalUrgency.ROUTINE,
        description="3-point logistics tour connecting two northern hospitals "
                    "and a rural sub-center. Balances energy and time.",
        dem_challenge="Triangular route: west to Calacalí (slight elevation "
                      "change), east to Calderón (crossing the Mitad del Mundo "
                      "ridge), south back to P.A.Suárez. Mixed terrain.",
    )

    # S4b: Southern valley multi-stop
    catalog["S4b"] = FlightScenario(
        name="Southern Valley 4-Stop Emergency Tour",
        scenario_id="S4b",
        legs=[
            MissionLeg(HOSPITAL_ENRIQUE_GARCES, CS_CONOCOTO,
                       payload_kg=2.0, description="Medicines to Conocoto"),
            MissionLeg(CS_CONOCOTO, CS_AMAGUANIA,
                       payload_kg=1.0, description="Partial supplies to Amaguaña"),
            MissionLeg(CS_AMAGUANIA, CS_GUANGOPOLO,
                       payload_kg=0.5, description="Samples via Guangopolo"),
            MissionLeg(CS_GUANGOPOLO, HOSPITAL_ENRIQUE_GARCES,
                       payload_kg=2.0, description="All samples to hospital"),
        ],
        priority=OptPriority.TIME,
        urgency=MedicalUrgency.URGENT,
        description="4-stop emergency tour through southern Quito valley. "
                    "Collect and distribute across multiple sub-centers.",
        dem_challenge="Complex terrain loop through the Los Chillos valley. "
                      "Must navigate around or over Ilaló hill (3169m). "
                      "Elevation varies 2520–2950m. Multiple cruise levels needed.",
    )

    return catalog


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_scenario(scenario_id: str) -> FlightScenario:
    """Get a specific scenario by ID."""
    catalog = build_scenario_catalog()
    if scenario_id not in catalog:
        raise KeyError(f"Unknown scenario '{scenario_id}'. "
                       f"Available: {list(catalog.keys())}")
    return catalog[scenario_id]


def get_scenarios_by_type(n_legs: int = None,
                          priority: OptPriority = None,
                          urgency: MedicalUrgency = None) -> List[FlightScenario]:
    """Filter scenarios by type, priority, or urgency."""
    catalog = build_scenario_catalog()
    result = list(catalog.values())

    if n_legs is not None:
        result = [s for s in result if s.n_legs == n_legs]
    if priority is not None:
        result = [s for s in result if s.priority == priority]
    if urgency is not None:
        result = [s for s in result if s.urgency == urgency]

    return result


def list_scenarios() -> None:
    """Print a compact summary of all scenarios."""
    catalog = build_scenario_catalog()
    print(f"{'ID':5s} {'Name':45s} {'Legs':4s} {'Priority':10s} {'Urgency':10s}")
    print("-" * 80)
    for sid, sc in catalog.items():
        print(f"{sid:5s} {sc.name:45s} {sc.n_legs:4d} "
              f"{sc.priority.value:10s} {sc.urgency.value:10s}")
