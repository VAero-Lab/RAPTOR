"""
Regulatory Profile — Codified UAS Operating Rules
==================================================

Separates *what the rules say* (this module) from *where they apply*
(``airspace``) and *how they are enforced* (``optimizer``).

A :class:`RegulatoryProfile` holds the numeric limits of a national UAS
regulation. An :class:`OperationalContext` describes the credentials and
nature of a specific flight (certificates, authorisations, urgency).
Together they answer the two questions the optimizer needs:

    1. What is the altitude ceiling here?          → ``ceiling_agl_m()``
    2. May we enter this zone, and at what cost?   → ``resolve(zone)``

The bundled profile is RDAC 101 (DGAC Ecuador, Resolution
DGAC-DGAC-2024-0100-R, 11 December 2024). The framework is
region-agnostic: build a different ``RegulatoryProfile`` for another
jurisdiction and the rest of the package is unchanged.

RDAC 101 sections encoded here
------------------------------
101.160  Urban operations — 30 m horizontal/vertical from people and
         buildings; never above 122 m AGL. Exempt for civil-emergency
         support by a public-service body, or with special authorisation.
101.165  Autonomous flight limited to 1 000 m from the launch point
         unless the operator holds a UOC or a special authorisation.
101.170  Carriage of blood, medicines or biological samples is
         prohibited *unless* the operator holds a UOC in the specific
         category plus the corresponding authorisation. Directly
         relevant to medical-delivery missions.
101.185  Maximum height 122 m (400 ft) AGL. Over a structure taller
         than 122 m: within 50 m of it, up to 15 m above its top.
101.190  Flight-restriction zones (FRZ). Restricted zones are
         overflyable *with* authorisation from the responsible body;
         prohibited zones are never overflyable.
101.195  Aerodromes. Controlled: prohibited rectangle 600 m either side
         of the runway centreline and 3 km beyond each threshold, plus
         5 km approach/departure cones; elsewhere within 5 km, flight is
         allowed up to 40 m AGL with the aerodrome operator's
         coordination. Uncontrolled/no-ATS: 1.8 km radius. Heliport:
         0.9 km radius.
101.210  BVLOS is a *specific-category* operation requiring a UOC or a
         special authorisation.

Author: Victor (LUAS-EPN / KU Leuven)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════
# PERMISSION CLASSES
# ═══════════════════════════════════════════════════════════════════════════

class PermissionClass(Enum):
    """
    How a volume of airspace may be used.

    This is the axis the optimizer reasons about. It is deliberately
    coarser than the zone *type*: many different zone types collapse to
    the same operational answer.

    FREE
        Fly without asking anyone. No penalty.
    PERMIT
        Overflight is legal but requires prior authorisation or
        coordination with the responsible body (RDAC 101.190(a)).
        The optimizer treats this as a *soft* cost: a route may cross a
        PERMIT zone if the detour would be worse, and the resulting plan
        carries an explicit list of the permits it needs.
    PROHIBITED
        No overflight under any circumstance (RDAC 101.190(b)).
        The optimizer treats this as a hard barrier.
    """
    FREE = "free"
    PERMIT = "permit"
    PROHIBITED = "prohibited"


#: Relative effort of obtaining each kind of permit, in "permit units".
#: Used to rank routes that need different authorisations. These are
#: planning heuristics, not legal quantities — override per study.
DEFAULT_PERMIT_COST: Dict[str, float] = {
    "aerodrome":              10.0,   # coordination with the aerodrome operator
    "heliport":                6.0,
    "controlled_airspace":     8.0,   # ATC coordination, Class B/C/D/E
    "military":               10.0,
    "government":             10.0,
    "hospital":                3.0,   # 101.190(a)(3) — surrounding hospitals
    "strategic_infrastructure": 6.0,
    "industrial":              3.0,
    "ecological":              5.0,   # MAATE / protected-area permit
    "crowd":                   7.0,   # 101.190(a)(5) — outdoor gatherings
    "segregated":              6.0,
    "populated":               2.0,   # local GAD ordinance, 101.160(d)
    "generic":                 5.0,
}


# ═══════════════════════════════════════════════════════════════════════════
# OPERATIONAL CONTEXT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class OperationalContext:
    """
    The credentials and nature of one specific operation.

    Which regulatory limits actually bind depends on this. A UOC holder
    flying an emergency medical delivery is governed by a different
    envelope than a recreational pilot, even over identical terrain.

    Attributes
    ----------
    category : str
        ``'open'``, ``'specific'`` or ``'certified'`` (RDAC 101.060).
        Medical cargo delivery is always ``'specific'`` — see 101.170.
    has_uoc : bool
        Operator holds a UAS Operating Certificate (UOC, Chapter E).
        Required for BVLOS, for autonomous flight beyond 1 000 m, and
        for carrying medical payloads.
    has_special_authorization : bool
        Holds a *special UAS flight authorisation* (Chapter G) covering
        deviations from Chapter B operating rules.
    bvlos : bool
        Operation is beyond visual line of sight (101.210).
    autonomous : bool
        Flight is executed autonomously from a pre-programmed plan
        (101.165).
    emergency_public_service : bool
        Civil-emergency support flown by a public-service body. Grants
        the 101.160(b) exemption from the urban-operation restrictions.
    carries_medical_payload : bool
        Blood, medicines or biological samples (101.170(a)(1)(ii)).
    authorized_zone_ids : list of str
        Zone IDs for which authorisation has *already* been obtained.
        These become effectively FREE for this operation — this is how
        you model "we have the hospital's standing permission".
    max_agl_override_m : float or None
        Ceiling granted by a special authorisation, in m AGL. Overrides
        the 122 m default when set.
    """
    category: str = "specific"
    has_uoc: bool = True
    has_special_authorization: bool = False
    bvlos: bool = True
    autonomous: bool = True
    emergency_public_service: bool = False
    carries_medical_payload: bool = True
    authorized_zone_ids: List[str] = field(default_factory=list)
    max_agl_override_m: Optional[float] = None

    # ── Compliance self-check ─────────────────────────────────────────

    def compliance_findings(self, profile: "RegulatoryProfile") -> List[str]:
        """
        Static (route-independent) regulatory findings for this context.

        Returns a list of human-readable strings describing requirements
        this operation triggers *before* any route is planned. An empty
        list means the operation as configured is self-consistent.
        """
        out: List[str] = []

        if self.carries_medical_payload:
            if not (self.has_uoc and self.category == "specific"):
                out.append(
                    "RDAC 101.170(a)(1)(ii): carriage of blood, medicines or "
                    "biological samples requires a UOC in the specific "
                    "category plus the corresponding authorisation."
                )
            else:
                out.append(
                    "RDAC 101.170: medical payload — UOC (specific category) "
                    "held; sectoral health-authority authorisation must be "
                    "on file for each flight."
                )

        if self.bvlos:
            if not (self.has_uoc or self.has_special_authorization):
                out.append(
                    "RDAC 101.210(a)(4): BVLOS requires a UOC or a special "
                    "UAS flight authorisation."
                )
            out.append(
                "RDAC 101.210(a)(3): request NOTAM publication if applicable; "
                "(a)(5) detect-and-avoid provision required for UOC holders."
            )

        if self.category == "open":
            out.append(
                "RDAC 101.310(a): the open category forbids BVLOS, commercial "
                "use and carriage of 101.170 cargo — medical delivery cannot "
                "be flown in the open category."
            )

        return out


# ═══════════════════════════════════════════════════════════════════════════
# REGULATORY PROFILE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RegulatoryProfile:
    """
    The numeric limits of a national UAS regulation.

    Defaults encode RDAC 101 (Ecuador, 2024). Every field is a plain
    number so the same optimizer runs unchanged against EASA, FAA Part
    107 or any other framework — swap the profile, not the code.
    """

    name: str = "RDAC 101 (DGAC Ecuador, 2024)"
    reference: str = "Resolución DGAC-DGAC-2024-0100-R, 11-dic-2024"

    # ── 101.185 — maximum height ──────────────────────────────────────
    max_agl_m: float = 122.0
    #: Within this radius of a structure taller than ``max_agl_m`` the
    #: aircraft may climb to (structure top + ``structure_overfly_margin_m``).
    structure_overfly_radius_m: float = 50.0
    structure_overfly_margin_m: float = 15.0

    # ── 101.160 — urban operations ────────────────────────────────────
    #: Horizontal separation from uninvolved persons and from
    #: structures, in a populated area [m].
    #:
    #: **Not checked, and it cannot be** — checking it needs a building
    #: footprint and crowd layer, and RAPTOR has terrain and airspace
    #: zones. The numbers are carried so a report can state the rule the
    #: operation is subject to, and :meth:`unenforceable_rules` names
    #: them so the omission is declared rather than merely absent.
    #:
    #: This matters for what a compliance verdict means: "compliant"
    #: from this package is compliant *with the rules it can see*.
    min_separation_persons_m: float = 30.0
    min_separation_buildings_m: float = 30.0

    # ── 101.195 — aerodromes ──────────────────────────────────────────
    #: Controlled aerodrome, prohibited rectangle: half-width either side
    #: of the runway centreline, and extension beyond each threshold.
    ctr_runway_half_width_m: float = 600.0
    ctr_threshold_extension_m: float = 3000.0
    #: Approach/departure cones beyond the prohibited rectangle.
    ctr_cone_length_m: float = 5000.0
    #: Outside the prohibited surfaces but within this radius, flight is
    #: permitted up to ``ctr_ring_max_agl_m`` with operator coordination.
    ctr_ring_radius_m: float = 5000.0
    ctr_ring_max_agl_m: float = 40.0
    uncontrolled_aerodrome_radius_m: float = 1800.0
    heliport_radius_m: float = 900.0

    # ── 101.165 — autonomous operations ───────────────────────────────
    autonomous_range_m: float = 1000.0

    # ── 101.215 — operating limitations ───────────────────────────────
    #: Minimum flight visibility [m]. Also unenforceable here: it is a
    #: weather minimum, and the wind model carries no visibility.
    min_visibility_m: float = 5000.0

    #: Effort weights for each permit family.
    permit_cost: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_PERMIT_COST)
    )

    # ── Ceiling resolution ────────────────────────────────────────────

    def ceiling_agl_m(self, ctx: OperationalContext) -> float:
        """
        The height ceiling that binds this operation, in m AGL.

        A special authorisation may raise it (101.035 / Chapter G);
        otherwise 101.185 applies at 122 m.
        """
        if ctx.max_agl_override_m is not None:
            return float(ctx.max_agl_override_m)
        return self.max_agl_m

    def structure_ceiling_agl_m(self, structure_height_m: float) -> float:
        """
        Ceiling in m AGL when overflying a structure taller than the
        general limit, per 101.185(a)(1).

        Only valid within ``structure_overfly_radius_m`` of the structure.
        """
        if structure_height_m <= self.max_agl_m:
            return self.max_agl_m
        return structure_height_m + self.structure_overfly_margin_m

    # ── Zone resolution ───────────────────────────────────────────────

    def resolve_permission(
        self,
        zone,
        ctx: OperationalContext,
    ) -> PermissionClass:
        """
        The effective permission class of ``zone`` for this operation.

        A zone the operator is already authorised for resolves to FREE.
        Everything else keeps the permission class the zone declares —
        a special authorisation does *not* unlock a prohibited zone,
        because 101.190(b) admits no exception.
        """
        if zone.zone_id in ctx.authorized_zone_ids:
            return PermissionClass.FREE
        return zone.permission

    def permit_cost_for(self, zone) -> float:
        """Effort weight of the permit that ``zone`` requires."""
        return self.permit_cost.get(zone.permit_family, self.permit_cost["generic"])

    def autonomous_range_limit_m(self, ctx: OperationalContext) -> Optional[float]:
        """
        Maximum horizontal distance from the launch point for an
        autonomous flight, or ``None`` if unlimited for this operation.

        RDAC 101.165(a)(2): 1 000 m, lifted by a UOC or a special
        authorisation.
        """
        if not ctx.autonomous:
            return None
        if ctx.has_uoc or ctx.has_special_authorization:
            return None
        return self.autonomous_range_m

    def unenforceable_rules(self) -> List[str]:
        """
        Rules this profile carries that nothing in the package checks.

        Every one of them needs data RAPTOR does not have — building
        footprints, crowd locations, a weather feed. Listing them is the
        difference between a limitation and an oversight: a reader of a
        compliance verdict is entitled to know which rules were tested
        and which were only named.
        """
        return [
            f"RDAC 101.160: {self.min_separation_persons_m:.0f} m from "
            f"uninvolved persons and {self.min_separation_buildings_m:.0f} m "
            f"from structures in populated areas — needs a building and "
            f"crowd layer; not modelled.",
            f"RDAC 101.215: {self.min_visibility_m/1000:.0f} km minimum "
            f"flight visibility — a weather minimum; not modelled.",
        ]

    def summary(self) -> str:
        """Human-readable description of the binding limits."""
        return "\n".join([
            f"Regulatory profile: {self.name}",
            f"  Reference:            {self.reference}",
            f"  Max height:           {self.max_agl_m:.0f} m AGL (101.185)",
            f"  Structure exemption:  +{self.structure_overfly_margin_m:.0f} m "
            f"within {self.structure_overfly_radius_m:.0f} m (101.185(a)(1))",
            f"  Person/building sep.: {self.min_separation_persons_m:.0f} m (101.160)",
            f"  Controlled aerodrome: {self.ctr_runway_half_width_m:.0f} m x "
            f"{self.ctr_threshold_extension_m/1000:.0f} km rectangle + "
            f"{self.ctr_cone_length_m/1000:.0f} km cones (101.195(a))",
            f"  Aerodrome ring:       {self.ctr_ring_radius_m/1000:.0f} km @ "
            f"{self.ctr_ring_max_agl_m:.0f} m AGL with coordination",
            f"  Uncontrolled/heliport:{self.uncontrolled_aerodrome_radius_m/1000:.1f} km / "
            f"{self.heliport_radius_m/1000:.1f} km",
            f"  Autonomous range:     {self.autonomous_range_m:.0f} m without UOC (101.165)",
        ])


#: The bundled Ecuadorian profile.
RDAC_101 = RegulatoryProfile()


#: Ready-made context for the medical-delivery case study: a certified
#: operator flying BVLOS autonomous missions with a medical payload.
MEDICAL_DELIVERY_CONTEXT = OperationalContext(
    category="specific",
    has_uoc=True,
    has_special_authorization=True,
    bvlos=True,
    autonomous=True,
    emergency_public_service=True,
    carries_medical_payload=True,
)
