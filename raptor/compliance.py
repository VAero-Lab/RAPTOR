"""
Compliance — What To Do When No Legal Path Exists
==================================================

Treating regulation as a switch — on, and the route is rejected; off, and
it is ignored — throws away the answer an operator actually needs. Some
Quito corridors have no compliant profile at any altitude: the ground
falls away faster than the aircraft can descend, so it is forced above
the 122 m ceiling however it is flown. Garcés → Lloa is one. Reporting
that as "infeasible" is true and useless.

The regulation itself is not binary. RDAC 101 distinguishes:

**Compliant.** Fly it. Nothing to file.

**Authorisation required** (101.190(a)). The route crosses restricted
airspace. Legal once the responsible body agrees; the plan carries the
list.

**Waiver required** (101.035, Chapter G). The route breaches an
operating rule that the DGAC *can* grant relief from — the height limit
of 101.185 being the one that binds here. This is a filing, not a
rejection, and a filing needs numbers: how much extra height, over what
length, where. That is what this module computes.

**Non-compliant.** The route enters prohibited airspace under
101.190(b), which admits no exception. No paperwork fixes it; the route
must change.

Keeping those four apart is what lets a planner answer "this corridor
needs a 103 m height waiver over 1.4 km near Lloa" instead of "no
solution found".

Author: Victor (LUAS-EPN / KU Leuven)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .regulations import OperationalContext, RegulatoryProfile, RDAC_101


class ComplianceLevel(Enum):
    """
    How a route stands against the regulation, and what it would take.

    Ordered by severity, so ``max()`` over a mission's legs gives the
    mission's level.
    """
    COMPLIANT = "compliant"
    AUTHORIZATION_REQUIRED = "authorization_required"
    WAIVER_REQUIRED = "waiver_required"
    NON_COMPLIANT = "non_compliant"

    @property
    def rank(self) -> int:
        return {
            "compliant": 0,
            "authorization_required": 1,
            "waiver_required": 2,
            "non_compliant": 3,
        }[self.value]

    @property
    def flyable(self) -> bool:
        """True if some combination of paperwork makes this route legal."""
        return self is not ComplianceLevel.NON_COMPLIANT

    def __lt__(self, other): return self.rank < other.rank
    def __gt__(self, other): return self.rank > other.rank


@dataclass
class Excursion:
    """One continuous stretch where the route leaves the height band."""
    start_distance_m: float
    end_distance_m: float
    max_agl_m: float
    max_excess_m: float
    lat: float          # where the worst point is
    lon: float

    @property
    def length_m(self) -> float:
        return max(self.end_distance_m - self.start_distance_m, 0.0)


@dataclass
class HeightWaiver:
    """
    The height relief a route needs, in the terms a filing requires.

    RDAC 101.710 asks a special-authorisation request to state the scope
    of the operation: areas, polygons or lines of flight, and the heights
    approved. These are those numbers.
    """
    limit_agl_m: float                  # the rule being exceeded
    requested_agl_m: float              # height that would make it legal
    max_excess_m: float
    total_length_m: float
    route_length_m: float
    excursions: List[Excursion] = field(default_factory=list)

    @property
    def affected_fraction(self) -> float:
        return (self.total_length_m / self.route_length_m
                if self.route_length_m > 0 else 0.0)

    @property
    def n_excursions(self) -> int:
        return len(self.excursions)

    def summary(self) -> str:
        lines = [
            f"Height waiver required (RDAC 101.185 → 101.035)",
            f"  Limit:              {self.limit_agl_m:.0f} m AGL",
            f"  Requested:          {self.requested_agl_m:.0f} m AGL "
            f"(+{self.max_excess_m:.0f} m)",
            f"  Affected:           {self.total_length_m/1000:.2f} km of "
            f"{self.route_length_m/1000:.2f} km "
            f"({self.affected_fraction*100:.1f} %), "
            f"in {self.n_excursions} stretch(es)",
        ]
        for i, e in enumerate(self.excursions, 1):
            lines.append(
                f"    {i}. km {e.start_distance_m/1000:5.2f}–"
                f"{e.end_distance_m/1000:5.2f}  peak {e.max_agl_m:.0f} m AGL "
                f"at {e.lat:.4f}, {e.lon:.4f}"
            )
        return "\n".join(lines)


@dataclass
class ComplianceAssessment:
    """The regulatory standing of one route, and how to act on it."""
    level: ComplianceLevel
    permits_required: Dict[str, str] = field(default_factory=dict)
    height_waiver: Optional[HeightWaiver] = None
    blocking_zones: List[str] = field(default_factory=list)
    clearance_violations: int = 0
    min_agl_m: float = float('nan')
    notes: List[str] = field(default_factory=list)

    @property
    def flyable(self) -> bool:
        return self.level.flyable

    @property
    def needs_filing(self) -> bool:
        return bool(self.permits_required) or self.height_waiver is not None

    def verdict(self) -> str:
        """One line, for tables."""
        if self.level is ComplianceLevel.NON_COMPLIANT:
            return f"NOT FLYABLE — prohibited airspace ({len(self.blocking_zones)})"
        if self.level is ComplianceLevel.WAIVER_REQUIRED:
            w = self.height_waiver
            return (f"WAIVER — +{w.max_excess_m:.0f} m AGL over "
                    f"{w.affected_fraction*100:.0f}% of route")
        if self.level is ComplianceLevel.AUTHORIZATION_REQUIRED:
            return f"AUTHORISED — {len(self.permits_required)} permit(s)"
        return "COMPLIANT"

    def filing_summary(self) -> str:
        """What would actually have to be submitted, and to whom."""
        lines = [f"Compliance: {self.level.value.replace('_', ' ').upper()}"]

        if self.level is ComplianceLevel.NON_COMPLIANT:
            lines += [
                "  This route cannot be authorised. RDAC 101.190(b) admits no",
                "  exception for prohibited airspace, so no filing helps — the",
                "  route has to change.",
                "  Prohibited zones entered: " + ", ".join(self.blocking_zones),
            ]
            return "\n".join(lines)

        if self.permits_required:
            lines.append(f"  Authorisations (RDAC 101.190(a)) — "
                         f"{len(self.permits_required)}:")
            for zid, fam in sorted(self.permits_required.items()):
                lines.append(f"    - {zid}  [{fam}]")

        if self.height_waiver is not None:
            lines.append("  " + self.height_waiver.summary().replace("\n", "\n  "))
            lines += [
                "  File under RDAC 101.710 (special UAS flight authorisation),",
                "  stating the line of flight and the heights approved.",
            ]

        if self.clearance_violations:
            lines.append(
                f"  ! {self.clearance_violations} waypoint(s) below the terrain "
                f"clearance floor (lowest {self.min_agl_m:.0f} m AGL). This is a "
                f"safety constraint, not a regulatory one — it is not waivable "
                f"and the route must be re-planned."
            )

        for n in self.notes:
            lines.append(f"  - {n}")

        if len(lines) == 1:
            lines.append("  Nothing to file.")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════════

def find_excursions(
    distances: np.ndarray,
    agl: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    limit_m: float,
    exempt: np.ndarray = None,
) -> List[Excursion]:
    """
    Continuous stretches where AGL exceeds ``limit_m``.

    Grouped into stretches rather than counted per waypoint because a
    waiver is granted over a *segment of the route*, not per sample: two
    hundred consecutive breaching waypoints are one filing, not two
    hundred.
    """
    over = np.asarray(agl) > limit_m
    if exempt is not None:
        over = over & ~np.asarray(exempt)
    if not over.any():
        return []

    out: List[Excursion] = []
    idx = np.flatnonzero(over)
    splits = np.flatnonzero(np.diff(idx) > 1)
    groups = np.split(idx, splits + 1)

    for g in groups:
        k = int(g[np.argmax(agl[g])])
        out.append(Excursion(
            start_distance_m=float(distances[g[0]]),
            end_distance_m=float(distances[g[-1]]),
            max_agl_m=float(agl[k]),
            max_excess_m=float(agl[k] - limit_m),
            lat=float(lats[k]),
            lon=float(lons[k]),
        ))
    return out


def assess_compliance(
    path,
    terrain_report,
    airspace_report=None,
    regulation: RegulatoryProfile = RDAC_101,
    context: OperationalContext = None,
    constraints=None,
) -> ComplianceAssessment:
    """
    Grade a route against the regulation and say what it would take to fly it.

    Parameters
    ----------
    path : FlightPath
    terrain_report : TerrainReport
        Supplies the AGL profile and which waypoints sit in the
        take-off/landing funnels, which are exempt from both walls of the
        corridor.
    airspace_report : AirspaceReport, optional
    regulation : RegulatoryProfile
    context : OperationalContext
        Zones already authorised are not counted again.
    constraints : MissionConstraints, optional
        Supplies the AGL limit; falls back to the regulatory profile.

    Returns
    -------
    ComplianceAssessment
    """
    limit = (constraints.max_agl if constraints is not None
             else regulation.max_agl_m)
    authorized = set(context.authorized_zone_ids) if context else set()

    waypoints = path.get_waypoints()
    lats = np.array([w.lat for w in waypoints])
    lons = np.array([w.lon for w in waypoints])
    dist = np.array([w.distance for w in waypoints])
    agl = np.asarray(terrain_report.agl_profile, dtype=float)

    # Terminal funnels are exempt: the aircraft has to pass through every
    # height in the band to reach its pad, and a waiver is not needed for
    # taking off.
    req = np.asarray(getattr(terrain_report, 'required_clearance', []), dtype=float)
    exempt = (req == 0.0) if req.size == len(agl) else np.zeros(len(agl), bool)

    excursions = find_excursions(dist, agl, lats, lons, limit, exempt)

    waiver = None
    if excursions:
        total = sum(e.length_m for e in excursions)
        worst = max(e.max_excess_m for e in excursions)
        waiver = HeightWaiver(
            limit_agl_m=float(limit),
            requested_agl_m=float(limit + worst),
            max_excess_m=float(worst),
            total_length_m=float(total),
            route_length_m=float(dist[-1] - dist[0]) if len(dist) > 1 else 0.0,
            excursions=excursions,
        )

    permits: Dict[str, str] = {}
    blocking: List[str] = []
    if airspace_report is not None:
        permits = {z: f for z, f in airspace_report.permits_required.items()
                   if z not in authorized}
        blocking = [z for z in airspace_report.prohibited_zones
                    if z not in authorized]

    if blocking:
        level = ComplianceLevel.NON_COMPLIANT
    elif waiver is not None:
        level = ComplianceLevel.WAIVER_REQUIRED
    elif permits:
        level = ComplianceLevel.AUTHORIZATION_REQUIRED
    else:
        level = ComplianceLevel.COMPLIANT

    notes: List[str] = []
    if context is not None:
        notes.extend(context.compliance_findings(regulation))

    return ComplianceAssessment(
        level=level,
        permits_required=permits,
        height_waiver=waiver,
        blocking_zones=blocking,
        clearance_violations=int(terrain_report.n_violations),
        # The lowest AGL among *violating* waypoints, not the lowest on
        # the path: the latter is 0 at every pad and would report a
        # perfectly good route as scraping the ground.
        min_agl_m=float(getattr(terrain_report, 'min_violating_agl',
                                terrain_report.min_agl)),
        notes=notes,
    )


def mission_compliance(assessments: List[ComplianceAssessment]
                       ) -> ComplianceAssessment:
    """
    Roll several legs up into one verdict for the whole mission.

    A mission is only as legal as its worst leg, and its filings are the
    union of the legs' — a permit is obtained once, not per crossing.
    """
    if not assessments:
        return ComplianceAssessment(level=ComplianceLevel.COMPLIANT)

    level = max((a.level for a in assessments), key=lambda x: x.rank)
    permits: Dict[str, str] = {}
    blocking: List[str] = []
    notes: List[str] = []
    for a in assessments:
        permits.update(a.permits_required)
        blocking.extend(z for z in a.blocking_zones if z not in blocking)
        notes.extend(n for n in a.notes if n not in notes)

    waivers = [a.height_waiver for a in assessments if a.height_waiver]
    waiver = None
    if waivers:
        route_len = sum(w.route_length_m for w in waivers)
        waiver = HeightWaiver(
            limit_agl_m=waivers[0].limit_agl_m,
            requested_agl_m=max(w.requested_agl_m for w in waivers),
            max_excess_m=max(w.max_excess_m for w in waivers),
            total_length_m=sum(w.total_length_m for w in waivers),
            route_length_m=route_len,
            excursions=[e for w in waivers for e in w.excursions],
        )

    return ComplianceAssessment(
        level=level,
        permits_required=permits,
        height_waiver=waiver,
        blocking_zones=blocking,
        clearance_violations=sum(a.clearance_violations for a in assessments),
        min_agl_m=min((a.min_agl_m for a in assessments), default=float('nan')),
        notes=notes,
    )
