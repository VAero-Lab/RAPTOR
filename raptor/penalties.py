"""
Constraint Handling — Turning Regulations Into Search Pressure
==============================================================

The optimizer minimises a scalar. Everything the regulation says has to
arrive there as a number, and *how* it arrives decides whether the search
works at all.

Three rules govern the design below.

**Commensurability.** The objective is normalised into roughly [0, 2].
A penalty of 1e6 is not "strong", it is *flat*: every infeasible design
scores about the same, differential evolution sees no slope, and the
population wanders. Penalties here are scaled to the objective and sized
so that any violating design loses to any compliant one, while still
ranking violations by severity.

**The gate must match the gradient.** If feasibility is judged at 60 m
clearance but the penalty is computed against 50 m, then designs between
the two are penalty-free and still reported infeasible — the optimizer is
told nothing about the very constraint that fails it. Every penalty here
is derived from the same report that decides feasibility.

**Permits are a cost, not a wall.** RDAC 101.190 distinguishes airspace
that may be flown with authorisation from airspace that may never be
flown. Collapsing both into "violation" throws away the choice the
planner actually wants to make — whether crossing a zone and filing for
permission beats a longer route. Prohibited zones are barriers here;
permit zones are priced.

Author: Victor (LUAS-EPN / KU Leuven)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .regulations import RegulatoryProfile, OperationalContext, RDAC_101


@dataclass
class PenaltyWeights:
    """
    Weights converting each constraint into objective units.

    The normalised objective (energy and/or time) lives in about [0, 2],
    so a weight of 1.0 means "as important as doubling the mission cost".

    Attributes
    ----------
    activation : float
        Flat cost added as soon as *any* hard constraint is violated. This
        is what guarantees a separation between the feasible and
        infeasible regions: without it, a tiny violation that saves a lot
        of energy can outscore a compliant design.
    corridor : float
        Multiplies the terrain report's ``corridor_penalty`` — mean
        excursion outside the vertical band, in units of band height.
    prohibited : float
        Per-unit-exposure cost of flying inside a prohibited zone.
    ceiling : float
        Multiplies mean exceedance of a zone ceiling, normalised by the
        regulatory ceiling height.
    soc : float
        Multiplies the shortfall below the required state of charge.
    time : float
        Multiplies fractional overrun of the flight-time limit.
    endpoint : float
        Multiplies endpoint altitude error, in km.
    permit : float
        Soft cost per permit-unit of route exposure. Deliberately small:
        it should break ties between comparable routes, not force long
        detours around a zone whose permit is routine.
    waiver : float
        Cost of height relief when the ceiling is treated as waivable
        rather than absolute — see ``ceiling_waivable`` in
        :func:`evaluate_penalties`. Sized between ``permit`` and
        ``ceiling``: a waiver is a real regulatory burden the optimizer
        should work to avoid, but a route needing one is still worth
        having when the alternative is no route at all.
    """
    activation: float = 5.0
    corridor: float = 40.0
    prohibited: float = 60.0
    ceiling: float = 40.0
    soc: float = 20.0
    time: float = 10.0
    endpoint: float = 5.0
    permit: float = 0.02
    waiver: float = 6.0


@dataclass
class PenaltyBreakdown:
    """Per-constraint contributions, for diagnostics and figures."""
    total: float = 0.0
    corridor: float = 0.0
    prohibited: float = 0.0
    ceiling: float = 0.0
    soc: float = 0.0
    time: float = 0.0
    endpoint: float = 0.0
    permit: float = 0.0
    waiver: float = 0.0
    activation: float = 0.0
    feasible: bool = True
    permits_required: Dict[str, str] = field(default_factory=dict)
    #: The reports this breakdown was computed from, attached by the
    #: optimizer so the final verdict and the search pressure provably
    #: come from the same evaluation.
    terrain_report: object = None
    airspace_report: object = None

    def as_dict(self) -> Dict[str, float]:
        return {
            'corridor': self.corridor,
            'prohibited': self.prohibited,
            'ceiling': self.ceiling,
            'soc': self.soc,
            'time': self.time,
            'endpoint': self.endpoint,
            'permit': self.permit,
            'waiver': self.waiver,
            'activation': self.activation,
            'total': self.total,
        }

    def summary(self) -> str:
        rows = [(k, v) for k, v in self.as_dict().items()
                if k != 'total' and abs(v) > 1e-9]
        if not rows:
            return "Penalties: none — all constraints satisfied"
        width = max(len(k) for k, _ in rows)
        lines = [f"Penalties (total {self.total:.3f}):"]
        for k, v in sorted(rows, key=lambda kv: -kv[1]):
            lines.append(f"  {k:<{width}s}  {v:8.3f}")
        if self.permits_required:
            lines.append(f"  permits: {', '.join(sorted(self.permits_required))}")
        return "\n".join(lines)


def evaluate_penalties(
    terrain_report,
    airspace_report,
    soc_final: float,
    min_soc: float,
    flight_time_s: float,
    max_flight_time_s: float,
    endpoint_alt_error_m: float = 0.0,
    weights: PenaltyWeights = None,
    regulation: RegulatoryProfile = RDAC_101,
    context: Optional[OperationalContext] = None,
    ceiling_waivable: bool = False,
) -> PenaltyBreakdown:
    """
    Score one candidate path against every constraint.

    Parameters
    ----------
    terrain_report : TerrainReport
        From ``TerrainAnalyzer.analyze``. Supplies ``corridor_penalty``,
        which already blends clearance shortfall and ceiling exceedance
        in units of the corridor's own height, so the weight below does
        not need retuning per region.
    airspace_report : AirspaceReport or None
        From ``AirspaceManager.check_path``.
    soc_final, min_soc : float
        Battery state of charge at arrival and the required reserve.
    flight_time_s, max_flight_time_s : float
        Achieved and permitted mission duration.
    endpoint_alt_error_m : float
        How far the path's final altitude misses the destination pad.
    weights : PenaltyWeights
        Scaling. Defaults are tuned for a normalised [0, 2] objective.
    regulation : RegulatoryProfile
        Supplies the ceiling used for normalisation and the permit costs.
    context : OperationalContext
        Zones already authorised for this operation are not charged.
    ceiling_waivable : bool
        Treat exceeding the AGL ceiling as a cost to minimise rather than
        a hard failure. Over terrain that outruns the aircraft's descent
        rate there is no compliant profile at any altitude, and rejecting
        every candidate leaves the optimizer nothing to rank — so the
        search stalls precisely where a planner most needs an answer.
        With this set, the optimizer instead finds the route needing the
        *smallest* height relief, which is what RDAC 101.035 asks a
        special-authorisation request to state. Prohibited airspace stays
        hard either way: 101.190(b) admits no exception.

    Returns
    -------
    PenaltyBreakdown
    """
    w = weights or PenaltyWeights()
    b = PenaltyBreakdown()

    # ── Vertical corridor: clearance floor and regulatory ceiling ─────
    if terrain_report is not None:
        cp = float(getattr(terrain_report, 'corridor_penalty', 0.0) or 0.0)
        if not np.isfinite(cp):
            cp = 1.0
        if cp > 0:
            b.corridor = w.corridor * cp

        floor_ok = getattr(terrain_report, 'is_feasible', True)
        ceiling_ok = getattr(terrain_report, 'ceiling_feasible', True)

        # The clearance floor is a safety constraint and never waivable.
        if not floor_ok:
            b.feasible = False

        if not ceiling_ok:
            if ceiling_waivable:
                excess = float(getattr(terrain_report, 'max_ceiling_excess', 0.0))
                frac = float(getattr(terrain_report,
                                     'ceiling_violation_fraction', 0.0))
                ceil = max(regulation.max_agl_m, 1.0)
                # Charge both how far outside and how much of the route:
                # a waiver for 10 m over 200 m of track is a different
                # filing from 150 m over half the leg.
                b.waiver = w.waiver * (excess / ceil + frac)
            else:
                b.feasible = False

    # ── Airspace ──────────────────────────────────────────────────────
    if airspace_report is not None:
        n = max(getattr(airspace_report, 'n_samples', 0), 1)
        authorized = set(context.authorized_zone_ids) if context else set()

        prohibited_hits = 0
        ceiling_excess = 0.0
        for v in airspace_report.hard_violations:
            if v.zone_id in authorized and v.violation_type == 'inside_zone':
                continue
            if v.violation_type == 'above_ceiling':
                ceiling_excess += v.excess_m
            else:
                prohibited_hits += 1

        if prohibited_hits:
            # Graded, not infinite: the optimizer must be able to feel its
            # way out of a zone it started inside. Exposure is the fraction
            # of the route spent there, so shaving the incursion pays.
            b.prohibited = w.prohibited * (1.0 + prohibited_hits / n)
            b.feasible = False

        if ceiling_excess > 0:
            ceil = max(regulation.max_agl_m, 1.0)
            b.ceiling = w.ceiling * (ceiling_excess / n) / ceil
            b.feasible = False

        # Permits: a soft cost proportional to how much of the route lies
        # in each zone, weighted by how hard that permit is to obtain.
        permits = {
            zid: fam for zid, fam in airspace_report.permits_required.items()
            if zid not in authorized
        }
        b.permits_required = permits
        if permits:
            cost = 0.0
            for zid, fam in permits.items():
                exposure = airspace_report.permit_exposure.get(zid, 0.0)
                unit = regulation.permit_cost.get(
                    fam, regulation.permit_cost["generic"]
                )
                cost += unit * (0.5 + exposure)
            b.permit = w.permit * cost

    # ── Battery reserve ───────────────────────────────────────────────
    deficit = min_soc - soc_final
    if deficit > 0:
        b.soc = w.soc * deficit
        b.feasible = False

    # ── Flight time ───────────────────────────────────────────────────
    if max_flight_time_s > 0:
        over = flight_time_s - max_flight_time_s
        if over > 0:
            b.time = w.time * (over / max_flight_time_s)
            b.feasible = False

    # ── Endpoint ──────────────────────────────────────────────────────
    if endpoint_alt_error_m > 50.0:
        b.endpoint = w.endpoint * (endpoint_alt_error_m / 1000.0)

    if not b.feasible:
        b.activation = w.activation

    b.total = (b.corridor + b.prohibited + b.ceiling + b.soc
               + b.time + b.endpoint + b.permit + b.waiver + b.activation)
    return b
