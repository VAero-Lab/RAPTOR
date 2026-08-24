"""
Mission Profiles — The Shapes a Delivery Flight Actually Takes
==============================================================

A single A→B leg is the exception in medical logistics, not the rule.
Real operations look like:

* **Out and back.** Deliver, return to base. The return leg is flown on
  what the outbound leg left in the battery, and usually empty — so the
  two legs are neither symmetric nor independent.
* **Sample collection.** Visit several health centres, picking up
  specimens. Mass *accumulates*, so the last leg is the heaviest and the
  battery is at its lowest — the worst combination, and one that a
  per-leg analysis never surfaces.
* **Supply tour.** The mirror image: leave with everything, get lighter.
* **Hub and spoke.** Return to base between deliveries, which trades
  flight time for a fresh battery each time.
* **Shuttle.** Repeat one leg until the battery says stop.

What makes these different from a list of legs is what happens *between*
them:

``service_time_s``
    Time on the ground loading or unloading. It does not consume
    propulsion energy, but it does consume the clock — which matters for
    a cold-chain payload and for a time-constrained mission.
``battery_action``
    Whether the aircraft can swap or recharge at that stop. A hospital
    hangar can; a rural health post cannot. This single attribute decides
    whether a four-stop tour is one mission or four.

Author: Victor (LUAS-EPN / KU Leuven)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

from .scenarios import Facility, FlightScenario, MedicalUrgency, MissionLeg, OptPriority


class MissionProfile(Enum):
    """The shape of a mission, independent of which facilities it visits."""
    ONE_WAY = "one_way"
    OUT_AND_BACK = "out_and_back"
    SAMPLE_COLLECTION = "sample_collection"
    SUPPLY_TOUR = "supply_tour"
    HUB_AND_SPOKE = "hub_and_spoke"
    SHUTTLE = "shuttle"


class BatteryAction(Enum):
    """What can be done to the battery at a stop."""
    NONE = "none"            # touch-and-go; the pack keeps depleting
    SWAP = "swap"            # fresh pack, SOC back to full
    RECHARGE = "recharge"    # charged in place, takes real time

    @property
    def restores_charge(self) -> bool:
        return self is not BatteryAction.NONE


@dataclass
class Stop:
    """
    One visit in a mission, and what happens on the ground there.

    Attributes
    ----------
    facility : Facility
    payload_delta_kg : float
        Mass loaded (positive) or unloaded (negative) here. Sequencing
        matters: a collection round gets heavier as it goes, and the
        heaviest leg therefore coincides with the emptiest battery.
    service_time_s : float
        Time on the ground. Costs no propulsion energy but does count
        against a mission time limit and against a payload's viable
        window.
    battery_action : BatteryAction
        Whether the pack can be swapped or charged here.
    recharge_rate_c : float
        Charge rate for RECHARGE stops, in C. 1 C fills an empty pack in
        an hour, which is usually far longer than the delivery itself.
    """
    facility: Facility
    payload_delta_kg: float = 0.0
    service_time_s: float = 300.0
    battery_action: BatteryAction = BatteryAction.NONE
    recharge_rate_c: float = 1.0

    def recharge_time_s(self, soc_from: float, soc_to: float = 1.0) -> float:
        """Ground time needed to bring the pack from one SOC to another."""
        if self.battery_action is BatteryAction.SWAP:
            return self.service_time_s
        if self.battery_action is not BatteryAction.RECHARGE:
            return self.service_time_s
        deficit = max(soc_to - soc_from, 0.0)
        return max(self.service_time_s, deficit / max(self.recharge_rate_c, 1e-6) * 3600.0)


@dataclass
class MissionPlan:
    """
    A mission as a sequence of stops, from which legs are derived.

    Holding stops rather than legs is what makes payload sequencing and
    ground operations expressible: a leg knows what it carries, but only
    the stop knows what changed hands.
    """
    name: str
    mission_id: str
    profile: MissionProfile
    stops: List[Stop]
    priority: OptPriority = OptPriority.ENERGY
    urgency: MedicalUrgency = MedicalUrgency.ROUTINE
    initial_payload_kg: float = 0.0
    description: str = ""

    @property
    def n_legs(self) -> int:
        return max(len(self.stops) - 1, 0)

    def payload_schedule(self) -> List[float]:
        """
        Payload carried on each leg [kg].

        Computed by walking the stops and applying each delta, so the
        answer reflects the order the mission is actually flown.
        """
        carried = self.initial_payload_kg
        out: List[float] = []
        for i in range(self.n_legs):
            carried = max(carried + self.stops[i].payload_delta_kg, 0.0)
            out.append(carried)
        return out

    @property
    def heaviest_leg(self) -> int:
        sched = self.payload_schedule()
        return int(max(range(len(sched)), key=lambda i: sched[i])) if sched else -1

    @property
    def total_service_time_s(self) -> float:
        return sum(s.service_time_s for s in self.stops)

    def to_legs(self) -> List[MissionLeg]:
        """Flatten into the leg list the planner consumes."""
        sched = self.payload_schedule()
        return [
            MissionLeg(
                origin=self.stops[i].facility,
                destination=self.stops[i + 1].facility,
                payload_kg=sched[i],
                description=self._leg_description(i, sched[i]),
            )
            for i in range(self.n_legs)
        ]

    def _leg_description(self, i: int, payload: float) -> str:
        if payload <= 0:
            return "repositioning, empty"
        delta = self.stops[i].payload_delta_kg
        if delta > 0:
            return f"carrying {payload:.1f} kg (picked up {delta:.1f} kg)"
        if delta < 0:
            return f"carrying {payload:.1f} kg (dropped {-delta:.1f} kg)"
        return f"carrying {payload:.1f} kg"

    def to_scenario(self) -> FlightScenario:
        """Convert to the ``FlightScenario`` the existing planner takes."""
        return FlightScenario(
            name=self.name,
            scenario_id=self.mission_id,
            legs=self.to_legs(),
            priority=self.priority,
            urgency=self.urgency,
            description=self.description or self.profile.value,
            base_payload_kg=max(self.payload_schedule(), default=0.0),
        )

    def summary(self) -> str:
        sched = self.payload_schedule()
        lines = [
            f"{self.name} [{self.mission_id}] — {self.profile.value}",
            f"  {self.n_legs} leg(s), {len(self.stops)} stop(s), "
            f"{self.urgency.value}, optimize for {self.priority.value}",
        ]
        for i in range(self.n_legs):
            a, b = self.stops[i], self.stops[i + 1]
            action = ("" if b.battery_action is BatteryAction.NONE
                      else f"  [{b.battery_action.value} at {b.facility.short_name}]")
            lines.append(
                f"    {i+1}. {a.facility.short_name:>16s} → "
                f"{b.facility.short_name:<16s} {sched[i]:5.1f} kg{action}")
        if sched:
            lines.append(f"  Heaviest leg: {self.heaviest_leg + 1} "
                         f"({max(sched):.1f} kg)")
        lines.append(f"  Ground time:  {self.total_service_time_s/60:.0f} min")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# BUILDERS
# ═══════════════════════════════════════════════════════════════════════════

def one_way(base: Facility, destination: Facility, payload_kg: float = 2.0,
            **kw) -> MissionPlan:
    """A single delivery, no return."""
    return MissionPlan(
        name=kw.pop("name", f"{base.short_name} → {destination.short_name}"),
        mission_id=kw.pop("mission_id", "OW"),
        profile=MissionProfile.ONE_WAY,
        stops=[Stop(base, payload_delta_kg=payload_kg),
               Stop(destination, payload_delta_kg=-payload_kg)],
        initial_payload_kg=0.0,
        **kw,
    )


def out_and_back(base: Facility, destination: Facility,
                 payload_out_kg: float = 2.0,
                 payload_back_kg: float = 0.0,
                 service_time_s: float = 300.0, **kw) -> MissionPlan:
    """
    Deliver and return to base.

    The return leg starts on whatever the outbound leg left in the pack,
    which is why a round trip is not two independent flights: the range
    that matters is half the one-way range, minus reserve, minus whatever
    the payload cost on the way out.
    """
    return MissionPlan(
        name=kw.pop("name",
                    f"{base.short_name} ⇄ {destination.short_name}"),
        mission_id=kw.pop("mission_id", "OB"),
        profile=MissionProfile.OUT_AND_BACK,
        stops=[
            Stop(base, payload_delta_kg=payload_out_kg,
                 service_time_s=service_time_s),
            Stop(destination,
                 payload_delta_kg=payload_back_kg - payload_out_kg,
                 service_time_s=service_time_s),
            Stop(base, service_time_s=service_time_s,
                 battery_action=BatteryAction.SWAP),
        ],
        **kw,
    )


def sample_collection(base: Facility, centres: Sequence[Facility],
                      per_stop_kg: float = 0.5,
                      service_time_s: float = 240.0, **kw) -> MissionPlan:
    """
    Tour several health centres collecting specimens, then return.

    Mass accumulates along the route, so the final leg — the one flown
    on the least remaining charge — is also the heaviest. That coupling
    is the whole point of planning this as one mission rather than as a
    set of legs.
    """
    stops = [Stop(base, service_time_s=service_time_s)]
    for c in centres:
        stops.append(Stop(c, payload_delta_kg=per_stop_kg,
                          service_time_s=service_time_s))
    stops.append(Stop(base, payload_delta_kg=-per_stop_kg * len(centres),
                      service_time_s=service_time_s,
                      battery_action=BatteryAction.SWAP))
    return MissionPlan(
        name=kw.pop("name", f"Sample collection from {len(centres)} centre(s)"),
        mission_id=kw.pop("mission_id", "SC"),
        profile=MissionProfile.SAMPLE_COLLECTION,
        stops=stops, **kw,
    )


def supply_tour(base: Facility, centres: Sequence[Facility],
                per_stop_kg: float = 0.7,
                service_time_s: float = 240.0, **kw) -> MissionPlan:
    """
    Leave loaded, drop supplies at each centre, return empty.

    The mirror of a collection round: heaviest at the start, when the
    battery is full. Comparing the two over the same facilities isolates
    what payload *sequencing* alone costs.
    """
    total = per_stop_kg * len(centres)
    stops = [Stop(base, payload_delta_kg=total, service_time_s=service_time_s)]
    for c in centres:
        stops.append(Stop(c, payload_delta_kg=-per_stop_kg,
                          service_time_s=service_time_s))
    stops.append(Stop(base, service_time_s=service_time_s,
                      battery_action=BatteryAction.SWAP))
    return MissionPlan(
        name=kw.pop("name", f"Supply tour to {len(centres)} centre(s)"),
        mission_id=kw.pop("mission_id", "ST"),
        profile=MissionProfile.SUPPLY_TOUR,
        stops=stops, **kw,
    )


def hub_and_spoke(base: Facility, destinations: Sequence[Facility],
                  payload_kg: float = 2.0,
                  service_time_s: float = 300.0,
                  battery_action: BatteryAction = BatteryAction.SWAP,
                  **kw) -> MissionPlan:
    """
    Return to base between deliveries.

    Trades flight time and energy for a fresh pack each time, which is
    what makes a set of destinations reachable that no single tour could
    cover. Set ``battery_action=NONE`` to model a base without a spare
    pack and see the difference.
    """
    stops = [Stop(base, payload_delta_kg=payload_kg,
                  service_time_s=service_time_s)]
    for i, d in enumerate(destinations):
        stops.append(Stop(d, payload_delta_kg=-payload_kg,
                          service_time_s=service_time_s))
        last = i == len(destinations) - 1
        stops.append(Stop(
            base,
            payload_delta_kg=0.0 if last else payload_kg,
            service_time_s=service_time_s,
            battery_action=battery_action,
        ))
    return MissionPlan(
        name=kw.pop("name", f"Hub and spoke, {len(destinations)} destination(s)"),
        mission_id=kw.pop("mission_id", "HS"),
        profile=MissionProfile.HUB_AND_SPOKE,
        stops=stops, **kw,
    )


def shuttle(base: Facility, destination: Facility, n_round_trips: int = 3,
            payload_kg: float = 2.0, service_time_s: float = 300.0,
            battery_action: BatteryAction = BatteryAction.SWAP,
            **kw) -> MissionPlan:
    """Repeat one delivery, to size a day's throughput on a corridor."""
    stops = [Stop(base, payload_delta_kg=payload_kg,
                  service_time_s=service_time_s)]
    for i in range(n_round_trips):
        stops.append(Stop(destination, payload_delta_kg=-payload_kg,
                          service_time_s=service_time_s))
        last = i == n_round_trips - 1
        stops.append(Stop(
            base,
            payload_delta_kg=0.0 if last else payload_kg,
            service_time_s=service_time_s,
            battery_action=battery_action,
        ))
    return MissionPlan(
        name=kw.pop("name", f"Shuttle ×{n_round_trips} to "
                            f"{destination.short_name}"),
        mission_id=kw.pop("mission_id", "SH"),
        profile=MissionProfile.SHUTTLE,
        stops=stops, **kw,
    )
