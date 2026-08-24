#!/usr/bin/env python3
"""
Demo: Mission Profiles for Medical Delivery
============================================

A single A→B leg is the exception in medical logistics. This compares
the shapes a real operation takes over the same set of facilities, and
shows what each one costs.

The comparison worth making is **sample collection against supply tour**:
identical facilities, identical total mass, opposite sequencing. A
collection round gets heavier as it goes, so its worst leg is flown on
the least remaining charge; a supply tour gets lighter. Everything else
is equal, so the difference is what payload *ordering* alone costs.

Usage:
    python -m examples.demo_missions
    python -m examples.demo_missions --maxiter 40 --plan
"""

from __future__ import annotations

import argparse
import sys

from raptor.airspace import build_airspace
from raptor.config import MissionConstraints, UAVConfig
from raptor.dem import DEMInterface, find_dem
from raptor.energy import AircraftEnergyParams
from raptor.mission_planner import MissionPlanner
from raptor.missions import (
    BatteryAction, hub_and_spoke, out_and_back, sample_collection,
    shuttle, supply_tour,
)
from raptor.scenarios import (
    CS_LLOA, CS_PINTAG, HOSPITAL_ENRIQUE_GARCES, HOSPITAL_EUGENIO_ESPEJO,
)


def rule(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxiter", type=int, default=25)
    ap.add_argument("--popsize", type=int, default=5)
    ap.add_argument("--plan", action="store_true",
                    help="actually optimize the missions (slow)")
    args = ap.parse_args()

    base = HOSPITAL_EUGENIO_ESPEJO
    centres = [CS_LLOA, CS_PINTAG]

    missions = [
        out_and_back(base, CS_LLOA, payload_out_kg=2.0),
        sample_collection(base, centres, per_stop_kg=0.7),
        supply_tour(base, centres, per_stop_kg=0.7),
        hub_and_spoke(base, centres, payload_kg=2.0),
        shuttle(base, CS_LLOA, n_round_trips=2, payload_kg=2.0),
        hub_and_spoke(base, centres, payload_kg=2.0,
                      battery_action=BatteryAction.NONE,
                      name="Hub and spoke, no spare pack",
                      mission_id="HS-nobat"),
    ]

    rule("1. THE MISSION SHAPES")
    for m in missions:
        print(m.summary())
        print()

    rule("2. PAYLOAD SEQUENCING")
    coll = sample_collection(base, centres, per_stop_kg=0.7)
    supply = supply_tour(base, centres, per_stop_kg=0.7)
    print(f"  Same facilities, same total mass ({0.7*len(centres):.1f} kg), "
          f"opposite order:")
    print(f"    collection: {coll.payload_schedule()} kg per leg  "
          f"→ heaviest leg is #{coll.heaviest_leg + 1} (last)")
    print(f"    supply:     {supply.payload_schedule()} kg per leg  "
          f"→ heaviest leg is #{supply.heaviest_leg + 1} (first)")
    print()
    print("  The collection round flies its heaviest leg on its lowest")
    print("  state of charge. That is the coupling a per-leg analysis")
    print("  cannot see, and the reason multi-stop missions are planned")
    print("  as one problem rather than several.")

    if not args.plan:
        print("\n(Pass --plan to optimize these; it takes a few minutes.)")
        return 0

    rule("3. FLYING THEM")
    dem = DEMInterface(find_dem())
    uav, con = UAVConfig(), MissionConstraints()
    ac = AircraftEnergyParams()
    airspace = build_airspace(dem=dem)
    planner = MissionPlanner(dem, uav, con, ac, airspace=airspace)

    print(f"{'mission':34s} {'legs':>4s} {'E [Wh]':>8s} {'flight':>8s} "
          f"{'ground':>7s} {'SOC min':>8s}  compliance")
    for m in missions:
        try:
            result = planner.plan(m, n_intermediate=1,
                                  maxiter=args.maxiter, popsize=args.popsize,
                                  verbose=False)
        except Exception as exc:
            print(f"{m.name[:34]:34s} failed: {type(exc).__name__}: {exc}")
            continue
        verdict = (result.compliance.verdict() if result.compliance
                   else ("feasible" if result.mission_feasible else "INFEASIBLE"))
        print(f"{m.name[:34]:34s} {result.scenario.n_legs:4d} "
              f"{result.total_energy_wh:8.1f} "
              f"{result.total_time_min:7.1f}m {result.ground_time_s/60:6.0f}m "
              f"{result.soc_min*100:7.1f}%  {verdict}")
        sys.stdout.flush()

    print("\n  A stop that can swap the pack resets the state of charge; one")
    print("  that cannot does not. Compare the two hub-and-spoke rows — the")
    print("  route is identical and only the ground equipment differs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
