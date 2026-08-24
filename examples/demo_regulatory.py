#!/usr/bin/env python3
"""
Demo: Regulation-Aware Planning
================================

Shows the three things that decide whether a medical delivery route is
flyable under RDAC 101, and how each is handled:

1. **The vertical corridor.** Compares a constant-altitude cruise with a
   terrain-following one over the same corridor, and reports how far
   each leaves the 122 m AGL band.
2. **Permission classes.** Prohibited zones are barriers; permit zones
   are priced and reported, so the plan comes with the list of
   authorisations it depends on.
3. **Operational credentials.** The same route is evaluated for an
   operator that already holds an authorisation and for one that does
   not, showing how the plan changes.

Usage:
    python -m examples.demo_regulatory
    python -m examples.demo_regulatory --maxiter 60
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from raptor.airspace import build_airspace
from raptor.builder import FacilityNode
from raptor.config import MissionConstraints, UAVConfig
from raptor.corridor import survey_corridor
from raptor.dem import DEMInterface
from raptor.energy import AircraftEnergyParams, analyze_path_energy
from raptor.optimizer import OptMode, PathOptimizer
from raptor.regulations import (
    MEDICAL_DELIVERY_CONTEXT, RDAC_101, OperationalContext,
)
from raptor.routed_path import RoutedPath
from raptor.terrain import TerrainAnalyzer


def find_dem():
    from raptor.dem import find_dem as _find
    return _find()


def rule(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxiter", type=int, default=40)
    ap.add_argument("--popsize", type=int, default=8)
    args = ap.parse_args()

    dem = DEMInterface(find_dem())
    uav, con, ac = UAVConfig(), MissionConstraints(), AircraftEnergyParams()
    airspace = build_airspace(dem=dem)
    ta = TerrainAnalyzer(dem, con)

    a = (-0.2000, -78.4930)     # Hospital Eugenio Espejo
    b = (-0.3242, -78.5493)     # Centro de Salud Guamaní
    orig = FacilityNode("H.Espejo", a[0], a[1], dem.elevation(*a))
    dest = FacilityNode("CS.Guamani", b[0], b[1], dem.elevation(*b))

    # ── 1. The vertical corridor ──────────────────────────────────────
    rule("1. THE VERTICAL CORRIDOR")
    print(con.describe_corridor())
    problems = con.validate()
    print("Constraint set:", "OK" if not problems else "PROBLEMS")
    for p in problems:
        print("  !", p)

    print()
    print(survey_corridor(dem, a, b, con, uav).summary())

    print("\nSame route, two ways of holding altitude:")
    print(f"  {'mode':<18s} {'segs':>5s} {'AGL min':>8s} {'AGL max':>8s} "
          f"{'over 122':>9s} {'E [Wh]':>8s}")
    for mode in ("amsl", "agl"):
        rp = RoutedPath(orig, dest, dem, uav, con,
                        n_intermediate=1, corridor_mode=mode)
        fp = rp.flight_path
        r = ta.analyze(fp)
        e = analyze_path_energy(fp, ac, SOC_min=0.2)
        label = ("constant altitude" if mode == "amsl" else "terrain-following")
        print(f"  {label:<18s} {fp.n_segments:5d} "
              f"{np.nanmin(r.agl_profile):8.0f} {np.nanmax(r.agl_profile):8.0f} "
              f"{r.n_ceiling_violations:9d} {e.total_energy_wh:8.1f}")
    print("\n  Holding one altitude cannot obey a height-above-ground rule over")
    print("  relief. Following the terrain costs energy — and is the only one")
    print("  of the two that is legal.")

    # ── 2. Permission classes ─────────────────────────────────────────
    rule("2. PERMISSION CLASSES")
    counts = {}
    for z in airspace.active_zones():
        counts[z.permission.value] = counts.get(z.permission.value, 0) + 1
    print(f"Airspace: {len(airspace.zones)} zones — "
          + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print("\n  PROHIBITED zones are barriers (RDAC 101.190(b)).")
    print("  PERMIT zones are priced, not forbidden (RDAC 101.190(a)): the")
    print("  optimizer weighs crossing them against flying around, and the")
    print("  plan carries the authorisations it needs.")

    optimizer = PathOptimizer(dem, uav, con, ac,
                              regulation=RDAC_101,
                              context=MEDICAL_DELIVERY_CONTEXT)
    rp = RoutedPath(orig, dest, dem, uav, con,
                    n_intermediate=2, corridor_mode="agl")
    print(f"\nOptimizing ({args.maxiter} generations)...")
    res = optimizer.optimize_routed(
        rp, mode=OptMode.ENERGY, payload_kg=1.5, airspace=airspace,
        maxiter=args.maxiter, popsize=args.popsize, verbose=False,
    )
    print(res.summary())

    # ── 3. Credentials change the plan ────────────────────────────────
    rule("3. OPERATIONAL CREDENTIALS")
    if res.permits_required:
        held = sorted(res.permits_required)[:1]
        print(f"Suppose the operator already holds: {held[0]}")
        ctx = OperationalContext(
            has_uoc=True, bvlos=True, autonomous=True,
            carries_medical_payload=True, authorized_zone_ids=held,
        )
        opt2 = PathOptimizer(dem, uav, con, ac,
                             regulation=RDAC_101, context=ctx)
        rp2 = RoutedPath(orig, dest, dem, uav, con,
                         n_intermediate=2, corridor_mode="agl")
        res2 = opt2.optimize_routed(
            rp2, mode=OptMode.ENERGY, payload_kg=1.5, airspace=airspace,
            maxiter=args.maxiter, popsize=args.popsize, verbose=False,
        )
        print(f"  without it: {res.energy_wh:6.1f} Wh, "
              f"{len(res.permits_required)} authorisation(s)")
        print(f"  with it:    {res2.energy_wh:6.1f} Wh, "
              f"{len(res2.permits_required)} authorisation(s)")
        print("\n  An authorisation already in hand costs nothing to use, so a")
        print("  route through it stops being a detour worth avoiding.")
    else:
        print("This route needs no authorisations.")

    print("\nRoute-independent obligations for this operation:")
    for note in MEDICAL_DELIVERY_CONTEXT.compliance_findings(RDAC_101):
        print(f"  - {note}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
