#!/usr/bin/env python3
"""
Feasibility triage — why a route can or cannot be flown legally.

Run this *before* optimizing. It answers, per corridor, the question the
optimizer cannot: is there any legal path here at all? Three things can
make the answer no, and they need completely different responses:

1. **The constraint set is impossible.** A clearance floor at or near the
   122 m AGL ceiling leaves no corridor to fly in, and every route fails
   regardless of routing. Fixed by editing ``MissionConstraints``.
2. **The relief is too steep.** Ground that falls away faster than the
   aircraft can descend forces it above the ceiling no matter what height
   it targets. Fixed by re-routing laterally, by a waiver under RDAC
   101.035, or by moving an endpoint.
3. **Airspace blocks the line.** A prohibited zone sits across the direct
   corridor. Fixed by routing around it — which is the optimizer's job.

Usage
-----
    python -m scripts.check_feasibility
    python -m scripts.check_feasibility --airspace data/ecuador_uas_zones.json
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np

from raptor.airspace import load_airspace_from_file, build_airspace
from raptor.config import MissionConstraints, UAVConfig
from raptor.corridor import survey_corridor
from raptor.dem import DEMInterface
from raptor.regulations import RDAC_101, MEDICAL_DELIVERY_CONTEXT


#: (label, (lat, lon), (lat, lon)) — the medical corridors of the DMQ case study.
DEFAULT_CORRIDORS: List[Tuple[str, Tuple[float, float], Tuple[float, float]]] = [
    ("H.Espejo -> CS.Guamani",   (-0.2000, -78.4930), (-0.3242, -78.5493)),
    ("H.Espejo -> H.Calderon",   (-0.2000, -78.4930), (-0.0978, -78.4239)),
    ("H.Espejo -> CS.Tumbaco",   (-0.2000, -78.4930), (-0.2158, -78.4085)),
    ("H.Garces -> CS.Lloa",      (-0.2641, -78.5504), (-0.2358, -78.5886)),
    ("H.Garces -> CS.Pintag",    (-0.2641, -78.5504), (-0.3833, -78.3889)),
    ("H.P.A.Suarez -> IESS Sur", (-0.1273, -78.4977), (-0.2578, -78.5253)),
]


def line_airspace_report(airspace, dem, a, b, alt_agl: float, n: int = 120):
    """Which zones the straight line crosses, at a nominal AGL."""
    lats = np.linspace(a[0], b[0], n)
    lons = np.linspace(a[1], b[1], n)
    terrain = dem.elevation_batch(lats, lons)
    alts = terrain + alt_agl

    prohibited, permits = set(), {}
    for zone in airspace.active_zones():
        if zone.is_global:
            continue
        inside = zone.horizontal_contains_many(lats, lons)
        if not inside.any():
            continue
        for i in np.flatnonzero(inside):
            v = zone.check_point(lats[i], lons[i], alts[i], terrain[i])
            if v is None:
                continue
            if v.is_hard:
                prohibited.add(zone.zone_id)
            else:
                permits[zone.zone_id] = zone.permit_family
    return sorted(prohibited), permits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dem", default=None,
                    help="DEM .npz (default: the finest one present)")
    ap.add_argument("--airspace", default=None,
                    help="airspace JSON (default: bundled DMQ file)")
    ap.add_argument("--cruise-agl", type=float, default=None,
                    help="AGL used for the airspace line check "
                         "(default: middle of the corridor)")
    args = ap.parse_args(argv)

    from raptor.dem import find_dem
    dem_path = args.dem or find_dem()
    if not os.path.exists(dem_path):
        print(f"DEM not found: {dem_path}", file=sys.stderr)
        return 1

    dem = DEMInterface(dem_path)
    print(f"DEM: {dem_path}  "
          f"({dem.metadata.n_lat}x{dem.metadata.n_lon}, "
          f"~{min(dem.metadata.dlat_m, dem.metadata.dlon_m):.0f} m cells)")
    uav = UAVConfig()
    con = MissionConstraints()
    airspace = (load_airspace_from_file(args.airspace, dem=dem)
                if args.airspace else build_airspace(dem=dem))

    print("=" * 74)
    print("FEASIBILITY TRIAGE")
    print("=" * 74)
    print(RDAC_101.summary())
    print()
    print(con.describe_corridor())

    # ── 1. Is the constraint set itself flyable? ──────────────────────
    problems = con.validate()
    if problems:
        print("\nCONSTRAINT SET PROBLEMS — fix these first:")
        for p in problems:
            print(f"  ! {p}")
        print("\n  Until the corridor has room, every route below will fail")
        print("  for reasons that have nothing to do with the route.")
    else:
        print("\nConstraint set: OK — the vertical corridor has room to fly in.")

    # ── 2. Static compliance of the operation ─────────────────────────
    notes = MEDICAL_DELIVERY_CONTEXT.compliance_findings(RDAC_101)
    if notes:
        print("\nOperation-level requirements (independent of route):")
        for n in notes:
            print(f"  - {n}")

    # ── 3. Per-corridor verdict ───────────────────────────────────────
    floor, ceiling = con.corridor("FW_CRUISE")
    probe_agl = args.cruise_agl if args.cruise_agl is not None else 0.5 * (floor + ceiling)

    print("\n" + "=" * 74)
    print(f"CORRIDORS  (airspace probed at {probe_agl:.0f} m AGL, {len(airspace.zones)} zones)")
    print("=" * 74)

    n_ok = 0
    for label, a, b in DEFAULT_CORRIDORS:
        survey = survey_corridor(dem, a, b, con, uav)
        prohibited, permits = line_airspace_report(airspace, dem, a, b, probe_agl)

        clear = survey.feasible and not prohibited
        n_ok += clear
        mark = "OK  " if clear else "BLOCKED"
        print(f"\n[{mark}] {label}")
        print(f"    distance {survey.direct_distance_m/1000:5.1f} km   "
              f"relief {survey.relief_m:4.0f} m   "
              f"best height {survey.best_target_agl:3.0f} m AGL")

        if not survey.feasible:
            print(f"    ! relief forces {survey.ceiling_excess_m:.0f} m above the "
                  f"{con.max_agl:.0f} m ceiling over "
                  f"{survey.ceiling_excess_fraction*100:.0f}% of the line")
            print(f"      -> needs a lateral detour, or a special authorisation "
                  f"(RDAC 101.035)")
        if prohibited:
            print(f"    ! prohibited airspace on the direct line: "
                  f"{', '.join(prohibited)}")
            print(f"      -> the optimizer must route around this")
        if permits:
            fams = ", ".join(f"{z} [{f}]" for z, f in sorted(permits.items()))
            print(f"    i authorisations if flown direct: {fams}")

    print("\n" + "=" * 74)
    print(f"{n_ok}/{len(DEFAULT_CORRIDORS)} corridors are legal as a straight line.")
    print("The rest are the ones worth optimizing — that is what the lateral")
    print("routing variables are for.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
