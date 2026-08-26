#!/usr/bin/env python3
"""
Do the facilities sit where they say they do?

Why this exists
---------------
Facility coordinates are hand-collected and nothing checked them. A
check against the DEM found six of fifteen sitting more than a corridor
height above or below the ground beneath them — and in the worst case,
Nanegalito, the latitude carried the wrong sign and the facility was
14.4 km and 505 m from where it belonged.

That kind of error is invisible: every corridor from it computes
cleanly, reports a plausible relief profile and returns a confident
verdict about a place the aircraft is not going.

The check is cheap and there is no reason not to run it whenever the
catalogue or the DEM changes.

Usage:
    python -m scripts.check_facilities
    python -m scripts.check_facilities --tolerance 30
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import raptor.scenarios as sc
from raptor.dem import DEMInterface, find_dem


def facilities():
    out = []
    for name in dir(sc):
        obj = getattr(sc, name)
        if hasattr(obj, "short_name") and hasattr(obj, "ground_elev"):
            out.append((name, obj))
    return sorted(out, key=lambda kv: kv[1].short_name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tolerance", type=float, default=15.0,
                    help="how far a facility may sit from the DEM [m]")
    ap.add_argument("--search-radius", type=float, default=6.0,
                    help="how far to look for better ground [km]")
    args = ap.parse_args(argv)

    dem = DEMInterface(find_dem())
    m = dem.metadata
    print(f"DEM: {m.provenance()}")
    print(f"     lat {m.lat_min:.4f}..{m.lat_max:.4f}, "
          f"lon {m.lon_min:.4f}..{m.lon_max:.4f}\n")

    print(f"{'facility':18s} {'lat':>9s} {'lon':>10s} {'declared':>9s} "
          f"{'DEM':>8s} {'delta':>7s}  status")
    print("-" * 74)

    problems = []
    for name, f in facilities():
        inside = (m.lat_min <= f.lat <= m.lat_max
                  and m.lon_min <= f.lon <= m.lon_max)
        z = dem.elevation(f.lat, f.lon) if inside else float("nan")
        delta = z - f.ground_elev

        if not inside:
            status = "OUTSIDE THE DEM"
            problems.append((name, f, status))
        elif not np.isfinite(z):
            status = "no terrain data"
            problems.append((name, f, status))
        elif abs(delta) > args.tolerance:
            status = f"off by more than {args.tolerance:.0f} m"
            problems.append((name, f, status))
        else:
            status = "ok"

        print(f"{f.short_name:18s} {f.lat:9.5f} {f.lon:10.5f} "
              f"{f.ground_elev:8.0f}m {z:7.0f}m {delta:+6.0f}m  {status}")

    if not problems:
        print(f"\nAll {len(facilities())} facilities agree with the DEM to "
              f"within {args.tolerance:.0f} m.")
        return 0

    # For each problem, say whether the elevation or the position is the
    # more likely culprit — ground at the declared height existing a
    # kilometre away is a coordinate error, not an elevation error.
    print(f"\n{len(problems)} problem(s):\n")
    R = args.search_radius / 111.32
    for name, f, status in problems:
        print(f"  {f.short_name} — {status}")
        la = dem.lat_1d[(dem.lat_1d >= f.lat - R) & (dem.lat_1d <= f.lat + R)]
        lo = dem.lon_1d[(dem.lon_1d >= f.lon - R) & (dem.lon_1d <= f.lon + R)]
        if len(la) < 2 or len(lo) < 2:
            print("      too close to the DEM edge to search around\n")
            continue
        Z = np.array([[dem.elevation(a, b) for b in lo] for a in la])
        hit = np.abs(Z - f.ground_elev) < 25
        if hit.any():
            LO, LA = np.meshgrid(lo, la)
            k = np.argmin((LA[hit] - f.lat) ** 2 + (LO[hit] - f.lon) ** 2)
            i, j = np.argwhere(hit)[k]
            d_km = 111.32 * np.hypot(la[i] - f.lat,
                                     (lo[j] - f.lon) * np.cos(np.radians(f.lat)))
            print(f"      ground at the declared {f.ground_elev:.0f} m exists "
                  f"{d_km:.1f} km away, at ({la[i]:.5f}, {lo[j]:.5f}).")
            print(f"      That points at the COORDINATE, not the elevation.\n")
        else:
            print(f"      no ground near {f.ground_elev:.0f} m within "
                  f"{args.search_radius:.0f} km — the declared elevation is "
                  f"the suspect.\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
