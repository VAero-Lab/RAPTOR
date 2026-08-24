#!/usr/bin/env python3
"""
Build a DEM for any region, from local tiles or from a terrain API.

RAPTOR is region-agnostic; this is how you point it at a new place.

From local GeoTIFF tiles (free, offline, source resolution)
-----------------------------------------------------------
Download 1-arc-second SRTM/NASADEM tiles covering your area from
https://earthexplorer.usgs.gov/ and:

    python -m scripts.build_dem tiles \\
        --tiff "data/TIF_files/*.tif" \\
        --bbox -0.42 0.08 -78.72 -78.28 \\
        --spacing 30 \\
        --out data/dmq_dem_30m.npz

From OpenTopoData (no files, works anywhere)
---------------------------------------------
Global coverage over HTTP. The public instance allows 100 000 points a
day, so a full 30 m raster over a city is out of reach — fetch the
corridors you actually plan over instead:

    python -m scripts.build_dem corridor \\
        --from -0.2000 -78.4930 --to -0.3242 -78.5493 \\
        --width 3000 --spacing 30 \\
        --out data/corridor_espejo_guamani.npz

    python -m scripts.build_dem grid \\
        --bbox -0.42 0.08 -78.72 -78.28 --spacing 90 \\
        --out data/dmq_dem_90m.npz

Self-host to lift the limits entirely:

    docker run -p 5000:5000 opentopodata/opentopodata
    python -m scripts.build_dem grid ... --api http://localhost:5000/v1

Add ``--estimate`` to any API command to price the request without
spending quota.
"""

from __future__ import annotations

import argparse
import glob
import sys

from raptor.dem_build import (
    PUBLIC_API, OpenTopoDataClient, build_dem_from_geotiff,
    build_dem_from_opentopodata, corridor_bbox,
)


def _report(result):
    import numpy as np
    e = result["elev"]
    print(f"\n  {len(result['lat_1d'])} x {len(result['lon_1d'])} cells")
    print(f"  elevation {np.nanmin(e):.0f} .. {np.nanmax(e):.0f} m AMSL")
    print(f"  relief    {np.nanmax(e) - np.nanmin(e):.0f} m")


def cmd_tiles(args) -> int:
    paths = []
    for pattern in args.tiff:
        paths.extend(sorted(glob.glob(pattern)))
    if not paths:
        print(f"No files matched: {args.tiff}", file=sys.stderr)
        return 1
    print(f"Reading {len(paths)} tile(s)")
    result = build_dem_from_geotiff(
        paths, *args.bbox, spacing_m=args.spacing,
        out_path=None if args.estimate else args.out, verbose=True,
    )
    _report(result)
    return 0


def cmd_grid(args) -> int:
    if args.estimate:
        return _estimate_grid(args, args.bbox)
    result = build_dem_from_opentopodata(
        *args.bbox, spacing_m=args.spacing, dataset=args.dataset,
        api_url=args.api, out_path=args.out, cache_path=args.cache,
        verbose=True, allow_over_quota=args.force,
    )
    _report(result)
    return 0


def cmd_corridor(args) -> int:
    bbox = corridor_bbox([tuple(args.origin), tuple(args.destination)],
                         half_width_m=args.width)
    print(f"Corridor bbox: lat {bbox[0]:.4f}..{bbox[1]:.4f}, "
          f"lon {bbox[2]:.4f}..{bbox[3]:.4f}")
    route = [tuple(args.origin), tuple(args.destination)]
    if args.estimate:
        return _estimate_grid(args, bbox, corridor=route,
                              half_width=args.width)
    result = build_dem_from_opentopodata(
        *bbox, spacing_m=args.spacing, dataset=args.dataset,
        api_url=args.api, out_path=args.out, cache_path=args.cache,
        verbose=True, allow_over_quota=args.force,
        corridor=route, corridor_half_width_m=args.width,
    )
    _report(result)
    return 0


def _estimate_grid(args, bbox, corridor=None, half_width=None) -> int:
    import numpy as np
    from raptor.dem_build import _grid_axes, route_mask
    lat_1d, lon_1d = _grid_axes(*bbox, args.spacing)
    client = OpenTopoDataClient(api_url=args.api, dataset=args.dataset)
    n_cells = len(lat_1d) * len(lon_1d)
    if corridor is not None:
        LON, LAT = np.meshgrid(lon_1d, lat_1d)
        n = int(route_mask(LAT, LON, corridor, half_width).sum())
        print(f"\n  grid          {len(lat_1d)} x {len(lon_1d)} = {n_cells:,} cells")
        print(f"  in swath      {n:,} points ({100*n/n_cells:.0f}% of the box)")
    else:
        n = n_cells
        print(f"\n  grid          {len(lat_1d)} x {len(lon_1d)} = {n:,} points")
    est = client.estimate(n)
    print(f"  API calls     {est['calls']:,}")
    print(f"  wall time     {est['seconds']/60:.1f} min")
    print(f"  daily quota   {est['days_of_quota']:.2f} days"
          f"  ({'fits' if est['within_daily_quota'] else 'DOES NOT FIT'})")
    if not est["within_daily_quota"]:
        print("\n  Options: coarsen --spacing, fetch a corridor instead of a")
        print("  grid, or self-host OpenTopoData and pass --api.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p, need_out=True):
        p.add_argument("--spacing", type=float, default=30.0,
                       help="output grid spacing [m] (default 30)")
        p.add_argument("--out", required=need_out, help="output .npz path")
        p.add_argument("--estimate", action="store_true",
                       help="report the cost and stop")

    def add_api(p):
        p.add_argument("--dataset", default="srtm30m",
                       help="OpenTopoData dataset (srtm30m, nasadem, "
                            "aster30m, eudem25m, mapzen, ...)")
        p.add_argument("--api", default=PUBLIC_API,
                       help="API base URL; point at a self-hosted instance "
                            "to lift the rate limits")
        p.add_argument("--cache", default="data/elevation_cache.npz",
                       help="point cache, so an interrupted fetch resumes")
        p.add_argument("--force", action="store_true",
                       help="proceed even if the request exceeds the daily "
                            "quota (it will stop when the quota runs out)")

    p = sub.add_parser("tiles", help="mosaic local GeoTIFF tiles")
    p.add_argument("--tiff", nargs="+", required=True,
                   help="tile paths or glob patterns")
    p.add_argument("--bbox", nargs=4, type=float, required=True,
                   metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"))
    add_common(p)
    p.set_defaults(func=cmd_tiles)

    p = sub.add_parser("grid", help="fetch a rectangle from OpenTopoData")
    p.add_argument("--bbox", nargs=4, type=float, required=True,
                   metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"))
    add_common(p)
    add_api(p)
    p.set_defaults(func=cmd_grid)

    p = sub.add_parser("corridor",
                       help="fetch a swath along one route from OpenTopoData")
    p.add_argument("--from", dest="origin", nargs=2, type=float, required=True,
                   metavar=("LAT", "LON"))
    p.add_argument("--to", dest="destination", nargs=2, type=float,
                   required=True, metavar=("LAT", "LON"))
    p.add_argument("--width", type=float, default=3000.0,
                   help="half-width of the swath [m] (default 3000)")
    add_common(p)
    add_api(p)
    p.set_defaults(func=cmd_corridor)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
