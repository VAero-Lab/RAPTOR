#!/usr/bin/env python3
"""
Every figure, for one case and one pair of facilities.

Why this is separate from ``make_figures``
-------------------------------------------
``scripts/make_figures.py`` draws the *network*: all the corridors, all
the airspace, the whole atlas. Those figures answer questions about the
region and there is one of each.

This one draws a *flight*. There is one of each per case and per pair,
which is a few hundred combinations, so it takes selectors and a sweep
rather than producing everything by default.

What it produces, per combination
----------------------------------
``plan_and_profile``   the route from above, its altitude, its height
                       above ground and its lateral offset, on one
                       distance axis — enough to reconstruct the flight
``path_3d``            the same route over the terrain, in three
                       dimensions
``optimizer_history``  the routes the search tried, and the objective
                       against the constraint penalties behind it
``battery``            state of charge, voltage, current and cumulative
                       energy, with the flight phases marked
``energy_breakdown``   which phase spent the energy, against which
                       phase spent the time

Usage
-----
    python -m scripts.make_case_figures --list
    python -m scripts.make_case_figures --case one_way --pair espejo-suarez
    python -m scripts.make_case_figures --case out_and_back --all-pairs
    python -m scripts.make_case_figures --all --maxiter 8 --popsize 6
    python -m scripts.make_case_figures --case one_way --pair garces-pintag \
        --objective time --urgency emergency

Objectives
----------
``--objective`` picks what the optimizer minimises: ``energy`` (the
default everywhere else in the package), ``time``, or ``balanced``.
``--urgency`` sets the regulatory time cap and state-of-charge floor —
routine is 60 min and 15 %, urgent 30 min and 15 %, emergency 15 min and
20 %. The two are separate on purpose: a mission can be *constrained* in
time without being *optimised* for it.

Author: Victor (LUAS-EPN / KU Leuven)
"""

from __future__ import annotations

import argparse
import os
import sys
import time as _time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from raptor.airspace import build_airspace
from raptor.config import MissionConstraints, UAVConfig
from raptor.dem import DEMInterface, find_dem
from raptor.energy import analyze_path_energy
from raptor.mission_planner import MissionPlanner
from raptor.mission_view import MissionPath
from raptor.missions import (
    BatteryAction, hub_and_spoke, one_way, out_and_back, sample_collection,
    shuttle, supply_tour,
)
from raptor.optimizer import OptMode, PathOptimizer
from raptor.routed_path import RoutedPath
from raptor.builder import FacilityNode
from raptor.scenarios import MedicalUrgency, OptPriority
from raptor.vehicles import get_vehicle
from raptor.viz_corridor import (
    plot_battery_timeline, plot_energy_breakdown, plot_optimizer_history,
    plot_path_3d, plot_plan_and_profile,
)
import raptor.scenarios as _sc

# ── Facilities and pairs ──────────────────────────────────────────────

FACILITIES = {
    "espejo": _sc.HOSPITAL_EUGENIO_ESPEJO,
    "garces": _sc.HOSPITAL_ENRIQUE_GARCES,
    "suarez": _sc.HOSPITAL_PABLO_ARTURO_SUAREZ,
    "calderon": _sc.HOSPITAL_DOCENTE_CALDERON,
    "lloa": _sc.CS_LLOA,
    "pintag": _sc.CS_PINTAG,
    "conocoto": _sc.CS_CONOCOTO,
    "tumbaco": _sc.CS_GUANGOPOLO,
}

#: The pairs worth drawing, chosen to span the range the atlas shows —
#: from the two that fit a straight corridor to the ones that miss by
#: hundreds of metres.
PAIRS = {
    "espejo-suarez":   ("espejo", "suarez"),     # fits, basin floor
    "suarez-calderon": ("suarez", "calderon"),   # fits, basin floor
    "espejo-calderon": ("espejo", "calderon"),   # misses by 82 m
    "garces-pintag":   ("garces", "pintag"),     # 709 m of relief
    "garces-lloa":     ("garces", "lloa"),       # short and steep
    "espejo-lloa":     ("espejo", "lloa"),       # onto the Pichincha flank
    "suarez-lloa":     ("suarez", "lloa"),       # the worst in the atlas
    "espejo-pintag":   ("espejo", "pintag"),     # long, moderate relief
}

# ── Cases ─────────────────────────────────────────────────────────────
#
# A case is a mission shape. Those that need a second destination take
# the pair's origin as base and both facilities as the stops.

CASES = {
    "corridor":               "one leg, five planning methods compared",
    "one_way":                "a single delivery, no return",
    "out_and_back":           "deliver and return, the pack carried over",
    "sample_collection":      "leave empty, collect at each stop, return heavy",
    "supply_tour":            "leave loaded, drop at each stop, return empty",
    "hub_and_spoke":          "out and back to each spoke, swapping at base",
    "hub_and_spoke_no_spare": "the same with no spare pack at base",
    "shuttle":                "the same delivery repeated, swapping between",
}

OBJECTIVES = {"energy": OptPriority.ENERGY,
              "time": OptPriority.TIME,
              "balanced": OptPriority.BALANCED}
URGENCIES = {"routine": MedicalUrgency.ROUTINE,
             "urgent": MedicalUrgency.URGENT,
             "emergency": MedicalUrgency.EMERGENCY}


def build_mission(case: str, a, b, payload_kg: float,
                  priority, urgency):
    """The MissionPlan for one case over one pair."""
    kw = dict(priority=priority, urgency=urgency)
    if case == "one_way":
        return one_way(a, b, payload_kg=payload_kg, **kw)
    if case == "out_and_back":
        return out_and_back(a, b, payload_out_kg=payload_kg, **kw)
    if case == "sample_collection":
        return sample_collection(a, [b], per_stop_kg=payload_kg, **kw)
    if case == "supply_tour":
        return supply_tour(a, [b], per_stop_kg=payload_kg, **kw)
    if case == "hub_and_spoke":
        return hub_and_spoke(a, [b], payload_kg=payload_kg, **kw)
    if case == "hub_and_spoke_no_spare":
        return hub_and_spoke(a, [b], payload_kg=payload_kg,
                             battery_action=BatteryAction.NONE,
                             name="Hub and spoke, no spare pack",
                             mission_id="HS-nobat", **kw)
    if case == "shuttle":
        return shuttle(a, b, n_round_trips=2, payload_kg=payload_kg, **kw)
    raise KeyError(case)


def save(fig, out_dir: str, name: str, dpi: int):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    {path}")
    return path


# ── The corridor case: five methods on one leg ────────────────────────

def run_corridor(ctx, a, b, out_dir, args):
    """Five planning methods on one leg, drawn together."""
    dem, con, ac, uav, airspace = ctx
    o = FacilityNode(a.short_name, a.lat, a.lon, dem.elevation(a.lat, a.lon))
    d = FacilityNode(b.short_name, b.lat, b.lon, dem.elevation(b.lat, b.lon))

    from raptor.astar_baseline import AStarGridPlanner

    paths, labels, wpts = [], [], []
    rp = RoutedPath(o, d, dem, uav, con, n_intermediate=0,
                    corridor_mode="amsl")
    paths.append(rp.flight_path); labels.append("straight line, constant altitude")
    wpts.append(rp._waypoint_coords)
    rp = RoutedPath(o, d, dem, uav, con, n_intermediate=0,
                    corridor_mode="agl")
    paths.append(rp.flight_path); labels.append("terrain-following, DGAC off")
    wpts.append(rp._waypoint_coords)

    res = AStarGridPlanner(dem, ac=ac, airspace=airspace,
                           grid_resolution_m=400.0).plan(
        (o.lat, o.lon, o.ground_elev), (d.lat, d.lon, d.ground_elev))
    if res.path_found:
        paths.append(res)
        labels.append(f"A* grid ({res.grid_resolution_m:.0f} m cells)")
        wpts.append(None)

    best = None
    for n in (1, 3):
        rp = RoutedPath(o, d, dem, uav, con, n_intermediate=n,
                        corridor_mode="agl")
        r = PathOptimizer(dem, uav, con, ac).optimize_routed(
            rp, mode=OptMode[args.objective.upper()],
            payload_kg=args.payload, airspace=airspace,
            maxiter=args.maxiter, popsize=args.popsize, seed=args.seed,
            verbose=False)
        paths.append(r.optimized_path)
        labels.append(f"optimised, DGAC on, N={n}  ({r.energy_wh:.0f} Wh)")
        wpts.append(getattr(rp, "_waypoint_coords", None))
        best = (r, rp)

    title = f"{a.short_name} → {b.short_name}"
    fig = plot_plan_and_profile(paths, labels, dem, con, airspace=airspace,
                                waypoints=wpts, suptitle=title,
                                endpoint_names=(a.short_name, b.short_name))
    save(fig, out_dir, "plan_and_profile.png", args.dpi)

    fig = plt.figure(figsize=(10.5, 8.0))
    ax = fig.add_subplot(111, projection="3d")
    plot_path_3d(paths, labels, dem, ax=ax, title=title)
    save(fig, out_dir, "path_3d.png", args.dpi)

    if best is not None:
        r, rp = best
        fig = plot_optimizer_history(
            r, rp, dem, airspace=airspace,
            suptitle=f"{title} — {args.objective} objective")
        save(fig, out_dir, "optimizer_history.png", args.dpi)

        er = analyze_path_energy(r.optimized_path,
                                 ac.with_payload(args.payload))
        fig = plot_battery_timeline(er, ac, suptitle=f"{title} — the pack")
        save(fig, out_dir, "battery.png", args.dpi)
        _, ax = plt.subplots(figsize=(8.2, 3.6))
        plot_energy_breakdown(er, ax=ax,
                              title=f"{title} — energy against time, by phase")
        save(ax.figure, out_dir, "energy_breakdown.png", args.dpi)
    return True


# ── The mission cases ─────────────────────────────────────────────────

def run_mission(case, ctx, a, b, out_dir, args):
    """One mission shape, planned and drawn."""
    dem, con, ac, uav, airspace = ctx
    planner = MissionPlanner(dem, uav, con, ac, airspace=airspace)
    mission = build_mission(case, a, b, args.payload,
                            OBJECTIVES[args.objective],
                            URGENCIES[args.urgency])
    result = planner.plan(mission, n_intermediate=args.n_intermediate,
                          maxiter=args.maxiter, popsize=args.popsize,
                          seed=args.seed, verbose=False)

    leg_paths = [lr.opt_result.optimized_path for lr in result.legs]
    leg_names = [f"leg {i+1}: {lr.leg.origin.short_name}→"
                 f"{lr.leg.destination.short_name}"
                 for i, lr in enumerate(result.legs)]
    mission_path = MissionPath(leg_paths, leg_names)

    title = (f"{mission.name} — {args.objective} objective, "
             f"{args.urgency}")
    subtitle = (f"{result.total_distance_km:.1f} km, "
                f"{result.total_time_min:.1f} min, "
                f"{result.total_energy_wh:.0f} Wh, "
                f"SOC min {100*result.soc_min:.1f} %")

    fig = plot_plan_and_profile(
        leg_paths, leg_names, dem, con, airspace=airspace,
        suptitle=f"{title}\n{subtitle}",
        endpoint_names=(a.short_name, b.short_name))
    save(fig, out_dir, "plan_and_profile.png", args.dpi)

    fig = plt.figure(figsize=(10.5, 8.0))
    ax = fig.add_subplot(111, projection="3d")
    plot_path_3d(leg_paths, leg_names, dem, ax=ax, title=title)
    save(fig, out_dir, "path_3d.png", args.dpi)

    # The pack across the whole mission, not per leg: the coupling
    # between legs is the thing a multi-leg mission has that a
    # single flight does not.
    er = analyze_path_energy(mission_path, ac.with_payload(args.payload))
    fig = plot_battery_timeline(
        er, ac, soc_reserve=URGENCIES[args.urgency].min_soc_percent(),
        suptitle=f"{title} — the pack over {len(leg_paths)} leg(s)")
    save(fig, out_dir, "battery.png", args.dpi)

    _, ax = plt.subplots(figsize=(8.2, 3.6))
    plot_energy_breakdown(er, ax=ax,
                          title=f"{title} — energy against time, by phase")
    save(ax.figure, out_dir, "energy_breakdown.png", args.dpi)

    # Optimizer history for the first leg — the search is per leg, so
    # one history per mission would be a fiction.
    first = result.legs[0]
    rp = first.routed_path
    if rp is not None and first.opt_result.parameter_history:
        fig = plot_optimizer_history(
            first.opt_result, rp, dem, airspace=airspace,
            suptitle=f"{title} — leg 1 of {len(result.legs)}")
        save(fig, out_dir, "optimizer_history.png", args.dpi)
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", choices=sorted(CASES))
    ap.add_argument("--pair", choices=sorted(PAIRS))
    ap.add_argument("--all", action="store_true",
                    help="every case over every pair (slow)")
    ap.add_argument("--all-pairs", action="store_true",
                    help="one case over every pair")
    ap.add_argument("--all-cases", action="store_true",
                    help="every case over one pair")
    ap.add_argument("--list", action="store_true",
                    help="show the cases and pairs and exit")
    ap.add_argument("--objective", choices=sorted(OBJECTIVES),
                    default="energy")
    ap.add_argument("--urgency", choices=sorted(URGENCIES), default="routine")
    ap.add_argument("--vehicle", default="va23")
    ap.add_argument("--payload", type=float, default=1.5)
    ap.add_argument("--n-intermediate", type=int, default=1)
    ap.add_argument("--maxiter", type=int, default=10)
    ap.add_argument("--popsize", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--out", default="figures/cases")
    args = ap.parse_args(argv)

    if args.list:
        print("cases:")
        for k, v in CASES.items():
            print(f"  {k:24s} {v}")
        print("\npairs:")
        for k, (x, y) in PAIRS.items():
            print(f"  {k:18s} {FACILITIES[x].short_name} → "
                  f"{FACILITIES[y].short_name}")
        print("\nobjectives:", ", ".join(sorted(OBJECTIVES)))
        print("urgencies :", ", ".join(sorted(URGENCIES)))
        return 0

    cases = (list(CASES) if (args.all or args.all_cases)
             else [args.case] if args.case else None)
    pairs = (list(PAIRS) if (args.all or args.all_pairs)
             else [args.pair] if args.pair else None)
    if not cases or not pairs:
        ap.error("give --case and --pair, or --all / --all-cases / --all-pairs")

    dem = DEMInterface(find_dem())
    con = MissionConstraints()
    ac = get_vehicle(args.vehicle).build_aero()
    uav = UAVConfig.for_vehicle(ac.with_payload(args.payload), altitude=2800.0)
    airspace = build_airspace(dem=dem)
    ctx = (dem, con, ac, uav, airspace)

    print(f"vehicle   {ac.name}, {args.payload} kg payload")
    print(f"terrain   {dem.metadata.provenance()}")
    print(f"objective {args.objective}, urgency {args.urgency} "
          f"({URGENCIES[args.urgency].max_flight_time_s()/60:.0f} min cap, "
          f"{100*URGENCIES[args.urgency].min_soc_percent():.0f} % floor)")
    print(f"effort    maxiter={args.maxiter} popsize={args.popsize} "
          f"seed={args.seed}")
    print(f"{len(cases)} case(s) x {len(pairs)} pair(s)\n")

    ok = fail = 0
    for case in cases:
        for pair in pairs:
            a = FACILITIES[PAIRS[pair][0]]
            b = FACILITIES[PAIRS[pair][1]]
            out_dir = os.path.join(args.out, case, pair)
            print(f"  {case} / {pair}  ({a.short_name} → {b.short_name})")
            t0 = _time.perf_counter()
            try:
                if case == "corridor":
                    run_corridor(ctx, a, b, out_dir, args)
                else:
                    run_mission(case, ctx, a, b, out_dir, args)
                ok += 1
            except Exception as exc:
                print(f"    FAILED: {type(exc).__name__}: {exc}",
                      file=sys.stderr)
                fail += 1
            print(f"    ({_time.perf_counter()-t0:.0f} s)\n")
            plt.close("all")

    print(f"{ok} combination(s) drawn, {fail} failed")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
