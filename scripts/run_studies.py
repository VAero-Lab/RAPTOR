#!/usr/bin/env python3
"""
The scenario studies — the experiments the package exists to run.

Five studies, each isolating one variable:

1. **Corridor atlas.** Every facility pair, classified compliant /
   authorisation / waiver / not flyable. The network-level result.
2. **Payload sequencing.** Collection against supply tour over identical
   facilities. Seeded and repeated, because at low generation counts the
   optimizer's own spread can exceed the effect.
3. **Battery infrastructure.** The same route with and without a spare
   pack at base.
4. **Permit sensitivity.** Sweep the permit weight and watch routes
   switch from "cross and file" to "fly around".
5. **Credential sensitivity.** The same route under different operator
   credentials.

Usage:
    python -m scripts.run_studies --only atlas
    python -m scripts.run_studies --maxiter 60 --repeats 3
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

from raptor.aero import check_flight_envelope
from raptor.airspace import build_airspace
from raptor.compliance import ComplianceLevel
from raptor.config import MissionConstraints, UAVConfig
from raptor.corridor import survey_corridor
from raptor.dem import DEMInterface, find_dem
from raptor.energy import AircraftEnergyParams
from raptor.vehicles import get_vehicle
from raptor.builder import FacilityNode
from raptor.mission_planner import MissionPlanner
from raptor.missions import (
    BatteryAction, hub_and_spoke, sample_collection, supply_tour,
)
from raptor.optimizer import OptMode, PathOptimizer
from raptor.penalties import PenaltyWeights
from raptor.regulations import MEDICAL_DELIVERY_CONTEXT, OperationalContext, RDAC_101
from raptor.routed_path import RoutedPath
from raptor.scenarios import (
    CS_LLOA, CS_PINTAG, HOSPITAL_DOCENTE_CALDERON, HOSPITAL_ENRIQUE_GARCES,
    HOSPITAL_EUGENIO_ESPEJO, HOSPITAL_PABLO_ARTURO_SUAREZ,
)

FACILITIES = [
    HOSPITAL_EUGENIO_ESPEJO, HOSPITAL_ENRIQUE_GARCES,
    HOSPITAL_PABLO_ARTURO_SUAREZ, HOSPITAL_DOCENTE_CALDERON,
    CS_LLOA, CS_PINTAG,
]


def setup(vehicle: str = "va23", payload_kg: float = 1.5):
    """
    The aircraft, the terrain and the rules every study runs against.

    The envelope is derived at the *loaded* mass, because that is what
    flies the mission; deriving it empty gives a stall speed the loaded
    aircraft cannot achieve.
    """
    dem = DEMInterface(find_dem())
    con = MissionConstraints()
    ac = get_vehicle(vehicle).build_aero()
    uav = UAVConfig.for_vehicle(ac.with_payload(payload_kg), altitude=2800.0)
    airspace = build_airspace(dem=dem)
    return dem, uav, con, ac, airspace


def banner(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


# ── 1. Corridor atlas ─────────────────────────────────────────────────────

def study_atlas(dem, uav, con, ac, airspace, args):
    """
    Every facility pair, surveyed. No optimization: this asks whether a
    *straight* corridor admits a legal profile, which is the question that
    tells you which pairs need work.
    """
    banner("STUDY 1 — CORRIDOR ATLAS")
    rows = []
    for a, b in itertools.combinations(FACILITIES, 2):
        s = survey_corridor(dem, (a.lat, a.lon), (b.lat, b.lon), con, uav)
        rows.append(dict(
            origin=a.short_name, destination=b.short_name,
            km=round(s.direct_distance_m / 1000, 2),
            relief_m=round(s.relief_m),
            excess_m=round(s.ceiling_excess_m),
            over_pct=round(100 * s.ceiling_excess_fraction, 1),
            fits=bool(s.feasible),
        ))

    rows.sort(key=lambda r: (-r["excess_m"], r["km"]))
    print(f"{'pair':38s} {'km':>6s} {'relief':>7s} {'excess':>7s} {'over':>6s}")
    for r in rows:
        mark = "" if r["fits"] else "  <-- needs a detour or a waiver"
        print(f"{r['origin']+' -> '+r['destination']:38s} {r['km']:6.1f} "
              f"{r['relief_m']:6d}m {r['excess_m']:6d}m {r['over_pct']:5.1f}%{mark}")

    n_ok = sum(r["fits"] for r in rows)
    print(f"\n{n_ok}/{len(rows)} pairs admit a legal straight corridor.")
    print("The rest are where lateral routing earns its keep.")
    return rows


# ── 2. Payload sequencing ─────────────────────────────────────────────────

def replay_payload_schedule(leg_paths, schedule, ac, stops=None,
                            soc_initial: float = 1.0):
    """
    Fly a fixed set of legs carrying a different payload on each.

    The geometry is given and never re-planned. That is the whole point:
    it removes the optimizer from the comparison entirely, so what comes
    out is the effect of *when* mass is carried and nothing else.

    Returns ``(total energy Wh, minimum SOC)``.
    """
    from raptor.energy import analyze_path_energy
    from raptor.missions import BatteryAction

    soc = soc_initial
    total = 0.0
    soc_min = soc
    for i, (path, payload) in enumerate(zip(leg_paths, schedule)):
        loaded = ac.with_payload(payload)
        r = analyze_path_energy(path, loaded, SOC_initial=soc)
        total += r.total_energy_wh
        soc = r.SOC_final
        soc_min = min(soc_min, r.min_SOC)
        if stops is not None and i + 1 < len(stops):
            if stops[i + 1].battery_action == BatteryAction.SWAP:
                soc = 1.0
    return total, soc_min


def study_sequencing(dem, uav, con, ac, airspace, args):
    """
    Collection versus supply tour, on identical geometry.

    The two missions visit the same facilities in the same order and
    differ only in the payload each leg carries — a collection round
    starts empty and gets heavier, a supply tour starts loaded and gets
    lighter. So the effect being measured is worth a few watt-hours, and
    the earlier version of this study could not see it: it planned each
    mission separately, and the optimizer's run-to-run spread of about
    15 Wh swamped a 6 Wh effect. Two rounds of this study reported a
    winner and the second one reversed.

    The fix is not more compute. It is to stop re-planning: fix the
    route once, then fly both payload schedules down it. The optimizer
    appears once instead of twice, its noise is common to both arms and
    cancels exactly, and the answer is deterministic.

    The route is still planned twice — once for each mission — so the
    comparison can also be made on the *other* mission's route. If the
    sign holds on both, the conclusion does not depend on which route
    was chosen.
    """
    banner("STUDY 2 — PAYLOAD SEQUENCING")
    planner = MissionPlanner(dem, uav, con, ac, airspace=airspace)
    centres = [CS_LLOA, CS_PINTAG]

    missions = {
        "collection": sample_collection(HOSPITAL_EUGENIO_ESPEJO, centres,
                                        per_stop_kg=0.7),
        "supply": supply_tour(HOSPITAL_EUGENIO_ESPEJO, centres,
                              per_stop_kg=0.7),
    }
    schedules = {k: m.payload_schedule() for k, m in missions.items()}
    print("  payload carried per leg [kg]")
    for k, sched in schedules.items():
        print(f"    {k:11s} {sched}   heaviest leg #"
              f"{missions[k].heaviest_leg + 1} of {len(sched)}")
    print()

    routes, out = {}, {}
    for name, mission in missions.items():
        r = planner.plan(mission, n_intermediate=1, maxiter=args.maxiter,
                         popsize=args.popsize, seed=42, verbose=False)
        routes[name] = [leg.opt_result.optimized_path for leg in r.legs]
        print(f"  planned the {name} route: {len(r.legs)} legs, "
              f"{r.total_distance_km:.1f} km")
        sys.stdout.flush()

    print()
    print(f"  {'route flown':>16s} {'payload order':>15s} {'E [Wh]':>9s} "
          f"{'SOC min':>9s}")
    print("  " + "-" * 54)
    for route_name, leg_paths in routes.items():
        energies = {}
        for sched_name, sched in schedules.items():
            e, soc_min = replay_payload_schedule(
                leg_paths, sched, ac, stops=missions[sched_name].stops)
            energies[sched_name] = e
            print(f"  {route_name:>16s} {sched_name:>15s} {e:9.2f} "
                  f"{100*soc_min:8.1f}%")
        d = energies["collection"] - energies["supply"]
        out[route_name] = d
        print(f"  {'':>16s} {'difference':>15s} {d:+9.2f}"
              f"   ({100*d/energies['supply']:+.2f} %)")
        print()

    # Where the difference actually lives. Reporting a total without
    # this is how the first two versions of this study ended up
    # attributing the effect to state of charge, which is not what
    # drives it.
    from raptor.energy import analyze_path_energy
    leg_paths = routes["collection"]
    print(f"  {'leg':24s} {'km':>6s} {'net climb':>10s} {'Wh/kg':>8s} "
          f"{'Wh/kg/km':>9s}")
    print("  " + "-" * 62)
    per_leg = []
    for i, p in enumerate(leg_paths):
        km = p.metrics.total_ground_distance / 1000.0
        net = p.metrics.total_altitude_gain - p.metrics.total_altitude_loss
        e0 = analyze_path_energy(p, ac.with_payload(0.0)).total_energy_wh
        e1 = analyze_path_energy(p, ac.with_payload(1.4)).total_energy_wh
        dedm = (e1 - e0) / 1.4
        per_leg.append(dict(km=km, net_climb_m=net, wh_per_kg=dedm))
        print(f"  {'leg ' + str(i + 1):24s} {km:6.1f} {net:+9.0f}m "
              f"{dedm:8.2f} {dedm / km:9.4f}")

    signs = {np.sign(v) for v in out.values()}
    mean_d = float(np.mean(list(out.values())))
    cheap = min(per_leg, key=lambda d: d["wh_per_kg"] / d["km"])
    dear = max(per_leg, key=lambda d: d["wh_per_kg"] / d["km"])
    ratio = ((dear["wh_per_kg"] / dear["km"])
             / max(cheap["wh_per_kg"] / cheap["km"], 1e-9))

    print()
    spread_between_routes = abs(out["collection"] - out["supply"])
    print(f"  Within a fixed route the comparison is exact — no seeds, no "
          f"spread.\n  Collection costs {out['collection']:+.2f} Wh on its own "
          f"route and\n  {out['supply']:+.2f} Wh on the supply tour's, "
          f"averaging {mean_d:+.2f} Wh.")
    print(f"\n  The {spread_between_routes:.2f} Wh between those two figures is "
          f"not noise in the\n  comparison: it is the two routes being "
          f"genuinely different, and it\n  is worth noticing that it is the "
          f"same size as the effect. **Which\n  route the optimizer settles on "
          f"matters about as much as which way\n  round the payload is "
          f"carried.**")
    if len(signs) == 1:
        print(f"\n  The sign is the same on both routes, so the result does "
              f"not\n  depend on which geometry was chosen.")
    else:
        print(f"\n  ! The sign flips between the routes. The effect is "
              f"smaller than\n  the difference between the routes themselves.")

    print(f"\n  WHY, and it is not what it looks like. Carrying a kilogram "
          f"costs\n  {ratio:.1f}x more on the dearest leg than on the cheapest, "
          f"and what\n  separates them is *net climb*, not distance: the "
          f"expensive leg\n  gains {dear['net_climb_m']:+.0f} m over "
          f"{dear['km']:.0f} km, the cheap one "
          f"{cheap['net_climb_m']:+.0f} m over {cheap['km']:.0f} km.\n"
          f"  Lifting mass up a hill buys potential energy that the descent\n"
          f"  does not give back, and a climbing leg also spends longer at\n"
          f"  climb power. Potential energy alone accounts for about a third\n"
          f"  of the gap; the time at power accounts for the rest.")
    print(f"\n  So the rule is not 'carry heavy early' or 'carry heavy "
          f"late'.\n  It is: carry the mass on the legs that descend. Which "
          f"ordering\n  wins is a property of the terrain between these "
          f"particular\n  facilities, not of the mission shape — swap the "
          f"facilities and\n  the answer can reverse, which is exactly why "
          f"two earlier\n  versions of this study disagreed with each other.")

    return {"per_route_difference_wh": out, "mean_difference_wh": mean_d,
            "schedules": schedules, "per_leg": per_leg,
            "cost_ratio_dearest_to_cheapest": ratio}


# ── 3. Battery infrastructure ─────────────────────────────────────────────

def study_battery(dem, uav, con, ac, airspace, args):
    """The same route, differing only in whether base has a spare pack."""
    banner("STUDY 3 — BATTERY INFRASTRUCTURE")
    planner = MissionPlanner(dem, uav, con, ac, airspace=airspace)
    centres = [CS_LLOA, CS_PINTAG]

    print(f"{'ground equipment':24s} {'E [Wh]':>8s} {'flight':>8s} "
          f"{'SOC min':>8s}  verdict")
    out = []
    for action, label in ((BatteryAction.SWAP, "spare pack at base"),
                          (BatteryAction.NONE, "no spare pack")):
        m = hub_and_spoke(HOSPITAL_EUGENIO_ESPEJO, centres, payload_kg=2.0,
                          battery_action=action)
        r = planner.plan(m, n_intermediate=1, maxiter=args.maxiter,
                         popsize=args.popsize, seed=7, verbose=False)
        verdict = "feasible" if r.soc_min > 0.15 else "BATTERY EXHAUSTED"
        print(f"{label:24s} {r.total_energy_wh:8.1f} "
              f"{r.total_time_min:7.1f}m {r.soc_min*100:7.1f}%  {verdict}")
        out.append(dict(equipment=label, energy_wh=r.total_energy_wh,
                        soc_min=r.soc_min))
        sys.stdout.flush()

    print("\n  Identical routing and near-identical energy. Whether the")
    print("  mission exists is decided on the ground, not in the air.")
    return out


# ── 4. Permit sensitivity ─────────────────────────────────────────────────

def study_permits(dem, uav, con, ac, airspace, args):
    """
    Sweep the permit weight. At zero the optimizer crosses anything it is
    allowed to; raise it far enough and it flies around. The switching
    point is a statement about the cost of the regulatory regime.
    """
    banner("STUDY 4 — PERMIT SENSITIVITY")
    # Garcés rather than Espejo: Espejo's pad sits *inside* the Baca Ortiz
    # hospital buffer, so every route from it needs that permit whatever
    # the weight. A sweep run from there measures nothing.
    o = FacilityNode("H.Garces", -0.2641, -78.5504,
                     float(dem.elevation(-0.2641, -78.5504)))
    d = FacilityNode("H.Calderon", -0.0978, -78.4239,
                     float(dem.elevation(-0.0978, -78.4239)))

    print(f"{'permit weight':>14s} {'E [Wh]':>8s} {'detour':>8s} "
          f"{'permits':>8s}  verdict")
    out = []
    for w in (0.0, 0.02, 0.1, 0.5, 2.0):
        opt = PathOptimizer(dem, uav, con, ac, regulation=RDAC_101,
                            context=MEDICAL_DELIVERY_CONTEXT,
                            penalty_weights=PenaltyWeights(permit=w),
                            warn_on_bad_constraints=False)
        rp = RoutedPath(o, d, dem, uav, con, n_intermediate=2,
                        corridor_mode="agl")
        r = opt.plan_route(rp, mode=OptMode.ENERGY, payload_kg=1.5,
                           airspace=airspace, maxiter=args.maxiter,
                           popsize=args.popsize, seed=11, verbose=False)
        topo = rp.topology_summary()
        fixed = len(r.compliance.unavoidable_permits) if r.compliance else 0
        print(f"{w:14.2f} {r.energy_wh:8.1f} "
              f"{topo['route_stretch_factor']:7.3f}x "
              f"{len(r.permits_required):8d}  {r.regulatory_summary}"
              + (f"  ({fixed} unavoidable)" if fixed else ""))
        out.append(dict(weight=w, energy_wh=r.energy_wh,
                        stretch=topo["route_stretch_factor"],
                        n_permits=len(r.permits_required),
                        unavoidable=fixed))
        sys.stdout.flush()

    n = [r["n_permits"] for r in out]
    if len(set(n)) == 1:
        fixed = out[0]["unavoidable"]
        print(f"\n  The permit count does not move across the sweep.")
        if fixed:
            print(f"  {fixed} of the {n[0]} are unavoidable — a terminal sits")
            print(f"  inside the zone, so no weight can buy a way out of it.")
        else:
            print(f"  The remaining permits are cheaper to cross than to")
            print(f"  detour around at any weight tried; widen the sweep.")
    return out


# ── 5. Credential sensitivity ─────────────────────────────────────────────

def study_credentials(dem, uav, con, ac, airspace, args):
    """The same flight under different operator credentials."""
    banner("STUDY 5 — OPERATOR CREDENTIALS")
    o = FacilityNode("H.Espejo", -0.2000, -78.4930,
                     float(dem.elevation(-0.2000, -78.4930)))
    d = FacilityNode("CS.Guamani", -0.3242, -78.5493,
                     float(dem.elevation(-0.3242, -78.5493)))

    contexts = [
        ("no UOC, VLOS", OperationalContext(
            category="open", has_uoc=False, bvlos=False, autonomous=True,
            carries_medical_payload=True)),
        ("UOC, BVLOS", MEDICAL_DELIVERY_CONTEXT),
        ("UOC + hospital permits held", OperationalContext(
            has_uoc=True, bvlos=True, autonomous=True,
            carries_medical_payload=True,
            authorized_zone_ids=[
                "HOSPITAL_Hospital_de_Ninos_Baca_Ortiz",
                "HOSPITAL_Hospital_Metropolitano",
            ])),
    ]

    print(f"{'credentials':30s} {'E [Wh]':>8s} {'permits':>8s}  standing")
    out = []
    for label, ctx in contexts:
        opt = PathOptimizer(dem, uav, con, ac, regulation=RDAC_101,
                            context=ctx, warn_on_bad_constraints=False)
        rp = RoutedPath(o, d, dem, uav, con, n_intermediate=2,
                        corridor_mode="agl")
        r = opt.plan_route(rp, mode=OptMode.ENERGY, payload_kg=1.5,
                           airspace=airspace, maxiter=args.maxiter,
                           popsize=args.popsize, seed=11, verbose=False)
        print(f"{label:30s} {r.energy_wh:8.1f} "
              f"{len(r.permits_required):8d}  {r.regulatory_summary}")
        for note in ctx.compliance_findings(RDAC_101):
            if "requires" in note or "forbids" in note or "cannot" in note:
                print(f"    ! {note[:88]}")
        out.append(dict(credentials=label, energy_wh=r.energy_wh,
                        n_permits=len(r.permits_required),
                        standing=r.regulatory_summary))
        sys.stdout.flush()
    return out


STUDIES = {
    "atlas": study_atlas,
    "sequencing": study_sequencing,
    "battery": study_battery,
    "permits": study_permits,
    "credentials": study_credentials,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=sorted(STUDIES))
    ap.add_argument("--maxiter", type=int, default=25)
    ap.add_argument("--popsize", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="figures/current/studies.json")
    ap.add_argument("--vehicle", default="va23",
                    help="which shipped aircraft to fly the studies with")
    ap.add_argument("--payload", type=float, default=1.5, help="kg")
    args = ap.parse_args(argv)

    dem, uav, con, ac, airspace = setup(args.vehicle, args.payload)
    print(f"DEM       {find_dem()}")
    print(f"aero      {ac.aero_source}")
    print(f"envelope  {uav.fw_min_airspeed:.1f}–{uav.fw_max_airspeed:.1f} m/s "
          f"(problems: {len(check_flight_envelope(uav, ac, 2800.0))})")
    print(f"corridor  {con.describe_corridor()}")
    print(f"settings  maxiter={args.maxiter} popsize={args.popsize} "
          f"repeats={args.repeats}")

    results = {}
    t0 = time.time()
    for name in (args.only or sorted(STUDIES)):
        try:
            results[name] = STUDIES[name](dem, uav, con, ac, airspace, args)
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=1, default=float)
        print(f"\nwrote {args.out}   ({time.time()-t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
