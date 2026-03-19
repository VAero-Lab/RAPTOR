#!/usr/bin/env python3
"""
Full Scenario Runner — Runs all 8 scenarios with results tables AND figures.

Usage:
    python run_scenario_catalog.py [--maxiter 60] [--popsize 12] [--seed 42]
    python run_scenario_catalog.py --maxiter 200 --popsize 20  # publication

Author: Victor (EPN / LUAS-EPN)
"""
import sys, os, argparse, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from raptor.dem import DEMInterface
from raptor.config import UAVConfig, MissionConstraints
from raptor.energy import AircraftEnergyParams
from raptor.airspace import build_airspace
from raptor.scenarios import build_scenario_catalog, list_scenarios
from raptor.mission_planner import MissionPlanner
from raptor.viz_airspace import plot_mission_soc


def find_dem():
    for p in ["data/dmq_dem.npz", "dmq_dem.npz", "../data/dmq_dem.npz"]:
        if os.path.exists(p): return p
    raise FileNotFoundError("dmq_dem.npz not found")

def main():
    parser = argparse.ArgumentParser(description="Run full scenario catalog")
    parser.add_argument('--maxiter', type=int, default=60)
    parser.add_argument('--popsize', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-wp', type=int, default=1)
    parser.add_argument('--scenarios', nargs='*', default=None)
    parser.add_argument('--no-airspace', action='store_true')
    parser.add_argument('--output', type=str, default='catalog_results.md')
    parser.add_argument('--plot', action='store_true', default=True, help='Generate figures')
    parser.add_argument('--no-plot', dest='plot', action='store_false')
    parser.add_argument('--fig-dir', type=str, default='scenario_figures')
    args = parser.parse_args()

    print("Loading framework...")
    dem = DEMInterface(find_dem())
    uav = UAVConfig(); constraints = MissionConstraints()
    ac = AircraftEnergyParams()
    airspace = build_airspace(dem=dem)
    planner = MissionPlanner(dem, uav, constraints, ac, airspace=airspace)
    catalog = build_scenario_catalog()
    os.makedirs(args.fig_dir, exist_ok=True)

    scenario_ids = args.scenarios or list(catalog.keys())
    print(f"\nRunning {len(scenario_ids)} scenarios: {scenario_ids}")
    print(f"Settings: maxiter={args.maxiter}, popsize={args.popsize}, n_wp={args.n_wp}")
    print("=" * 80)

    all_results = {}
    t0 = time.time()

    for sid in scenario_ids:
        if sid not in catalog:
            print(f"WARNING: '{sid}' not in catalog, skipping"); continue
        scenario = catalog[sid]
        print(f"\n{'─'*60}\nSCENARIO {sid}: {scenario.name}")
        print(f"  Legs: {scenario.n_legs}, Priority: {scenario.priority.value}")

        configs = [('OFF', False)]
        if not args.no_airspace:
            configs.append(('ON', True))

        scenario_results = {}
        for label, use_air in configs:
            print(f"\n  Airspace {label}:")
            result = planner.plan_mission(scenario, n_intermediate=args.n_wp,
                maxiter=args.maxiter, popsize=args.popsize,
                use_airspace=use_air, seed=args.seed, verbose=True)
            scenario_results[label] = result
            print(f"  → E={result.total_energy_wh:.0f}Wh, t={result.total_time_min:.1f}min, "
                  f"SOC={result.soc_final*100:.1f}%, "
                  f"{'FEASIBLE' if result.mission_feasible else 'INFEASIBLE'}")

        all_results[sid] = scenario_results

        # Generate per-scenario SOC figure (for multi-leg scenarios)
        best = scenario_results.get('ON', scenario_results.get('OFF'))
        if best and best.legs:
            leg_data = []
            for lr in best.legs:
                leg_data.append({
                    'label': f"{lr.leg.origin.short_name}→{lr.leg.destination.short_name}",
                    'E': lr.energy_wh, 't_min': lr.time_min,
                    'soc_i': lr.soc_start, 'soc_f': lr.soc_end,
                    'dist_km': lr.distance_km, 'payload': lr.leg.payload_kg,
                })
            if args.plot:
                fig = plot_mission_soc(leg_data, title=f'{sid}: {scenario.name}',
                    save_path=f'{args.fig_dir}/{sid}_soc_profile.png')
                plt.close(fig)
                print(f"  Saved {sid}_soc_profile.png")

    print(f"\n{'='*80}\nAll scenarios completed in {time.time()-t0:.0f}s")

    # Write results markdown
    write_results(all_results, catalog, args)
    print(f"Results written to {args.output}")
    print(f"Figures saved to {args.fig_dir}/")


def write_results(all_results, catalog, args):
    lines = ["# Scenario Catalog Results",
             f"\nSettings: maxiter={args.maxiter}, popsize={args.popsize}, n_wp={args.n_wp}",
             f"Framework: v0.0.1\n"]

    for sid, configs in all_results.items():
        sc = catalog[sid]
        lines.append(f"## {sid}: {sc.name}")
        lines.append(f"Priority: {sc.priority.value} | Urgency: {sc.urgency.value} | Legs: {sc.n_legs}\n")
        for label, result in configs.items():
            lines.append(f"### Airspace {label}")
            lines.append(result.summary_table())
            lines.append("")

    lines.append("## Summary: OFF vs ON\n")
    lines.append(f"| {'ID':5s} | {'Legs':>4s} | {'E_OFF':>7s} | {'E_ON':>7s} | {'ΔE%':>6s} | {'SOC_OFF':>7s} | {'SOC_ON':>7s} |")
    lines.append("|" + "|".join(["-"*w for w in [7,6,9,9,8,9,9]]) + "|")
    for sid, configs in all_results.items():
        r_off = configs.get('OFF'); r_on = configs.get('ON')
        if r_off and r_on:
            de = (r_on.total_energy_wh - r_off.total_energy_wh) / max(r_off.total_energy_wh, 1) * 100
            lines.append(f"| {sid:5s} | {catalog[sid].n_legs:4d} | {r_off.total_energy_wh:7.0f} | "
                        f"{r_on.total_energy_wh:7.0f} | {de:+6.1f} | {r_off.soc_final*100:7.1f} | {r_on.soc_final*100:7.1f} |")

    with open(args.output, 'w') as f:
        f.write("\n".join(lines))


if __name__ == '__main__':
    main()
