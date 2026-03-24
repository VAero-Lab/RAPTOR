#!/usr/bin/env python3
"""
Run Experiments & Generate Figures
====================================

Combines experiment execution with figure generation.
Produces both text tables (always) and publication figures (--plot).

Usage:
    python -m scripts.run_experiments                           # tables only
    python -m scripts.run_experiments --plot                    # tables + 7 figures
    python -m scripts.run_experiments --plot --maxiter 200      # publication quality

Author: Victor (LUAS-EPN / KU Leuven)
"""
import sys, os, argparse, time, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from raptor.dem import DEMInterface
from raptor.config import UAVConfig, MissionConstraints
from raptor.builder import FacilityNode, PathBuilder, PathStrategy
from raptor.routed_path import RoutedPath
from raptor.energy import AircraftEnergyParams, analyze_path_energy
from raptor.terrain import TerrainAnalyzer
from raptor.airspace import build_airspace
from raptor.optimizer import PathOptimizer, OptMode
from raptor.scenarios import ALL_FACILITIES
from raptor.astar_baseline import AStarGridPlanner


def find_dem():
    """Look for DEM file in common locations."""
    for p in ['data/dmq_dem.npz', 'dmq_dem.npz', '../data/dmq_dem.npz']:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Cannot find dmq_dem.npz. Place it in data/ or current directory.")


def main():
    parser = argparse.ArgumentParser(description="Run experiments and generate figures")
    parser.add_argument('--maxiter', type=int, default=60, help='DE iterations (default: 60)')
    parser.add_argument('--popsize', type=int, default=12, help='DE population (default: 12)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--plot', action='store_true', help='Generate publication figures')
    parser.add_argument('--no-plot', dest='plot', action='store_false')
    parser.add_argument('--fig-dir', type=str, default='figures', help='Figure output directory')
    parser.add_argument('--plotly-dir', type=str, default='figures_interactive',
                        help='Interactive HTML figure output directory')
    parser.set_defaults(plot=True)
    args = parser.parse_args()

    MI, POP, SEED = args.maxiter, args.popsize, args.seed
    dem_path = find_dem()

    print(f"Settings: maxiter={MI}, popsize={POP}, seed={SEED}, plot={args.plot}")
    print("Loading framework...")

    dem = DEMInterface(dem_path)
    uav = UAVConfig(); constraints = MissionConstraints()
    ac = AircraftEnergyParams()
    ta = TerrainAnalyzer(dem, constraints)
    optimizer = PathOptimizer(dem, uav, constraints, ac)
    airspace = build_airspace(dem=dem)
    builder = PathBuilder(dem, uav, constraints)

    # Reference route
    orig = FacilityNode('H.Espejo', -0.2000, -78.4930, 2788.0)
    dest = FacilityNode('CS.Conocoto', -0.3080, -78.4710, 2510.0)

    # ═══════════════════════════════════════════════════════
    # EXPERIMENT 1: Three-path progressive comparison
    # ═══════════════════════════════════════════════════════
    print("\n" + "█" * 70)
    print("  EXPERIMENT 1: Progressive Constraint Analysis")
    print("█" * 70)

    print("\n  A. Straight line (no optimization)...", end=" ", flush=True)
    fp_s = builder.build(orig, dest, PathStrategy.HIGH_OVERFLY)
    e_s = analyze_path_energy(fp_s, ac)
    ar_s = airspace.check_path(fp_s)
    print(f"E={e_s.total_energy_wh:.0f}Wh, AirV={ar_s.n_violations}")

    print("  B. Optimized direct N=0 (no airspace)...", end=" ", flush=True)
    rp_d = RoutedPath(orig, dest, dem, uav, constraints, n_intermediate=0)
    optimizer.optimize_routed(rp_d, mode=OptMode.ENERGY, payload_kg=1.5,
        airspace=None, maxiter=MI, popsize=POP, verbose=False, seed=SEED)
    e_d = analyze_path_energy(rp_d.flight_path, ac)
    ar_d = airspace.check_path(rp_d.flight_path)
    print(f"E={e_d.total_energy_wh:.0f}Wh, AirV={ar_d.n_violations}")

    print("  C. Optimized routed N=1 (RDAC 101)...", end=" ", flush=True)
    rp_r = RoutedPath(orig, dest, dem, uav, constraints, n_intermediate=1)
    optimizer.optimize_routed(rp_r, mode=OptMode.ENERGY, payload_kg=1.5,
        airspace=airspace, maxiter=MI, popsize=POP, verbose=False, seed=SEED)
    e_r = analyze_path_energy(rp_r.flight_path, ac)
    ar_r = airspace.check_path(rp_r.flight_path)
    print(f"E={e_r.total_energy_wh:.0f}Wh, AirV={ar_r.n_violations}")

    # Results table
    print(f"\n{'Config':<30s} | {'E[Wh]':>7s} | {'t[min]':>7s} | {'AirV':>5s} | {'Feasible':>8s}")
    print("-" * 70)
    for lbl, e, ar in [("A: Straight line", e_s, ar_s),
                        ("B: Optimized direct", e_d, ar_d),
                        ("C: Optimized RDAC", e_r, ar_r)]:
        f = "YES" if ar.feasible else "NO"
        print(f"{lbl:<30s} | {e.total_energy_wh:7.0f} | {e.total_time/60:7.1f} | {ar.n_violations:5d} | {f:>8s}")

    # ═══════════════════════════════════════════════════════
    # EXPERIMENT 2: A* baseline comparison
    # ═══════════════════════════════════════════════════════
    print("\n" + "█" * 70)
    print("  EXPERIMENT 2: A* Grid vs DE Continuous")
    print("█" * 70)

    astar = AStarGridPlanner(dem, ac, airspace=airspace, grid_resolution_m=500)
    routes = [
        ('Espejo→Conocoto', (-0.2000,-78.4930,2788), (-0.3080,-78.4710,2510),
         FacilityNode('E',-0.2000,-78.4930,2788), FacilityNode('C',-0.3080,-78.4710,2510)),
        ('Garcés→Lloa', (-0.2641,-78.5504,2869), (-0.2480,-78.5800,3048),
         FacilityNode('G',-0.2641,-78.5504,2869), FacilityNode('L',-0.2480,-78.5800,3048)),
        ('PASuárez→Nono', (-0.1128,-78.4907,2758), (-0.0580,-78.5770,2673),
         FacilityNode('S',-0.1128,-78.4907,2758), FacilityNode('N',-0.0580,-78.5770,2673)),
    ]

    labels, ea, ed, ta_l, td = [], [], [], [], []
    ra_reference = None   # A* result for the reference route (Espejo→Conocoto)
    print(f"\n{'Route':<20s} | {'E_A*':>7s} | {'E_DE':>7s} | {'t_A*':>6s} | {'t_DE':>6s}")
    print("-" * 55)
    for i, (name, start, end, o, d) in enumerate(routes):
        ra = astar.plan(start, end)
        if i == 0:
            ra_reference = ra  # capture Espejo→Conocoto A* path for 2D/3D plots
        rp = RoutedPath(o, d, dem, uav, constraints, n_intermediate=1)
        optimizer.optimize_routed(rp, mode=OptMode.ENERGY, payload_kg=1.5,
            airspace=airspace, maxiter=MI//2, popsize=POP//2, verbose=False, seed=SEED)
        er = analyze_path_energy(rp.flight_path, ac)
        labels.append(name.split('→')[1])
        ea.append(ra.total_energy_wh); ed.append(er.total_energy_wh)
        ta_l.append(ra.total_time_s/60); td.append(er.total_time/60)
        print(f"{name:<20s} | {ra.total_energy_wh:7.0f} | {er.total_energy_wh:7.0f} | "
              f"{ra.total_time_s/60:6.1f} | {er.total_time/60:6.1f}")

    # ═══════════════════════════════════════════════════════
    # FIGURES (if --plot)
    # ═══════════════════════════════════════════════════════
    if args.plot:
        os.makedirs(args.fig_dir, exist_ok=True)
        from raptor.visualization import (
            plot_airspace_map, plot_stall_envelope, plot_mission_soc,
            plot_constraint_budget, plot_three_path_comparison,
            plot_vehicle_comparison, plot_astar_vs_de,
            plot_path_2d, plot_path_3d, plot_energy_profile,
            plot_path_vs_ceiling, plot_topology_comparison,
        )

        print(f"\nGenerating figures in {args.fig_dir}/...")

        fig = plot_three_path_comparison(
            dem, airspace, orig, dest,
            fp_s, rp_d.flight_path, rp_r.flight_path,
            rp_direct=rp_d, rp_rdac=rp_r,
            e_straight=e_s, e_direct=e_d, e_rdac=e_r,
            title='Espejo → Conocoto: Progressive Constraint Analysis',
            save_path=f'{args.fig_dir}/fig01_three_path_comparison.png')
        plt.close(fig); print("  ✓ fig01_three_path_comparison.png")

        fig = plot_airspace_map(dem, airspace, facilities=ALL_FACILITIES,
            title='Quito DMQ — RDAC 101 Airspace Restrictions',
            lat_range=(-0.40, 0.06), lon_range=(-78.60, -78.35),
            save_path=f'{args.fig_dir}/fig02_airspace_overview.png', figsize=(11, 9))
        plt.close(fig); print("  ✓ fig02_airspace_overview.png")

        fig = plot_stall_envelope(ac, save_path=f'{args.fig_dir}/fig03_stall_envelope.png')
        plt.close(fig); print("  ✓ fig03_stall_envelope.png")

        fig = plot_vehicle_comparison(save_path=f'{args.fig_dir}/fig04_vehicle_comparison.png')
        plt.close(fig); print("  ✓ fig04_vehicle_comparison.png")

        leg_data = [
            {'label':'Garcés→Conocoto','E':88,'t_min':8,'soc_i':1.0,'soc_f':0.853,'dist_km':10.1,'payload':2.0},
            {'label':'Conocoto→Amaguaña','E':105,'t_min':9.1,'soc_i':0.853,'soc_f':0.678,'dist_km':8.3,'payload':0.5},
            {'label':'Amaguaña→Garcés','E':96,'t_min':9.9,'soc_i':0.678,'soc_f':0.519,'dist_km':13.0,'payload':0.0},
        ]
        fig = plot_mission_soc(leg_data, title='Multi-Point Mission: Garcés → Conocoto → Amaguaña → Garcés',
            save_path=f'{args.fig_dir}/fig05_mission_soc.png')
        plt.close(fig); print("  ✓ fig05_mission_soc.png")

        sc_data = {
            'S1a: Garcés→Lloa': [{'label':'A','E':317},{'label':'B','E':68},{'label':'C','E':68},{'label':'D','E':68}],
            'S2a: Espejo→Guang.': [{'label':'A','E':820},{'label':'B','E':126},{'label':'C','E':329},{'label':'D','E':231}],
            'S3b: Garcés→Píntag': [{'label':'A','E':878},{'label':'B','E':146},{'label':'C','E':165},{'label':'D','E':149}],
        }
        fig = plot_constraint_budget(sc_data, save_path=f'{args.fig_dir}/fig06_constraint_budget.png')
        plt.close(fig); print("  ✓ fig06_constraint_budget.png")

        fig = plot_astar_vs_de(labels, ea, ed, ta_l, td,
            title="A* Grid vs DE Continuous (RDAC 101, incl. VTOL phases)",
            save_path=f'{args.fig_dir}/fig07_astar_vs_de.png')
        plt.close(fig); print("  ✓ fig07_astar_vs_de.png")

        all_paths = {
            'Straight': fp_s,
            'Direct (opt)': rp_d.flight_path,
            'RDAC (opt)': rp_r.flight_path,
        }
        if ra_reference and ra_reference.path_found:
            all_paths['A* Grid'] = ra_reference
        facilities = [orig, dest]

        fig = plot_path_2d(dem, all_paths, facilities=facilities,
            title='Espejo → Conocoto: Flight Paths over DEM',
            save_path=f'{args.fig_dir}/fig08_path_2d.png')
        plt.close(fig); print("  ✓ fig08_path_2d.png")

        fig = plot_path_3d(dem, all_paths,
            title='Espejo → Conocoto: 3D Flight Paths over DEM',
            save_path=f'{args.fig_dir}/fig09_path_3d.png')
        plt.close(fig); print("  ✓ fig09_path_3d.png")

        for i, (lbl, er) in enumerate([
                ('straight', e_s), ('direct', e_d), ('rdac', e_r)], 10):
            fig = plot_energy_profile(er,
                title=f'Energy & SOC Profile — {lbl.capitalize()}',
                save_path=f'{args.fig_dir}/fig{i:02d}_energy_{lbl}.png')
            plt.close(fig); print(f"  ✓ fig{i:02d}_energy_{lbl}.png")

        fig = plot_path_vs_ceiling(dem, airspace, all_paths,
            title='Espejo → Conocoto: Flight Profile with Regulatory Ceiling',
            save_path=f'{args.fig_dir}/fig13_path_vs_ceiling.png')
        plt.close(fig); print("  ✓ fig13_path_vs_ceiling.png")

        fig = plot_topology_comparison(
            dem, airspace, rp_d.flight_path, rp_r.flight_path,
            rp_off=rp_d, rp_on=rp_r, facilities=facilities,
            title='Topology Change: Airspace OFF vs ON',
            save_path=f'{args.fig_dir}/fig14_topology_comparison.png')
        plt.close(fig); print("  ✓ fig14_topology_comparison.png")

        # ── Interactive Plotly figures (HTML) ──────────────────────────────
        print(f"\nGenerating interactive HTML figures in {args.plotly_dir}/...")
        os.makedirs(args.plotly_dir, exist_ok=True)
        from raptor.visualization_plotly import (
            plot_airspace_map as iplot_airspace_map,
            plot_stall_envelope as iplot_stall_envelope,
            plot_mission_soc as iplot_mission_soc,
            plot_constraint_budget as iplot_constraint_budget,
            plot_three_path_comparison as iplot_three_path_comparison,
            plot_vehicle_comparison as iplot_vehicle_comparison,
            plot_astar_vs_de as iplot_astar_vs_de,
            plot_path_2d as iplot_path_2d,
            plot_path_3d as iplot_path_3d,
            plot_energy_profile as iplot_energy_profile,
            plot_path_vs_ceiling as iplot_path_vs_ceiling,
            plot_topology_comparison as iplot_topology_comparison,
        )

        iplot_three_path_comparison(
            dem, airspace, orig, dest,
            fp_s, rp_d.flight_path, rp_r.flight_path,
            rp_direct=rp_d, rp_rdac=rp_r,
            e_straight=e_s, e_direct=e_d, e_rdac=e_r,
            title='Espejo → Conocoto: Progressive Constraint Analysis',
            save_path=f'{args.plotly_dir}/fig01_three_path_comparison.html')
        print("  ✓ fig01_three_path_comparison.html")

        iplot_airspace_map(dem, airspace, facilities=ALL_FACILITIES,
            title='Quito DMQ — RDAC 101 Airspace Restrictions',
            lat_range=(-0.40, 0.06), lon_range=(-78.60, -78.35),
            save_path=f'{args.plotly_dir}/fig02_airspace_overview.html', figsize=(11, 9))
        print("  ✓ fig02_airspace_overview.html")

        iplot_stall_envelope(ac, save_path=f'{args.plotly_dir}/fig03_stall_envelope.html')
        print("  ✓ fig03_stall_envelope.html")

        iplot_vehicle_comparison(save_path=f'{args.plotly_dir}/fig04_vehicle_comparison.html')
        print("  ✓ fig04_vehicle_comparison.html")

        iplot_mission_soc(leg_data,
            title='Multi-Point Mission: Garcés → Conocoto → Amaguaña → Garcés',
            save_path=f'{args.plotly_dir}/fig05_mission_soc.html')
        print("  ✓ fig05_mission_soc.html")

        iplot_constraint_budget(sc_data,
            save_path=f'{args.plotly_dir}/fig06_constraint_budget.html')
        print("  ✓ fig06_constraint_budget.html")

        iplot_astar_vs_de(labels, ea, ed, ta_l, td,
            title="A* Grid vs DE Continuous (RDAC 101, incl. VTOL phases)",
            save_path=f'{args.plotly_dir}/fig07_astar_vs_de.html')
        print("  ✓ fig07_astar_vs_de.html")

        iplot_path_2d(dem, all_paths, facilities=facilities,
            title='Espejo → Conocoto: Flight Paths over DEM',
            save_path=f'{args.plotly_dir}/fig08_path_2d.html')
        print("  ✓ fig08_path_2d.html")

        iplot_path_3d(dem, all_paths,
            title='Espejo → Conocoto: 3D Flight Paths over DEM',
            save_path=f'{args.plotly_dir}/fig09_path_3d.html')
        print("  ✓ fig09_path_3d.html")

        for i, (lbl, er) in enumerate([
                ('straight', e_s), ('direct', e_d), ('rdac', e_r)], 10):
            iplot_energy_profile(er,
                title=f'Energy & SOC Profile — {lbl.capitalize()}',
                save_path=f'{args.plotly_dir}/fig{i:02d}_energy_{lbl}.html')
            print(f"  ✓ fig{i:02d}_energy_{lbl}.html")

        iplot_path_vs_ceiling(dem, airspace, all_paths,
            title='Espejo → Conocoto: Flight Profile with Regulatory Ceiling',
            save_path=f'{args.plotly_dir}/fig13_path_vs_ceiling.html')
        print("  ✓ fig13_path_vs_ceiling.html")

        iplot_topology_comparison(
            dem, airspace, rp_d.flight_path, rp_r.flight_path,
            rp_off=rp_d, rp_on=rp_r, facilities=facilities,
            title='Topology Change: Airspace OFF vs ON',
            save_path=f'{args.plotly_dir}/fig14_topology_comparison.html')
        print("  ✓ fig14_topology_comparison.html")

    print("\nDone.")


if __name__ == '__main__':
    main()
