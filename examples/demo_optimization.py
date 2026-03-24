#!/usr/bin/env python3
"""
Demo: Path Optimization with Airspace Constraints
====================================================

Demonstrates: RoutedPath, DE optimizer, airspace zones, topology changes.

Usage:
    python -m examples.demo_optimization
    python -m examples.demo_optimization --plot --maxiter 100
"""
import sys, os, argparse, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from raptor.dem import DEMInterface
from raptor.config import UAVConfig, MissionConstraints
from raptor.builder import FacilityNode, PathBuilder, PathStrategy
from raptor.routed_path import RoutedPath
from raptor.energy import AircraftEnergyParams, analyze_path_energy
from raptor.terrain import TerrainAnalyzer
from raptor.airspace import build_airspace
from raptor.optimizer import PathOptimizer, OptMode
from raptor.astar_baseline import AStarGridPlanner

def find_dem():
    for p in ['data/dmq_dem.npz', 'dmq_dem.npz', '../data/dmq_dem.npz']:
        if os.path.exists(p): return p
    raise FileNotFoundError("dmq_dem.npz not found")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--maxiter', type=int, default=20)
    parser.add_argument('--popsize', type=int, default=20)
    parser.add_argument('--plot', action='store_true', default=True)
    parser.add_argument('--no-plot', dest='plot', action='store_false')
    parser.add_argument('--fig-dir', type=str, default='figures')
    args = parser.parse_args()

    # Output directories
    args._static_dir = os.path.join(args.fig_dir, 'demo_optimization', 'static')
    args._interactive_dir = os.path.join(args.fig_dir, 'demo_optimization', 'interactive')

    print("=== Demo: Optimization with Airspace ===\n")
    dem = DEMInterface(find_dem())
    uav = UAVConfig(); constraints = MissionConstraints()
    ac = AircraftEnergyParams()
    optimizer = PathOptimizer(dem, uav, constraints, ac)
    airspace = build_airspace(dem=dem)
    builder = PathBuilder(dem, uav, constraints)

    orig = FacilityNode('P.A.Suarez', -0.1128,-78.4907,2758)
    dest = FacilityNode('Lloa', -0.2480,-78.5800,3048)

    # A* baseline (for path comparison plots)
    print("  A*. Running A* grid planner...", end=" ", flush=True)
    astar = AStarGridPlanner(dem, ac, airspace=airspace, grid_resolution_m=500)
    ra = astar.plan(
        (orig.lat, orig.lon, orig.ground_elev),
        (dest.lat, dest.lon, dest.ground_elev),
    )
    print(f"E={ra.total_energy_wh:.0f}Wh, found={ra.path_found}")

    # A. Baseline (straight, no optimization)
    fp_base = builder.build(orig, dest, PathStrategy.HIGH_OVERFLY)
    e_base = analyze_path_energy(fp_base, ac)
    ar_base = airspace.check_path(fp_base)

    # B. Optimized without airspace (N=0)
    print("  B. Optimizing direct (no airspace)...", end=" ", flush=True)
    rp_direct = RoutedPath(orig, dest, dem, uav, constraints, n_intermediate=3)
    optimizer.optimize_routed(rp_direct, mode=OptMode.MULTI, payload_kg=1.5,
        airspace=None, maxiter=args.maxiter, popsize=args.popsize,
        verbose=False, seed=42)
    e_direct = analyze_path_energy(rp_direct.flight_path, ac)
    ar_direct = airspace.check_path(rp_direct.flight_path)
    print(f"E={e_direct.total_energy_wh:.0f}Wh, AirV={ar_direct.n_violations}")

    # C. Optimized with RDAC 101 (N=1)
    print("  C. Optimizing with RDAC 101...", end=" ", flush=True)
    rp = RoutedPath(orig, dest, dem, uav, constraints, n_intermediate=3)
    r = optimizer.optimize_routed(rp, mode=OptMode.MULTI, payload_kg=1.5,
        airspace=airspace, maxiter=args.maxiter, popsize=args.popsize,
        verbose=False, seed=42)
    e_opt = analyze_path_energy(rp.flight_path, ac)
    ar_opt = airspace.check_path(rp.flight_path)
    topo = rp.topology_summary()
    print(f"E={e_opt.total_energy_wh:.0f}Wh, AirV={ar_opt.n_violations}")

    print(f"\n{'Metric':<25s} | {'Baseline':>10s} | {'Direct opt':>10s} | {'RDAC opt':>10s}")
    print("-" * 62)
    print(f"{'Energy [Wh]':<25s} | {e_base.total_energy_wh:10.0f} | {e_direct.total_energy_wh:10.0f} | {e_opt.total_energy_wh:10.0f}")
    print(f"{'Time [min]':<25s} | {e_base.total_time/60:10.1f} | {e_direct.total_time/60:10.1f} | {e_opt.total_time/60:10.1f}")
    print(f"{'Airspace violations':<25s} | {ar_base.n_violations:10d} | {ar_direct.n_violations:10d} | {ar_opt.n_violations:10d}")
    print(f"{'Lateral deviation [m]':<25s} | {'0':>10s} | {'0':>10s} | {topo['max_lateral_deviation_m']:10.0f}")

    if args.plot:
        import matplotlib; matplotlib.use('Agg')
        from raptor.visualization import plot_all
        from raptor.visualization_plotly import plot_all as iplot_all

        paths = {
            'Baseline': fp_base,
            'Direct (opt)': rp_direct.flight_path,
            'RDAC (opt)': rp.flight_path,
        }
        if ra.path_found:
            paths['A* Grid'] = ra

        common_kwargs = dict(
            dem=dem, paths=paths, facilities=[orig, dest],
            energy_results=[e_base, e_direct, e_opt],
            opt_results=[r] if r else None,
            ac=ac, airspace=airspace,
            orig=orig, dest=dest,
            fp_straight=fp_base,
            fp_direct=rp_direct.flight_path,
            fp_rdac=rp.flight_path,
            rp_direct=rp_direct, rp_rdac=rp,
            e_straight=e_base, e_direct=e_direct, e_rdac=e_opt,
            title_prefix="Demo: ",
        )

        # Matplotlib figures (PNG)
        figs = plot_all(**common_kwargs, save_dir=args._static_dir)
        import matplotlib.pyplot as plt
        for fig in figs.values():
            plt.close(fig)
        print(f"\nSaved {len(figs)} PNG figures to {args._static_dir}/")

        # Plotly figures (HTML)
        ifigs = iplot_all(**common_kwargs, save_dir=args._interactive_dir)
        print(f"Saved {len(ifigs)} HTML figures to {args._interactive_dir}/")

if __name__ == '__main__':
    main()
