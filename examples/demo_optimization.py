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

def find_dem():
    for p in ['data/dmq_dem.npz', 'dmq_dem.npz', '../data/dmq_dem.npz']:
        if os.path.exists(p): return p
    raise FileNotFoundError("dmq_dem.npz not found")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--maxiter', type=int, default=60)
    parser.add_argument('--popsize', type=int, default=12)
    parser.add_argument('--plot', action='store_true', default=True)
    parser.add_argument('--no-plot', dest='plot', action='store_false')
    args = parser.parse_args()

    print("=== Demo: Optimization with Airspace ===\n")
    dem = DEMInterface(find_dem())
    uav = UAVConfig(); constraints = MissionConstraints()
    ac = AircraftEnergyParams()
    optimizer = PathOptimizer(dem, uav, constraints, ac)
    airspace = build_airspace(dem=dem)
    builder = PathBuilder(dem, uav, constraints)

    orig = FacilityNode('H.Espejo', -0.2000, -78.4930, 2788.0)
    dest = FacilityNode('CS.Conocoto', -0.3080, -78.4710, 2510.0)

    # Baseline
    fp_base = builder.build(orig, dest, PathStrategy.HIGH_OVERFLY)
    e_base = analyze_path_energy(fp_base, ac)
    ar_base = airspace.check_path(fp_base)

    # Optimized with RDAC 101
    rp = RoutedPath(orig, dest, dem, uav, constraints, n_intermediate=1)
    r = optimizer.optimize_routed(rp, mode=OptMode.ENERGY, payload_kg=1.5,
        airspace=airspace, maxiter=args.maxiter, popsize=args.popsize,
        verbose=True, seed=42)
    e_opt = analyze_path_energy(rp.flight_path, ac)
    ar_opt = airspace.check_path(rp.flight_path)
    topo = rp.topology_summary()

    print(f"\n{'Metric':<25s} | {'Baseline':>10s} | {'Optimized':>10s}")
    print("-" * 50)
    print(f"{'Energy [Wh]':<25s} | {e_base.total_energy_wh:10.0f} | {e_opt.total_energy_wh:10.0f}")
    print(f"{'Time [min]':<25s} | {e_base.total_time/60:10.1f} | {e_opt.total_time/60:10.1f}")
    print(f"{'Airspace violations':<25s} | {ar_base.n_violations:10d} | {ar_opt.n_violations:10d}")
    print(f"{'Lateral deviation [m]':<25s} | {'0':>10s} | {topo['max_lateral_deviation_m']:10.0f}")

    if args.plot:
        import matplotlib; matplotlib.use('Agg')
        from raptor.viz_airspace import plot_three_path_comparison
        fig = plot_three_path_comparison(
            dem, airspace, orig, dest,
            fp_base, fp_base, rp.flight_path,  # straight = direct for 2-col
            rp_rdac=rp, e_straight=e_base, e_direct=e_base, e_rdac=e_opt,
            title='Demo: Baseline vs RDAC-Optimized',
            save_path='demo_optimization.png')
        print("\nSaved demo_optimization.png")

if __name__ == '__main__':
    main()
