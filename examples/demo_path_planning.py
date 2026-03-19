#!/usr/bin/env python3
"""
Demo: Basic Path Planning Pipeline
=====================================

Demonstrates: DEM loading → path construction → terrain analysis → plotting.

Usage:
    python -m examples.demo_path_planning
    python -m examples.demo_path_planning --plot
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from raptor.dem import DEMInterface
from raptor.config import UAVConfig, MissionConstraints
from raptor.builder import PathBuilder, FacilityNode, PathStrategy
from raptor.terrain import TerrainAnalyzer

def find_dem():
    for p in ['data/dmq_dem.npz', 'dmq_dem.npz', '../data/dmq_dem.npz']:
        if os.path.exists(p): return p
    raise FileNotFoundError("dmq_dem.npz not found")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true', default=True)
    parser.add_argument('--no-plot', dest='plot', action='store_false')
    args = parser.parse_args()

    print("=== Demo: Path Planning Pipeline ===\n")
    dem = DEMInterface(find_dem())
    uav = UAVConfig()
    constraints = MissionConstraints()
    builder = PathBuilder(dem, uav, constraints)
    ta = TerrainAnalyzer(dem, constraints)

    orig = FacilityNode('H.Espejo', -0.2000, -78.4930, 2788.0)
    dest = FacilityNode('CS.Conocoto', -0.3080, -78.4710, 2510.0)

    print(f"Origin: {orig.name} ({orig.lat:.4f}, {orig.lon:.4f}) alt={orig.ground_elev:.0f}m")
    print(f"Dest:   {dest.name} ({dest.lat:.4f}, {dest.lon:.4f}) alt={dest.ground_elev:.0f}m")
    print(f"Direct: {DEMInterface.haversine(orig.lat, orig.lon, dest.lat, dest.lon)/1000:.1f} km\n")

    paths = {}
    for strategy in [PathStrategy.HIGH_OVERFLY, PathStrategy.TERRAIN_FOLLOW]:
        fp = builder.build(orig, dest, strategy)
        tr = ta.analyze(fp)
        wp = fp.get_waypoints_array()
        paths[strategy.value] = fp
        print(f"  {strategy.value:18s}: {fp.n_segments} segs, "
              f"max_alt={wp[:,2].max():.0f}m, min_AGL={tr.min_agl:.0f}m, "
              f"terrain_ok={tr.is_feasible}")

    if args.plot:
        import matplotlib; matplotlib.use('Agg')
        from raptor.visualization import plot_path_2d
        fig = plot_path_2d(dem, paths, facilities=[orig, dest],
            title="Demo: Path Strategies", save_path='demo_paths.png')
        print("\nSaved demo_paths.png")

if __name__ == '__main__':
    main()
