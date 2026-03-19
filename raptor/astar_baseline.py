"""
A* Grid Baseline — Comparison Benchmark
=========================================

Provides a simplified A*-based path planner on a discretized 3D grid
for comparison against the continuous DE-based framework.

The A* planner:
    - Discretizes the horizontal space into a lat/lon grid
    - Uses terrain-aware altitude levels
    - Applies airspace zone penalties as edge costs
    - Returns a waypoint sequence with total energy estimate

This is NOT the primary method — it serves as a baseline to quantify
the advantage of continuous optimization over graph-based planning.

Author: Victor (EPN / LUAS-EPN)
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import numpy as np
import heapq

from .dem import DEMInterface
from .energy import AircraftEnergyParams, power_fw_cruise, power_fw_climb
from .airspace import AirspaceManager
from .atmosphere import isa_density


@dataclass
class AStarResult:
    """Result from A* grid planner."""
    waypoints: List[Tuple[float, float, float]]  # (lat, lon, alt)
    total_energy_wh: float
    total_distance_m: float
    total_time_s: float
    n_nodes_expanded: int
    grid_resolution_m: float
    n_altitude_levels: int
    path_found: bool


class AStarGridPlanner:
    """
    A* path planner on a discretized 3D grid.

    The grid is constructed around the origin-destination line with
    lateral extent. Altitude levels are sampled between terrain floor
    and a maximum ceiling. Edge costs are physics-based energy estimates.

    Parameters
    ----------
    dem : DEMInterface
    ac : AircraftEnergyParams
    airspace : AirspaceManager, optional
    grid_resolution_m : float
        Horizontal grid spacing [m]. Smaller = more accurate but slower.
    n_altitude_levels : int
        Number of vertical layers in the 3D grid.
    lateral_extent_m : float
        How far the grid extends perpendicular to the direct line [m].
    """

    def __init__(
        self,
        dem: DEMInterface,
        ac: AircraftEnergyParams = None,
        airspace: AirspaceManager = None,
        grid_resolution_m: float = 500.0,
        n_altitude_levels: int = 5,
        lateral_extent_m: float = 5000.0,
    ):
        self.dem = dem
        self.ac = ac or AircraftEnergyParams()
        self.airspace = airspace
        self.grid_res = grid_resolution_m
        self.n_alt = n_altitude_levels
        self.lat_extent = lateral_extent_m

    def plan(
        self,
        origin: Tuple[float, float, float],
        destination: Tuple[float, float, float],
        max_altitude: float = None,
    ) -> AStarResult:
        """
        Run A* from origin to destination.

        Parameters
        ----------
        origin, destination : (lat, lon, ground_elev)
        max_altitude : float, optional
            Maximum allowed altitude [m AMSL]. Default: terrain + 120m.
        """
        lat_o, lon_o, elev_o = origin
        lat_d, lon_d, elev_d = destination
        direct_dist = DEMInterface.haversine(lat_o, lon_o, lat_d, lon_d)
        direct_brg = DEMInterface.bearing(lat_o, lon_o, lat_d, lon_d)
        perp_brg = (direct_brg + 90.0) % 360.0

        # ── Build grid ────────────────────────────────────────────────
        # Points along the direct line
        n_along = max(3, int(direct_dist / self.grid_res) + 1)
        along_dists = np.linspace(0, direct_dist, n_along)

        # Lateral offsets
        n_lateral = max(1, int(self.lat_extent / self.grid_res))
        lat_offsets = np.linspace(-self.lat_extent, self.lat_extent, 2 * n_lateral + 1)

        # Build node positions
        nodes = {}  # (i_along, i_lat, i_alt) → (lat, lon, alt)
        node_to_idx = {}
        idx = 0

        for ia, ad in enumerate(along_dists):
            base_lat, base_lon = DEMInterface.destination_point(
                lat_o, lon_o, direct_brg, ad
            )
            for il, lo in enumerate(lat_offsets):
                if abs(lo) > 1.0:
                    nlat, nlon = DEMInterface.destination_point(
                        base_lat, base_lon, perp_brg, lo
                    )
                else:
                    nlat, nlon = base_lat, base_lon

                terrain = self.dem.elevation(nlat, nlon)
                if np.isnan(terrain):
                    continue

                # Altitude levels: from terrain+50 to terrain+120 (or max_altitude)
                ceil = terrain + 120.0 if max_altitude is None else min(terrain + 120.0, max_altitude)
                floor = terrain + 50.0
                if ceil <= floor:
                    alt_levels = [floor]
                else:
                    alt_levels = np.linspace(floor, ceil, self.n_alt)

                for ial, alt in enumerate(alt_levels):
                    key = (ia, il, ial)
                    nodes[key] = (nlat, nlon, float(alt))
                    node_to_idx[key] = idx
                    idx += 1

        n_nodes = len(nodes)
        if n_nodes == 0:
            return AStarResult([], 0, 0, 0, 0, self.grid_res, self.n_alt, False)

        # ── Find start and goal nodes ─────────────────────────────────
        # Start: closest node to origin
        start_key = min(nodes.keys(), key=lambda k: (k[0], abs(k[1] - n_lateral)))
        # Goal: closest node to destination
        goal_key = min(
            [k for k in nodes.keys() if k[0] == n_along - 1],
            key=lambda k: abs(k[1] - n_lateral),
            default=None,
        )
        if goal_key is None:
            return AStarResult([], 0, 0, 0, 0, self.grid_res, self.n_alt, False)

        # ── A* search ─────────────────────────────────────────────────
        def heuristic(key):
            """Euclidean distance to goal node."""
            n = nodes[key]
            g = nodes[goal_key]
            return DEMInterface.haversine(n[0], n[1], g[0], g[1])

        def edge_cost(k1, k2):
            """Physics-based energy cost between two nodes."""
            n1 = nodes[k1]
            n2 = nodes[k2]
            dist = DEMInterface.haversine(n1[0], n1[1], n2[0], n2[1])
            alt_diff = n2[2] - n1[2]

            # Cruise speed
            V = 25.0  # m/s

            # Energy = power × time
            alt_mid = (n1[2] + n2[2]) / 2
            if alt_diff > 5:
                gamma = np.degrees(np.arctan2(alt_diff, dist))
                P = power_fw_climb(self.ac, V, gamma, alt_mid) / self.ac.eta_prop
            else:
                P = power_fw_cruise(self.ac, V, alt_mid) / self.ac.eta_prop

            t = dist / V  # seconds
            E_wh = P * t / 3600.0

            # Airspace penalty
            if self.airspace:
                vs = self.airspace.check_point(n2[0], n2[1], n2[2])
                for v in vs:
                    E_wh += 50.0  # Heavy penalty per violation

            return E_wh

        def neighbors(key):
            """Get adjacent nodes (forward + lateral + vertical transitions)."""
            ia, il, ial = key
            nbrs = []
            # Forward (required: can only go forward along the route)
            for dia in [1]:
                for dil in [-1, 0, 1]:
                    for dial in [-1, 0, 1]:
                        nk = (ia + dia, il + dil, ial + dial)
                        if nk in nodes:
                            nbrs.append(nk)
            return nbrs

        # Priority queue: (f_cost, key)
        open_set = [(0 + heuristic(start_key), start_key)]
        g_cost = {start_key: 0.0}
        came_from = {}
        expanded = 0

        while open_set:
            f, current = heapq.heappop(open_set)
            expanded += 1

            if current == goal_key:
                break

            if expanded > n_nodes * 3:
                break  # Safety limit

            for nbr in neighbors(current):
                new_g = g_cost[current] + edge_cost(current, nbr)
                if nbr not in g_cost or new_g < g_cost[nbr]:
                    g_cost[nbr] = new_g
                    f_new = new_g + heuristic(nbr) * 0.01  # Scale heuristic
                    heapq.heappush(open_set, (f_new, nbr))
                    came_from[nbr] = current

        # ── Reconstruct path ──────────────────────────────────────────
        if goal_key not in came_from and start_key != goal_key:
            return AStarResult([], 0, 0, 0, expanded, self.grid_res, self.n_alt, False)

        path_keys = []
        current = goal_key
        while current != start_key:
            path_keys.append(current)
            current = came_from.get(current)
            if current is None:
                return AStarResult([], 0, 0, 0, expanded, self.grid_res, self.n_alt, False)
        path_keys.append(start_key)
        path_keys.reverse()

        waypoints = [nodes[k] for k in path_keys]

        # Compute total metrics
        total_e = 0.0
        total_d = 0.0
        for i in range(len(path_keys) - 1):
            total_e += edge_cost(path_keys[i], path_keys[i + 1])
            n1 = nodes[path_keys[i]]
            n2 = nodes[path_keys[i + 1]]
            total_d += DEMInterface.haversine(n1[0], n1[1], n2[0], n2[1])

        total_t = total_d / 25.0  # Approximate at cruise speed

        # Add VTOL departure and arrival energy for fair comparison
        # with the segment-based framework (which includes VTOL phases)
        from .energy import power_hover, power_vertical_ascent
        alt_start = waypoints[0][2] if waypoints else origin[2]
        alt_end = waypoints[-1][2] if waypoints else destination[2]
        terr_start = origin[2]
        terr_end = destination[2]
        vtol_climb = max(alt_start - terr_start, 50.0)
        vtol_desc = max(alt_end - terr_end, 50.0)
        p_hover_start = power_hover(self.ac, terr_start)
        p_hover_end = power_hover(self.ac, terr_end)
        vtol_ascend_time = vtol_climb / 2.5  # 2.5 m/s climb rate
        vtol_descend_time = vtol_desc / 2.0  # 2.0 m/s descent rate
        transition_time = 30.0  # 2 × 15s transitions
        vtol_energy = (
            power_vertical_ascent(self.ac, 2.5, terr_start) * vtol_ascend_time / 3600 +
            p_hover_end * vtol_descend_time / 3600 +
            (p_hover_start + p_hover_end) * 0.5 * transition_time / 3600
        )
        total_e += vtol_energy
        total_t += vtol_ascend_time + vtol_descend_time + transition_time

        return AStarResult(
            waypoints=waypoints,
            total_energy_wh=total_e,
            total_distance_m=total_d,
            total_time_s=total_t,
            n_nodes_expanded=expanded,
            grid_resolution_m=self.grid_res,
            n_altitude_levels=self.n_alt,
            path_found=True,
        )


def benchmark_astar_vs_de(
    dem: DEMInterface,
    origin: Tuple[float, float, float],
    destination: Tuple[float, float, float],
    ac: AircraftEnergyParams = None,
    airspace: AirspaceManager = None,
    grid_resolutions: List[float] = None,
) -> Dict:
    """
    Run A* at multiple grid resolutions and compare with DE results.

    Returns a dictionary with comparison metrics for each resolution.
    """
    if grid_resolutions is None:
        grid_resolutions = [1000, 500, 250]
    if ac is None:
        ac = AircraftEnergyParams()

    results = {}
    for res in grid_resolutions:
        import time
        t0 = time.time()
        planner = AStarGridPlanner(dem, ac, airspace, grid_resolution_m=res)
        astar_result = planner.plan(origin, destination)
        dt = time.time() - t0

        results[res] = {
            'energy_wh': astar_result.total_energy_wh,
            'distance_m': astar_result.total_distance_m,
            'time_s': astar_result.total_time_s,
            'nodes_expanded': astar_result.n_nodes_expanded,
            'wall_time_s': dt,
            'path_found': astar_result.path_found,
            'n_waypoints': len(astar_result.waypoints),
        }

    return results
