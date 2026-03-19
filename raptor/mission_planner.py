"""
Mission Planner — Multi-Point Mission Orchestration
=====================================================

Orchestrates multi-leg missions across the scenario catalog:
    - Sequential leg optimization with SOC coupling
    - Feasibility checking (battery, time, terrain, airspace)
    - Mission-level metrics aggregation
    - Visit sequence reordering (nearest-neighbor TSP heuristic)

Author: Victor (EPN / LUAS-EPN)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
import time as time_module

from .dem import DEMInterface
from .config import UAVConfig, MissionConstraints
from .builder import FacilityNode
from .routed_path import RoutedPath
from .energy import AircraftEnergyParams, analyze_path_energy, MissionEnergyResult
from .terrain import TerrainAnalyzer, TerrainReport
from .airspace import AirspaceManager, AirspaceReport
from .optimizer import PathOptimizer, OptMode, OptimizationResult
from .scenarios import FlightScenario, MissionLeg, OptPriority


# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LegResult:
    """Result of optimizing a single mission leg."""
    leg_index: int
    leg: MissionLeg
    routed_path: RoutedPath
    opt_result: OptimizationResult
    energy_result: MissionEnergyResult
    terrain_report: TerrainReport
    airspace_report: Optional[AirspaceReport]

    # SOC coupling
    soc_start: float          # SOC at leg departure [0-1]
    soc_end: float            # SOC at leg arrival [0-1]

    # Topology
    n_intermediate: int       # Waypoints used
    lateral_deviation_m: float
    cruise_altitudes: List[float]

    @property
    def energy_wh(self) -> float:
        return self.energy_result.total_energy_wh

    @property
    def time_min(self) -> float:
        return self.energy_result.total_time / 60.0

    @property
    def distance_km(self) -> float:
        wp = self.routed_path.flight_path.get_waypoints_array()
        return wp[-1, 4] / 1000.0

    @property
    def feasible(self) -> bool:
        return self.soc_end > 0 and self.terrain_report.is_feasible


@dataclass
class MissionResult:
    """Aggregated result of a complete multi-leg mission."""
    scenario: FlightScenario
    legs: List[LegResult]
    wall_time_s: float

    # Mission-level aggregates
    total_energy_wh: float = 0.0
    total_time_min: float = 0.0
    total_distance_km: float = 0.0
    battery_capacity_wh: float = 600.0
    soc_final: float = 1.0
    soc_min: float = 1.0
    mission_feasible: bool = True

    # Constraint summary
    total_airspace_violations: int = 0
    total_terrain_violations: int = 0

    def __post_init__(self):
        if self.legs:
            self.total_energy_wh = sum(l.energy_wh for l in self.legs)
            self.total_time_min = sum(l.time_min for l in self.legs)
            self.total_distance_km = sum(l.distance_km for l in self.legs)
            self.soc_final = self.legs[-1].soc_end
            self.soc_min = min(l.soc_end for l in self.legs)
            self.mission_feasible = all(l.feasible for l in self.legs) and self.soc_min > 0
            self.total_airspace_violations = sum(
                l.airspace_report.n_violations if l.airspace_report else 0
                for l in self.legs
            )
            self.total_terrain_violations = sum(
                l.terrain_report.n_violations for l in self.legs
            )

    @property
    def battery_utilization(self) -> float:
        """Fraction of battery used."""
        return 1.0 - self.soc_final

    def summary_table(self) -> str:
        """Format a publication-ready summary table."""
        lines = []
        lines.append(f"Mission: {self.scenario.name} ({self.scenario.scenario_id})")
        lines.append(f"{'Leg':30s} | {'d[km]':>6s} | {'P[kg]':>5s} | {'E[Wh]':>7s} | "
                      f"{'t[min]':>6s} | {'SOC_i':>5s} | {'SOC_f':>5s} | {'LatDev':>7s}")
        lines.append("-" * 90)
        for lr in self.legs:
            label = f"{lr.leg.origin.short_name}→{lr.leg.destination.short_name}"
            lines.append(
                f"{label:30s} | {lr.distance_km:6.1f} | {lr.leg.payload_kg:5.1f} | "
                f"{lr.energy_wh:7.1f} | {lr.time_min:6.1f} | "
                f"{lr.soc_start*100:5.1f} | {lr.soc_end*100:5.1f} | "
                f"{lr.lateral_deviation_m:7.0f}"
            )
        lines.append("-" * 90)
        lines.append(
            f"{'TOTAL':30s} | {self.total_distance_km:6.1f} | {'':>5s} | "
            f"{self.total_energy_wh:7.1f} | {self.total_time_min:6.1f} | "
            f"{'100.0':>5s} | {self.soc_final*100:5.1f} |"
        )
        lines.append(f"\nFeasible: {'YES' if self.mission_feasible else 'NO'} | "
                      f"Battery util: {self.battery_utilization*100:.1f}% | "
                      f"SOC_min: {self.soc_min*100:.1f}%")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# MISSION PLANNER
# ═══════════════════════════════════════════════════════════════════════════

class MissionPlanner:
    """
    Orchestrates multi-leg mission planning and optimization.

    Handles SOC coupling between legs: each leg starts with the SOC
    that the previous leg ended at. The optimizer's energy penalty
    scales with remaining SOC to avoid depleting the battery.

    Parameters
    ----------
    dem : DEMInterface
    uav : UAVConfig
    constraints : MissionConstraints
    ac : AircraftEnergyParams
    airspace : AirspaceManager, optional
    battery_wh : float
        Total battery capacity [Wh].
    min_soc : float
        Minimum allowed SOC fraction [0-1].
    """

    def __init__(
        self,
        dem: DEMInterface,
        uav: UAVConfig,
        constraints: MissionConstraints,
        ac: AircraftEnergyParams,
        airspace: AirspaceManager = None,
        battery_wh: float = 600.0,
        min_soc: float = 0.15,
    ):
        self.dem = dem
        self.uav = uav
        self.constraints = constraints
        self.ac = ac
        self.airspace = airspace
        self.battery_wh = battery_wh
        self.min_soc = min_soc
        self.optimizer = PathOptimizer(dem, uav, constraints, ac)
        self.terrain_analyzer = TerrainAnalyzer(dem, constraints)

    def plan_mission(
        self,
        scenario: FlightScenario,
        n_intermediate: int = 1,
        maxiter: int = 60,
        popsize: int = 12,
        use_airspace: bool = True,
        seed: int = None,
        verbose: bool = False,
    ) -> MissionResult:
        """
        Plan and optimize a complete multi-leg mission.

        Each leg is optimized sequentially. The SOC at the start of
        leg k is the SOC at the end of leg k-1. This captures the
        cumulative energy coupling that makes multi-point missions
        harder than the sum of independent legs.
        """
        t_start = time_module.time()

        # Map priority to optimizer mode
        mode_map = {
            OptPriority.ENERGY: OptMode.ENERGY,
            OptPriority.TIME: OptMode.TIME,
            OptPriority.BALANCED: OptMode.MULTI,
        }
        mode = mode_map.get(scenario.priority, OptMode.ENERGY)
        air = self.airspace if use_airspace else None

        current_soc = 1.0
        leg_results = []

        for leg_idx, leg in enumerate(scenario.legs):
            if verbose:
                print(f"  Leg {leg_idx+1}/{scenario.n_legs}: "
                      f"{leg.origin.short_name}→{leg.destination.short_name} "
                      f"(SOC={current_soc*100:.1f}%)...", end=" ", flush=True)

            # Build origin/destination nodes
            orig_node = FacilityNode(
                leg.origin.short_name,
                leg.origin.lat, leg.origin.lon,
                leg.origin.ground_elev,
            )
            dest_node = FacilityNode(
                leg.destination.short_name,
                leg.destination.lat, leg.destination.lon,
                leg.destination.ground_elev,
            )

            # Create and optimize routed path
            rp = RoutedPath(orig_node, dest_node, self.dem, self.uav,
                           self.constraints, n_intermediate=n_intermediate)

            opt_result = self.optimizer.optimize_routed(
                rp, mode=mode,
                payload_kg=leg.payload_kg,
                airspace=air,
                maxiter=maxiter,
                popsize=popsize,
                verbose=False,
                seed=seed + leg_idx if seed else None,
            )

            # Evaluate the optimized path
            fp = rp.flight_path
            energy_result = analyze_path_energy(fp, self.ac, SOC_min=self.min_soc)
            terrain_report = self.terrain_analyzer.analyze(fp)
            airspace_report = air.check_path(fp) if air else None

            # SOC coupling
            soc_start = current_soc
            energy_used_frac = energy_result.total_energy_wh / self.battery_wh
            soc_end = max(soc_start - energy_used_frac, 0.0)

            # Topology info
            topo = rp.topology_summary()

            leg_result = LegResult(
                leg_index=leg_idx,
                leg=leg,
                routed_path=rp,
                opt_result=opt_result,
                energy_result=energy_result,
                terrain_report=terrain_report,
                airspace_report=airspace_report,
                soc_start=soc_start,
                soc_end=soc_end,
                n_intermediate=rp.n_intermediate,
                lateral_deviation_m=topo['max_lateral_deviation_m'],
                cruise_altitudes=list(rp.cruise_altitudes),
            )
            leg_results.append(leg_result)

            if verbose:
                print(f"E={energy_result.total_energy_wh:.0f}Wh, "
                      f"t={energy_result.total_time/60:.1f}min, "
                      f"SOC→{soc_end*100:.1f}%")

            current_soc = soc_end

        wall_time = time_module.time() - t_start

        return MissionResult(
            scenario=scenario,
            legs=leg_results,
            wall_time_s=wall_time,
            battery_capacity_wh=self.battery_wh,
        )

    def check_feasibility(
        self,
        scenario: FlightScenario,
        use_airspace: bool = True,
    ) -> Dict:
        """
        Quick feasibility check without full optimization.

        Uses default parameters (no lateral routing) to estimate
        whether the mission is achievable within battery constraints.
        """
        current_soc = 1.0
        results = []

        for leg in scenario.legs:
            orig = FacilityNode(leg.origin.short_name,
                               leg.origin.lat, leg.origin.lon,
                               leg.origin.ground_elev)
            dest = FacilityNode(leg.destination.short_name,
                               leg.destination.lat, leg.destination.lon,
                               leg.destination.ground_elev)

            rp = RoutedPath(orig, dest, self.dem, self.uav,
                           self.constraints, n_intermediate=0)
            fp = rp.flight_path
            er = analyze_path_energy(fp, self.ac)
            soc_used = er.total_energy_wh / self.battery_wh
            soc_end = max(current_soc - soc_used, 0.0)

            results.append({
                'leg': f"{leg.origin.short_name}→{leg.destination.short_name}",
                'dist_km': DEMInterface.haversine(
                    leg.origin.lat, leg.origin.lon,
                    leg.destination.lat, leg.destination.lon
                ) / 1000,
                'energy_wh': er.total_energy_wh,
                'time_min': er.total_time / 60,
                'soc_start': current_soc,
                'soc_end': soc_end,
                'feasible': soc_end > self.min_soc,
            })
            current_soc = soc_end

        return {
            'scenario_id': scenario.scenario_id,
            'n_legs': scenario.n_legs,
            'legs': results,
            'soc_final': current_soc,
            'mission_feasible': all(r['feasible'] for r in results),
        }

    @staticmethod
    def reorder_nearest_neighbor(
        legs: List[MissionLeg],
        start_facility=None,
    ) -> List[MissionLeg]:
        """
        Reorder legs using nearest-neighbor TSP heuristic.

        Useful for multi-stop tours where the visit order is flexible.
        Returns a new list of MissionLeg with optimized ordering.
        """
        if len(legs) <= 2:
            return legs  # No reordering needed

        # Extract unique waypoints (excluding return to start)
        facilities = []
        for leg in legs:
            if leg.origin not in facilities:
                facilities.append(leg.origin)
            if leg.destination not in facilities:
                facilities.append(leg.destination)

        if start_facility is None:
            start_facility = legs[0].origin

        # Nearest-neighbor ordering
        visited = [start_facility]
        remaining = [f for f in facilities if f != start_facility]

        while remaining:
            current = visited[-1]
            dists = [
                DEMInterface.haversine(current.lat, current.lon, f.lat, f.lon)
                for f in remaining
            ]
            nearest_idx = np.argmin(dists)
            visited.append(remaining.pop(nearest_idx))

        # Build new legs from the ordering
        # Return to start
        visited.append(start_facility)

        new_legs = []
        for i in range(len(visited) - 1):
            # Find original leg payload or use default
            orig_leg = None
            for leg in legs:
                if (leg.origin == visited[i] and
                    leg.destination == visited[i + 1]):
                    orig_leg = leg
                    break
            if orig_leg:
                new_legs.append(orig_leg)
            else:
                new_legs.append(MissionLeg(
                    origin=visited[i],
                    destination=visited[i + 1],
                    payload_kg=0.0,
                    description="Reordered leg",
                ))

        return new_legs
