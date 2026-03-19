"""
Path Optimization Engine — Energy / Time / Multi-Objective
============================================================

Optimizes eVTOL flight path parameters over real terrain using
scipy.optimize.differential_evolution (global stochastic optimizer).

Three optimization modes:
    1. ENERGY:  min E_total   s.t.  t ≤ t_max, terrain clearance
    2. TIME:    min t_total   s.t.  SOC ≥ SOC_min, terrain clearance
    3. MULTI:   min w_E·Ê + w_t·t̂ s.t.  both constraints

Design variables:
    Segment parameter vectors (airspeed, climb angle, altitude gains, etc.)

Constraints (handled via penalty functions):
    - DEM terrain clearance at all waypoints
    - Battery SOC ≥ minimum reserve
    - Flight time ≤ maximum allowed
    - UAV envelope limits (bounds on design variables)

Why differential_evolution?
    - Global optimizer: finds good solutions in non-convex terrain landscape
    - Derivative-free: objective involves DEM lookups (non-differentiable)
    - Handles bounds natively on all design variables
    - Penalty-based constraint handling integrates naturally
    - Robust convergence with moderate computational cost

Author: Victor (EPN / LUAS-EPN)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Callable
from enum import Enum
import numpy as np
import time as time_module
from scipy.optimize import differential_evolution, OptimizeResult
import copy
import warnings

from .config import UAVConfig, MissionConstraints
from .dem import DEMInterface
from .segments import (
    SegmentType, FlightSegment,
    VTOLAscend, VTOLDescend, FWClimb, FWDescend, FWCruise, Transition,
    SegmentState
)
from .path import FlightPath
from .terrain import TerrainAnalyzer
from .energy import (
    AircraftEnergyParams, analyze_path_energy, MissionEnergyResult,
)
from .builder import PathBuilder, FacilityNode, PathStrategy


# ═══════════════════════════════════════════════════════════════════════════
# OPTIMIZATION MODE
# ═══════════════════════════════════════════════════════════════════════════

class OptMode(Enum):
    """Optimization objective type."""
    ENERGY = "energy"       # Minimize battery consumption
    TIME = "time"           # Minimize flight time
    MULTI = "multi"         # Weighted sum of energy + time


# ═══════════════════════════════════════════════════════════════════════════
# OPTIMIZATION RESULT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class OptimizationResult:
    """
    Complete result of a path optimization run.

    Contains the optimized path, energy analysis, constraint evaluation,
    convergence history, and comparison with the initial (un-optimized) path.
    """
    # Final solution
    mode: str
    optimized_path: FlightPath
    optimized_energy: MissionEnergyResult
    parameter_vector: np.ndarray

    # Objective values
    objective_value: float
    energy_wh: float
    flight_time_s: float

    # Constraints
    terrain_feasible: bool
    min_agl: float
    soc_final: float
    battery_feasible: bool
    time_feasible: bool
    fully_feasible: bool

    # Convergence
    n_evaluations: int
    n_iterations: int
    wall_time_s: float
    convergence_history: List[float]  # best objective per generation
    parameter_history: List[np.ndarray]  # best parameters per generation (sampled)

    # Comparison with initial
    initial_energy_wh: float
    initial_time_s: float
    energy_improvement_pct: float
    time_improvement_pct: float

    # Optimizer info
    success: bool
    message: str
    weights: Tuple[float, float] = (1.0, 0.0)

    def summary(self) -> str:
        """Human-readable result summary."""
        lines = [
            "=" * 70,
            f"OPTIMIZATION RESULT — Mode: {self.mode.upper()}",
            "=" * 70,
            f"Weights: w_energy={self.weights[0]:.2f}, w_time={self.weights[1]:.2f}",
            f"",
            f"{'Metric':<30s} {'Initial':>12s} {'Optimized':>12s} {'Change':>10s}",
            f"{'-'*64}",
            f"{'Energy [Wh]':<30s} {self.initial_energy_wh:>12.1f} "
            f"{self.energy_wh:>12.1f} {self.energy_improvement_pct:>+9.1f}%",
            f"{'Flight time [s]':<30s} {self.initial_time_s:>12.1f} "
            f"{self.flight_time_s:>12.1f} {self.time_improvement_pct:>+9.1f}%",
            f"{'Flight time [min]':<30s} {self.initial_time_s/60:>12.1f} "
            f"{self.flight_time_s/60:>12.1f} {'':>10s}",
            f"{'SOC final [%]':<30s} {'—':>12s} {self.soc_final*100:>11.1f}%",
            f"{'Min AGL [m]':<30s} {'—':>12s} {self.min_agl:>12.1f}",
            f"",
            f"Terrain feasible:  {'✓' if self.terrain_feasible else '✗'}",
            f"Battery feasible:  {'✓' if self.battery_feasible else '✗'}",
            f"Time feasible:     {'✓' if self.time_feasible else '✗'}",
            f"Fully feasible:    {'✓' if self.fully_feasible else '✗'}",
            f"",
            f"Evaluations: {self.n_evaluations}",
            f"Generations: {self.n_iterations}",
            f"Wall time:   {self.wall_time_s:.1f} s",
            f"Converged:   {'Yes' if self.success else 'No'} — {self.message}",
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# PATH OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════

class PathOptimizer:
    """
    Optimizes flight path parameters using differential evolution.

    The optimizer takes an initial path (from PathBuilder), extracts
    its design variable vector, and searches for better parameters
    that reduce energy and/or time while satisfying terrain and
    battery constraints.

    Parameters
    ----------
    dem : DEMInterface
        Terrain model for clearance evaluation.
    uav : UAVConfig
        Aircraft performance limits (bound design variables).
    constraints : MissionConstraints
        Mission-level constraints (clearance, time, range).
    ac_params : AircraftEnergyParams
        Aircraft energy model parameters.
    """

    def __init__(
        self,
        dem: DEMInterface,
        uav: UAVConfig,
        constraints: MissionConstraints,
        ac_params: AircraftEnergyParams,
    ):
        self.dem = dem
        self.uav = uav
        self.constraints = constraints
        self.ac_params = ac_params
        self.terrain_analyzer = TerrainAnalyzer(dem, constraints)

        # Penalty weights (tuned for problem scale)
        # These must be large enough to dominate the normalized [0,1] objective
        self.penalty_terrain = 500.0     # terrain violation (must dominate)
        self.penalty_soc = 100.0         # SOC below minimum
        self.penalty_time = 20.0         # time exceeding limit
        self.penalty_endpoint = 10.0     # endpoint altitude mismatch

        # Tracking
        self._eval_count = 0
        self._best_obj = np.inf
        self._convergence_history = []
        self._parameter_history = []
        self._generation = 0
        self._callback_interval = 1  # record every N generations

    # ── Main optimization interface ───────────────────────────────────────

    def optimize(
        self,
        initial_path: FlightPath,
        mode: OptMode = OptMode.ENERGY,
        payload_kg: float = 0.0,
        max_flight_time: float = None,
        min_soc: float = 0.15,
        w_energy: float = None,
        w_time: float = None,
        maxiter: int = 80,
        popsize: int = 20,
        tol: float = 1e-4,
        seed: int = 42,
        verbose: bool = True,
    ) -> OptimizationResult:
        """
        Run path optimization.

        Parameters
        ----------
        initial_path : FlightPath
            Starting path (from PathBuilder). Provides segment structure
            and initial parameter values.
        mode : OptMode
            Optimization objective type.
        payload_kg : float
            Payload mass to add to aircraft m_tow for energy calculations.
        max_flight_time : float
            Maximum allowed flight time [s]. Default from constraints.
        min_soc : float
            Minimum battery SOC fraction (0–1).
        w_energy, w_time : float
            Weights for multi-objective mode. Auto-set if None.
        maxiter : int
            Max DE generations.
        popsize : int
            DE population size multiplier.
        tol : float
            Convergence tolerance.
        seed : int
            Random seed for reproducibility.
        verbose : bool
            Print progress updates.

        Returns
        -------
        OptimizationResult
        """
        # ── Setup ─────────────────────────────────────────────────────────
        if max_flight_time is None:
            max_flight_time = self.constraints.max_flight_time

        # Set weights based on mode
        if mode == OptMode.ENERGY:
            w_energy = w_energy if w_energy is not None else 1.0
            w_time = w_time if w_time is not None else 0.0
        elif mode == OptMode.TIME:
            w_energy = w_energy if w_energy is not None else 0.0
            w_time = w_time if w_time is not None else 1.0
        else:  # MULTI
            w_energy = w_energy if w_energy is not None else 0.5
            w_time = w_time if w_time is not None else 0.5

        # Create aircraft params with payload
        ac = copy.deepcopy(self.ac_params)
        ac.m_tow = self.ac_params.m_tow + payload_kg

        # ── Get initial solution metrics ──────────────────────────────────
        initial_energy = analyze_path_energy(initial_path, ac, SOC_min=min_soc)
        initial_energy_wh = initial_energy.total_energy_wh
        initial_time_s = initial_energy.total_time

        # ── Normalization scales (for multi-objective) ────────────────────
        # Scale objectives to [0, 1] range approximately
        E_scale = max(initial_energy_wh, 50.0)  # Wh
        T_scale = max(initial_time_s, 60.0)       # s

        # ── Prepare bounds ────────────────────────────────────────────────
        bounds = initial_path.parameter_bounds
        theta_0 = initial_path.parameter_vector.copy()
        n_params = len(theta_0)

        # Clip initial vector to bounds (safety)
        for i, (lo, hi) in enumerate(bounds):
            theta_0[i] = np.clip(theta_0[i], lo, hi)

        if verbose:
            print(f"\n{'='*60}")
            print(f"PATH OPTIMIZATION — Mode: {mode.value.upper()}")
            print(f"{'='*60}")
            print(f"Design variables:  {n_params}")
            print(f"Weights:           w_E={w_energy:.2f}, w_t={w_time:.2f}")
            print(f"Payload:           {payload_kg:.1f} kg (m_tow={ac.m_tow:.1f} kg)")
            print(f"Initial energy:    {initial_energy_wh:.1f} Wh")
            print(f"Initial time:      {initial_time_s:.1f} s ({initial_time_s/60:.1f} min)")
            print(f"Max flight time:   {max_flight_time:.0f} s ({max_flight_time/60:.0f} min)")
            print(f"Min SOC:           {min_soc*100:.0f}%")
            print(f"Population:        {popsize * n_params}")
            print(f"Max generations:   {maxiter}")
            print(f"")

        # ── Reset tracking ────────────────────────────────────────────────
        self._eval_count = 0
        self._best_obj = np.inf
        self._convergence_history = []
        self._parameter_history = []
        self._generation = 0

        # ── Build objective function ──────────────────────────────────────

        # We need a deep copy of the path structure for thread-safety
        # in the objective function
        path_template = initial_path

        def objective(theta: np.ndarray) -> float:
            """Penalized objective function for DE."""
            self._eval_count += 1

            try:
                # Set parameters and re-propagate the path
                path_template.parameter_vector = theta

                # ── Evaluate energy ───────────────────────────────
                energy_result = analyze_path_energy(
                    path_template, ac, SOC_min=min_soc
                )

                E_wh = energy_result.total_energy_wh
                t_s = energy_result.total_time
                soc_final = energy_result.SOC_final

                # ── Evaluate terrain constraints ──────────────────
                terrain_report = self.terrain_analyzer.analyze(path_template)
                terrain_penalty = terrain_report.constraint_penalty

                # ── Build penalized objective ─────────────────────
                # Normalized objectives
                f_energy = E_wh / E_scale
                f_time = t_s / T_scale

                # Weighted objective
                f_obj = w_energy * f_energy + w_time * f_time

                # ── Constraint penalties ──────────────────────────
                penalty = 0.0

                # Terrain clearance: aggressive barrier
                if terrain_penalty > 0:
                    n_wp = max(len(terrain_report.agl_profile), 1)
                    # Max violation in meters
                    agl = terrain_report.agl_profile
                    valid = ~np.isnan(agl)
                    if np.any(valid):
                        clearance_req = 50.0  # min AGL
                        deficits = np.maximum(clearance_req - agl[valid], 0)
                        max_deficit = np.max(deficits)
                        frac_violated = np.sum(deficits > 0) / np.sum(valid)
                        # Strong penalty: proportional to max deficit + fraction
                        penalty += self.penalty_terrain * (
                            max_deficit / 100.0 + frac_violated * 5.0
                        )

                # SOC below minimum
                soc_deficit = min_soc - soc_final
                if soc_deficit > 0:
                    penalty += self.penalty_soc * soc_deficit

                # Time exceeding limit
                time_excess = t_s - max_flight_time
                if time_excess > 0:
                    penalty += self.penalty_time * (time_excess / max_flight_time)

                # Endpoint altitude: final segment should end near destination
                end_state = path_template.end_state
                if end_state is not None:
                    dest_elev = path_template.destination[2]
                    alt_error = abs(end_state.alt - dest_elev)
                    if alt_error > 100:  # allow 100m tolerance
                        penalty += self.penalty_endpoint * (alt_error / 1000)

                return f_obj + penalty

            except Exception:
                return 1e10  # Return large value on failure

        # ── DE callback for tracking ──────────────────────────────────────
        def de_callback(xk, convergence):
            self._generation += 1
            obj_val = objective(xk)

            if obj_val < self._best_obj:
                self._best_obj = obj_val

            self._convergence_history.append(self._best_obj)

            # Sample parameter snapshots periodically
            if self._generation % max(1, maxiter // 20) == 0:
                self._parameter_history.append(xk.copy())

            if verbose and self._generation % 5 == 0:
                # Decode current best for display
                path_template.parameter_vector = xk
                try:
                    er = analyze_path_energy(path_template, ac, SOC_min=min_soc)
                    print(f"  Gen {self._generation:4d} | "
                          f"obj={self._best_obj:.4f} | "
                          f"E={er.total_energy_wh:.1f} Wh | "
                          f"t={er.total_time:.0f}s | "
                          f"SOC={er.SOC_final*100:.1f}% | "
                          f"conv={convergence:.4f}")
                except Exception:
                    print(f"  Gen {self._generation:4d} | "
                          f"obj={self._best_obj:.4f} | "
                          f"conv={convergence:.4f}")

        # ── Run differential evolution ────────────────────────────────────
        t_start = time_module.time()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = differential_evolution(
                objective,
                bounds=bounds,
                x0=theta_0,
                maxiter=maxiter,
                popsize=popsize,
                tol=tol,
                seed=seed,
                callback=de_callback,
                polish=True,        # Local refinement at the end
                mutation=(0.5, 1.0),
                recombination=0.8,
                strategy='best1bin',
                atol=0.0,
                updating='deferred',
                workers=1,
            )

        wall_time = time_module.time() - t_start

        # ── Decode final solution ─────────────────────────────────────────
        theta_opt = result.x
        path_template.parameter_vector = theta_opt

        # Final evaluations
        final_energy = analyze_path_energy(path_template, ac, SOC_min=min_soc)
        final_terrain = self.terrain_analyzer.analyze(path_template)

        # Compute improvements
        energy_imp = ((initial_energy_wh - final_energy.total_energy_wh)
                      / initial_energy_wh * 100) if initial_energy_wh > 0 else 0
        time_imp = ((initial_time_s - final_energy.total_time)
                    / initial_time_s * 100) if initial_time_s > 0 else 0

        opt_result = OptimizationResult(
            mode=mode.value,
            optimized_path=path_template,
            optimized_energy=final_energy,
            parameter_vector=theta_opt,
            objective_value=result.fun,
            energy_wh=final_energy.total_energy_wh,
            flight_time_s=final_energy.total_time,
            terrain_feasible=final_terrain.is_feasible,
            min_agl=final_terrain.min_agl,
            soc_final=final_energy.SOC_final,
            battery_feasible=final_energy.SOC_final >= min_soc,
            time_feasible=final_energy.total_time <= max_flight_time,
            fully_feasible=(final_terrain.is_feasible and
                           final_energy.SOC_final >= min_soc and
                           final_energy.total_time <= max_flight_time),
            n_evaluations=self._eval_count,
            n_iterations=self._generation,
            wall_time_s=wall_time,
            convergence_history=self._convergence_history.copy(),
            parameter_history=self._parameter_history.copy(),
            initial_energy_wh=initial_energy_wh,
            initial_time_s=initial_time_s,
            energy_improvement_pct=energy_imp,
            time_improvement_pct=time_imp,
            success=result.success,
            message=result.message,
            weights=(w_energy, w_time),
        )

        if verbose:
            print(f"\n{opt_result.summary()}")

        return opt_result

    # ── Multi-objective Pareto sweep ──────────────────────────────────────

    def pareto_sweep(
        self,
        initial_path: FlightPath,
        payload_kg: float = 0.0,
        max_flight_time: float = None,
        min_soc: float = 0.15,
        n_weights: int = 7,
        maxiter: int = 50,
        popsize: int = 15,
        seed: int = 42,
        verbose: bool = True,
    ) -> List[OptimizationResult]:
        """
        Sweep weight combinations to approximate the Pareto front.

        Parameters
        ----------
        n_weights : int
            Number of weight combinations to evaluate.

        Returns
        -------
        List of OptimizationResult, one per weight combination.
        """
        weights = np.linspace(0, 1, n_weights)
        results = []

        for i, w_e in enumerate(weights):
            w_t = 1.0 - w_e
            if verbose:
                print(f"\n{'#'*60}")
                print(f"Pareto sweep {i+1}/{n_weights}: "
                      f"w_energy={w_e:.2f}, w_time={w_t:.2f}")
                print(f"{'#'*60}")

            # Rebuild initial path for each sweep point
            path_copy = self._clone_path(initial_path)

            res = self.optimize(
                path_copy,
                mode=OptMode.MULTI,
                payload_kg=payload_kg,
                max_flight_time=max_flight_time,
                min_soc=min_soc,
                w_energy=w_e,
                w_time=w_t,
                maxiter=maxiter,
                popsize=popsize,
                seed=seed + i,
                verbose=verbose,
            )
            results.append(res)

        return results

    # ── Scenario-level optimization ───────────────────────────────────────

    def optimize_scenario(
        self,
        scenario,
        initial_strategy: PathStrategy = PathStrategy.TERRAIN_FOLLOW,
        maxiter: int = 60,
        popsize: int = 15,
        verbose: bool = True,
    ) -> Dict[str, List[OptimizationResult]]:
        """
        Optimize all legs of a FlightScenario.

        For each leg, runs the three optimization modes based on
        the scenario's priority flag.

        Parameters
        ----------
        scenario : FlightScenario
            Medical delivery scenario with legs and priority.
        initial_strategy : PathStrategy
            Strategy to build the initial path for each leg.
        maxiter : int
            Max DE generations per optimization.
        popsize : int
            Population size multiplier.
        verbose : bool
            Print progress.

        Returns
        -------
        Dict mapping leg descriptions to lists of OptimizationResult.
        """
        from .scenarios import OptPriority

        builder = PathBuilder(self.dem, self.uav, self.constraints)
        results = {}

        for i, leg in enumerate(scenario.legs):
            leg_key = f"Leg{i+1}_{leg.origin.short_name}_to_{leg.destination.short_name}"

            if verbose:
                print(f"\n{'═'*60}")
                print(f"Optimizing {leg_key}")
                print(f"  Payload: {leg.payload_kg:.1f} kg")
                print(f"  {leg.origin.name} → {leg.destination.name}")
                print(f"{'═'*60}")

            # Build initial path
            origin_node = FacilityNode(
                leg.origin.name, leg.origin.lat, leg.origin.lon,
                leg.origin.ground_elev
            )
            dest_node = FacilityNode(
                leg.destination.name, leg.destination.lat,
                leg.destination.lon, leg.destination.ground_elev
            )
            initial_path = builder.build(origin_node, dest_node, initial_strategy)

            # Determine urgency constraints
            max_time = scenario.urgency.max_flight_time_s()
            min_soc = scenario.urgency.min_soc_percent()

            # Run optimization modes based on priority
            leg_results = []

            modes_to_run = []
            if scenario.priority == OptPriority.ENERGY:
                modes_to_run = [OptMode.ENERGY]
            elif scenario.priority == OptPriority.TIME:
                modes_to_run = [OptMode.TIME]
            else:  # BALANCED
                modes_to_run = [OptMode.ENERGY, OptMode.TIME, OptMode.MULTI]

            for mode in modes_to_run:
                path_copy = self._clone_path(initial_path)
                res = self.optimize(
                    path_copy,
                    mode=mode,
                    payload_kg=leg.payload_kg,
                    max_flight_time=max_time,
                    min_soc=min_soc,
                    maxiter=maxiter,
                    popsize=popsize,
                    verbose=verbose,
                )
                leg_results.append(res)

            results[leg_key] = leg_results

        return results

    # ── Utilities ─────────────────────────────────────────────────────────

    def _clone_path(self, path: FlightPath) -> FlightPath:
        """Deep-copy a FlightPath preserving structure and parameters."""
        new_path = FlightPath(
            path.origin[0], path.origin[1], path.origin[2],
            path.destination[0], path.destination[1], path.destination[2],
        )
        for seg in path.segments:
            new_seg = self._clone_segment(seg)
            new_path.add_segment(new_seg)
        return new_path

    @staticmethod
    def _clone_segment(seg: FlightSegment) -> FlightSegment:
        """Deep-copy a flight segment."""
        t = seg.segment_type
        if t == SegmentType.VTOL_ASCEND:
            return VTOLAscend(seg.altitude_gain, seg.climb_rate)
        elif t == SegmentType.VTOL_DESCEND:
            return VTOLDescend(seg.altitude_loss, seg.descent_rate)
        elif t == SegmentType.FW_CLIMB:
            return FWClimb(seg.altitude_gain, seg.climb_angle_deg, seg.airspeed)
        elif t == SegmentType.FW_DESCEND:
            return FWDescend(seg.altitude_loss, seg.descent_angle_deg, seg.airspeed)
        elif t == SegmentType.FW_CRUISE:
            return FWCruise(seg.ground_distance_param, seg.airspeed)
        elif t == SegmentType.TRANSITION:
            return Transition(seg.duration, seg.altitude_change_param,
                            seg.ground_distance_param)
        raise ValueError(f"Unknown segment type: {t}")

    def evaluate_path(
        self,
        path: FlightPath,
        payload_kg: float = 0.0,
        min_soc: float = 0.15,
    ) -> dict:
        """
        Evaluate a path's energy, time, and terrain metrics.

        Returns a dict with all key metrics.
        """
        ac = copy.deepcopy(self.ac_params)
        ac.m_tow = self.ac_params.m_tow + payload_kg

        energy = analyze_path_energy(path, ac, SOC_min=min_soc)
        terrain = self.terrain_analyzer.analyze(path)

        return {
            'energy_wh': energy.total_energy_wh,
            'time_s': energy.total_time,
            'soc_final': energy.SOC_final,
            'terrain_feasible': terrain.is_feasible,
            'min_agl': terrain.min_agl,
            'energy_result': energy,
            'terrain_report': terrain,
        }

    # ── Routed path optimization (with airspace constraints) ──────────

    def optimize_routed(
        self,
        routed_path,
        mode: OptMode = OptMode.ENERGY,
        payload_kg: float = 0.0,
        max_flight_time: float = None,
        min_soc: float = 0.15,
        w_energy: float = None,
        w_time: float = None,
        airspace=None,
        maxiter: int = 100,
        popsize: int = 20,
        tol: float = 1e-4,
        seed: int = 42,
        verbose: bool = True,
    ) -> OptimizationResult:
        """
        Optimize a RoutedPath: jointly searches over lateral routing
        and vertical flight parameters to find the energy/time-optimal
        path that satisfies terrain, airspace, and battery constraints.

        This is the core method for regulation-aware path planning. The
        optimizer adjusts intermediate waypoint lateral offsets, per-leg
        cruise altitudes, per-leg airspeeds, and VTOL rates simultaneously.

        Parameters
        ----------
        routed_path : RoutedPath
            The routed path to optimize. Provides the routing design
            variable structure and bounds.
        mode : OptMode
            Optimization objective: ENERGY, TIME, or MULTI.
        payload_kg : float
            Payload mass [kg].
        max_flight_time : float
            Maximum flight time [s].
        min_soc : float
            Minimum battery SOC fraction.
        w_energy, w_time : float
            Weights for multi-objective mode.
        airspace : AirspaceManager or None
            If provided, airspace violations are penalized. This is
            what drives the optimizer to discover lateral detours.
        maxiter : int
            Maximum DE generations.
        popsize : int
            DE population size multiplier.
        tol : float
            Convergence tolerance.
        seed : int
            Random seed.
        verbose : bool
            Print progress.

        Returns
        -------
        OptimizationResult with the optimized path and routing metrics.
        """
        if max_flight_time is None:
            max_flight_time = self.constraints.max_flight_time

        # Set weights
        if mode == OptMode.ENERGY:
            w_energy = w_energy if w_energy is not None else 1.0
            w_time = w_time if w_time is not None else 0.0
        elif mode == OptMode.TIME:
            w_energy = w_energy if w_energy is not None else 0.0
            w_time = w_time if w_time is not None else 1.0
        else:
            w_energy = w_energy if w_energy is not None else 0.5
            w_time = w_time if w_time is not None else 0.5

        # Aircraft params with payload
        ac = copy.deepcopy(self.ac_params)
        ac.m_tow = self.ac_params.m_tow + payload_kg

        # ── Initial solution metrics ─────────────────────────────────
        initial_path = routed_path.flight_path
        initial_energy = analyze_path_energy(initial_path, ac, SOC_min=min_soc)
        initial_energy_wh = initial_energy.total_energy_wh
        initial_time_s = initial_energy.total_time

        E_scale = max(initial_energy_wh, 50.0)
        T_scale = max(initial_time_s, 60.0)

        # ── Bounds and initial vector ────────────────────────────────
        bounds = routed_path.parameter_bounds
        theta_0 = routed_path.parameter_vector.copy()
        n_params = len(theta_0)

        for i, (lo, hi) in enumerate(bounds):
            theta_0[i] = np.clip(theta_0[i], lo, hi)

        if verbose:
            print(f"\n{'='*60}")
            print(f"ROUTED PATH OPTIMIZATION — Mode: {mode.value.upper()}")
            print(f"{'='*60}")
            print(f"Route:             {routed_path.origin.name} → "
                  f"{routed_path.destination.name}")
            print(f"Intermediate WPs:  {routed_path.n_intermediate}")
            print(f"Design variables:  {n_params}")
            print(f"Weights:           w_E={w_energy:.2f}, w_t={w_time:.2f}")
            print(f"Payload:           {payload_kg:.1f} kg (m_tow={ac.m_tow:.1f} kg)")
            print(f"Initial energy:    {initial_energy_wh:.1f} Wh")
            print(f"Initial time:      {initial_time_s:.1f} s ({initial_time_s/60:.1f} min)")
            print(f"Airspace zones:    "
                  f"{'ENABLED (' + str(len(airspace.active_zones())) + ' zones)' if airspace else 'DISABLED'}")
            print(f"Population:        {popsize * n_params}")
            print(f"Max generations:   {maxiter}")
            print()

        # ── Tracking ─────────────────────────────────────────────────
        self._eval_count = 0
        self._best_obj = np.inf
        self._convergence_history = []
        self._parameter_history = []
        self._generation = 0

        # ── Airspace penalty weight ──────────────────────────────────
        penalty_airspace = 500.0  # Strong penalty to enforce zone avoidance

        # ── Objective function ───────────────────────────────────────
        def objective(theta: np.ndarray) -> float:
            self._eval_count += 1

            try:
                # Set routing parameters → triggers path rebuild
                routed_path.parameter_vector = theta

                # Get the built flight path
                fp = routed_path.flight_path

                # ── Energy evaluation ────────────────────────
                energy_result = analyze_path_energy(fp, ac, SOC_min=min_soc)
                E_wh = energy_result.total_energy_wh
                t_s = energy_result.total_time
                soc_final = energy_result.SOC_final

                # ── Terrain constraints ──────────────────────
                terrain_report = self.terrain_analyzer.analyze(fp)

                # ── Normalized objectives ────────────────────
                f_energy = E_wh / E_scale
                f_time = t_s / T_scale
                f_obj = w_energy * f_energy + w_time * f_time

                # ── Constraint penalties ─────────────────────
                penalty = 0.0

                # Terrain clearance
                if terrain_report.constraint_penalty > 0:
                    agl = terrain_report.agl_profile
                    valid = ~np.isnan(agl)
                    if np.any(valid):
                        clearance_req = 50.0
                        deficits = np.maximum(clearance_req - agl[valid], 0)
                        max_deficit = np.max(deficits)
                        frac_violated = np.sum(deficits > 0) / np.sum(valid)
                        penalty += self.penalty_terrain * (
                            max_deficit / 100.0 + frac_violated * 5.0
                        )

                # SOC below minimum
                soc_deficit = min_soc - soc_final
                if soc_deficit > 0:
                    penalty += self.penalty_soc * soc_deficit

                # Time exceeding limit
                time_excess = t_s - max_flight_time
                if time_excess > 0:
                    penalty += self.penalty_time * (time_excess / max_flight_time)

                # ── Airspace constraints ───────────────────
                if airspace is not None:
                    air_report = airspace.check_path(fp)
                    if not air_report.feasible:
                        # Separate hard barriers (PROHIBITED) from soft penalties
                        from .airspace import ZoneType
                        hard_violations = sum(
                            1 for v in air_report.violations
                            if v.zone_type in (ZoneType.PROHIBITED, ZoneType.AERODROME_CTR)
                        )
                        if hard_violations > 0:
                            # PROHIBITED/AERODROME: absolute barrier
                            penalty += 1e6 * hard_violations
                        # Graduated penalty for all zones
                        penalty += penalty_airspace * air_report.total_penalty

                return f_obj + penalty

            except Exception:
                return 1e10

        # ── DE callback ──────────────────────────────────────────────
        def de_callback(xk, convergence):
            self._generation += 1
            obj_val = objective(xk)
            if obj_val < self._best_obj:
                self._best_obj = obj_val
            self._convergence_history.append(self._best_obj)

            if self._generation % max(1, maxiter // 20) == 0:
                self._parameter_history.append(xk.copy())

            if verbose and self._generation % 5 == 0:
                routed_path.parameter_vector = xk
                try:
                    fp = routed_path.flight_path
                    er = analyze_path_energy(fp, ac, SOC_min=min_soc)
                    topo = routed_path.topology_summary()
                    lat_dev = topo['max_lateral_deviation_m']
                    print(f"  Gen {self._generation:4d} | "
                          f"obj={self._best_obj:.4f} | "
                          f"E={er.total_energy_wh:.1f} Wh | "
                          f"t={er.total_time:.0f}s | "
                          f"SOC={er.SOC_final*100:.1f}% | "
                          f"lat_dev={lat_dev:.0f}m | "
                          f"conv={convergence:.4f}")
                except Exception:
                    print(f"  Gen {self._generation:4d} | "
                          f"obj={self._best_obj:.4f} | "
                          f"conv={convergence:.4f}")

        # ── Run DE ───────────────────────────────────────────────────
        t_start = time_module.time()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = differential_evolution(
                objective,
                bounds=bounds,
                x0=theta_0,
                maxiter=maxiter,
                popsize=popsize,
                tol=tol,
                seed=seed,
                callback=de_callback,
                polish=True,
                mutation=(0.5, 1.0),
                recombination=0.8,
                strategy='best1bin',
                atol=0.0,
                updating='deferred',
                workers=1,
            )

        wall_time = time_module.time() - t_start

        # ── Decode final solution ────────────────────────────────────
        theta_opt = result.x
        routed_path.parameter_vector = theta_opt
        final_path = routed_path.flight_path

        final_energy = analyze_path_energy(final_path, ac, SOC_min=min_soc)
        final_terrain = self.terrain_analyzer.analyze(final_path)

        # Airspace check on final solution
        airspace_feasible = True
        if airspace is not None:
            final_air = airspace.check_path(final_path)
            airspace_feasible = final_air.feasible

        energy_imp = ((initial_energy_wh - final_energy.total_energy_wh)
                      / initial_energy_wh * 100) if initial_energy_wh > 0 else 0
        time_imp = ((initial_time_s - final_energy.total_time)
                    / initial_time_s * 100) if initial_time_s > 0 else 0

        opt_result = OptimizationResult(
            mode=mode.value,
            optimized_path=final_path,
            optimized_energy=final_energy,
            parameter_vector=theta_opt,
            objective_value=result.fun,
            energy_wh=final_energy.total_energy_wh,
            flight_time_s=final_energy.total_time,
            terrain_feasible=final_terrain.is_feasible,
            min_agl=final_terrain.min_agl,
            soc_final=final_energy.SOC_final,
            battery_feasible=final_energy.SOC_final >= min_soc,
            time_feasible=final_energy.total_time <= max_flight_time,
            fully_feasible=(final_terrain.is_feasible and
                           final_energy.SOC_final >= min_soc and
                           final_energy.total_time <= max_flight_time and
                           airspace_feasible),
            n_evaluations=self._eval_count,
            n_iterations=self._generation,
            wall_time_s=wall_time,
            convergence_history=self._convergence_history.copy(),
            parameter_history=self._parameter_history.copy(),
            initial_energy_wh=initial_energy_wh,
            initial_time_s=initial_time_s,
            energy_improvement_pct=energy_imp,
            time_improvement_pct=time_imp,
            success=result.success,
            message=result.message,
            weights=(w_energy, w_time),
        )

        if verbose:
            print(f"\n{opt_result.summary()}")
            topo = routed_path.topology_summary()
            print(f"\n  Routing summary:")
            print(f"    Cruise legs:        {topo['n_cruise_legs']}")
            print(f"    Max lateral dev:    {topo['max_lateral_deviation_m']:.0f} m")
            print(f"    Route stretch:      {topo['route_stretch_factor']:.3f}x")
            print(f"    Direct path:        {topo['is_direct']}")
            if airspace is not None:
                final_air = airspace.check_path(final_path)
                print(f"    Airspace feasible:  {final_air.feasible}")
                if not final_air.feasible:
                    print(f"    Violations:         {final_air.n_violations} "
                          f"in {final_air.violated_zones}")

        return opt_result

    # ── Pareto sweep for routed paths ─────────────────────────────────

    def pareto_sweep_routed(
        self,
        origin,
        destination,
        n_intermediate: int = 1,
        payload_kg: float = 1.5,
        airspace=None,
        n_weights: int = 7,
        maxiter: int = 50,
        popsize: int = 12,
        seed: int = 42,
        verbose: bool = True,
    ) -> List[OptimizationResult]:
        """
        Sweep energy-vs-time weights for routed paths with airspace.

        Returns a list of OptimizationResult, one per weight combination,
        approximating the Pareto front under the given constraint config.
        """
        from .routed_path import RoutedPath

        weights = np.linspace(0, 1, n_weights)
        results = []

        for i, w_e in enumerate(weights):
            w_t = 1.0 - w_e
            if verbose:
                print(f"  Pareto {i+1}/{n_weights}: w_E={w_e:.2f}, w_t={w_t:.2f}...",
                      end=" ", flush=True)

            rp = RoutedPath(origin, destination, self.dem, self.uav,
                           self.constraints, n_intermediate=n_intermediate)

            res = self.optimize_routed(
                rp, mode=OptMode.MULTI,
                payload_kg=payload_kg,
                airspace=airspace,
                w_energy=w_e, w_time=w_t,
                maxiter=maxiter, popsize=popsize,
                seed=seed + i, verbose=False,
            )
            results.append(res)
            if verbose:
                print(f"E={res.energy_wh:.0f}Wh, t={res.flight_time_s/60:.1f}min")

        return results
