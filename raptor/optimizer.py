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

Author: Victor (LUAS-EPN / KU Leuven)
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
from .regulations import (
    RegulatoryProfile, OperationalContext, RDAC_101, MEDICAL_DELIVERY_CONTEXT,
)
from .penalties import PenaltyWeights, PenaltyBreakdown, evaluate_penalties
from .compliance import ComplianceAssessment, ComplianceLevel, assess_compliance


# ═══════════════════════════════════════════════════════════════════════════
# OPTIMIZATION MODE
# ═══════════════════════════════════════════════════════════════════════════

#: The constraint terms a penalty breakdown carries, in the order a
#: history figure should stack them. Named here rather than read off the
#: dataclass so the figure's legend order is stable.
PENALTY_TERMS = ("corridor", "prohibited", "ceiling", "soc", "time",
                 "endpoint", "permit", "waiver", "activation")


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

    # ── Regulatory outcome ────────────────────────────────────────────
    #: Path stays inside the AGL corridor (clearance floor and 101.185 ceiling).
    ceiling_feasible: bool = True
    #: Worst exceedance of the AGL ceiling [m].
    max_ceiling_excess_m: float = 0.0
    #: No prohibited zone entered and no ceiling busted.
    airspace_feasible: bool = True
    #: Prohibited zones the route still enters, if any.
    prohibited_zones: List[str] = field(default_factory=list)
    #: zone_id → permit family for every authorisation this route needs.
    permits_required: Dict[str, str] = field(default_factory=dict)
    #: Total permit effort under the active regulatory profile.
    permit_cost: float = 0.0
    #: Per-constraint penalty contributions of the final design.
    penalties: Optional[PenaltyBreakdown] = None
    #: Static compliance notes for the operation (payload, BVLOS, …).
    compliance_notes: List[str] = field(default_factory=list)
    #: Graded regulatory standing: compliant, needs authorisation, needs a
    #: height waiver, or not flyable at all. Replaces the binary verdict —
    #: "no legal path exists" and "needs a 103 m waiver over 1.4 km" are
    #: different answers, and only one of them is actionable.
    compliance: Optional[ComplianceAssessment] = None
    #: Whether the starting path satisfied every constraint. When it did
    #: not, the energy and time "improvements" below are measured against
    #: a route that could not legally be flown, so a negative number is
    #: the price of compliance rather than a failure to optimize.
    initial_feasible: bool = True
    #: Per-generation record of the objective *and* the constraint
    #: penalties behind it. The convergence curve alone says the search
    #: improved; this says which constraint it stopped violating, which
    #: is the half a reader needs in order to trust the answer.
    penalty_history: List[dict] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable result summary."""
        lines = [
            "=" * 70,
            f"OPTIMIZATION RESULT — Mode: {self.mode.upper()}",
            "=" * 70,
            f"Weights: w_energy={self.weights[0]:.2f}, w_time={self.weights[1]:.2f}",
            f"",
            (f"{'Metric':<30s} {'Initial':>12s} {'Optimized':>12s} {'Change':>10s}"
             if self.initial_feasible else
             f"{'Metric':<30s} {'Initial*':>12s} {'Optimized':>12s} {'Change':>10s}"),
            f"{'-'*64}",
            f"{'Energy [Wh]':<30s} {self.initial_energy_wh:>12.1f} "
            f"{self.energy_wh:>12.1f} {self.energy_improvement_pct:>+9.1f}%",
            f"{'Flight time [s]':<30s} {self.initial_time_s:>12.1f} "
            f"{self.flight_time_s:>12.1f} {self.time_improvement_pct:>+9.1f}%",
            f"{'Flight time [min]':<30s} {self.initial_time_s/60:>12.1f} "
            f"{self.flight_time_s/60:>12.1f} {'':>10s}",
            f"{'SOC final [%]':<30s} {'—':>12s} {self.soc_final*100:>11.1f}%",
            f"{'Min AGL [m]':<30s} {'—':>12s} {self.min_agl:>12.1f}",
            (None if self.initial_feasible else
             "\n* the starting path violated a constraint, so these changes are\n"
             "  measured against a route that could not legally be flown."),
            f"",
            f"Terrain clearance: {'✓' if self.terrain_feasible else '✗'}",
            f"AGL ceiling:       {'✓' if self.ceiling_feasible else '✗'}"
            + (f"  (exceeded by {self.max_ceiling_excess_m:.0f} m)"
               if not self.ceiling_feasible else ""),
            f"Airspace:          {'✓' if self.airspace_feasible else '✗'}"
            + (f"  (prohibited: {', '.join(self.prohibited_zones)})"
               if self.prohibited_zones else ""),
            f"Battery feasible:  {'✓' if self.battery_feasible else '✗'}",
            f"Time feasible:     {'✓' if self.time_feasible else '✗'}",
            f"Fully feasible:    {'✓' if self.fully_feasible else '✗'}",
            f"",
            self._permit_block(),
            f"Evaluations: {self.n_evaluations}",
            f"Generations: {self.n_iterations}",
            f"Wall time:   {self.wall_time_s:.1f} s",
            f"Converged:   {'Yes' if self.success else 'No'} — {self.message}",
        ]
        return "\n".join(l for l in lines if l is not None)

    def _permit_block(self) -> str:
        """The authorisations this plan depends on."""
        waiver = self.compliance.height_waiver if self.compliance else None
        if not self.permits_required:
            if waiver is None:
                return "Authorisations:    none required\n"
            return ("Authorisations:    height waiver only\n  "
                    + waiver.summary().replace("\n", "\n  ") + "\n")
        lines = [f"Authorisations:    {len(self.permits_required)} "
                 f"(effort {self.permit_cost:.0f})"]
        for zid, fam in sorted(self.permits_required.items()):
            lines.append(f"  - {zid:<28s} [{fam}]")
        for note in self.compliance_notes:
            lines.append(f"  ! {note}")
        if self.compliance is not None and self.compliance.height_waiver:
            lines.append("  " + self.compliance.height_waiver.summary()
                         .replace("\n", "\n  "))
        return "\n".join(lines) + "\n"

    @property
    def regulatory_summary(self) -> str:
        """Compact one-line verdict, for tables and logs."""
        if self.compliance is not None:
            return self.compliance.verdict()   # accounts for safety too
        if not self.fully_feasible:
            return "INFEASIBLE"
        if self.permits_required:
            return f"LEGAL with {len(self.permits_required)} authorisation(s)"
        return "LEGAL, no authorisation required"


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
        regulation: RegulatoryProfile = RDAC_101,
        context: OperationalContext = None,
        penalty_weights: PenaltyWeights = None,
        warn_on_bad_constraints: bool = True,
    ):
        self.dem = dem
        self.uav = uav
        self.constraints = constraints
        self.ac_params = ac_params
        self.terrain_analyzer = TerrainAnalyzer(dem, constraints)

        #: Which rulebook applies, and under what credentials.
        self.regulation = regulation
        self.context = context if context is not None else MEDICAL_DELIVERY_CONTEXT
        #: How each constraint converts into objective units.
        self.penalty_weights = penalty_weights or PenaltyWeights()

        # A constraint set whose vertical corridor is empty makes every
        # path infeasible no matter how well it is routed. That failure is
        # silent and looks like an optimizer problem, so say it out loud.
        self.constraint_problems = constraints.validate()
        if warn_on_bad_constraints and self.constraint_problems:
            for msg in self.constraint_problems:
                warnings.warn(f"MissionConstraints: {msg}", stacklevel=2)

        # Legacy scalar weights, retained so existing scripts that poke at
        # them keep running; the penalty model above is what is used.
        self.penalty_terrain = 500.0
        self.penalty_soc = 100.0
        self.penalty_time = 20.0
        self.penalty_endpoint = 10.0

    def _envelope_bounds(self, path: FlightPath):
        """
        Segment parameter bounds, narrowed to the aircraft's real envelope.

        ``FWCruise.parameter_bounds`` and friends are class constants —
        (15, 35) m/s regardless of the wing. Once the envelope is derived
        from the aerodynamics those constants are no longer consistent
        with it, and the widest bound wins unless they are intersected.
        """
        bounds = list(path.parameter_bounds)
        names = []
        for seg in path.segments:
            names.extend(seg.parameter_names)

        v_lo, v_hi = self.uav.fw_min_airspeed, self.uav.fw_max_airspeed
        out = []
        for i, (lo, hi) in enumerate(bounds):
            name = names[i] if i < len(names) else ""
            if name == "airspeed":
                lo, hi = max(lo, v_lo), min(hi, v_hi)
                if lo >= hi:                 # envelope disjoint from the class bound
                    lo, hi = v_lo, v_hi
            elif name == "climb_angle_deg":
                hi = min(hi, self.uav.fw_max_climb_angle)
            elif name == "descent_angle_deg":
                hi = min(hi, self.uav.fw_max_descent_angle)
            out.append((lo, hi))
        return out

    # ── Shared constraint evaluation ──────────────────────────────────────

    def _score(self, fp: FlightPath, energy_result, min_soc: float,
               max_flight_time: float, airspace=None,
               ceiling_waivable: bool = False) -> PenaltyBreakdown:
        """
        Evaluate every constraint for one candidate path.

        Both optimizers route through here, so the penalty that steers the
        search and the verdict reported at the end are computed from the
        same numbers — a mismatch between the two is what lets a design be
        penalty-free and infeasible at the same time.
        """
        terrain_report = self.terrain_analyzer.analyze(fp)
        air_report = airspace.check_path(fp) if airspace is not None else None

        alt_error = 0.0
        end_state = fp.end_state
        if end_state is not None:
            alt_error = abs(end_state.alt - fp.destination[2])

        b = evaluate_penalties(
            terrain_report=terrain_report,
            airspace_report=air_report,
            soc_final=energy_result.SOC_final,
            min_soc=min_soc,
            flight_time_s=energy_result.total_time,
            max_flight_time_s=max_flight_time,
            endpoint_alt_error_m=alt_error,
            weights=self.penalty_weights,
            regulation=self.regulation,
            context=self.context,
            ceiling_waivable=ceiling_waivable,
        )
        b.terrain_report = terrain_report
        b.airspace_report = air_report
        return b

        # Tracking
        self._eval_count = 0
        self._best_obj = np.inf
        self._convergence_history = []
        self._parameter_history = []
        self._penalty_history = []
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
        ac = self.ac_params.with_payload(payload_kg)

        # ── Get initial solution metrics ──────────────────────────────────
        initial_energy = analyze_path_energy(initial_path, ac, SOC_min=min_soc)
        initial_energy_wh = initial_energy.total_energy_wh
        initial_time_s = initial_energy.total_time
        initial_feasible = self._score(
            initial_path, initial_energy, min_soc, max_flight_time
        ).feasible

        # ── Normalization scales (for multi-objective) ────────────────────
        # Scale objectives to [0, 1] range approximately
        E_scale = max(initial_energy_wh, 50.0)  # Wh
        T_scale = max(initial_time_s, 60.0)       # s

        # ── Prepare bounds ────────────────────────────────────────────────
        # Segment classes declare generic bounds that know nothing about
        # this aircraft. Intersect them with the actual flight envelope,
        # or the optimizer is free to select an airspeed below the wing's
        # stall speed and be rewarded for it.
        bounds = self._envelope_bounds(initial_path)
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
        self._penalty_history = []
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

                # ── Normalized objectives ─────────────────────────
                f_obj = w_energy * (E_wh / E_scale) + w_time * (t_s / T_scale)

                # ── Constraints ───────────────────────────────────
                b = self._score(path_template, energy_result,
                                min_soc, max_flight_time)

                return f_obj + b.total

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
        final_b = self._score(path_template, final_energy,
                              min_soc, max_flight_time)
        final_terrain = final_b.terrain_report

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
            fully_feasible=final_b.feasible,
            ceiling_feasible=final_terrain.ceiling_feasible,
            max_ceiling_excess_m=final_terrain.max_ceiling_excess,
            penalties=final_b,
            compliance=assess_compliance(
                final_path, final_terrain, final_air,
                regulation=self.regulation, context=self.context,
                constraints=self.constraints, airspace=airspace,
            ),
            compliance_notes=self.context.compliance_findings(self.regulation),
            initial_feasible=initial_feasible,
            n_evaluations=self._eval_count,
            n_iterations=self._generation,
            wall_time_s=wall_time,
            convergence_history=self._convergence_history.copy(),
            parameter_history=self._parameter_history.copy(),
            penalty_history=list(self._penalty_history),
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
        ac = self.ac_params.with_payload(payload_kg)

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
        feasibility_first: bool = True,
        feasibility_maxiter: int = None,
        allow_height_waiver: bool = False,
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
        feasibility_first : bool
            When the starting route violates a constraint, run a short
            search on the constraint penalty alone before optimizing
            energy. Minimising ``energy + penalty`` from an infeasible
            start asks the optimizer to do two things at once, and cheap
            infeasible routes routinely outscore expensive legal ones
            long enough for the population to converge around them. A
            feasible seed removes the ambiguity.
        feasibility_maxiter : int
            Generations for that first stage. Defaults to half of
            ``maxiter``.
        allow_height_waiver : bool
            Treat the AGL ceiling as waivable under RDAC 101.035 instead
            of absolute. Over terrain that outruns the aircraft's descent
            rate no compliant profile exists at any altitude, and the
            strict search then has nothing to rank; this mode finds the
            route needing the *least* height relief and reports it as a
            waiver to file. Prohibited airspace stays hard regardless.
            :meth:`plan_route` tries strict first and falls back to this.

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
        ac = self.ac_params.with_payload(payload_kg)

        # ── Initial solution metrics ─────────────────────────────────
        initial_path = routed_path.flight_path
        initial_energy = analyze_path_energy(initial_path, ac, SOC_min=min_soc)
        initial_energy_wh = initial_energy.total_energy_wh
        initial_time_s = initial_energy.total_time
        initial_feasible = self._score(
            initial_path, initial_energy, min_soc, max_flight_time,
            airspace=airspace, ceiling_waivable=allow_height_waiver,
        ).feasible

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
        self._penalty_history = []
        self._generation = 0

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

                # ── Normalized objectives ────────────────────
                f_obj = w_energy * (E_wh / E_scale) + w_time * (t_s / T_scale)

                # ── Constraints, including airspace ──────────
                # Prohibited zones and ceiling busts are scored as graded,
                # finite penalties rather than a 1e6 wall. A wall flattens
                # the landscape: every infeasible design scores the same,
                # so DE cannot tell a route that clips a zone corner from
                # one that flies straight through, and never learns which
                # way to move. Graded penalties leave a slope to descend.
                b = self._score(fp, energy_result, min_soc,
                                max_flight_time, airspace=airspace,
                                ceiling_waivable=allow_height_waiver)

                return f_obj + b.total

            except Exception:
                return 1e10

        # ── DE callback ──────────────────────────────────────────────
        def de_callback(xk, convergence):
            self._generation += 1
            obj_val = objective(xk)
            if obj_val < self._best_obj:
                self._best_obj = obj_val
            self._convergence_history.append(self._best_obj)

            # The penalty breakdown at this generation. The objective
            # alone says the search improved; this says which constraint
            # it stopped violating to do it, which is the half a reader
            # needs to trust the answer.
            try:
                routed_path.parameter_vector = xk
                fp_h = routed_path.flight_path
                er_h = analyze_path_energy(fp_h, ac, SOC_min=min_soc)
                sc = self._score(fp_h, er_h, min_soc, max_flight_time,
                                 airspace=airspace,
                                 ceiling_waivable=allow_height_waiver)
                self._penalty_history.append(dict(
                    generation=self._generation,
                    total=float(sc.total),
                    energy_wh=float(er_h.total_energy_wh),
                    time_s=float(er_h.total_time),
                    soc_final=float(er_h.SOC_final),
                    feasible=bool(sc.feasible),
                    **{k: float(getattr(sc, k)) for k in PENALTY_TERMS},
                ))
            except Exception:
                pass

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

        # ── Stage 1: find something legal to start from ──────────────
        t_start = time_module.time()
        feas_evals = 0

        def constraint_only(theta: np.ndarray) -> float:
            """Constraint violation alone — no energy, no time."""
            try:
                routed_path.parameter_vector = theta
                fp_c = routed_path.flight_path
                er = analyze_path_energy(fp_c, ac, SOC_min=min_soc)
                return self._score(fp_c, er, min_soc, max_flight_time,
                                   airspace=airspace,
                                   ceiling_waivable=allow_height_waiver).total
            except Exception:
                return 1e10

        if feasibility_first and constraint_only(theta_0) > 0:
            fmax = (feasibility_maxiter if feasibility_maxiter is not None
                    else max(maxiter // 2, 10))
            if verbose:
                print(f"  Stage 1/2 — restoring feasibility "
                      f"({fmax} generations)")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                feas = differential_evolution(
                    constraint_only,
                    bounds=bounds,
                    x0=theta_0,
                    maxiter=fmax,
                    popsize=max(popsize // 2, 4),
                    tol=tol,
                    seed=seed,
                    polish=False,
                    mutation=(0.5, 1.0),
                    recombination=0.8,
                    strategy='best1bin',
                    atol=0.0,
                    updating='deferred',
                    workers=1,
                )
            feas_evals = int(getattr(feas, 'nfev', 0))
            if feas.fun < constraint_only(theta_0):
                theta_0 = np.clip(
                    feas.x,
                    [b[0] for b in bounds],
                    [b[1] for b in bounds],
                )
            if verbose:
                status = "feasible" if feas.fun <= 0 else f"penalty {feas.fun:.3f}"
                print(f"  Stage 1 done — {status}")
                print(f"  Stage 2/2 — minimising {mode.value} "
                      f"({maxiter} generations)")

        # ── Stage 2: optimize the objective from that seed ───────────
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
        self._eval_count += feas_evals

        # ── Decode final solution ────────────────────────────────────
        # `polish` runs an L-BFGS-B pass that ignores the bounds' meaning
        # for the penalty landscape and can return a point worse than the
        # population best, so keep whichever actually scores better.
        theta_opt = result.x
        if objective(np.asarray(result.x)) > objective(np.asarray(theta_0)):
            theta_opt = theta_0
        routed_path.parameter_vector = theta_opt
        final_path = routed_path.flight_path

        final_energy = analyze_path_energy(final_path, ac, SOC_min=min_soc)
        final_b = self._score(final_path, final_energy, min_soc,
                              max_flight_time, airspace=airspace,
                              ceiling_waivable=allow_height_waiver)
        final_terrain = final_b.terrain_report
        final_air = final_b.airspace_report

        airspace_feasible = final_air.feasible if final_air is not None else True
        permits = dict(final_b.permits_required)
        permit_cost = sum(
            self.regulation.permit_cost.get(
                fam, self.regulation.permit_cost["generic"])
            for fam in permits.values()
        )

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
            # Needing a permit does not make a route infeasible — that is
            # the distinction RDAC 101.190 draws between restricted and
            # prohibited airspace, and collapsing it would reject perfectly
            # flyable plans over paperwork.
            fully_feasible=final_b.feasible,
            ceiling_feasible=final_terrain.ceiling_feasible,
            max_ceiling_excess_m=final_terrain.max_ceiling_excess,
            airspace_feasible=airspace_feasible,
            prohibited_zones=(final_air.prohibited_zones
                              if final_air is not None else []),
            permits_required=permits,
            permit_cost=permit_cost,
            penalties=final_b,
            compliance=assess_compliance(
                final_path, final_terrain, final_air,
                regulation=self.regulation, context=self.context,
                constraints=self.constraints,
            ),
            compliance_notes=self.context.compliance_findings(self.regulation),
            initial_feasible=initial_feasible,
            n_evaluations=self._eval_count,
            n_iterations=self._generation,
            wall_time_s=wall_time,
            convergence_history=self._convergence_history.copy(),
            parameter_history=self._parameter_history.copy(),
            penalty_history=list(self._penalty_history),
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
            print(f"    Corridor mode:      {topo.get('corridor_mode', 'amsl')}")
            if topo.get('cruise_agl_m'):
                agls = ', '.join(f"{a:.0f}" for a in topo['cruise_agl_m'])
                print(f"    Cruise height:      {agls} m AGL")
            if final_air is not None:
                print(f"    Airspace feasible:  {final_air.feasible}")
                if not final_air.feasible:
                    print(f"    Hard violations:    "
                          f"{len(final_air.hard_violations)} in "
                          f"{final_air.prohibited_zones + final_air.ceiling_zones}")
            print()
            print(final_b.summary())

        return opt_result

    # ── Planning entry point: strict first, waiver only if needed ─────

    def plan_route(
        self,
        routed_path,
        mode: OptMode = OptMode.ENERGY,
        payload_kg: float = 0.0,
        max_flight_time: float = None,
        min_soc: float = 0.15,
        airspace=None,
        maxiter: int = 100,
        popsize: int = 20,
        seed: int = 42,
        verbose: bool = True,
        allow_waiver_fallback: bool = True,
        seed_with_astar: bool = True,
        astar_resolution_m: float = 400.0,
        **kwargs,
    ) -> OptimizationResult:
        """
        Plan a route, escalating through the regulation rather than failing.

        Runs the strict search first. If it produces a compliant plan —
        with or without authorisations — that is the answer, because a
        route that needs no waiver is always preferable to one that does.

        If it does not, and the only thing standing in the way is the
        height ceiling, the search is repeated with the ceiling treated
        as waivable. That second pass does not ignore the rule: it
        minimises the relief needed, so the result carries the smallest
        height waiver that makes the corridor flyable, quantified for a
        filing under RDAC 101.035.

        The fallback is deliberately *not* attempted when prohibited
        airspace is the blocker. There is no waiver for 101.190(b), so
        relaxing anything else would only produce a route that is still
        illegal but now looks acceptable — the worst possible output.

        Parameters
        ----------
        allow_waiver_fallback : bool
            Set False to keep the strict verdict and let the caller
            decide what to do with an infeasible corridor.
        seed_with_astar : bool
            Start the search from an A* route rather than the direct
            line. A* already knows how to get round a prohibited zone;
            without it, differential evolution spends its early
            generations rediscovering that a detour is needed at all.
        astar_resolution_m : float
            Grid resolution for the seeding search. Coarse is fine — it
            only has to find the right side of the obstacle.

        Returns
        -------
        OptimizationResult whose ``compliance`` grades the outcome.
        """
        if seed_with_astar and routed_path.n_intermediate > 0:
            from .astar_baseline import AStarGridPlanner
            ac_seed = self.ac_params.with_payload(payload_kg)
            planner = AStarGridPlanner(
                self.dem, ac_seed, airspace=airspace,
                grid_resolution_m=astar_resolution_m,
            )
            if verbose:
                print("  Seeding from A* grid search...")
            routed_path.seed_from_astar(planner, verbose=verbose)

        strict = self.optimize_routed(
            routed_path, mode=mode, payload_kg=payload_kg,
            max_flight_time=max_flight_time, min_soc=min_soc,
            airspace=airspace, maxiter=maxiter, popsize=popsize,
            seed=seed, verbose=verbose, allow_height_waiver=False, **kwargs,
        )

        if strict.fully_feasible or not allow_waiver_fallback:
            return strict

        blocked = strict.compliance.blocking_zones if strict.compliance else []
        if blocked:
            if verbose:
                print(f"\n  No waiver applies: the route enters prohibited "
                      f"airspace ({', '.join(blocked)}).")
                print(f"  RDAC 101.190(b) admits no exception — the route must "
                      f"change, not the paperwork.")
            return strict

        if not strict.ceiling_feasible:
            # Attempt the fallback even when terrain clearance is also
            # violated. The relaxed pass still penalises the floor — it
            # only stops treating the ceiling as fatal — so it can fix
            # both at once. Refusing to try because a second constraint
            # is unmet leaves the corridor reported as impossible when a
            # perfectly filable route exists.
            if verbose:
                print(f"\n  No compliant profile exists on this corridor "
                      f"(ceiling exceeded by "
                      f"{strict.max_ceiling_excess_m:.0f} m).")
                print(f"  Re-planning for the smallest height waiver that "
                      f"makes it flyable...")
            relaxed = self.optimize_routed(
                routed_path, mode=mode, payload_kg=payload_kg,
                max_flight_time=max_flight_time, min_soc=min_soc,
                airspace=airspace, maxiter=maxiter, popsize=popsize,
                seed=seed, verbose=verbose, allow_height_waiver=True, **kwargs,
            )
            if verbose and not relaxed.terrain_feasible:
                print(f"\n  Terrain clearance is still violated "
                      f"({relaxed.compliance.clearance_violations} waypoint(s), "
                      f"lowest {relaxed.compliance.min_agl_m:.0f} m AGL). "
                      f"That is a safety floor, not a regulatory limit, and no "
                      f"waiver covers it — the route needs re-planning.")
            return relaxed

        if verbose and not strict.terrain_feasible:
            print(f"\n  Terrain clearance is violated "
                  f"({strict.compliance.clearance_violations} waypoint(s), "
                  f"lowest {strict.compliance.min_agl_m:.0f} m AGL). That is a "
                  f"safety floor, not a regulatory limit, and is not waivable.")
        return strict

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
