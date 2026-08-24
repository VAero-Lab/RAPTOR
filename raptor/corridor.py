"""
Terrain-Following Corridor — Flyable Reference Profiles Over Real Relief
========================================================================

A constant-altitude (AMSL) cruise leg cannot obey an AGL rule over
anything but a plain. Hold 2 950 m AMSL across a corridor whose ground
runs from 2 429 m to 2 904 m, and the height above ground swings from
46 m to 521 m: the aircraft is simultaneously too low over the ridge and
several hundred metres above the 122 m ceiling over the valley. No choice
of that single altitude fixes it, and no amount of lateral re-routing
helps, because the problem is vertical.

This module builds the reference the cruise must actually follow: the
lowest profile that stays a chosen height above the ground *and* that the
aircraft can physically fly.

The construction
----------------
1. Sample terrain along the leg.
2. **Cone dilation.** Sweep forward limiting how fast the reference may
   descend, then backward limiting how fast it may climb. The result is
   the pointwise-lowest profile that never dips below the terrain and
   never demands a flight-path angle the aircraft does not have. Two
   O(n) passes; it is the standard morphological envelope.
3. Offset by the target AGL.
4. Simplify to a handful of breakpoints, so the leg becomes a short
   sequence of climb/cruise/descend segments rather than hundreds of
   micro-steps the energy model would have to integrate.

Where terrain falls away faster than the aircraft can descend, the
envelope stays above ``target_agl``. That residual is reported rather
than hidden: it is a genuine conflict between the relief and the
regulatory ceiling, and it is exactly what a feasibility study of drone
corridors over Andean terrain needs to quantify.

Author: Victor (LUAS-EPN / KU Leuven)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class CorridorProfile:
    """
    A flyable reference profile along one leg.

    Attributes
    ----------
    distances : ndarray
        Along-track distance of each breakpoint [m].
    altitudes : ndarray
        Reference altitude at each breakpoint [m AMSL].
    sample_distances, sample_altitudes, sample_terrain : ndarray
        The dense pre-simplification profile, for plotting and analysis.
    agl : ndarray
        Height above ground at the dense samples [m].
    target_agl : float
        The AGL the leg was asked to hold [m].
    max_agl : float
        Regulatory ceiling used when scoring the profile [m AGL].
    ceiling_excess_m : float
        Worst exceedance of ``max_agl`` forced by the relief [m]. Zero
        means the whole leg fits inside the corridor.
    ceiling_excess_fraction : float
        Fraction of the leg above ``max_agl``.
    infeasible_relief : bool
        True when the terrain is simply too steep for the corridor —
        a finding about the route, not a failure of the optimizer.
    """
    distances: np.ndarray
    altitudes: np.ndarray
    sample_distances: np.ndarray
    sample_altitudes: np.ndarray
    sample_terrain: np.ndarray
    agl: np.ndarray
    target_agl: float
    max_agl: float
    ceiling_excess_m: float = 0.0
    ceiling_excess_fraction: float = 0.0
    infeasible_relief: bool = False

    @property
    def n_breakpoints(self) -> int:
        return len(self.distances)

    @property
    def start_alt(self) -> float:
        return float(self.altitudes[0])

    @property
    def end_alt(self) -> float:
        return float(self.altitudes[-1])

    @property
    def length_m(self) -> float:
        return float(self.distances[-1] - self.distances[0])


# ═══════════════════════════════════════════════════════════════════════════
# ENVELOPE CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

def climb_limited_envelope(
    terrain: np.ndarray,
    step_m: float,
    max_climb_slope: float,
    max_descent_slope: float,
) -> np.ndarray:
    """
    Lowest profile above ``terrain`` that the aircraft can actually fly.

    Parameters
    ----------
    terrain : ndarray
        Ground elevation at evenly spaced samples [m].
    step_m : float
        Spacing between samples [m].
    max_climb_slope, max_descent_slope : float
        tan(gamma) limits — how much altitude may be gained or lost per
        metre of ground track.

    Returns
    -------
    ndarray, same shape as ``terrain``.

    Notes
    -----
    Two sweeps. Forward, ``env[i] >= env[i-1] - step*tan(descent)`` caps
    how fast the profile may sink; backward, ``env[i] >= env[i+1] -
    step*tan(climb)`` caps how fast it may rise. Together they give the
    pointwise minimum of all admissible profiles, so no flyable reference
    sits lower anywhere.
    """
    env = np.array(terrain, dtype=float, copy=True)
    n = len(env)
    if n < 2:
        return env

    down = step_m * max_descent_slope
    up = step_m * max_climb_slope
    i = np.arange(n, dtype=float)

    # Both sweeps are running maxima with a constant drift, which
    # `np.maximum.accumulate` does exactly once the drift is folded into
    # the values. Writing them as Python loops costs a few thousand
    # interpreter steps per leg, and this runs inside every objective
    # evaluation of every generation — on a 30 m DEM it was most of the
    # per-evaluation time.
    #
    #   forward:  env[i] >= env[i-1] - down
    #             let f = env + i*down; then f[i] >= f[i-1]
    #   backward: env[i] >= env[i+1] - up
    #             let g = env + (n-1-i)*up; then g[i] >= g[i+1]
    env = np.maximum.accumulate(env + i * down) - i * down
    rev = (n - 1 - i) * up
    env = np.maximum.accumulate((env + rev)[::-1])[::-1] - rev
    return env


def _simplify(distances: np.ndarray, altitudes: np.ndarray,
              tol_m: float, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduce a dense profile to breakpoints within ``tol_m`` vertically.

    Ramer–Douglas–Peucker on the (distance, altitude) polyline, measuring
    error vertically rather than perpendicularly — the vertical error is
    what the clearance and ceiling checks care about. If the result still
    exceeds ``max_points``, the tolerance is relaxed until it fits, so the
    number of flight segments stays bounded whatever the terrain.
    """
    n = len(distances)
    if n <= 2:
        return distances.copy(), altitudes.copy()

    def rdp(i0: int, i1: int, tol: float, keep: List[int]):
        if i1 <= i0 + 1:
            return
        d0, d1 = distances[i0], distances[i1]
        a0, a1 = altitudes[i0], altitudes[i1]
        span = d1 - d0
        if span <= 0:
            return
        t = (distances[i0:i1 + 1] - d0) / span
        interp = a0 + t * (a1 - a0)
        err = np.abs(altitudes[i0:i1 + 1] - interp)
        k = int(np.argmax(err))
        if err[k] <= tol:
            return
        k += i0
        keep.append(k)
        rdp(i0, k, tol, keep)
        rdp(k, i1, tol, keep)

    tol = tol_m
    for _ in range(12):
        keep: List[int] = [0, n - 1]
        rdp(0, n - 1, tol, keep)
        idx = np.array(sorted(set(keep)))
        if len(idx) <= max_points:
            break
        tol *= 1.6
    return distances[idx], altitudes[idx]


def build_corridor_profile(
    dem,
    start: Tuple[float, float],
    end: Tuple[float, float],
    target_agl: float,
    max_climb_angle_deg: float,
    max_descent_angle_deg: float,
    max_agl: float = 122.0,
    sample_spacing_m: float = None,
    simplify_tol_m: float = 12.0,
    max_breakpoints: int = 12,
    angle_margin: float = 0.75,
) -> CorridorProfile:
    """
    Build a flyable terrain-following reference for one leg.

    Parameters
    ----------
    dem : DEMInterface
        Terrain model.
    start, end : (lat, lon)
        Leg endpoints.
    target_agl : float
        Height above ground the leg should hold [m].
    max_climb_angle_deg, max_descent_angle_deg : float
        Aircraft flight-path-angle limits [deg].
    max_agl : float
        Regulatory ceiling used to score the result [m AGL].
    sample_spacing_m : float or None
        Terrain sampling interval [m]. ``None`` derives it from the DEM's
        own cell size (half a cell), which is what you want: sampling at
        or above the cell size steps over ridges between samples, and the
        envelope then clears terrain only at the points it looked at. On
        the bundled ~186 m DMQ grid, 200 m sampling leaves the reference
        profile dipping ~30 m below its target height on real corridors.
    simplify_tol_m : float
        Vertical tolerance when reducing the profile to breakpoints [m].
        Must stay well inside the corridor height.
    max_breakpoints : int
        Cap on breakpoints, bounding the number of flight segments.
    angle_margin : float
        Fraction of the aircraft's limit angles used for the envelope,
        leaving authority for disturbance rejection and altitude hold.

    Returns
    -------
    CorridorProfile
    """
    from .dem import DEMInterface

    if sample_spacing_m is None:
        meta = getattr(dem, 'metadata', None)
        cell = min(abs(getattr(meta, 'dlat_m', 200.0)),
                   abs(getattr(meta, 'dlon_m', 200.0))) if meta else 200.0
        # Well below the cell size, not merely at it. The DEM interpolates
        # bilinearly, so along a transect that crosses the grid diagonally
        # the ground between two samples is not the straight line between
        # them: a crest can sit above both. Sampling at the cell size left
        # the reference passing ~17 m lower than asked over such crests on
        # real corridors; a tenth of a cell brings that inside a metre.
        # The envelope is two linear sweeps over a vectorised DEM query,
        # so the extra samples cost about 0.3 ms per leg.
        sample_spacing_m = float(np.clip(cell * 0.1, 15.0, 25.0))

    dist = DEMInterface.haversine(start[0], start[1], end[0], end[1])
    n = int(max(dist / max(sample_spacing_m, 1.0), 2)) + 1
    n = int(np.clip(n, 3, 1200))

    profile = dem.terrain_profile(start, end, n=n)
    d = np.asarray(profile['distances'], dtype=float)
    terr = np.asarray(profile['elevations'], dtype=float)

    # Outside-DEM gaps: interpolate across rather than letting NaN
    # propagate into the envelope and wipe out the leg.
    if np.isnan(terr).any():
        good = ~np.isnan(terr)
        if not good.any():
            terr = np.zeros_like(terr)
        else:
            terr = np.interp(d, d[good], terr[good])

    step = float(d[1] - d[0]) if n > 1 else max(dist, 1.0)
    tan_climb = np.tan(np.radians(max_climb_angle_deg * angle_margin))
    tan_desc = np.tan(np.radians(max_descent_angle_deg * angle_margin))

    env = climb_limited_envelope(terr, step, tan_climb, tan_desc)
    ref = env + float(target_agl)

    agl = ref - terr
    excess = np.maximum(agl - max_agl, 0.0)

    bp_d, bp_a = _simplify(d, ref, simplify_tol_m, max_breakpoints)

    return CorridorProfile(
        distances=bp_d,
        altitudes=bp_a,
        sample_distances=d,
        sample_altitudes=ref,
        sample_terrain=terr,
        agl=agl,
        target_agl=float(target_agl),
        max_agl=float(max_agl),
        ceiling_excess_m=float(np.max(excess)) if len(excess) else 0.0,
        ceiling_excess_fraction=float(np.mean(excess > 0)) if len(excess) else 0.0,
        infeasible_relief=bool(np.any(excess > 0)),
    )


# ═══════════════════════════════════════════════════════════════════════════
# CORRIDOR SURVEY — is this route flyable at all?
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CorridorSurvey:
    """Verdict on whether a straight corridor admits a legal profile."""
    direct_distance_m: float
    terrain_min_m: float
    terrain_max_m: float
    relief_m: float
    best_target_agl: float
    ceiling_excess_m: float
    ceiling_excess_fraction: float
    feasible: bool
    note: str = ""

    def summary(self) -> str:
        verdict = "FITS" if self.feasible else "DOES NOT FIT"
        lines = [
            f"Corridor {verdict} inside the vertical band",
            f"  Direct distance:     {self.direct_distance_m/1000:.2f} km",
            f"  Terrain:             {self.terrain_min_m:.0f} – "
            f"{self.terrain_max_m:.0f} m AMSL (relief {self.relief_m:.0f} m)",
            f"  Best cruise height:  {self.best_target_agl:.0f} m AGL",
        ]
        if not self.feasible:
            lines += [
                f"  Ceiling exceeded by: {self.ceiling_excess_m:.0f} m over "
                f"{self.ceiling_excess_fraction*100:.1f}% of the leg",
            ]
        if self.note:
            lines.append(f"  {self.note}")
        return "\n".join(lines)


def survey_corridor(
    dem,
    start: Tuple[float, float],
    end: Tuple[float, float],
    constraints,
    uav,
    n_trials: int = 9,
) -> CorridorSurvey:
    """
    Ask whether *any* terrain-following height makes this leg legal.

    Sweeps candidate cruise heights across the corridor band and keeps
    the one with the least ceiling exceedance. A negative answer is
    informative: it says the relief along this straight line beats the
    aircraft's descent rate, so the leg needs a lateral detour, a waiver
    under RDAC 101.035, or a different pair of endpoints.
    """
    floor, ceiling = constraints.corridor("FW_CRUISE")
    if ceiling <= floor:
        return CorridorSurvey(
            direct_distance_m=0.0, terrain_min_m=np.nan, terrain_max_m=np.nan,
            relief_m=np.nan, best_target_agl=np.nan,
            ceiling_excess_m=np.inf, ceiling_excess_fraction=1.0,
            feasible=False,
            note="Constraint set has an empty corridor; fix MissionConstraints first.",
        )

    best: Optional[CorridorProfile] = None
    for target in np.linspace(floor, ceiling * 0.9, n_trials):
        prof = build_corridor_profile(
            dem, start, end, target,
            uav.fw_max_climb_angle, uav.fw_max_descent_angle,
            max_agl=constraints.max_agl,
        )
        if best is None or prof.ceiling_excess_m < best.ceiling_excess_m:
            best = prof
        if prof.ceiling_excess_m <= 0:
            break

    terr = best.sample_terrain
    from .dem import DEMInterface
    return CorridorSurvey(
        direct_distance_m=DEMInterface.haversine(start[0], start[1], end[0], end[1]),
        terrain_min_m=float(np.min(terr)),
        terrain_max_m=float(np.max(terr)),
        relief_m=float(np.max(terr) - np.min(terr)),
        best_target_agl=best.target_agl,
        ceiling_excess_m=best.ceiling_excess_m,
        ceiling_excess_fraction=best.ceiling_excess_fraction,
        feasible=best.ceiling_excess_m <= 0.0,
        note=("Relief along this line outruns the aircraft's descent capability."
              if best.ceiling_excess_m > 0 else ""),
    )
