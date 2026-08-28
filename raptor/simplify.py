"""
Turning a Flown Profile into a Flight Plan
===========================================

A terrain-following leg comes out of the planner as thirty-odd climbs
and descents. On one 21 km leg the count is 35, with a median altitude
change of 17 m and a quarter of them under 10 m — the aircraft nodding
its way along the ground to stay inside a 62 m band.

That is a correct answer to the question the optimizer was asked and a
bad answer to the question an operator has. Nobody files that. Mission
Planner wants a handful of waypoints, and a pilot reviewing the plan
wants to see the shape of the flight rather than count ripples.

So this simplifies the *vertical* profile after the fact. The ground
track is untouched — lateral routing was the optimizer's job and it
already produces few waypoints — and only the altitude polyline is
redrawn with as few straight segments as will still fit inside the
corridor.

How it works
------------
The legal altitudes at each point along the leg form a tube: terrain
plus the clearance floor below, terrain plus the regulatory ceiling
above. Finding the fewest straight segments that stay inside a tube is
a classic problem with a greedy solution that is provably minimal — walk
forward from the current waypoint maintaining the cone of slopes that
remain feasible, and stop at the last station where that cone is still
open.

Two departures from the textbook version, both operational:

* the cone is clipped to the aircraft's own climb and descent angles, so
  a segment the airframe cannot fly is never proposed;
* where the cone permits it, the slope nearest to level is chosen. Given
  a free choice the aeroplane should cruise, not drift.

What it deliberately does not do
--------------------------------
It does not touch the take-off and landing funnels, which are exempt
from the corridor and are where the vertical manoeuvring is real.

And it never silently replaces the flown path. The flown path is what
the energy model integrated and what every number in the results came
from; the simplification is reported beside it, with the manoeuvre count
and whatever margin the tidying cost. A simplification that cannot stay
inside the band is reported as a failure rather than shipped — if the
terrain genuinely demands continuous tracking, that is a finding about
the corridor.

Author: Victor (LUAS-EPN / KU Leuven)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class SimplifiedPath:
    """
    A flown path redrawn with as few vertical segments as the corridor allows.

    Quacks like a ``FlightPath`` for the figures — it answers
    ``get_waypoints_array()`` — and carries the numbers that say what the
    tidying cost.

    Attributes
    ----------
    waypoints : ndarray, shape (N, 5)
        ``lat, lon, alt, time_s, distance_m`` at the kept breakpoints.
        This is the list to hand a flight planner.
    n_before, n_after : int
        Vertical manoeuvres before and after. The ratio is the headline.
    max_agl_m, min_agl_m : float
        Height above ground of the simplified profile, over the stretch
        the corridor applies to.
    within_band : bool
        Whether it stayed inside. False means the simplification failed
        and the flown path is the only legal one.
    energy_wh, energy_wh_flown : float
        Energy of the simplified and flown profiles, so the cost of
        tidying is visible rather than assumed to be zero.
    """
    waypoints: np.ndarray
    dense_distance_m: np.ndarray
    dense_alt_m: np.ndarray
    dense_terrain_m: np.ndarray
    n_before: int
    n_after: int
    max_agl_m: float
    min_agl_m: float
    within_band: bool
    floor_m: float
    ceiling_m: float
    energy_wh: float = float("nan")
    energy_wh_flown: float = float("nan")
    notes: List[str] = field(default_factory=list)

    def get_waypoints_array(self) -> np.ndarray:
        return self.waypoints

    @property
    def reduction(self) -> float:
        """Fraction of the vertical manoeuvres removed."""
        return 1.0 - self.n_after / max(self.n_before, 1)

    @property
    def energy_delta_pct(self) -> float:
        if not np.isfinite(self.energy_wh) or not np.isfinite(self.energy_wh_flown):
            return float("nan")
        return 100.0 * (self.energy_wh / max(self.energy_wh_flown, 1e-9) - 1.0)

    def summary(self) -> str:
        lines = [
            f"{self.n_before} vertical manoeuvres -> {self.n_after} "
            f"({100*self.reduction:.0f} % fewer), "
            f"{len(self.waypoints)} waypoints",
            f"  height above ground {self.min_agl_m:.0f}–{self.max_agl_m:.0f} m "
            f"against a {self.floor_m:.0f}–{self.ceiling_m:.0f} m band"
            + ("" if self.within_band else "   OUTSIDE"),
        ]
        if np.isfinite(self.energy_delta_pct):
            lines.append(f"  energy {self.energy_wh:.1f} Wh against "
                         f"{self.energy_wh_flown:.1f} Wh flown "
                         f"({self.energy_delta_pct:+.1f} %)")
        lines += [f"  {n}" for n in self.notes]
        return "\n".join(lines)


def min_link_profile(distance_m: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                     start_alt: float, end_alt: float,
                     max_climb_tan: float, max_descent_tan: float,
                     fallback_alt: Optional[np.ndarray] = None
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fewest straight segments through a tube, respecting slope limits.

    Walks forward from the current breakpoint narrowing a cone of
    feasible slopes: every station ahead contributes an upper bound from
    its ceiling and a lower bound from its floor, and the segment can be
    extended for as long as the two have not crossed. When they cross,
    the previous station is the furthest a single straight line reaches,
    and a new breakpoint starts there.

    Given a free choice within the surviving cone, the slope closest to
    level is taken — an aeroplane with no reason to climb should not.

    ``fallback_alt`` is the profile actually flown. Where the tube is
    tighter than the aircraft's own slope limits — steep ground under a
    62 m band — no straight segment fits, and the walk falls back to
    this profile for one station rather than inventing an altitude. That
    keeps the output legal by construction: every breakpoint is either
    inside the tube or a point the aircraft already flew. Without it the
    walk could hand back a tidy profile that busts the ceiling, which is
    the one outcome a simplification must never produce.

    Returns ``(breakpoint distances, breakpoint altitudes)``.
    """
    n = len(distance_m)
    if n < 2:
        return distance_m.copy(), np.array([start_alt] * n)

    d_out = [float(distance_m[0])]
    a_out = [float(np.clip(start_alt, lo[0], hi[0]))]
    i = 0

    while i < n - 1:
        s_max, s_min = max_climb_tan, -max_descent_tan
        best_j, best_slope = i + 1, None

        for j in range(i + 1, n):
            dx = distance_m[j] - d_out[-1]
            if dx <= 0:
                continue
            up = (hi[j] - a_out[-1]) / dx
            dn = (lo[j] - a_out[-1]) / dx
            s_max = min(s_max, up)
            s_min = max(s_min, dn)
            if s_min > s_max:
                break
            # Prefer level flight; otherwise the gentlest slope allowed.
            best_slope = float(np.clip(0.0, s_min, s_max))
            best_j = j

        if best_slope is None:
            # Cannot advance even one station — the tube is tighter here
            # than the aircraft's own slope limits. Take the station as
            # the aircraft actually flew it, which is inside the band by
            # construction, rather than inventing an altitude that is
            # not.
            best_j = i + 1
            if fallback_alt is not None:
                a_new = float(np.clip(fallback_alt[best_j],
                                      lo[best_j], hi[best_j]))
            else:
                a_new = float(0.5 * (lo[best_j] + hi[best_j]))
        else:
            a_new = a_out[-1] + best_slope * (
                float(distance_m[best_j]) - d_out[-1])

        d_out.append(float(distance_m[best_j]))
        a_out.append(a_new)
        i = best_j

    # The final breakpoint hands back to the landing funnel, which was
    # never simplified, so it should sit where the flown path was — but
    # only if moving it there keeps the last segment inside the tube.
    # Overwriting it unconditionally would tilt that segment and undo
    # the guarantee the whole walk just established.
    target = float(np.clip(end_alt, lo[-1], hi[-1]))
    if len(a_out) >= 2:
        d_prev, a_prev = d_out[-2], a_out[-2]
        dx = d_out[-1] - d_prev
        if dx > 0:
            k0 = int(np.searchsorted(distance_m, d_prev, side="left"))
            seg_d = distance_m[k0:]
            seg_a = a_prev + (target - a_prev) * (seg_d - d_prev) / dx
            slope = (target - a_prev) / dx
            if (np.all(seg_a >= lo[k0:] - 1e-6)
                    and np.all(seg_a <= hi[k0:] + 1e-6)
                    and -max_descent_tan - 1e-9 <= slope <= max_climb_tan + 1e-9):
                a_out[-1] = target
    return np.asarray(d_out), np.asarray(a_out)


def _band_window(agl: np.ndarray, floor: float, ceiling: float,
                 min_run: int = 8) -> Tuple[int, int]:
    """
    The stretch of a flown path the corridor rule actually governs.

    Not the fixed-wing segments — the aircraft is still climbing out
    when the first of those begins, and a simplification that starts
    there is asked to reach the band from below in one grid step, which
    no climb angle allows. The honest window is where the *flown* path
    is already inside the band, because that is the part the rule
    constrains and the only part worth redrawing.

    The take-off and landing funnels are returned untouched.
    """
    inside = (agl >= floor - 1e-6) & (agl <= ceiling + 1e-6)
    if not inside.any():
        return -1, -1

    # Longest continuous run inside the band, so a single stray sample
    # in the climb-out does not anchor the window at the pad.
    best = (0, -1, -1)
    k = 0
    while k < len(inside):
        if not inside[k]:
            k += 1
            continue
        j = k
        while j + 1 < len(inside) and inside[j + 1]:
            j += 1
        if j - k > best[0]:
            best = (j - k, k, j)
        k = j + 1
    if best[0] < min_run:
        return -1, -1
    return best[1], best[2]


def simplify_path(path, dem, constraints, uav=None, ac=None,
                  margin_m: float = 2.0) -> SimplifiedPath:
    """
    Redraw a flown path's altitude profile with as few segments as fit.

    Parameters
    ----------
    path : FlightPath or MissionPath
    dem, constraints : as everywhere else
    uav : UAVConfig, optional
        Supplies the climb and descent angles the simplification must
        respect. Without it the corridor's own limits are used.
    ac : AircraftEnergyParams, optional
        Given one, the energy of both profiles is computed so the cost of
        tidying is reported rather than assumed.
    margin_m : float
        Shrink the band by this much before simplifying, so a segment
        that grazes the ceiling in the model does not bust it in the air.
    """
    from .energy import power_fw_climb, power_fw_cruise, power_fw_descent

    arr = np.asarray(path.get_waypoints_array(), dtype=float)
    d = arr[:, 4]
    alt = arr[:, 2]
    terr = np.asarray(dem.elevation_batch(arr[:, 0], arr[:, 1]), dtype=float)
    good = np.isfinite(terr)
    terr = np.interp(d, d[good], terr[good]) if not good.all() else terr

    floor, ceiling = constraints.corridor("FW_CRUISE")
    agl_flown = alt - terr
    i0, i1 = _band_window(agl_flown, floor, ceiling)
    if i0 < 0 or i1 - i0 < 4:
        return SimplifiedPath(
            waypoints=arr.copy(), dense_distance_m=d, dense_alt_m=alt,
            dense_terrain_m=terr, n_before=0, n_after=0,
            max_agl_m=float(np.max(agl_flown)),
            min_agl_m=float(np.min(agl_flown)), within_band=False,
            floor_m=floor, ceiling_m=ceiling,
            notes=["the flown path is never inside the band for long "
                   "enough to simplify"])

    lo = terr[i0:i1 + 1] + floor + margin_m
    hi = terr[i0:i1 + 1] + ceiling - margin_m
    # A tube can invert where the band is thinner than twice the margin;
    # open it back up rather than handing the search an impossible
    # constraint it would answer with a step at every station.
    bad = lo >= hi
    if bad.any():
        mid = 0.5 * (lo + hi)
        lo = np.where(bad, mid - 1.0, lo)
        hi = np.where(bad, mid + 1.0, hi)

    if uav is not None:
        tc = np.tan(np.radians(uav.fw_max_climb_angle))
        td = np.tan(np.radians(uav.fw_max_descent_angle))
    else:
        tc = td = np.tan(np.radians(10.0))

    d_bp, a_bp = min_link_profile(d[i0:i1 + 1], lo, hi,
                                  float(alt[i0]), float(alt[i1]), tc, td,
                                  fallback_alt=alt[i0:i1 + 1])

    # Re-check on the dense profile: a straight segment between two legal
    # breakpoints can still leave the band in between if the tube bends.
    a_dense = np.interp(d[i0:i1 + 1], d_bp, a_bp)
    agl = a_dense - terr[i0:i1 + 1]
    within = bool(np.all(agl >= floor - 1e-6) and np.all(agl <= ceiling + 1e-6))

    # Stitch the simplified cruise back between the untouched funnels.
    keep_lat = np.interp(d_bp, d, arr[:, 0])
    keep_lon = np.interp(d_bp, d, arr[:, 1])
    keep_t = np.interp(d_bp, d, arr[:, 3])
    mid = np.column_stack([keep_lat, keep_lon, a_bp, keep_t, d_bp])
    # The funnels are kept as flown — one waypoint each, since a flight
    # planner is given a take-off and a landing, not a sampled climb.
    pieces = [arr[:1]] if i0 > 0 else []
    pieces.append(mid)
    if i1 < len(arr) - 1:
        pieces.append(arr[-1:])
    waypoints = np.vstack(pieces)

    n_before = sum(1 for s in getattr(path, "segments", [])
                   if s.segment_type.value in ("FW_CLIMB", "FW_DESCEND"))
    n_after = int(np.sum(np.abs(np.diff(a_bp)) > 3.0))

    out = SimplifiedPath(
        waypoints=waypoints,
        dense_distance_m=d[i0:i1 + 1], dense_alt_m=a_dense,
        dense_terrain_m=terr[i0:i1 + 1],
        n_before=n_before, n_after=n_after,
        max_agl_m=float(np.max(agl)), min_agl_m=float(np.min(agl)),
        within_band=within, floor_m=floor, ceiling_m=ceiling,
    )
    if not within:
        out.notes.append(
            f"leaves the band by "
            f"{max(np.max(agl) - ceiling, floor - np.min(agl)):.0f} m — the "
            f"terrain here demands continuous tracking, and the flown "
            f"path is the only legal one")

    if ac is not None and uav is not None:
        out.energy_wh = _profile_energy(d_bp, a_bp, uav, ac,
                                        power_fw_climb, power_fw_cruise,
                                        power_fw_descent)
        out.energy_wh_flown = _profile_energy(
            d[i0:i1 + 1], alt[i0:i1 + 1], uav, ac,
            power_fw_climb, power_fw_cruise, power_fw_descent)
    return out


def _profile_energy(d, a, uav, ac, p_climb, p_cruise, p_desc) -> float:
    """
    Energy of a fixed-wing altitude profile [Wh].

    Only the cruise portion, and only for comparing one profile against
    another over the same ground — the VTOL phases are identical between
    them and would drown the difference being measured.
    """
    V = float(uav.fw_cruise_airspeed)
    total = 0.0
    for k in range(len(d) - 1):
        dx = float(d[k + 1] - d[k])
        dz = float(a[k + 1] - a[k])
        if dx <= 0:
            continue
        alt_mid = 0.5 * (a[k] + a[k + 1])
        gamma = np.degrees(np.arctan2(dz, dx))
        if dz > 3.0:
            P = p_climb(ac, V, gamma, alt_mid) / ac.eta_cruise_effective
        elif dz < -3.0:
            P = p_desc(ac, V, abs(gamma), alt_mid) / ac.eta_cruise_effective
        else:
            P = p_cruise(ac, V, alt_mid) / ac.eta_cruise_effective
        total += P * (np.hypot(dx, dz) / V) / 3600.0
    return float(total)


def to_waypoint_table(simplified: SimplifiedPath, name: str = "route"
                      ) -> List[dict]:
    """
    The waypoint list, numbered, ready for a flight planner.

    Numbers start at 1 and are the same numbers the figures draw, so a
    point picked off a map can be found in the table without counting.
    """
    out = []
    wp = simplified.waypoints
    terr = np.interp(wp[:, 4], simplified.dense_distance_m,
                     simplified.dense_terrain_m)
    for k, row in enumerate(wp):
        out.append(dict(
            n=k + 1, name=f"{name}_{k+1:02d}",
            lat=round(float(row[0]), 7), lon=round(float(row[1]), 7),
            alt_amsl_m=round(float(row[2]), 1),
            agl_m=round(float(row[2] - terr[k]), 1),
            distance_km=round(float(row[4]) / 1000.0, 3),
        ))
    return out
