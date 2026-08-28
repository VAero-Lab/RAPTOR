"""
Regression tests for turning an optimised profile into a flight plan.

The defect these pin is the one that matters for this module: a
simplification is allowed to be worse than the flown path, and it is
allowed to give up — but it is **never** allowed to hand back a tidy
route that leaves the legal band. The first working version could, and
reported an 43 % manoeuvre reduction while doing it.

    pytest tests/test_simplify.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from raptor.simplify import min_link_profile, to_waypoint_table


def _tube(terrain, floor=60.0, ceiling=122.0, margin=2.0):
    return terrain + floor + margin, terrain + ceiling - margin


def test_min_link_stays_inside_a_flat_tube():
    """Flat ground under a wide band collapses to a single segment."""
    d = np.linspace(0.0, 10_000.0, 201)
    terr = np.full_like(d, 2800.0)
    lo, hi = _tube(terr)
    d_bp, a_bp = min_link_profile(d, lo, hi, 2880.0, 2880.0, 0.18, 0.18,
                                  fallback_alt=np.full_like(d, 2880.0))
    assert len(d_bp) == 2, "flat ground should need no intermediate breakpoint"
    a = np.interp(d, d_bp, a_bp)
    assert np.all(a >= lo - 1e-6) and np.all(a <= hi + 1e-6)


def test_min_link_never_leaves_the_tube_on_steep_ground():
    """
    The case the first version got wrong.

    A ramp steeper than the aircraft's climb limit makes the tube
    unreachable by any straight segment. The walk must fall back to a
    profile that is inside the tube, not invent one and clip it.
    """
    d = np.linspace(0.0, 4_000.0, 201)
    # 25 % ground slope against a 10 % climb limit.
    terr = 2800.0 + 0.25 * d
    lo, hi = _tube(terr)
    flown = np.clip(0.5 * (lo + hi), lo, hi)
    d_bp, a_bp = min_link_profile(d, lo, hi, float(flown[0]), float(flown[-1]),
                                  np.tan(np.radians(6.0)),
                                  np.tan(np.radians(6.0)),
                                  fallback_alt=flown)
    a = np.interp(d, d_bp, a_bp)
    assert np.all(a >= lo - 1e-6), "dipped below the floor"
    assert np.all(a <= hi + 1e-6), "climbed through the ceiling"


def test_min_link_respects_the_slope_limits_it_is_given():
    d = np.linspace(0.0, 6_000.0, 121)
    terr = 2800.0 + 40.0 * np.sin(d / 900.0)
    lo, hi = _tube(terr)
    flown = 0.5 * (lo + hi)
    tan_lim = np.tan(np.radians(8.0))
    d_bp, a_bp = min_link_profile(d, lo, hi, float(flown[0]), float(flown[-1]),
                                  tan_lim, tan_lim, fallback_alt=flown)
    slopes = np.diff(a_bp) / np.diff(d_bp)
    assert np.all(np.abs(slopes) <= tan_lim + 1e-6)


def test_min_link_reduces_the_breakpoint_count():
    """A wandering profile inside a wide band should collapse a long way."""
    d = np.linspace(0.0, 12_000.0, 401)
    terr = 2800.0 + 15.0 * np.sin(d / 300.0)
    lo, hi = _tube(terr)
    flown = 0.5 * (lo + hi)
    d_bp, _ = min_link_profile(d, lo, hi, float(flown[0]), float(flown[-1]),
                               0.18, 0.18, fallback_alt=flown)
    assert len(d_bp) < 0.25 * len(d)


def test_waypoint_table_numbers_from_one():
    """
    The numbers in the table are the numbers drawn on the figures, and
    a flight planner counts from 1.
    """
    from raptor.simplify import SimplifiedPath

    wp = np.array([[-0.21, -78.49, 2880.0, 0.0, 0.0],
                   [-0.20, -78.49, 2910.0, 60.0, 1200.0],
                   [-0.19, -78.49, 2890.0, 120.0, 2400.0]])
    sp = SimplifiedPath(
        waypoints=wp, dense_distance_m=wp[:, 4], dense_alt_m=wp[:, 2],
        dense_terrain_m=wp[:, 2] - 80.0, n_before=9, n_after=2,
        max_agl_m=80.0, min_agl_m=80.0, within_band=True,
        floor_m=60.0, ceiling_m=122.0)
    rows = to_waypoint_table(sp, name="t")
    assert [r["n"] for r in rows] == [1, 2, 3]
    assert rows[0]["alt_amsl_m"] == pytest.approx(2880.0)
    assert rows[-1]["distance_km"] == pytest.approx(2.4)
