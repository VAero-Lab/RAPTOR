"""
Presenting a Multi-Leg Mission as One Flight
=============================================

Every profile figure in this package plots something against distance
flown, and every one of them reads that distance off a
``get_waypoints_array``. A mission is not one of those: it is a list of
legs, each with its own array starting at zero.

Concatenating them by hand at each call site is how two figures end up
disagreeing about where leg 3 starts. This does it once.

The ground time between legs is deliberately *not* inserted into the
distance axis — an aircraft on a pad covers no ground — but it is
recorded, so a figure plotting against time can put it back.

Author: Victor (LUAS-EPN / KU Leuven)
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Sequence

import numpy as np


class MissionPath:
    """
    Several legs presented as one path.

    Quacks like a ``FlightPath`` for the two things the figures ask of
    one — ``get_waypoints_array()`` and ``segments`` — with distance and
    time made cumulative across legs.

    Attributes
    ----------
    leg_bounds : list of (start_km, end_km)
        Where each leg begins and ends on the concatenated distance
        axis, so a figure can mark the stops.
    """

    def __init__(self, leg_paths: Sequence, leg_names: Sequence[str] = None,
                 ground_time_s: Sequence[float] = None):
        if not leg_paths:
            raise ValueError("a mission needs at least one leg")
        self.legs = list(leg_paths)
        self.leg_names = list(leg_names or
                              [f"leg {i+1}" for i in range(len(self.legs))])
        self.ground_time_s = list(ground_time_s or [0.0] * len(self.legs))

        arrays, d0, t0 = [], 0.0, 0.0
        self.leg_bounds: List[tuple] = []
        for i, p in enumerate(self.legs):
            a = np.asarray(p.get_waypoints_array(), dtype=float).copy()
            a[:, 3] += t0
            a[:, 4] += d0
            arrays.append(a)
            self.leg_bounds.append((d0 / 1000.0, a[-1, 4] / 1000.0))
            d0 = a[-1, 4]
            # Ground time advances the clock and not the odometer.
            t0 = a[-1, 3] + (self.ground_time_s[i]
                             if i < len(self.ground_time_s) else 0.0)
        self._array = np.vstack(arrays)

        self.segments = [s for p in self.legs
                         for s in getattr(p, "segments", [])]

    def get_waypoints_array(self) -> np.ndarray:
        return self._array

    @property
    def metrics(self):
        """
        Path metrics for the concatenated mission.

        Present because the checks that run on a finished path — the
        envelope check, the range limit — read ``path.metrics``, and a
        mission that could not answer them would silently skip them.
        """
        from .path import PathMetrics
        arr = self._array
        dz = np.diff(arr[:, 2])
        seg = [s.segment_type for s in self.segments]
        return PathMetrics(
            total_time=float(arr[-1, 3]),
            total_ground_distance=float(arr[-1, 4]),
            total_altitude_gain=float(dz[dz > 0].sum()),
            total_altitude_loss=float(-dz[dz < 0].sum()),
            max_altitude=float(arr[:, 2].max()),
            min_altitude=float(arr[:, 2].min()),
            n_segments=len(self.segments),
            n_vtol_segments=sum(1 for s in seg if "VTOL" in str(s)),
            n_fw_segments=sum(1 for s in seg if "FW" in str(s)),
            segment_summary={},
        )

    @property
    def total_distance_m(self) -> float:
        return float(self._array[-1, 4])

    @property
    def total_time_s(self) -> float:
        return float(self._array[-1, 3])

    def __len__(self) -> int:
        return len(self.legs)


def stitch_energy(leg_results):
    """
    One mission-wide energy result from a sequence of per-leg results.

    Why this exists rather than re-analysing the concatenated path:
    re-analysis has to be told *a* payload, and a mission does not have
    one. An out-and-back carries the cargo out and comes home empty; a
    supply tour sheds a box at every stop. Re-analysing the whole path
    at the departure payload charges the aircraft for cargo it already
    delivered, and the figure then disagrees with the planner's own
    headline number — by 8 % on a two-leg round trip, which is more than
    most of the effects the figures exist to show.

    The planner already analysed each leg at that leg's payload and with
    that leg's starting state of charge. Concatenating those results
    keeps the figures and the numbers the same arithmetic.

    Parameters
    ----------
    leg_results : sequence of LegResult
        In flight order.

    Returns
    -------
    MissionEnergyResult
        Timeline and segments in mission time; scalars summed, or taken
        from the last leg where that is what they mean.
    """
    from .energy import MissionEnergyResult

    segments, timeline = [], []
    t_offset = 0.0
    wh_offset = 0.0
    mah_offset = 0.0
    for lr in leg_results:
        er = lr.energy_result
        segments.extend(er.segments)
        for st in er.battery_timeline:
            timeline.append(replace(st, time=st.time + t_offset,
                                    energy_consumed_wh=st.energy_consumed_wh
                                    + wh_offset,
                                    mah_consumed=st.mah_consumed + mah_offset))
        t_offset += er.total_time
        wh_offset += er.total_energy_wh
        mah_offset += er.total_mah_consumed

    last = leg_results[-1].energy_result
    return MissionEnergyResult(
        segments=segments,
        total_energy_wh=sum(lr.energy_result.total_energy_wh
                            for lr in leg_results),
        total_energy_j=sum(lr.energy_result.total_energy_j
                           for lr in leg_results),
        total_mah_consumed=mah_offset,
        total_time=t_offset,
        SOC_final=last.SOC_final,
        percent_remaining=last.percent_remaining,
        mah_remaining=last.mah_remaining,
        feasible=all(lr.energy_result.feasible for lr in leg_results),
        min_SOC=min(lr.energy_result.min_SOC for lr in leg_results),
        battery_timeline=timeline,
        energy_from_cells_wh=sum(lr.energy_result.energy_from_cells_wh
                                 for lr in leg_results),
        ohmic_loss_wh=sum(lr.energy_result.ohmic_loss_wh
                          for lr in leg_results),
        max_c_rate=max(lr.energy_result.max_c_rate for lr in leg_results),
        c_rate_exceeded=any(lr.energy_result.c_rate_exceeded
                            for lr in leg_results),
        power_limited=any(lr.energy_result.power_limited
                          for lr in leg_results),
    )
