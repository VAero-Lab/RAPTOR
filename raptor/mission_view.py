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
