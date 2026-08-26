"""
Wind — The Thing That Decides Whether a Corridor Is Actually Flyable
=====================================================================

Nothing in the package modelled wind. Over Andean terrain, in a 62 m
vertical corridor, that is not a second-order omission:

* **Range.** Ground speed is airspeed plus the along-track wind. A 6 m/s
  headwind on a 25 m/s cruise costs a quarter of the range, and a round
  trip pays it on one leg without recovering it on the other — the slow
  leg lasts longer, so it spends more energy than the fast leg saves.
* **Track.** A crosswind requires a crab angle. Fly the same heading
  without it and the aircraft drifts out of its corridor, which is a
  regulatory as well as a navigational problem when the corridor is
  drawn around prohibited airspace.
* **Terrain.** Wind over a ridge is not the wind at the airport. This
  module carries a simple terrain speed-up so a corridor squeezing
  through a saddle is not planned as if it were over flat ground.

The model is deliberately modest — a steady wind with an optional
logarithmic boundary layer and terrain speed-up. It is not a forecast
and not a CFD result. Its purpose is to make the *sensitivity* visible:
a route whose feasibility disappears in 5 m/s of wind is a different
operational proposition from one that does not, and a planner that
models no wind at all cannot tell them apart.

Author: Victor (LUAS-EPN / KU Leuven)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class WindField:
    """
    A steady wind, optionally sheared with height and sped up over terrain.

    Attributes
    ----------
    speed_ref : float
        Wind speed at the reference height [m/s].
    direction_deg : float
        Meteorological convention — the direction the wind blows *from*,
        degrees clockwise from north. A "northerly" is 0° and pushes an
        aircraft southward.
    reference_height_m : float
        Height above ground at which ``speed_ref`` applies [m AGL].
        Anemometers usually sit at 10 m; a corridor flies at 60–120 m,
        where the wind is stronger.
    roughness_length_m : float
        Surface roughness for the logarithmic profile [m]. 0.03 open
        country, 0.3 suburban, 1.0 dense urban or forest. Quito's valley
        floor is closer to the latter.
    terrain_speedup : float
        Multiplier applied where the ground rises steeply — wind
        accelerates over a ridge. 1.0 disables it.
    reference_roughness_m : float
        Surface roughness at the *measuring site*, which is usually not
        the roughness the corridor flies over. An airport anemometer
        stands on mown grass; the corridor crosses a city. Converting
        between them needs both numbers — see :meth:`speed_at`.
    terrain_speedup : float
        Multiplier applied where the ground rises steeply — wind
        accelerates over a ridge. 1.0 disables it.
    gust_factor : float
        Ratio of gust to mean, for the margin check. 1.4 is typical over
        complex terrain.
    source : str
        Where ``speed_ref`` came from. A wind default with no
        attribution is a guess wearing a number.
    """
    speed_ref: float = 0.0
    direction_deg: float = 0.0
    reference_height_m: float = 10.0
    roughness_length_m: float = 0.3
    reference_roughness_m: float = 0.03
    terrain_speedup: float = 1.0
    gust_factor: float = 1.4
    source: str = "unspecified"

    # ── Profile ───────────────────────────────────────────────────────

    def speed_at(self, agl_m: float, slope_deg: float = 0.0) -> float:
        """
        Wind speed at a height above ground [m/s].

        Logarithmic boundary layer, which is the standard neutral-stability
        profile and the reason a 10 m anemometer reading understates what a
        corridor at 90 m actually flies in — by about 40 % over suburban
        roughness.
        """
        z0 = max(self.roughness_length_m, 1e-3)
        z0_ref = max(self.reference_roughness_m, 1e-3)
        z = max(float(agl_m), z0 * 1.1)
        z_ref = max(self.reference_height_m, z0_ref * 1.1)

        if abs(z0 - z0_ref) < 1e-9:
            shear = np.log(z / z0) / np.log(z_ref / z0)
        else:
            # Two different surfaces, so one logarithmic profile will not
            # do. Go up from the anemometer using the *measuring site's*
            # roughness to a blending height above both surface layers,
            # where the flow has forgotten what it was over, then back
            # down using the corridor's roughness.
            #
            # This is not fussiness. Reading a 3.1 m/s airport
            # measurement straight up to 90 m with an urban roughness
            # gives 6.0 m/s; doing it through the blending height gives
            # 4.1 m/s. A 47 % error in wind speed is a 47 % error in
            # every crab angle and round-trip penalty computed from it.
            z_blend = max(BLENDING_HEIGHT_M, z * 1.5, z_ref * 1.5)
            up = np.log(z_blend / z0_ref) / np.log(z_ref / z0_ref)
            down = np.log(z / z0) / np.log(z_blend / z0)
            shear = up * down

        # Terrain speed-up, ramped in with slope. Crude, but it puts the
        # acceleration where the ridges are instead of nowhere.
        speedup = 1.0 + (self.terrain_speedup - 1.0) * min(
            abs(float(slope_deg)) / 30.0, 1.0)
        return float(self.speed_ref * shear * speedup)

    def components(self, agl_m: float, slope_deg: float = 0.0
                   ) -> Tuple[float, float]:
        """
        Wind vector as (north, east) components [m/s].

        Meteorological direction is where the wind comes *from*, so the
        vector points the opposite way — a frequent sign error, and one
        that silently turns every headwind into a tailwind.
        """
        v = self.speed_at(agl_m, slope_deg)
        theta = np.radians(self.direction_deg)
        return (-v * np.cos(theta), -v * np.sin(theta))

    # ── Effect on a leg ───────────────────────────────────────────────

    def resolve(self, bearing_deg: float, agl_m: float,
                slope_deg: float = 0.0) -> Tuple[float, float]:
        """
        Head/tail and cross components along a track [m/s].

        Returns ``(along, cross)``: ``along`` positive for a tailwind,
        ``cross`` positive from the right.
        """
        n, e = self.components(agl_m, slope_deg)
        brg = np.radians(bearing_deg)
        along = n * np.cos(brg) + e * np.sin(brg)
        cross = -n * np.sin(brg) + e * np.cos(brg)
        return float(along), float(cross)

    def ground_speed(self, airspeed: float, bearing_deg: float,
                     agl_m: float, slope_deg: float = 0.0) -> float:
        """
        Ground speed on a track, flying a crab angle to hold it [m/s].

        The aircraft must point partly into a crosswind to track straight,
        which spends some of its airspeed sideways. Ignoring that both
        overstates ground speed and lets the planner think it is holding
        a corridor it is drifting out of.

        Returns 0 when the wind is too strong to hold the track at all —
        the caller should treat that as infeasible rather than as slow.
        """
        along, cross = self.resolve(bearing_deg, agl_m, slope_deg)
        if abs(cross) >= airspeed:
            return 0.0
        return float(np.sqrt(airspeed ** 2 - cross ** 2) + along)

    def crab_angle_deg(self, airspeed: float, bearing_deg: float,
                       agl_m: float, slope_deg: float = 0.0) -> float:
        """Heading offset needed to hold the track [deg]."""
        _, cross = self.resolve(bearing_deg, agl_m, slope_deg)
        if abs(cross) >= airspeed:
            return float(np.sign(cross) * 90.0)
        return float(np.degrees(np.arcsin(cross / airspeed)))

    def summary(self, agl_m: float = 90.0) -> str:
        if self.speed_ref <= 0:
            return "Wind: calm (not modelled)"
        v = self.speed_at(agl_m)
        return "\n".join([
            f"Wind: {self.speed_ref:.1f} m/s from {self.direction_deg:.0f}° "
            f"at {self.reference_height_m:.0f} m",
            f"  at {agl_m:.0f} m AGL:  {v:.1f} m/s "
            f"(gusting {v*self.gust_factor:.1f})",
            f"  roughness {self.roughness_length_m:.2f} m, "
            f"terrain speed-up ×{self.terrain_speedup:.2f}",
        ])


#: Height at which the flow is taken to have forgotten the surface
#: beneath it, for converting a measurement over one roughness into a
#: wind over another. 200 m is the usual choice for near-surface work.
BLENDING_HEIGHT_M = 200.0

#: Roughness lengths [m]. Standard values; the corridor over Quito's
#: built-up basin is the 1.0 case, the airport anemometer the 0.03 one.
ROUGHNESS = {
    "water": 0.0002,
    "open country": 0.03,
    "farmland": 0.10,
    "suburban": 0.30,
    "dense urban": 1.00,
}

CALM = WindField(speed_ref=0.0, source="no wind — the baseline to "
                                       "compare everything else against")


# ═══════════════════════════════════════════════════════════════════════════
# QUITO
# ═══════════════════════════════════════════════════════════════════════════
#
# Two sources, and they disagree, which is worth recording rather than
# resolving by picking the more convenient one.
#
# **Station record.** Thirty years (1980-2012) at the old Mariscal Sucre
# airport inside the basin: annual mean 11.09 km/h = 3.08 m/s, strongest
# in August (12.9 km/h = 3.58 m/s), weakest in April (9.7 km/h =
# 2.69 m/s), most frequent direction north-north-west.
# G. Guzman et al., "Local wind in urban canyons of a residential area in
# Quito-Ecuador", MOJ Ecology & Environmental Sciences.
#
# **Reanalysis.** WeatherSpark's MERRA-2 series at 10 m gives an annual
# mean near 1.65 m/s, monthly means 1.30 m/s (April) to 2.37 m/s (July),
# and a wind from the *east* for nine months of the year, up to 92 % of
# the time in early July, turning westerly from late October to late
# January.
# https://weatherspark.com/y/20030/Average-Weather-in-Quito-Ecuador-Year-Round
#
# They agree on the season — windiest in the June-August dry months,
# calmest around April — and disagree on both magnitude (by a factor of
# 1.9) and direction (NNW against E). Neither is wrong: a reanalysis
# cell is tens of kilometres across and cannot see a valley, while a
# single station sees the flow its own valley channels.
#
# So the defaults below take the *station* magnitudes, being a real
# thirty-year measurement, and the direction is left as a parameter to
# sweep rather than a number to trust. A route whose feasibility depends
# on which of these two is right is a route that needs a met campaign,
# not a better default.

QUITO_ANNUAL_MEAN = WindField(
    speed_ref=3.08, direction_deg=340.0,
    reference_height_m=10.0, reference_roughness_m=0.03,
    roughness_length_m=ROUGHNESS["dense urban"],
    terrain_speedup=1.3, gust_factor=1.4,
    source="Mariscal Sucre station, 1980-2012 annual mean 11.09 km/h, "
           "most frequent direction NNW",
)

QUITO_WINDIEST_MONTH = WindField(
    speed_ref=3.58, direction_deg=340.0,
    reference_height_m=10.0, reference_roughness_m=0.03,
    roughness_length_m=ROUGHNESS["dense urban"],
    terrain_speedup=1.3, gust_factor=1.4,
    source="Mariscal Sucre station, August mean 12.9 km/h",
)

QUITO_CALMEST_MONTH = WindField(
    speed_ref=2.69, direction_deg=340.0,
    reference_height_m=10.0, reference_roughness_m=0.03,
    roughness_length_m=ROUGHNESS["dense urban"],
    terrain_speedup=1.3, gust_factor=1.4,
    source="Mariscal Sucre station, April mean 9.7 km/h",
)

#: The reanalysis view of the same place, for the sensitivity study.
#: Half the speed and a different direction — if a conclusion survives
#: both, it does not depend on which source is right.
QUITO_REANALYSIS = WindField(
    speed_ref=1.65, direction_deg=90.0,
    reference_height_m=10.0, reference_roughness_m=0.03,
    roughness_length_m=ROUGHNESS["dense urban"],
    terrain_speedup=1.3, gust_factor=1.4,
    source="WeatherSpark MERRA-2 at 10 m, annual mean, wind from the east",
)

#: Beaufort force to wind speed band [m/s]. The reference aircraft state
#: a Beaufort rating rather than a number, and it is a band, not a value.
BEAUFORT_MS = {
    0: (0.0, 0.2), 1: (0.3, 1.5), 2: (1.6, 3.3), 3: (3.4, 5.4),
    4: (5.5, 7.9), 5: (8.0, 10.7), 6: (10.8, 13.8), 7: (13.9, 17.1),
    8: (17.2, 20.7),
}


def beaufort_limit_ms(force: int, conservative: bool = True) -> float:
    """
    Wind speed a Beaufort rating corresponds to [m/s].

    Manufacturers quote "wind resistance level 5", which is a band from
    8.0 to 10.7 m/s, not a number. ``conservative`` takes the bottom of
    the band, which is the only defensible reading of a limit: an
    aircraft rated to survive force 5 has not been shown to survive
    10.7 m/s.
    """
    if force not in BEAUFORT_MS:
        raise KeyError(f"Beaufort force {force} not tabulated; "
                       f"known: {sorted(BEAUFORT_MS)}")
    lo, hi = BEAUFORT_MS[force]
    return float(lo if conservative else hi)


def wind_penalty_report(path, wind: WindField, uav, dem=None) -> dict:
    """
    What a wind field does to an already-planned path.

    Applied after planning rather than inside it: the point is to show
    how much margin a route has, and a route that is only feasible in
    still air is worth knowing about even if it cannot be fixed.

    Returns
    -------
    dict with the time penalty, the worst crab angle, and whether any
    segment becomes untrackable.
    """
    if wind.speed_ref <= 0:
        return {"wind": "calm", "time_penalty_s": 0.0, "feasible": True}

    still_time = 0.0
    wind_time = 0.0
    worst_crab = 0.0
    untrackable = 0
    drift_m = 0.0

    # Only fixed-wing segments are judged for trackability. A hovering or
    # transitioning aircraft does not "hold a track" in the same sense: it
    # is part rotor-borne, flying at a fraction of cruise speed, and it
    # drifts rather than failing. Judging a 10 m/s transition against a
    # 13 m/s crosswind by the fixed-wing criterion declares an ordinary
    # breezy day infeasible.
    FW = {"FW_CRUISE", "FW_CLIMB", "FW_DESCEND"}

    for seg in path.segments:
        k = seg.kinematics
        name = seg.segment_type.value

        if name not in FW:
            # Rotor-borne: report how far the wind pushes it instead.
            s = seg.start_state
            agl = 90.0
            if dem is not None:
                try:
                    terrain = dem.elevation(s.lat, s.lon)
                    if np.isfinite(terrain):
                        agl = max(0.5 * (s.alt + seg.end_state.alt)
                                  - terrain, 1.0)
                except Exception:
                    pass
            drift_m += wind.speed_at(agl) * k.duration
            still_time += k.duration
            wind_time += k.duration
            continue

        if k.ground_distance <= 0 or k.airspeed <= 0:
            still_time += k.duration
            wind_time += k.duration
            continue

        s, e = seg.start_state, seg.end_state
        agl = 90.0
        slope = 0.0
        if dem is not None:
            try:
                terrain = dem.elevation(s.lat, s.lon)
                if np.isfinite(terrain):
                    agl = max(0.5 * (s.alt + e.alt) - terrain, 1.0)
                slope = float(dem.slope_at(s.lat, s.lon))
            except Exception:
                pass

        gs = wind.ground_speed(k.airspeed, s.bearing, agl, slope)
        crab = abs(wind.crab_angle_deg(k.airspeed, s.bearing, agl, slope))
        worst_crab = max(worst_crab, crab)

        still_time += k.duration
        if gs <= 0.1:
            untrackable += 1
            wind_time += k.duration * 3.0     # nominal, it is infeasible anyway
        else:
            wind_time += k.ground_distance / gs

    return {
        "wind": wind.summary(),
        "still_air_time_s": still_time,
        "wind_time_s": wind_time,
        "time_penalty_s": wind_time - still_time,
        "time_penalty_pct": (100.0 * (wind_time - still_time)
                             / max(still_time, 1e-9)),
        "max_crab_deg": worst_crab,
        "untrackable_segments": untrackable,
        #: How far the wind pushes the aircraft during rotor-borne phases,
        #: which it must correct for on the pad rather than in cruise.
        "vtol_drift_m": drift_m,
        "feasible": untrackable == 0,
    }
