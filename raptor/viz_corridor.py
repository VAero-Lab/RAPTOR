"""
Corridor Figures — The Plot the Argument Rests On
=================================================

The existing figures show a path over terrain. That was the right picture
when a route was a line at an altitude. It is the wrong one now, because
the quantity that decides whether a route is legal is not its altitude —
it is its height above the ground, and the band it has to stay inside.

Three figures live here:

:func:`plot_corridor`
    Terrain, the AGL band drawn as a ribbon that follows the ground, the
    flown profile inside it, and the stretches where relief forces the
    aircraft out. This is the paper's central figure.

:func:`plot_permission_map`
    Airspace coloured by *permission* — free, permit, prohibited — rather
    than by zone type. Type is the cartographic convention; permission is
    what the optimizer reasons about, and colouring by type hides the
    distinction the whole method turns on.

:func:`plot_waiver_map`
    Where on a network height relief is needed and how much. A
    regulator-facing view.

Author: Victor (LUAS-EPN / KU Leuven)
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

# Palette shared with the rest of the package's figures.
C_TERRAIN = "#8D6E63"
C_TERRAIN_FILL = "#D7CCC8"
C_BAND = "#90A4AE"
C_PATH = "#1565C0"
C_BUST = "#C62828"
C_FLOOR = "#EF6C00"
C_FREE = "#2E7D32"
C_PERMIT = "#F9A825"
C_PROHIBITED = "#C62828"
#: The simplified, flight-plannable path — distinct from every route
#: colour so it never reads as one more method.
C_SIMPLE = "#212121"


def _mpl():
    import matplotlib.pyplot as plt
    return plt


def _sample_path(path, dem):
    """
    Waypoints of a path with terrain and AGL attached.

    Reads ``get_waypoints_array()`` rather than ``get_waypoints()``.
    Both ``FlightPath`` and ``AStarResult`` provide the array form —
    that is the interface the A* result was written to satisfy — but
    only ``FlightPath`` has the object form, so a function reaching for
    the latter silently accepts one planner's output and rejects the
    other's, which is exactly the comparison these figures exist to
    make.
    """
    arr = np.asarray(path.get_waypoints_array(), dtype=float)
    lats, lons, alts = arr[:, 0], arr[:, 1], arr[:, 2]
    dist = arr[:, 4] / 1000.0
    terrain = np.asarray(dem.elevation_batch(lats, lons), dtype=float)
    return dist, alts, terrain, lats, lons


def plot_corridor(path, dem, constraints, ax=None, title: str = None,
                  show_band: bool = True, annotate_busts: bool = True):
    """
    Terrain, the legal AGL band, and the flown profile inside it.

    The band is drawn as a ribbon offset from the ground rather than as
    two horizontal lines, because that is what the rule actually is. Draw
    it as horizontal lines and a reader concludes the aircraft is
    comfortably inside a wide corridor; draw it following the terrain and
    they can see it is threading a 62 m gap that moves under them.

    Parameters
    ----------
    path : FlightPath
    dem : DEMInterface
    constraints : MissionConstraints
        Supplies the clearance floor and the regulatory ceiling.
    ax : matplotlib axis, optional
    show_band : bool
        Shade the legal band. Off gives a cleaner comparison figure.
    annotate_busts : bool
        Mark the stretches above the ceiling.
    """
    plt = _mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3.6))

    dist, alts, terrain, _, _ = _sample_path(path, dem)
    good = np.isfinite(terrain)
    floor, ceiling = constraints.corridor("FW_CRUISE")

    # Frame the interesting band, not the whole column of air down to sea
    # level. `fill_between(x, y)` fills to zero by default, which drags the
    # y-limits to 0 and squeezes a 62 m corridor over 2 500 m of terrain
    # into invisibility — the figure then shows exactly the thing it was
    # drawn to show, too small to see.
    lo = float(np.nanmin(terrain[good]))
    hi = max(float(np.nanmax(alts)),
             float(np.nanmax(terrain[good])) + ceiling)
    pad = 0.12 * max(hi - lo, 50.0)
    y_bottom = lo - pad

    ax.fill_between(dist[good], terrain[good], y_bottom,
                    color=C_TERRAIN_FILL, zorder=1, lw=0)
    ax.plot(dist[good], terrain[good], color=C_TERRAIN, lw=1.0,
            zorder=2, label="terrain")

    if show_band:
        ax.fill_between(dist[good], terrain[good] + floor,
                        terrain[good] + ceiling, color=C_BAND, alpha=0.30,
                        zorder=2, lw=0,
                        label=f"legal band, {floor:.0f}–{ceiling:.0f} m AGL")
        ax.plot(dist[good], terrain[good] + ceiling, color=C_BUST, lw=0.9,
                ls="--", alpha=0.75, zorder=3,
                label=f"{ceiling:.0f} m AGL ceiling")
        ax.plot(dist[good], terrain[good] + floor, color=C_FLOOR, lw=0.8,
                ls=":", alpha=0.7, zorder=3)

    ax.plot(dist, alts, color=C_PATH, lw=1.9, zorder=5, label="flown profile")

    if annotate_busts:
        agl = alts - terrain
        over = good & (agl > ceiling)
        if over.any():
            ax.plot(dist[over], alts[over], ".", color=C_BUST, ms=3.5,
                    zorder=6,
                    label=f"above ceiling ({100*over.mean():.0f} % of route)")

    ax.set_xlabel("along-track distance [km]")
    ax.set_ylabel("altitude [m AMSL]")
    ax.set_xlim(dist.min(), dist.max())
    ax.set_ylim(y_bottom, hi + pad)
    ax.spines[["top", "right"]].set_visible(False)
    # Below the axes: at this aspect ratio the profile fills the frame, and
    # a legend inside it covers the data it describes.
    ax.legend(fontsize=7, framealpha=0.92, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.24))
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")
    return ax


def plot_agl_profile(path, dem, constraints, ax=None, title: str = None):
    """
    The same information in the frame the rule is written in.

    Plotting height above ground directly makes the band two horizontal
    lines and the question trivial to read: is the blue line between
    them? The AMSL view shows where the aircraft is; this one shows
    whether it is allowed to be there.
    """
    plt = _mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 2.6))

    dist, alts, terrain, _, _ = _sample_path(path, dem)
    good = np.isfinite(terrain)
    agl = alts - terrain
    floor, ceiling = constraints.corridor("FW_CRUISE")

    ax.axhspan(floor, ceiling, color=C_BAND, alpha=0.30, lw=0,
               label=f"legal band {floor:.0f}–{ceiling:.0f} m")
    ax.axhline(ceiling, color=C_BUST, lw=1.0, ls="--", alpha=0.8)
    ax.axhline(floor, color=C_FLOOR, lw=0.9, ls=":", alpha=0.8)
    ax.axhline(0.0, color=C_TERRAIN, lw=1.0)

    ax.plot(dist[good], agl[good], color=C_PATH, lw=1.8, label="height AGL")
    over = good & (agl > ceiling)
    under = good & (agl < floor) & (agl > 1.0)
    if over.any():
        ax.plot(dist[over], agl[over], ".", color=C_BUST, ms=3.5,
                label="above ceiling")
    if under.any():
        ax.plot(dist[under], agl[under], ".", color=C_FLOOR, ms=3.5,
                label="below floor")

    ax.set_xlabel("along-track distance [km]")
    ax.set_ylabel("height AGL [m]")
    ax.set_xlim(dist.min(), dist.max())
    ax.set_ylim(-10, max(ceiling * 1.8, float(np.nanmax(agl[good])) * 1.1))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, framealpha=0.92, ncol=4,
              loc="upper center", bbox_to_anchor=(0.5, -0.28))
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")
    return ax


def plot_corridor_comparison(paths, labels, dem, constraints,
                             title: str = None, figsize=(9.5, 7.5)):
    """
    Several profiles in the AGL frame, stacked for comparison.

    Built for the one comparison the package exists to make: a
    constant-altitude cruise against a terrain-following one over the
    same ground.
    """
    plt = _mpl()
    fig, axes = plt.subplots(len(paths), 1, figsize=figsize, sharex=True)
    if len(paths) == 1:
        axes = [axes]
    for ax, p, label in zip(axes, paths, labels):
        plot_agl_profile(p, dem, constraints, ax=ax, title=label)
        ax.set_xlabel("")
    axes[-1].set_xlabel("along-track distance [km]")
    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig



#: Style for the numbered waypoints a simplified path carries. The same
#: number means the same point in every view — map, altitude, height
#: above ground, lateral offset and the three-dimensional plot — so a
#: point picked off one panel can be found in the others without
#: counting along the line.
WP_STYLE = dict(marker="o", ms=11, mfc="white", mew=1.4, zorder=11)
WP_TEXT = dict(ha="center", va="center", fontsize=6.5, fontweight="bold",
               zorder=12)


def draw_numbered(ax, x, y, colour="#212121", every: int = 1,
                  skip_ends: bool = False):
    """
    Draw numbered waypoint markers on any axis.

    Returns the number of markers drawn. Numbers start at 1 and count
    the waypoints of the simplified path, which is the list a flight
    planner would be handed — so the numbers on the figure and the rows
    of the exported table are the same numbers.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    idx = range(len(x))
    if skip_ends and len(x) > 2:
        idx = range(1, len(x) - 1)
    n = 0
    for k in idx:
        if k % every:
            continue
        ax.plot(x[k], y[k], mec=colour, **WP_STYLE)
        ax.annotate(str(k + 1), (x[k], y[k]), color=colour, **WP_TEXT)
        n += 1
    return n


def plot_altitude_profile(paths, labels, dem, ax=None, title: str = None,
                          show_terrain: bool = True, show_band=None,
                          route_legend: bool = True, simplified=None,
                          number_waypoints: bool = True):
    """
    Altitude above sea level against distance flown — the side view.

    The companion the AGL panel cannot replace. Height above ground
    answers the regulatory question and is the *wrong* frame for
    picturing a flight: a constant-altitude cruise appears in it as an
    inverted terrain profile, which reads as violent manoeuvring when
    the aircraft is holding one number.

    This panel plots what the aircraft is actually doing. Put beside a
    plan view on the same distance axis, the two together reconstruct
    the flight: the map says where, this says how high.

    Parameters
    ----------
    show_terrain : bool
        Fill the ground under the first path's track. With several
        routes the terrain differs beneath each, so only the first is
        drawn and it is labelled as such.
    show_band : tuple, optional
        ``(floor, ceiling)`` in metres AGL, drawn as a ribbon following
        the first path's terrain.
    """
    plt = _mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(9.0, 3.4))

    cmap = plt.get_cmap("tab10")
    first_terrain = None
    for i, (p, label) in enumerate(zip(paths, labels)):
        dist_km, alts, terrain, _, _ = _sample_path(p, dem)
        if i == 0:
            first_terrain = (dist_km, terrain)
        ax.plot(dist_km, alts, lw=1.8, color=cmap(i % 10), zorder=5,
                label=label)

    if simplified is not None:
        wp = np.asarray(simplified.get_waypoints_array(), float)
        ax.plot(wp[:, 4] / 1000.0, wp[:, 2], lw=2.2, color=C_SIMPLE,
                zorder=8, solid_capstyle="round",
                label=f"simplified — {simplified.n_after} manoeuvres, "
                      f"{len(wp)} waypoints")
        if number_waypoints:
            draw_numbered(ax, wp[:, 4] / 1000.0, wp[:, 2], C_SIMPLE)

    if show_terrain and first_terrain is not None:
        d, terr = first_terrain
        good = np.isfinite(terr)
        lo = float(np.nanmin(terr[good]))
        ax.fill_between(d[good], terr[good], lo - 0.15 * (np.nanmax(terr[good]) - lo + 50),
                        color=C_TERRAIN_FILL, zorder=1, lw=0)
        ax.plot(d[good], terr[good], color=C_TERRAIN, lw=1.1, zorder=2,
                label="terrain under the first route")
        if show_band:
            floor, ceiling = show_band
            ax.fill_between(d[good], terr[good] + floor, terr[good] + ceiling,
                            color=C_BAND, alpha=0.28, lw=0, zorder=2,
                            label=f"legal band, {floor:.0f}–{ceiling:.0f} m AGL")
        ax.set_ylim(bottom=lo - 0.15 * (np.nanmax(terr[good]) - lo + 50))

    ax.set_xlabel("distance flown [km]")
    ax.set_ylabel("altitude [m AMSL]")
    handles, names = ax.get_legend_handles_labels()
    if not route_legend:
        # On a combined sheet the routes are already named on the map
        # and in the panel below; repeating them here costs a third of
        # the panel and covers the terrain the panel exists to show.
        # The simplified path stays, because it is the only curve in
        # this panel that is not in the others — and the match must be
        # exact: a baseline called "terrain-following" is a route, and
        # a prefix test let it through.
        keep = [(h, n) for h, n in zip(handles, names)
                if n == "terrain under the first route"
                or n.startswith(("legal band", "simplified"))]
        handles, names = ([h for h, _ in keep], [n for _, n in keep])
    if handles:
        ax.legend(handles, names, fontsize=6.5, loc="lower left",
                  framealpha=0.93, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")
    return ax


def plot_agl_overlay(paths, labels, dem, constraints, ax=None,
                     title: str = None, simplified=None,
                     number_waypoints: bool = True):
    """
    Several AGL profiles on one axis, inside the corridor band.

    :func:`plot_corridor_comparison` stacks one panel per route, which
    is right when each is being read on its own. This overlays them,
    which is right when the question is which one leaves the band and
    where — a stacked view makes the reader hold two pictures in mind
    to answer it.
    """
    plt = _mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(9.0, 3.4))

    floor, ceiling = constraints.corridor("FW_CRUISE")
    ax.axhspan(floor, ceiling, color=C_BAND, alpha=0.28, lw=0, zorder=1,
               label=f"legal corridor, {floor:.0f}–{ceiling:.0f} m AGL")
    ax.axhline(ceiling, color=C_BUST, lw=1.2, ls="--", zorder=3)
    ax.axhline(floor, color=C_FLOOR, lw=1.2, ls=":", zorder=3)

    cmap = plt.get_cmap("tab10")
    for i, (p, label) in enumerate(zip(paths, labels)):
        dist_km, alts, terrain, _, _ = _sample_path(p, dem)
        agl = alts - terrain
        over = float(np.max(np.maximum(agl - ceiling, 0.0)))
        ax.plot(dist_km, agl, lw=1.7, color=cmap(i % 10), zorder=5,
                label=f"{label}" + (f"  (+{over:.0f} m over)" if over > 1 else ""))

    if simplified is not None:
        wp = np.asarray(simplified.get_waypoints_array(), float)
        # Terrain under each waypoint from the DEM, not interpolated
        # across the simplified window. The window covers only the
        # cruise; the take-off and landing waypoints sit outside it, so
        # interpolation clamped their terrain to the window edge and
        # drew the landing pad at 130 m *below* ground.
        terr_wp = np.asarray(dem.elevation_batch(wp[:, 0], wp[:, 1]),
                             dtype=float)
        agl_dense = simplified.dense_alt_m - simplified.dense_terrain_m
        ax.plot(simplified.dense_distance_m / 1000.0, agl_dense,
                lw=2.0, color=C_SIMPLE, zorder=8,
                label="simplified" + ("" if simplified.within_band
                                      else "  (LEAVES THE BAND)"))
        if number_waypoints:
            draw_numbered(ax, wp[:, 4] / 1000.0, wp[:, 2] - terr_wp, C_SIMPLE)

    ax.set_xlabel("distance flown [km]")
    ax.set_ylabel("height above ground [m]")
    ax.set_ylim(bottom=min(-10.0, float(ax.get_ylim()[0])))
    ax.legend(fontsize=6.5, loc="upper right", framealpha=0.93, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")
    return ax


# ═══════════════════════════════════════════════════════════════════════════
# PLAN VIEW — THE ROUTE SEEN FROM ABOVE
# ═══════════════════════════════════════════════════════════════════════════
#
# Every figure the package produced showed a route in *profile*: distance
# along the track against height. That view answers "does it clear the
# ground and stay under the ceiling", which is the regulatory question,
# and it answers nothing about the question a reader asks first — where
# does it go?
#
# A profile cannot show a detour. Two routes with identical profiles can
# differ by two kilometres of lateral deviation, and the deviation is the
# whole content of the comparison between planning methods: the straight
# line is what you would fly with no rules, and every method after it
# buys legality with distance. Drawing that on a map makes the trade
# visible; drawing it in profile hides it exactly.


def lateral_deviation(path, origin=None, destination=None,
                      x_mode: str = "projected"):
    """
    Signed perpendicular offset of a path from its own direct line [m].

    Positive is left of the origin-to-destination bearing.

    Parameters
    ----------
    x_mode : {'projected', 'flown'}
        Which distance to return alongside the offset. ``projected`` is
        distance along the direct line — natural for the deviation on
        its own. ``flown`` is the path's own cumulative distance, which
        is what every other profile in the package uses.

        The distinction is not cosmetic: a detour is *longer* than the
        line it departs from, so the two axes end at different values,
        and putting a deviation trace on one axis beside an altitude
        profile on the other silently misaligns them. Panels that share
        an x-axis must all use ``flown``.

    This is the number that says what a detour cost in space, as against
    the energy and time figures that say what it cost to fly. A route
    that is 4 % longer but 1.8 km off the direct line is threading
    something, and the profile view cannot show it.
    """
    wp = path.get_waypoints_array()
    lat, lon = wp[:, 0], wp[:, 1]

    o = origin if origin is not None else (lat[0], lon[0])
    d = destination if destination is not None else (lat[-1], lon[-1])

    # Local flat-earth frame on the origin: over a 30 km leg the error
    # is centimetres, and it keeps the geometry readable.
    lat0 = np.radians(float(o[0]))
    m_lat = 111_320.0
    m_lon = 111_320.0 * np.cos(lat0)

    x = (lon - o[1]) * m_lon          # east
    y = (lat - o[0]) * m_lat          # north
    dx = (d[1] - o[1]) * m_lon
    dy = (d[0] - o[0]) * m_lat
    leg = np.hypot(dx, dy)
    if leg < 1e-6:
        return np.zeros_like(x), np.zeros_like(x)

    ux, uy = dx / leg, dy / leg
    along = x * ux + y * uy
    cross = -x * uy + y * ux          # left-positive

    if x_mode == "flown":
        arr = np.asarray(path.get_waypoints_array(), dtype=float)
        return arr[:, 4], cross
    if x_mode != "projected":
        raise ValueError(f"x_mode must be 'projected' or 'flown', "
                         f"got {x_mode!r}")
    return along, cross


def plot_plan_view(paths, labels, dem, ax=None, bbox=None, airspace=None,
                   context=None, title: str = None, margin_m: float = 2500.0,
                   show_direct: bool = True, show_waypoints: bool = True,
                   waypoints=None, colour_terrain: bool = False,
                   annotate_ends: bool = True, endpoint_names=None,
                   simplified=None, number_waypoints: bool = True):
    """
    One or more routes drawn on shaded terrain, seen from above.

    Parameters
    ----------
    paths, labels : sequence
        Routes to draw and what to call them. The first path's endpoints
        define the direct line the others are compared against.
    airspace : AirspaceManager, optional
        Draws the zones the routes are threading, coloured by permission.
        Without it the detours look arbitrary.
    context : OperationalContext, optional
        Authorisations already held, so a zone the operator may enter is
        drawn as free rather than as an obstacle.
    waypoints : sequence of sequences of (lat, lon), optional
        The routing waypoints of each path, if the caller knows them —
        ``RoutedPath._waypoint_coords`` holds them. Without this the
        turn points are inferred from the track, which is only an
        approximation of where the planner actually put them.
    margin_m : float
        Padding around the routes' bounding box.
    colour_terrain : bool
        Colour relief instead of grey. Off by default: when several
        coloured routes and three permission classes are already on the
        page, a coloured ground competes with all of them.
    endpoint_names : (str, str), optional
        Names for the origin and destination. Given them, the markers
        are labelled with the facilities rather than with "origin" and
        "destination" — on a sheet of several corridors the generic
        words say nothing, because every panel has both.
    """
    plt = _mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 7.0))

    arrays = [p.get_waypoints_array() for p in paths]

    if bbox is None:
        lats = np.concatenate([a[:, 0] for a in arrays])
        lons = np.concatenate([a[:, 1] for a in arrays])
        dlat = margin_m / 111_320.0
        dlon = dlat / max(np.cos(np.radians(float(np.mean(lats)))), 1e-6)
        bbox = (float(lats.min()) - dlat, float(lats.max()) + dlat,
                float(lons.min()) - dlon, float(lons.max()) + dlon)
        # Clip to the DEM, so a route near the edge does not ask for
        # ground that was never loaded.
        m = dem.metadata
        bbox = (max(bbox[0], m.lat_min), min(bbox[1], m.lat_max),
                max(bbox[2], m.lon_min), min(bbox[3], m.lon_max))

    terrain_backdrop(ax, dem, bbox=bbox, colour=colour_terrain,
                     alpha=0.95, zorder=0)

    if airspace is not None:
        _draw_zones(ax, airspace, bbox, context=context, zorder=2)

    if show_direct and len(arrays):
        a = arrays[0]
        ax.plot([a[0, 1], a[-1, 1]], [a[0, 0], a[-1, 0]],
                ls=(0, (6, 4)), lw=1.4, color="0.25", zorder=5,
                label="direct line")

    cmap = plt.get_cmap("tab10")
    for i, (arr, label) in enumerate(zip(arrays, labels)):
        colour = cmap(i % 10)
        ax.plot(arr[:, 1], arr[:, 0], lw=2.0, color=colour, zorder=6,
                label=label, solid_capstyle="round")
        if show_waypoints:
            if waypoints is not None and i < len(waypoints) and waypoints[i]:
                wp = np.asarray(waypoints[i], dtype=float)
                # Endpoints get their own markers below.
                inner = wp[1:-1] if len(wp) > 2 else wp[:0]
                if len(inner):
                    ax.plot(inner[:, 1], inner[:, 0], "o", ms=5.5,
                            color=colour, mec="white", mew=0.9, zorder=7)
            else:
                turns = _turn_points(arr)
                if len(turns):
                    ax.plot(arr[turns, 1], arr[turns, 0], "o", ms=4.2,
                            color=colour, mec="white", mew=0.7, zorder=7)

    if simplified is not None:
        wp = np.asarray(simplified.get_waypoints_array(), float)
        ax.plot(wp[:, 1], wp[:, 0], lw=2.4, color=C_SIMPLE, zorder=8,
                solid_capstyle="round",
                label=f"simplified ({len(wp)} waypoints)")
        if number_waypoints:
            draw_numbered(ax, wp[:, 1], wp[:, 0], C_SIMPLE)

    if annotate_ends and len(arrays):
        a = arrays[0]
        o_name, d_name = endpoint_names or ("origin", "destination")
        # A round trip ends where it started. Two markers and two labels
        # on the same pixel read as a mislabelled destination, which is
        # exactly how it looked before: the pad at the bottom of the map
        # carried the *far* facility's name.
        ends = (((a[0, 0], a[0, 1]), o_name, "^", "#1B5E20"),
                ((a[-1, 0], a[-1, 1]), d_name, "s", "#B71C1C"))
        if (abs(a[0, 0] - a[-1, 0]) < 5e-4
                and abs(a[0, 1] - a[-1, 1]) < 5e-4):
            ends = (((a[0, 0], a[0, 1]),
                     f"{o_name} — start and end", "^", "#1B5E20"),)
        for (lat, lon), name, marker, colour in ends:
            ax.plot(lon, lat, marker, ms=10, color=colour, mec="white",
                    mew=1.4, zorder=9)
            ax.annotate(name, (lon, lat), xytext=(7, 7),
                        textcoords="offset points", fontsize=7.5,
                        fontweight="bold", zorder=10, color="#212121",
                        bbox=dict(boxstyle="round,pad=0.22", fc="white",
                                  ec=colour, lw=0.8, alpha=0.88))

    _scale_bar(ax, bbox)
    ax.set_xlim(bbox[2], bbox[3])
    ax.set_ylim(bbox[0], bbox[1])
    # Equal aspect in metres, not in degrees: a degree of longitude near
    # the equator is a degree of latitude, but the axes do not know that
    # and an unset aspect stretches the map.
    ax.set_aspect(1.0 / max(np.cos(np.radians(0.5 * (bbox[0] + bbox[1]))), 1e-6))
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.legend(fontsize=7, loc="best", framealpha=0.92)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")
    return ax


def plot_lateral_deviation(paths, labels, ax=None, title: str = None,
                           origin=None, destination=None,
                           x_mode: str = "projected", simplified=None,
                           number_waypoints: bool = True):
    """
    How far each route strays from the direct line, along its length.

    The companion to the plan view: the map shows *where* a route
    detours, this shows *how much* and lets several be compared on one
    axis.
    """
    plt = _mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(9.0, 3.0))

    cmap = plt.get_cmap("tab10")
    ax.axhline(0.0, color="0.25", ls=(0, (6, 4)), lw=1.2, zorder=2,
               label="direct line")

    worst = 0.0
    up, down = 0.0, 0.0
    for i, (p, label) in enumerate(zip(paths, labels)):
        along, cross = lateral_deviation(p, origin, destination,
                                         x_mode=x_mode)
        ax.plot(along / 1000.0, cross, lw=1.8, color=cmap(i % 10),
                label=f"{label}  (max {np.abs(cross).max():.0f} m)", zorder=4)
        worst = max(worst, float(np.abs(cross).max()))
        up = max(up, float(np.max(cross)))
        down = min(down, float(np.min(cross)))

    if simplified is not None:
        along, cross = lateral_deviation(simplified, origin, destination,
                                         x_mode=x_mode)
        ax.plot(along / 1000.0, cross, lw=2.0, color=C_SIMPLE, zorder=6,
                label="simplified")
        if number_waypoints:
            draw_numbered(ax, along / 1000.0, cross, C_SIMPLE)
        worst = max(worst, float(np.abs(cross).max()))
        up = max(up, float(np.max(cross)))
        down = min(down, float(np.min(cross)))

    ax.set_xlabel("distance flown [km]" if x_mode == "flown"
                  else "distance along the direct line [km]")
    ax.set_ylabel("lateral offset [m]\n(+ left of track)")

    # Routes usually detour to one side, so leave the headroom where the
    # legend has to go rather than padding both sides equally and then
    # dropping the legend on top of the curves.
    # Routes may detour to either side, and with several on one axis
    # both halves can be occupied. Pad generously and let matplotlib
    # find the emptiest corner rather than guessing at a fixed corner
    # and landing on a curve.
    span = max(worst, 50.0)
    ax.set_ylim(-1.15 * span - 0.15 * span * (up > 0),
                1.15 * span + 0.15 * span * (down < 0))
    ax.legend(fontsize=6.5, ncol=1, loc="best", framealpha=0.93,
              borderpad=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")
    return ax


def plot_plan_and_profile(paths, labels, dem, constraints, airspace=None,
                          context=None, waypoints=None, suptitle: str = None,
                          endpoint_names=None, simplified=None,
                          number_waypoints: bool = True,
                          figsize=(19.0, 11.5)):
    """
    A route in every view needed to reconstruct it, on one page.

    Four panels. The map says where the aircraft goes and what it
    threads. The three profiles beside it say, at the same distance
    flown:

    * **altitude AMSL** — what the aircraft is actually doing. Read with
      the map, this is the flight path.
    * **height AGL** — whether it is allowed to be there.
    * **lateral offset** — what the avoidance cost in distance.

    All three share one x-axis, and that axis is *distance flown by each
    route*. The deviation panel used to be drawn against distance along
    the direct line instead, which is a different and shorter axis — a
    detour is longer than the line it leaves — so a reader lining the
    panels up vertically was comparing points that are not the same
    point. Routes now end at different x, which is honest: they are
    different lengths.
    """
    plt = _mpl()
    fig = plt.figure(figsize=figsize)
    # Room on the right for the profile legends, which are moved out of
    # the axes below. With four baselines, several legs and the
    # simplified path there are seven or eight curves per panel, and a
    # legend inside the frame lands on the data every time.
    gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1.0],
                          height_ratios=[1, 1, 1], hspace=0.16, wspace=0.16,
                          left=0.045, right=0.79, top=0.92, bottom=0.06)

    ax_map = fig.add_subplot(gs[:, 0])
    ax_alt = fig.add_subplot(gs[0, 1])
    ax_agl = fig.add_subplot(gs[1, 1], sharex=ax_alt)
    ax_dev = fig.add_subplot(gs[2, 1], sharex=ax_alt)

    plot_plan_view(paths, labels, dem, ax=ax_map, airspace=airspace,
                   context=context, waypoints=waypoints,
                   endpoint_names=endpoint_names, simplified=simplified,
                   number_waypoints=number_waypoints, title="From above")
    plot_altitude_profile(paths, labels, dem, ax=ax_alt, route_legend=False,
                          show_band=constraints.corridor("FW_CRUISE"),
                          simplified=simplified,
                          number_waypoints=number_waypoints,
                          title="Altitude — what the aircraft is doing")
    plot_agl_overlay(paths, labels, dem, constraints, ax=ax_agl,
                     simplified=simplified,
                     number_waypoints=number_waypoints,
                     title="Height above ground — whether it is allowed")
    plot_lateral_deviation(paths, labels, ax=ax_dev, x_mode="flown",
                           simplified=simplified,
                           number_waypoints=number_waypoints,
                           title="Offset from the direct line")

    for ax in (ax_alt, ax_agl):
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)
    ax_dev.set_xlabel("distance flown [km]")

    # Legends out to the right, each beside the panel it belongs to.
    for ax in (ax_alt, ax_agl, ax_dev):
        old = ax.get_legend()
        if old is None:
            continue
        handles = list(getattr(old, "legend_handles", []))
        texts = [t.get_text() for t in old.get_texts()]
        old.remove()
        if handles:
            ax.legend(handles, texts, fontsize=7, loc="center left",
                      bbox_to_anchor=(1.015, 0.5), framealpha=0.95,
                      borderpad=0.4, handlelength=1.6)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12.5, fontweight="bold")
    return fig


def _turn_points(arr: np.ndarray, threshold_deg: float = 12.0,
                 min_separation: int = 15) -> np.ndarray:
    """
    Indices where the ground track really changes heading.

    A terrain-following leg is hundreds of sampled states, and the
    heading wobbles by a degree or two between most of them. Marking
    every one draws a caterpillar rather than a route, so a turn has to
    clear both a heading threshold and a minimum separation from the
    last one kept.
    """
    if len(arr) < 3:
        return np.array([], dtype=int)
    dlat = np.diff(arr[:, 0])
    dlon = np.diff(arr[:, 1]) * np.cos(np.radians(float(np.mean(arr[:, 0]))))
    heading = np.degrees(np.arctan2(dlon, dlat))
    turn = np.abs((np.diff(heading) + 180.0) % 360.0 - 180.0)

    kept: List[int] = []
    for i in np.argsort(turn)[::-1]:
        if turn[i] <= threshold_deg:
            break
        idx = int(i) + 1
        if all(abs(idx - k) >= min_separation for k in kept):
            kept.append(idx)
    return np.array(sorted(kept), dtype=int)


def _scale_bar(ax, bbox, fraction: float = 0.22):
    """A scale bar, because a map in degrees has no readable scale."""
    lat0, lat1, lon0, lon1 = bbox
    mid_lat = 0.5 * (lat0 + lat1)
    span_m = (lon1 - lon0) * 111_320.0 * np.cos(np.radians(mid_lat))
    if span_m <= 0:
        return
    # Round the bar to something a reader can do arithmetic with.
    nice = np.array([100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000])
    target = span_m * fraction
    length = float(nice[np.argmin(np.abs(nice - target))])

    dlon = length / (111_320.0 * np.cos(np.radians(mid_lat)))
    x0 = lon1 - 0.06 * (lon1 - lon0) - dlon
    y0 = lat0 + 0.045 * (lat1 - lat0)
    ax.plot([x0, x0 + dlon], [y0, y0], lw=3.0, color="black",
            solid_capstyle="butt", zorder=10)
    ax.annotate(f"{length/1000:g} km" if length >= 1000 else f"{length:g} m",
                (x0 + dlon / 2, y0), xytext=(0, 4),
                textcoords="offset points", ha="center", fontsize=7.5,
                fontweight="bold", zorder=10,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                          alpha=0.8))


def _draw_zones(ax, airspace, bbox, context=None, zorder: int = 2):
    """Airspace zones inside *bbox*, coloured by permission."""
    from matplotlib.patches import Polygon as MplPolygon

    from .airspace import CircularZone, PolygonalZone
    from .regulations import PermissionClass

    colour = {PermissionClass.FREE: C_FREE,
              PermissionClass.PERMIT: C_PERMIT,
              PermissionClass.PROHIBITED: C_PROHIBITED}
    alpha = {PermissionClass.FREE: 0.08,
             PermissionClass.PERMIT: 0.20,
             PermissionClass.PROHIBITED: 0.34}

    lat0, lat1, lon0, lon1 = bbox
    held = set(context.authorized_zone_ids) if context else set()
    counts = {k: 0 for k in colour}

    for z in airspace.active_zones():
        if z.is_global or z.geometry is None:
            continue
        perm = PermissionClass.FREE if z.zone_id in held else z.permission
        c, a = colour[perm], alpha[perm]

        if isinstance(z.geometry, CircularZone):
            g = z.geometry
            rl = g.radius_m / 111_320.0
            rn = rl / max(np.cos(np.radians(g.center_lat)), 1e-6)
            if (g.center_lat + rl < lat0 or g.center_lat - rl > lat1
                    or g.center_lon + rn < lon0 or g.center_lon - rn > lon1):
                continue
            t = np.linspace(0, 2 * np.pi, 96)
            ax.fill(g.center_lon + rn * np.cos(t),
                    g.center_lat + rl * np.sin(t),
                    color=c, alpha=a, lw=0.6, ec=c, zorder=zorder)
        elif isinstance(z.geometry, PolygonalZone):
            verts = [(lo, la) for la, lo in z.geometry.vertices]
            vlat = [v[1] for v in verts]
            vlon = [v[0] for v in verts]
            if (max(vlat) < lat0 or min(vlat) > lat1
                    or max(vlon) < lon0 or min(vlon) > lon1):
                continue
            ax.add_patch(MplPolygon(verts, closed=True, fc=c, alpha=a,
                                    ec=c, lw=0.6, zorder=zorder))
        else:
            continue
        counts[perm] += 1

    handles = [
        _mpl().Line2D([], [], marker="s", ls="", markersize=8,
                      markerfacecolor=colour[k], markeredgecolor=colour[k],
                      alpha=max(alpha[k], 0.35),
                      label=f"{k.value} ({counts.get(k, 0)})")
        for k in (PermissionClass.PROHIBITED, PermissionClass.PERMIT)
        if counts.get(k, 0)
    ]
    return handles


# ═══════════════════════════════════════════════════════════════════════════
# TERRAIN BACKDROP
# ═══════════════════════════════════════════════════════════════════════════
#
# Plan views need the ground under them to be readable, and a raw
# elevation raster is not: over Quito the useful relief is a few hundred
# metres riding on a 2 800 m plateau, so a linear grey ramp renders the
# whole city as one flat tone. Hillshading solves that by drawing the
# *slope* rather than the height — ridges, quebradas and the crater rim
# appear because they face the light differently, not because they are
# high.
#
# This is computed from the DEM the planner actually uses, rather than
# read from a pre-rendered raster supplied with the terrain data. Two
# reasons: a shipped hillshade is tied to one region and one sun angle,
# while RAPTOR is meant to work anywhere; and a backdrop rendered from a
# different grid than the one being planned over can disagree with it
# without anyone noticing.


def _land_cmap():
    """Matplotlib's ``terrain`` ramp with its water colours removed.

    The lower quarter of ``terrain`` is blue, meant for bathymetry. The
    Quito basin floor sits at 2 400 m and would be painted with it,
    which reads as a lake where there is a city. Starting the ramp above
    the water band keeps green-to-brown-to-white for land only.
    """
    from matplotlib.colors import LinearSegmentedColormap
    plt = _mpl()
    base = plt.get_cmap("terrain")
    return LinearSegmentedColormap.from_list(
        "terrain_land", base(np.linspace(0.28, 1.0, 256)))


def dem_cell_metres(dem) -> Tuple[float, float]:
    """North–south and east–west cell size of a DEM, in metres."""
    import math
    mid_lat = float(np.mean(dem.lat_1d))
    dlat_m = abs(float(dem.lat_1d[1] - dem.lat_1d[0])) * 111_320.0
    dlon_m = (abs(float(dem.lon_1d[1] - dem.lon_1d[0])) * 111_320.0
              * math.cos(math.radians(mid_lat)))
    return dlat_m, dlon_m


def hillshade(elev: np.ndarray, dlat_m: float, dlon_m: float,
              azimuth_deg: float = 315.0, elevation_deg: float = 45.0,
              vert_exag: float = 1.0) -> np.ndarray:
    """
    Shaded relief in [0, 1] from a north-up elevation grid.

    Light from the north-west at 45° is the cartographic convention. It
    is not arbitrary: humans read a lit-from-above scene as convex, so
    lighting from below inverts the apparent relief and ridges read as
    valleys.
    """
    from matplotlib.colors import LightSource
    ls = LightSource(azdeg=azimuth_deg, altdeg=elevation_deg)
    # Rows run south to north, so dy is negated to match the screen
    # orientation imshow(origin="lower") produces.
    return ls.hillshade(np.asarray(elev, dtype=float), vert_exag=vert_exag,
                        dx=dlon_m, dy=-dlat_m)


def terrain_backdrop(ax, dem, bbox=None, colour: bool = True,
                     vert_exag: float = 1.0, alpha: float = 1.0,
                     zorder: int = 0):
    """
    Draw shaded terrain as the base layer of a plan view.

    Parameters
    ----------
    colour : bool
        ``True`` tints the shading with a terrain ramp, which reads as a
        map. ``False`` leaves it grey, which is what you want when
        coloured overlays (permission classes, waiver severity) have to
        carry the meaning and must not compete with the ground.
    vert_exag : float
        Vertical exaggeration of the shading only — it changes how the
        relief *looks*, never any number the planner uses.

    Returns
    -------
    (lat_min, lat_max, lon_min, lon_max) actually drawn.
    """
    from matplotlib.colors import LightSource

    plt = _mpl()
    if bbox is None:
        bbox = (dem.metadata.lat_min, dem.metadata.lat_max,
                dem.metadata.lon_min, dem.metadata.lon_max)
    lat0, lat1, lon0, lon1 = bbox

    lm = (dem.lat_1d >= lat0) & (dem.lat_1d <= lat1)
    lnm = (dem.lon_1d >= lon0) & (dem.lon_1d <= lon1)
    if lm.sum() < 2 or lnm.sum() < 2:
        raise ValueError(
            f"bbox {bbox} selects {int(lm.sum())}x{int(lnm.sum())} DEM cells; "
            f"it lies outside or below the resolution of this DEM."
        )
    elev = np.asarray(dem.elev_grid[np.ix_(lm, lnm)], dtype=float)
    dlat_m, dlon_m = dem_cell_metres(dem)
    extent = [float(dem.lon_1d[lnm][0]), float(dem.lon_1d[lnm][-1]),
              float(dem.lat_1d[lm][0]), float(dem.lat_1d[lm][-1])]

    if colour:
        ls = LightSource(azdeg=315, altdeg=45)
        rgb = ls.shade(elev, cmap=_land_cmap(), vert_exag=vert_exag,
                       dx=dlon_m, dy=-dlat_m, blend_mode="soft")
        ax.imshow(rgb, extent=extent, origin="lower", aspect="auto",
                  alpha=alpha, zorder=zorder)
    else:
        shade = hillshade(elev, dlat_m, dlon_m, vert_exag=vert_exag)
        # Compressed into the upper part of the grey range so coloured
        # overlays drawn on top stay legible over every part of the
        # ground, while the relief is still readable underneath.
        ax.imshow(0.45 + 0.55 * shade, extent=extent, origin="lower",
                  cmap="Greys_r", vmin=0.0, vmax=1.0, aspect="auto",
                  alpha=alpha, zorder=zorder)

    return (extent[2], extent[3], extent[0], extent[1])


# ═══════════════════════════════════════════════════════════════════════════
# PERMISSION MAP
# ═══════════════════════════════════════════════════════════════════════════

def plot_permission_map(airspace, dem, paths=None, labels=None, ax=None,
                        bbox=None, title: str = None, context=None):
    """
    Airspace coloured by what may be done in it, not by what it contains.

    A hospital buffer and a military zone are different things and the
    conventional map colours them differently. To a planner they are the
    same thing — airspace needing a permit — while a prison and an
    airport approach are a different thing entirely. Colouring by
    permission puts the map in the same terms as the decision.
    """
    from matplotlib.patches import Polygon as MplPolygon

    from .airspace import CircularZone, PolygonalZone
    from .regulations import PermissionClass

    plt = _mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 7.0))

    colour = {
        PermissionClass.FREE: C_FREE,
        PermissionClass.PERMIT: C_PERMIT,
        PermissionClass.PROHIBITED: C_PROHIBITED,
    }
    alpha = {
        PermissionClass.FREE: 0.10,
        PermissionClass.PERMIT: 0.22,
        PermissionClass.PROHIBITED: 0.38,
    }

    if bbox is None:
        bbox = (dem.metadata.lat_min, dem.metadata.lat_max,
                dem.metadata.lon_min, dem.metadata.lon_max)
    lat0, lat1, lon0, lon1 = bbox

    # Grey, not coloured: the permission classes own the colour here.
    terrain_backdrop(ax, dem, bbox=bbox, colour=False, alpha=0.9, zorder=0)

    held = set(context.authorized_zone_ids) if context else set()
    counts = {k: 0 for k in colour}

    for z in airspace.active_zones():
        if z.is_global or z.geometry is None:
            continue
        perm = (PermissionClass.FREE if z.zone_id in held else z.permission)
        counts[perm] = counts.get(perm, 0) + 1
        c, a = colour[perm], alpha[perm]

        if isinstance(z.geometry, CircularZone):
            g = z.geometry
            rl = g.radius_m / 111_320.0
            rn = rl / max(np.cos(np.radians(g.center_lat)), 1e-6)
            t = np.linspace(0, 2 * np.pi, 64)
            ax.fill(g.center_lon + rn * np.cos(t),
                    g.center_lat + rl * np.sin(t),
                    color=c, alpha=a, lw=0.5, ec=c, zorder=2)
        elif isinstance(z.geometry, PolygonalZone):
            verts = [(lo, la) for la, lo in z.geometry.vertices]
            ax.add_patch(MplPolygon(verts, closed=True, fc=c, alpha=a,
                                    ec=c, lw=0.5, zorder=2))

    if paths:
        cmap = plt.get_cmap("tab10")
        for i, p in enumerate(paths):
            wp = p.get_waypoints_array()
            ax.plot(wp[:, 1], wp[:, 0], lw=1.8, color=cmap(i % 10), zorder=6,
                    label=(labels[i] if labels and i < len(labels) else None))

    handles = [
        plt.Line2D([], [], marker="s", ls="", markersize=9,
                   markerfacecolor=colour[k], markeredgecolor=colour[k],
                   alpha=max(alpha[k], 0.35),
                   label=f"{k.value} ({counts.get(k, 0)})")
        for k in (PermissionClass.PROHIBITED, PermissionClass.PERMIT,
                  PermissionClass.FREE)
    ]
    if paths and labels:
        handles += [plt.Line2D([], [], color=plt.get_cmap("tab10")(i % 10),
                               lw=1.8, label=l)
                    for i, l in enumerate(labels)]
    ax.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.92)

    ax.set_xlim(lon0, lon1)
    ax.set_ylim(lat0, lat1)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")
    return ax


# ═══════════════════════════════════════════════════════════════════════════
# WAIVER MAP
# ═══════════════════════════════════════════════════════════════════════════

def plot_waiver_map(assessments, labels, dem, ax=None, bbox=None,
                    title: str = None):
    """
    Where on a network height relief is needed, and how much.

    Each excursion is drawn at its own location, sized and coloured by the
    relief it needs. A regulator looking at this can see immediately
    whether a proposed network asks for a scatter of small local
    exemptions or a blanket raise of the ceiling — which are very
    different requests.
    """
    plt = _mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 7.0))

    if bbox is None:
        bbox = (dem.metadata.lat_min, dem.metadata.lat_max,
                dem.metadata.lon_min, dem.metadata.lon_max)
    lat0, lat1, lon0, lon1 = bbox

    lm = (dem.lat_1d >= lat0) & (dem.lat_1d <= lat1)
    lnm = (dem.lon_1d >= lon0) & (dem.lon_1d <= lon1)
    ax.imshow(dem.elev_grid[np.ix_(lm, lnm)],
              extent=[dem.lon_1d[lnm][0], dem.lon_1d[lnm][-1],
                      dem.lat_1d[lm][0], dem.lat_1d[lm][-1]],
              origin="lower", cmap="Greys_r", alpha=0.6, aspect="auto",
              zorder=0)

    lats, lons, excess = [], [], []
    for a in assessments:
        if a is None or a.height_waiver is None:
            continue
        for e in a.height_waiver.excursions:
            lats.append(e.lat)
            lons.append(e.lon)
            excess.append(e.max_excess_m)

    if excess:
        sc = ax.scatter(lons, lats, c=excess, s=[20 + 4 * x for x in excess],
                        cmap="YlOrRd", vmin=0, edgecolors="k", linewidths=0.4,
                        zorder=5)
        cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("height relief needed [m]", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    else:
        ax.text(0.5, 0.5, "no height relief needed anywhere",
                transform=ax.transAxes, ha="center", fontsize=10,
                color=C_FREE, fontweight="bold")

    ax.set_xlim(lon0, lon1)
    ax.set_ylim(lat0, lat1)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")
    return ax


def plot_drag_breakdown(ac, V: float = 30.0, altitude: float = 2800.0,
                        ax=None, title: str = None):
    """
    Where an airframe's drag comes from, part by part.

    Worth plotting because the answer is usually surprising: on a
    lift+cruise machine the stopped rotors typically out-drag the
    fuselage, and a single tuned ``C_D0`` hides that entirely.
    """
    plt = _mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 3.2))

    buildup = ac.drag_buildup
    if buildup is None:
        ax.text(0.5, 0.5, "no drag build-up attached\n(ac.build_drag())",
                transform=ax.transAxes, ha="center", va="center", fontsize=9)
        return ax

    parts = buildup.breakdown(V, altitude)
    rho_q = 0.5 * __import__("raptor.atmosphere", fromlist=["isa_density"]) \
        .isa_density(altitude) * V ** 2

    names = list(parts.keys())
    drag_n = [parts[n] * rho_q for n in names]

    # The wing, from the polar, for scale.
    C_L = ac.W / (rho_q * ac.S_ref)
    wing_cd = (ac.aero_polar.C_D(C_L, ac.reynolds_at(V, altitude))
               if ac.aero_polar is not None
               else ac.C_D0 + ac.k_drag * C_L ** 2)
    names.append("wing (profile + induced)")
    drag_n.append(wing_cd * rho_q * ac.S_ref)

    order = np.argsort(drag_n)
    y = np.arange(len(names))
    colours = ["#5C6BC0" if "wing" in names[i] else "#8D6E63" for i in order]
    ax.barh(y, [drag_n[i] for i in order], color=colours, height=0.68)
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order], fontsize=8)
    ax.set_xlabel(f"drag at {V:.0f} m/s, {altitude:.0f} m [N]")
    ax.spines[["top", "right"]].set_visible(False)
    total = sum(drag_n)
    ax.set_title(title or f"Drag build-up — total {total:.1f} N, "
                          f"L/D {ac.W/total:.1f}",
                 fontsize=10, fontweight="bold")
    return ax


# ═══════════════════════════════════════════════════════════════════════════
# THREE DIMENSIONS
# ═══════════════════════════════════════════════════════════════════════════

def plot_path_3d(paths, labels, dem, ax=None, margin_m: float = 2000.0,
                 title: str = None, decimate: int = None,
                 elev: float = 32.0, azim: float = -120.0,
                 show_ground_track: bool = True, colour_terrain: bool = True,
                 simplified=None, number_waypoints: bool = True):
    """
    The routes over the terrain they actually cross, in three dimensions.

    The plan and profile views each drop one axis, and a reader has to
    put them back together. This does not — but it is the *worst* view
    for reading a number off, because perspective destroys the one thing
    the regulation is written about: height above ground. Use it to see
    the shape of the problem and the other panels to measure it.

    Parameters
    ----------
    decimate : int
        Terrain subsampling. The DEM is 3 million cells; drawing them
        all is neither possible nor useful. Defaults to whatever keeps
        the surface near 120 cells a side.
    show_ground_track : bool
        Drop each route onto the terrain as a shadow. Without it the
        lines float and the eye cannot place them.
    elev, azim : float
        Viewing angles. With an interactive backend the figure is
        draggable and these are only the starting point; with Agg they
        are the whole picture, so they matter.
    """
    plt = _mpl()
    from matplotlib.colors import LightSource

    arrays = [np.asarray(p.get_waypoints_array(), float) for p in paths]
    lats = np.concatenate([a[:, 0] for a in arrays])
    lons = np.concatenate([a[:, 1] for a in arrays])
    dlat = margin_m / 111_320.0
    dlon = dlat / max(np.cos(np.radians(float(np.mean(lats)))), 1e-6)
    m = dem.metadata
    lat0 = max(float(lats.min()) - dlat, m.lat_min)
    lat1 = min(float(lats.max()) + dlat, m.lat_max)
    lon0 = max(float(lons.min()) - dlon, m.lon_min)
    lon1 = min(float(lons.max()) + dlon, m.lon_max)

    lm = (dem.lat_1d >= lat0) & (dem.lat_1d <= lat1)
    lnm = (dem.lon_1d >= lon0) & (dem.lon_1d <= lon1)
    if decimate is None:
        decimate = max(1, int(max(lm.sum(), lnm.sum()) // 120))
    la = dem.lat_1d[lm][::decimate]
    lo = dem.lon_1d[lnm][::decimate]
    z = np.asarray(dem.elev_grid[np.ix_(lm, lnm)], float)[::decimate, ::decimate]
    LON, LAT = np.meshgrid(lo, la)

    if ax is None:
        fig = plt.figure(figsize=(10.5, 8.0))
        ax = fig.add_subplot(111, projection="3d")

    dlat_m, dlon_m = dem_cell_metres(dem)
    ls = LightSource(azdeg=315, altdeg=45)
    if colour_terrain:
        rgb = ls.shade(z, cmap=_land_cmap(), vert_exag=1.0,
                       dx=dlon_m * decimate, dy=-dlat_m * decimate,
                       blend_mode="soft")
    else:
        shade = ls.hillshade(z, vert_exag=1.0,
                             dx=dlon_m * decimate, dy=-dlat_m * decimate)
        g = 0.55 + 0.45 * shade
        rgb = np.dstack([g, g, g])

    ax.plot_surface(LON, LAT, z, facecolors=rgb, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False, zorder=1)

    floor = float(np.nanmin(z))
    cmap = plt.get_cmap("tab10")
    for i, (arr, label) in enumerate(zip(arrays, labels)):
        c = cmap(i % 10)
        ax.plot(arr[:, 1], arr[:, 0], arr[:, 2], lw=2.2, color=c,
                label=label, zorder=6)
        if show_ground_track:
            ax.plot(arr[:, 1], arr[:, 0],
                    np.full(len(arr), floor), lw=1.0, color=c, alpha=0.45,
                    zorder=5)

    if simplified is not None:
        wp = np.asarray(simplified.get_waypoints_array(), float)
        ax.plot(wp[:, 1], wp[:, 0], wp[:, 2], lw=2.6, color=C_SIMPLE,
                zorder=8, label=f"simplified ({len(wp)} waypoints)")
        if number_waypoints:
            for k, row in enumerate(wp):
                ax.text(row[1], row[0], row[2], str(k + 1), color=C_SIMPLE,
                        fontsize=7, fontweight="bold", zorder=12,
                        ha="center", va="center",
                        bbox=dict(boxstyle="circle,pad=0.14", fc="white",
                                  ec=C_SIMPLE, lw=0.9, alpha=0.92))

    a = arrays[0]
    ax.scatter([a[0, 1]], [a[0, 0]], [a[0, 2]], s=55, marker="^",
               color="#1B5E20", edgecolor="white", linewidth=1.2, zorder=9)
    ax.scatter([a[-1, 1]], [a[-1, 0]], [a[-1, 2]], s=55, marker="s",
               color="#B71C1C", edgecolor="white", linewidth=1.2, zorder=9)

    ax.set_xlabel("longitude", labelpad=8)
    ax.set_ylabel("latitude", labelpad=8)
    ax.set_zlabel("altitude [m AMSL]", labelpad=8)
    ax.set_xlim(lon0, lon1)
    ax.set_ylim(lat0, lat1)
    # Frame the ground and the routes, nothing above them. Matplotlib's
    # own z-range rounds outward to a tidy number and left a third of
    # the box empty sky.
    z_lo = float(np.nanmin(z))
    z_hi = max(float(np.nanmax(z)),
               max(float(np.nanmax(np.asarray(a, float)[:, 2]))
                   for a in arrays) if arrays else float(np.nanmax(z)))
    ax.set_zlim(z_lo - 0.02 * (z_hi - z_lo), z_hi + 0.05 * (z_hi - z_lo))
    ax.view_init(elev=elev, azim=azim)
    # Degrees of latitude and longitude are the same length here, and
    # the vertical is compressed on purpose: at true scale a 3 km relief
    # over a 25 km route is a barely visible ripple.
    ax.set_box_aspect((lon1 - lon0, lat1 - lat0,
                       0.55 * max(lat1 - lat0, lon1 - lon0)))
    ax.legend(fontsize=7, loc="upper left")
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")
    return ax


# ═══════════════════════════════════════════════════════════════════════════
# WHAT THE OPTIMIZER DID
# ═══════════════════════════════════════════════════════════════════════════

def plot_optimizer_history(result, routed_path, dem, airspace=None,
                           suptitle: str = None, figsize=(16.0, 6.6),
                           n_snapshots: int = 6):
    """
    How the search got there: the routes it tried, and why it moved.

    Two panels, because a convergence curve on its own is not evidence.
    It says the objective fell; it does not say whether the fall came
    from finding a cheaper route or from giving up on a constraint. The
    right-hand panel separates those: the objective against the
    penalties that make it up, generation by generation.

    A search whose objective drops while a penalty rises is not
    converging, it is trading legality for energy — and that is exactly
    the failure the permit-sensitivity study caught at high penalty
    weight.

    The left panel always shows the same number of routes — by default
    the starting guess, four evenly spaced intermediates and the final
    answer — whatever ``maxiter`` the run used. Earlier versions
    subsampled at a rate that depended on the run length, so two cases
    drew a different number of lines and could not be read side by
    side.
    """
    plt = _mpl()
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.22)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_obj = fig.add_subplot(gs[0, 1])

    # ── left: the routes tried ────────────────────────────────────────
    # The starting guess first, then the recorded generations. The
    # optimiser keeps every generation; the fixed count is applied here
    # so it holds across runs of different length.
    gens = list(result.parameter_history or [])
    snaps, snap_labels = [], []
    theta_0 = getattr(result, "theta_initial", None)
    if theta_0 is not None:
        snaps.append(np.asarray(theta_0, float))
        snap_labels.append("initial guess")

    n_mid = max(n_snapshots - len(snaps) - 1, 0)
    if gens:
        # Evenly spaced through the recorded generations, final last.
        if len(gens) <= n_mid + 1:
            pick = list(range(len(gens)))
        else:
            pick = sorted(set(
                np.linspace(0, len(gens) - 1, n_mid + 1).astype(int).tolist()))
        for k in pick:
            snaps.append(np.asarray(gens[k], float))
            snap_labels.append(f"gen {k + 1}")

    trials, trial_labels = [], []
    saved = getattr(routed_path, "parameter_vector", None)
    try:
        for theta, lab in zip(snaps, snap_labels):
            routed_path.parameter_vector = theta
            trials.append(routed_path.flight_path)
            trial_labels.append(lab)
    finally:
        if saved is not None:
            routed_path.parameter_vector = saved

    terrain_backdrop(ax_map, dem, bbox=None if not trials else _bbox_of(
        [result.optimized_path] + trials, dem), colour=False, alpha=0.9)
    if airspace is not None:
        _draw_zones(ax_map, airspace,
                    _bbox_of([result.optimized_path] + trials, dem))

    n = max(len(trials), 1)
    for k, fp in enumerate(trials):
        arr = np.asarray(fp.get_waypoints_array(), float)
        # Faded early, solid late: the reader should see the search
        # sweeping toward the answer, not a bowl of spaghetti.
        ax_map.plot(arr[:, 1], arr[:, 0], lw=1.2,
                    color=plt.get_cmap("viridis")(k / max(n - 1, 1)),
                    alpha=0.45 + 0.45 * k / max(n - 1, 1), zorder=4,
                    label=trial_labels[k])

    best = np.asarray(result.optimized_path.get_waypoints_array(), float)
    ax_map.plot(best[:, 1], best[:, 0], lw=2.4, color=C_PATH, zorder=7,
                label=f"final ({result.energy_wh:.0f} Wh)")
    ax_map.plot([best[0, 1], best[-1, 1]], [best[0, 0], best[-1, 0]],
                ls=(0, (6, 4)), lw=1.3, color="0.25", zorder=6,
                label="direct line")
    ax_map.set_xlabel("longitude")
    ax_map.set_ylabel("latitude")
    ax_map.legend(fontsize=6.8, loc="best", framealpha=0.92, ncol=2)
    ax_map.set_title("Routes the search tried", fontsize=10, fontweight="bold")

    # ── right: objective and the constraints behind it ────────────────
    hist = list(result.penalty_history or [])
    conv = list(result.convergence_history or [])
    if hist:
        gen = [h["generation"] for h in hist]
        ax_obj.plot(gen, [h["energy_wh"] for h in hist], lw=1.8,
                    color=C_PATH, label="energy [Wh]")
        ax_obj.set_ylabel("energy [Wh]")
        ax_obj.set_xlabel("generation")

        ax_p = ax_obj.twinx()
        from .optimizer import PENALTY_TERMS
        active = [t for t in PENALTY_TERMS
                  if any(h.get(t, 0.0) > 1e-9 for h in hist)]
        for j, term in enumerate(active):
            ax_p.plot(gen, [h.get(term, 0.0) for h in hist], lw=1.3, ls="--",
                      color=plt.get_cmap("tab10")((j + 1) % 10),
                      label=f"{term} penalty")
        ax_p.set_ylabel("constraint penalty")
        ax_p.set_ylim(bottom=0)

        h1, l1 = ax_obj.get_legend_handles_labels()
        h2, l2 = ax_p.get_legend_handles_labels()
        # Outside the frame: with every active penalty plus the
        # objective there are up to seven curves, and a two-column
        # legend inside the panel overlapped itself.
        ax_p.legend(h1 + h2, l1 + l2, fontsize=7, loc="center left",
                    bbox_to_anchor=(1.09, 0.5), framealpha=0.95,
                    handlelength=1.6)
        if not active:
            ax_p.annotate("no constraint was ever violated",
                          xy=(0.5, 0.5), xycoords="axes fraction",
                          ha="center", fontsize=8, color="0.4")
    elif conv:
        ax_obj.plot(range(1, len(conv) + 1), conv, lw=1.8, color=C_PATH)
        ax_obj.set_xlabel("generation")
        ax_obj.set_ylabel("objective")
    ax_obj.set_title("Objective, and the constraints behind it",
                     fontsize=10, fontweight="bold")
    ax_obj.spines[["top"]].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    fig.subplots_adjust(top=0.88 if suptitle else 0.95, bottom=0.11,
                        left=0.05, right=0.80)
    return fig


def _bbox_of(paths, dem, margin_m: float = 2000.0):
    """Bounding box covering several paths, clipped to the DEM."""
    lats, lons = [], []
    for p in paths:
        arr = np.asarray(p.get_waypoints_array(), float)
        lats.append(arr[:, 0])
        lons.append(arr[:, 1])
    lats = np.concatenate(lats)
    lons = np.concatenate(lons)
    dlat = margin_m / 111_320.0
    dlon = dlat / max(np.cos(np.radians(float(np.mean(lats)))), 1e-6)
    m = dem.metadata
    return (max(float(lats.min()) - dlat, m.lat_min),
            min(float(lats.max()) + dlat, m.lat_max),
            max(float(lons.min()) - dlon, m.lon_min),
            min(float(lons.max()) + dlon, m.lon_max))


# ═══════════════════════════════════════════════════════════════════════════
# THE PACK, THROUGH THE FLIGHT
# ═══════════════════════════════════════════════════════════════════════════

#: Colours for the six flight phases, used wherever a figure marks them.
PHASE_COLOUR = {
    "VTOL_ASCEND": "#6A1B9A",
    "TRANSITION": "#00838F",
    "FW_CLIMB": "#C62828",
    "FW_CRUISE": "#1565C0",
    "FW_DESCEND": "#2E7D32",
    "VTOL_DESCEND": "#EF6C00",
}


def plot_battery_timeline(energy_result, ac, fig=None, suptitle: str = None,
                          figsize=(13.0, 9.0), soc_reserve: float = 0.15):
    """
    What the pack does over a flight, and where the energy went.

    Four stacked panels on one time axis, with the flight phases shaded
    behind all of them:

    * **state of charge**, against the reserve the mission is planned to
    * **terminal voltage**, which is not a restatement of SOC — it sags
      under load and recovers, and it is what a low-voltage cutout
      actually watches
    * **pack current**, against the pack's own limit, where the hover
      phases show themselves
    * **cumulative energy**, split into what reached the drivetrain and
      what became heat in the pack

    The phase shading is the point of putting them together: on a VTOL
    the two shortest phases are the two most expensive, and no single
    panel shows that.
    """
    plt = _mpl()
    if fig is None:
        fig = plt.figure(figsize=figsize)
    axes = fig.subplots(4, 1, sharex=True,
                        gridspec_kw=dict(hspace=0.12,
                                         height_ratios=[1.1, 1, 1, 1.1]))

    tl = list(energy_result.battery_timeline)
    t = np.array([s.time for s in tl]) / 60.0
    soc = np.array([s.SOC for s in tl]) * 100.0
    volt = np.array([s.voltage for s in tl])
    amps = np.array([s.current_draw for s in tl])
    wh = np.array([s.energy_consumed_wh for s in tl])

    # Phase bands. Consecutive segments of the same type are merged
    # first, and slivers are left unshaded: a terrain-following leg is
    # forty alternating climbs and descents, and shading each one turns
    # the background into a uniform wash that hides exactly the two
    # blocks — the rotor-borne phases — the panel exists to point at.
    runs, t0 = [], 0.0
    for seg in energy_result.segments:
        t1 = t0 + seg.duration / 60.0
        if runs and runs[-1][0] == seg.segment_type:
            runs[-1][2] = t1
        else:
            runs.append([seg.segment_type, t0, t1])
        t0 = t1
    total_min = max(t0, 1e-9)
    # The rotor-borne phases are always shaded however short they are.
    # They are the shortest blocks on the clock and the most expensive
    # on the battery, which is the whole reason for the panel; filtering
    # them out by duration would remove the finding.
    ALWAYS = {"VTOL_ASCEND", "VTOL_DESCEND", "TRANSITION"}
    seen = set()
    for kind, a, b in runs:
        if kind not in ALWAYS and (b - a) / total_min < 0.015:
            continue
        c = PHASE_COLOUR.get(kind, "0.7")
        for ax in axes:
            ax.axvspan(a, b, color=c, alpha=0.16, lw=0, zorder=0)
        seen.add(kind)

    axes[0].plot(t, soc, lw=1.9, color=C_PATH, label="state of charge")
    axes[0].axhline(100 * soc_reserve, color=C_BUST, ls="--", lw=1.1)
    axes[0].annotate(f"reserve {100*soc_reserve:.0f} %", xy=(0.995, 100*soc_reserve),
                     xycoords=("axes fraction", "data"), ha="right",
                     va="bottom", fontsize=7, color=C_BUST)
    axes[0].set_ylabel("SOC [%]")
    axes[0].set_ylim(0, 105)
    axes[0].legend(fontsize=7, loc="lower left", framealpha=0.92)

    axes[1].plot(t, volt, lw=1.6, color="#6A1B9A", label="terminal voltage")
    n_cells = max(ac.battery_cells_series, 1)
    try:
        from .battery import CELL_CHEMISTRY
        cut = CELL_CHEMISTRY[ac.cell_chemistry]["v_cutoff"] * n_cells
        axes[1].axhline(cut, color=C_BUST, ls="--", lw=1.1)
        axes[1].annotate(f"cell cut-off ({cut:.0f} V)", xy=(0.015, cut),
                         xycoords=("axes fraction", "data"), ha="left",
                         va="bottom", fontsize=7, color=C_BUST)
        # The cut-off is far below anything a flight reaches, so pinning
        # the axis to it wastes the panel. Frame the data and let the
        # line sit at the edge as a reference.
        lo = float(np.nanmin(volt))
        axes[1].set_ylim(min(cut - 1.0, lo - 2.0), float(np.nanmax(volt)) + 1.5)
    except Exception:
        pass
    axes[1].set_ylabel("pack voltage [V]")
    axes[1].legend(fontsize=7, loc="lower left", framealpha=0.92)

    # Current in amps only. The C-rate that used to share this panel is
    # the same curve divided by a constant, so the twin axis added a
    # second scale and no second fact; the pack's own current limit is
    # drawn instead, which is what the amps have to be read against.
    axes[2].plot(t, amps, lw=1.6, color="#EF6C00",
                 label="pack current")
    i_lim = ac.battery_max_c_rate * ac.battery_capacity_ah
    if np.isfinite(i_lim) and i_lim > 0:
        axes[2].axhline(i_lim, color=C_BUST, ls="--", lw=1.0,
                        label=f"pack limit {i_lim:.0f} A "
                              f"({ac.battery_max_c_rate:.0f} C)")
        axes[2].set_ylim(0, max(i_lim * 1.12, float(np.nanmax(amps)) * 1.2))
    axes[2].set_ylabel("current [A]")
    axes[2].legend(fontsize=7, loc="upper right", framealpha=0.92)

    axes[3].plot(t, wh, lw=1.8, color=C_PATH, label="to the drivetrain")
    axes[3].fill_between(t, 0, wh, color=C_PATH, alpha=0.12, lw=0)
    axes[3].set_ylabel("energy [Wh]")
    axes[3].set_xlabel("time [min]")
    axes[3].annotate(
        f"{energy_result.total_energy_wh:.1f} Wh delivered, "
        f"{energy_result.ohmic_loss_wh:.1f} Wh ({100*energy_result.ohmic_loss_wh/max(energy_result.energy_from_cells_wh,1e-9):.1f} %) "
        f"lost as heat in the pack",
        xy=(0.02, 0.9), xycoords="axes fraction", fontsize=7.5, color="0.25")
    axes[3].legend(fontsize=7, loc="center left", framealpha=0.92)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=0)

    handles = [plt.Line2D([], [], marker="s", ls="", markersize=8,
                          markerfacecolor=PHASE_COLOUR[k], alpha=0.45,
                          markeredgecolor="none", label=k.replace("_", " ").lower())
               for k in PHASE_COLOUR if k in seen]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.005))
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    return fig


def plot_energy_breakdown(energy_result, ax=None, title: str = None):
    """
    Where the energy went, by flight phase, against where the time went.

    Two bars per phase, normalised — because the interesting fact about
    a VTOL mission is that the two do not match. The hover and
    transition phases are a small share of the clock and a large share
    of the battery, and that ratio is what makes every extra stop in a
    multi-stop mission expensive.
    """
    plt = _mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(8.0, 3.4))

    by = {}
    for seg in energy_result.segments:
        e = by.setdefault(seg.segment_type, [0.0, 0.0])
        e[0] += seg.energy_wh
        e[1] += seg.duration

    order = [k for k in PHASE_COLOUR if k in by]
    e_tot = sum(v[0] for v in by.values()) or 1.0
    t_tot = sum(v[1] for v in by.values()) or 1.0
    x = np.arange(len(order))
    e_share = [100 * by[k][0] / e_tot for k in order]
    t_share = [100 * by[k][1] / t_tot for k in order]

    ax.bar(x - 0.2, e_share, width=0.4, label="share of energy",
           color=[PHASE_COLOUR[k] for k in order])
    ax.bar(x + 0.2, t_share, width=0.4, label="share of time",
           color=[PHASE_COLOUR[k] for k in order], alpha=0.42)
    for xi, (e, tm) in enumerate(zip(e_share, t_share)):
        if tm > 0.5:
            ax.annotate(f"×{e/tm:.1f}", (xi, max(e, tm) + 1.5), ha="center",
                        fontsize=7, color="0.3")

    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("_", "\n").lower() for k in order],
                       fontsize=7.5)
    ax.set_ylabel("share of mission [%]")
    ax.legend(fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.annotate("×n is energy share divided by time share — above 1 the "
                "phase costs more than its share of the clock",
                xy=(0.5, -0.30), xycoords="axes fraction", ha="center",
                fontsize=7, color="0.4")
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")
    return ax


# ═══════════════════════════════════════════════════════════════════════════
# COMPARING AIRCRAFT
# ═══════════════════════════════════════════════════════════════════════════
#
# Everything above draws one flight. These draw several against each
# other, which needs a different discipline: the quantity that separates
# two aircraft is rarely the one that separates two routes, and a figure
# that shows only energy will make the smallest aircraft look best on
# every case while hiding that it cannot carry the payload.


def plot_uav_routes(runs, dem, constraints, airspace=None, ax=None,
                    endpoint_names=None, title: str = None,
                    number_waypoints: bool = True,
                    figsize=(16.0, 6.8)):
    """
    Do different aircraft route differently, or only cost differently?

    Two panels: the routes on one map, and their heights above ground on
    one axis. The question is worth asking explicitly because the
    intuition is that a bigger aircraft flies the same line more
    expensively — and it does not. Climb angle is set by lift-to-drag and
    installed power, not by mass, so the aircraft disagree about which
    ground they can follow, and that changes the line.

    Parameters
    ----------
    runs : sequence of dict
        One per aircraft, each with ``label``, ``path`` and optionally
        ``waypoints``.
    """
    plt = _mpl()
    fig = None
    if ax is None:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.20)
        ax = fig.add_subplot(gs[0, 0])
        ax_agl = fig.add_subplot(gs[0, 1])
    else:
        ax_agl = None

    paths = [r["path"] for r in runs]
    labels = [r["label"] for r in runs]
    plot_plan_view(paths, labels, dem, ax=ax, airspace=airspace,
                   waypoints=[r.get("waypoints") for r in runs],
                   endpoint_names=endpoint_names, title="Routes")
    if ax_agl is not None:
        plot_agl_overlay(paths, labels, dem, constraints, ax=ax_agl,
                         title="Height above ground")

    # One aircraft's simplified path, numbered. Three sets of numbers on
    # one map would be unreadable, and the point here is the aircraft
    # rather than the plan — so the first is drawn and the rest are not.
    cmap = plt.get_cmap("tab10")
    for i, r in enumerate(runs):
        simp = r.get("simplified")
        if simp is None:
            continue
        wp = np.asarray(simp.get_waypoints_array(), float)
        ax.plot(wp[:, 1], wp[:, 0], lw=1.8, ls=(0, (5, 3)),
                color=cmap(i % 10), zorder=7,
                label=f"{r['label']} simplified ({len(wp)} wp)")
        if number_waypoints and i == 0:
            draw_numbered(ax, wp[:, 1], wp[:, 0], cmap(i % 10))
        if ax_agl is not None:
            ax_agl.plot(simp.dense_distance_m / 1000.0,
                        simp.dense_alt_m - simp.dense_terrain_m,
                        lw=1.5, ls=(0, (5, 3)), color=cmap(i % 10), zorder=7)
    ax.legend(fontsize=6.5, loc="best", framealpha=0.92)
    if fig is not None and title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    return fig if fig is not None else ax


def plot_uav_battery(runs, ax=None, title: str = None, figsize=(13.0, 9.2),
                     soc_reserve: float = 0.15):
    """
    The pack of every aircraft on one set of axes.

    Four panels against time: state of charge, per-cell voltage, pack
    current and cumulative energy.

    Voltage is shown **per cell**, not per pack. The three aircraft run
    different series counts, so their pack voltages are three unrelated
    numbers and overlaying them compares nothing; divided by the series
    count they land on the same 3.0–4.2 V scale, where the sag under a
    hover load and the recovery afterwards can be read across aircraft
    and against one cut-off line.

    Current stays in amps, which is what a pack, an ESC and a fuse are
    each rated in, with every aircraft's own limit drawn as a dashed
    line in its own colour.

    Time rather than distance on the x-axis, deliberately: a faster
    aircraft finishing sooner is the result, and a distance axis would
    hide it.
    """
    plt = _mpl()
    fig = plt.figure(figsize=figsize)
    axes = fig.subplots(4, 1, sharex=True,
                        gridspec_kw=dict(hspace=0.13,
                                         height_ratios=[1.2, 1, 1, 1]))
    cmap = plt.get_cmap("tab10")
    cutoffs = []
    v_lo, v_hi = None, None

    for i, r in enumerate(runs):
        er = r["energy"]
        ac = r["aircraft"]
        tl = list(er.battery_timeline)
        if not tl:
            continue
        t = np.array([s.time for s in tl]) / 60.0
        c = cmap(i % 10)
        n_cells = max(getattr(ac, "battery_cells_series", 0), 1)
        axes[0].plot(t, [100 * s.SOC for s in tl], lw=1.9, color=c,
                     label=f"{r['label']}  ({er.total_energy_wh:.0f} Wh, "
                           f"{er.total_time/60:.1f} min)")
        v_cell = np.array([s.voltage for s in tl]) / n_cells
        axes[1].plot(t, v_cell, lw=1.5, color=c,
                     label=f"{r['label']}  ({n_cells}S)")
        v_lo = float(np.min(v_cell)) if v_lo is None else min(v_lo, float(np.min(v_cell)))
        v_hi = float(np.max(v_cell)) if v_hi is None else max(v_hi, float(np.max(v_cell)))
        amps = np.array([s.current_draw for s in tl])
        axes[2].plot(t, amps, lw=1.4, color=c, label=r["label"])
        i_lim = ac.battery_max_c_rate * ac.battery_capacity_ah
        if np.isfinite(i_lim) and i_lim > 0:
            axes[2].axhline(i_lim, color=c, ls="--", lw=0.9, alpha=0.7)
            axes[2].annotate(f"{r['label']} limit {i_lim:.0f} A",
                             xy=(0.005, i_lim),
                             xycoords=("axes fraction", "data"), ha="left",
                             va="bottom", fontsize=6.5, color=c)
        axes[3].plot(t, [s.energy_consumed_wh for s in tl], lw=1.7, color=c,
                     label=f"{r['label']}  ({er.total_energy_wh:.0f} Wh)")
        # Where each aircraft finishes, so the eye can find the end.
        axes[0].plot(t[-1], 100 * tl[-1].SOC, "o", ms=5, color=c,
                     mec="white", mew=0.9)
        try:
            from .battery import CELL_CHEMISTRY
            cutoffs.append(CELL_CHEMISTRY[ac.cell_chemistry]["v_cutoff"])
        except Exception:
            pass

    axes[0].axhline(100 * soc_reserve, color=C_BUST, ls="--", lw=1.1,
                    label=f"planned reserve {100*soc_reserve:.0f} %")
    axes[0].set_ylabel("SOC [%]")
    axes[0].set_ylim(0, 105)
    axes[0].legend(fontsize=7, loc="lower left", framealpha=0.93)

    if cutoffs and v_lo is not None:
        cut = max(cutoffs)
        margin = v_lo - cut
        # A mission that never gets near the cut-off does not need the
        # cut-off in frame: keeping it there spends two thirds of the
        # panel on empty space and flattens the sag under load, which is
        # the one thing this panel shows that the SOC panel does not.
        # Above a third of a volt of margin the line becomes a legend
        # entry carrying the number instead.
        if margin > 0.35:
            axes[1].plot([], [], ls="--", lw=1.0, color=C_BUST,
                         label=f"cell cut-off {cut:.2f} V — "
                               f"{margin:.2f} V below this panel")
            axes[1].set_ylim(v_lo - 0.06, v_hi + 0.05)
        else:
            axes[1].axhline(cut, color=C_BUST, ls="--", lw=1.0,
                            label=f"cell cut-off {cut:.2f} V "
                                  f"({margin:.2f} V of margin)")
            axes[1].set_ylim(cut - 0.05, v_hi + 0.06)
    axes[1].set_ylabel("cell voltage [V]")
    axes[1].legend(fontsize=7, loc="lower left", framealpha=0.93, ncol=2)

    axes[2].set_ylabel("pack current [A]")
    axes[2].legend(fontsize=7, loc="upper right", framealpha=0.93)

    axes[3].set_ylabel("energy drawn [Wh]")
    axes[3].set_xlabel("time [min]")
    axes[3].legend(fontsize=7, loc="upper left", framealpha=0.93)

    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.margins(x=0.01)
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.subplots_adjust(top=0.94 if title else 0.98, bottom=0.07,
                        left=0.08, right=0.97)
    return fig


#: The metrics an aircraft comparison turns on, and which way is better.
SCORECARD = [
    ("energy_wh",            "energy [Wh]",              "lower"),
    ("time_min",             "flight time [min]",        "lower"),
    ("soc_min_pct",          "lowest SOC [%]",           "higher"),
    ("wh_per_km",            "Wh per km",                "lower"),
    ("wh_per_payload_km",    "Wh per payload-km",        "lower"),
    ("peak_c_rate",          "peak C-rate",              "lower"),
]


def plot_uav_scorecard(rows, ax=None, title: str = None, figsize=(15.0, 7.6)):
    """
    The headline aircraft comparison, six metrics at once.

    Grouped bars: one group per metric, one bar per aircraft, hatched by
    payload mode. Two of the six carry the argument.

    **Wh per km** flatters whichever aircraft is smallest, because a
    lighter machine spends less to move itself. **Wh per payload-km**
    charges each aircraft for what it actually delivered, and reverses
    the ranking whenever the small aircraft is carrying a third of what
    the large one is. Showing one without the other is how a comparison
    reaches the wrong recommendation while every number in it is right.

    Parameters
    ----------
    rows : sequence of dict
        One per (aircraft, payload mode), carrying the keys in
        :data:`SCORECARD` plus ``uav``, ``payload_mode`` and
        ``payload_kg``.
    """
    plt = _mpl()
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.ravel()
    uavs = list(dict.fromkeys(r["uav"] for r in rows))
    modes = list(dict.fromkeys(r["payload_mode"] for r in rows))
    cmap = plt.get_cmap("tab10")
    hatch = {m: h for m, h in zip(modes, ["", "///"])}

    for ax, (key, label, better) in zip(axes, SCORECARD):
        x = np.arange(len(uavs))
        w = 0.8 / max(len(modes), 1)
        for j, mode in enumerate(modes):
            vals, kg = [], []
            for u in uavs:
                hit = [r for r in rows
                       if r["uav"] == u and r["payload_mode"] == mode]
                vals.append(hit[0].get(key, np.nan) if hit else np.nan)
                kg.append(hit[0].get("payload_kg", np.nan) if hit else np.nan)
            pos = x - 0.4 + w * (j + 0.5)
            ax.bar(pos, vals, width=w * 0.92,
                   color=[cmap(i % 10) for i in range(len(uavs))],
                   alpha=1.0 if j == 0 else 0.55, hatch=hatch.get(mode, ""),
                   edgecolor="white", linewidth=0.6)
            for xi, v in zip(pos, vals):
                if np.isfinite(v):
                    ax.annotate(f"{v:.3g}", (xi, v), ha="center",
                                va="bottom", fontsize=6.5, color="0.25")
        ax.set_xticks(x)
        ax.set_xticklabels(uavs, fontsize=7.5)
        ax.set_title(f"{label}   ({better} is better)", fontsize=9,
                     fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(y=0.18)

    # Two legends, because the bars encode two things. Colour is the
    # aircraft and the hatch is the payload mode; the aircraft key was
    # missing, which left the reader matching colours to the tick
    # labels by position.
    kg = {}
    for r in rows:
        kg.setdefault(r["uav"], []).append(r.get("payload_kg", float("nan")))
    ac_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=cmap(i % 10), edgecolor="white",
                      label=f"{u}  ({', '.join(f'{v:.1f}' for v in kg[u])} kg)")
        for i, u in enumerate(uavs)]
    mode_handles = [plt.Rectangle((0, 0), 1, 1, facecolor="0.6",
                                  hatch=hatch.get(m, ""), edgecolor="white",
                                  alpha=1.0 if j == 0 else 0.55,
                                  label=f"{m} payload")
                    for j, m in enumerate(modes)]
    fig.legend(handles=ac_handles + mode_handles, loc="lower center",
               ncol=len(ac_handles) + len(mode_handles), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.035, 1, 0.955))
    return fig


def plot_driver_comparison(rows, title: str = None, figsize=(12.5, 4.4)):
    """
    Energy against balanced against time, and whether the cap was met.

    Three panels. The first two are the trade — what hurrying costs in
    energy and buys in time. The third is the one that decides whether
    the mission is flyable at all: an urgency carries a time cap, and an
    objective that minimises energy can miss it.

    A bar over the cap line is a mission that did not arrive in time. No
    amount of energy saving makes that acceptable for a delivery with a
    deadline, which is why the cap is drawn rather than left in a table.
    """
    plt = _mpl()
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    labels = [r["driver"] for r in rows]
    x = np.arange(len(rows))
    cmap = plt.get_cmap("Set2")
    colours = [cmap(i % 8) for i in range(len(rows))]

    axes[0].bar(x, [r["energy_wh"] for r in rows], color=colours)
    axes[0].set_ylabel("energy [Wh]")
    axes[0].set_title("What it costs", fontsize=9, fontweight="bold")

    axes[1].bar(x, [r["time_min"] for r in rows], color=colours)
    for i, r in enumerate(rows):
        cap = r.get("time_cap_min")
        if cap:
            axes[1].plot([i - 0.42, i + 0.42], [cap, cap], color=C_BUST,
                         lw=1.8, ls="--")
            if r["time_min"] > cap:
                axes[1].annotate("over cap", (i, r["time_min"]), ha="center",
                                 va="bottom", fontsize=7, color=C_BUST,
                                 fontweight="bold")
    axes[1].set_ylabel("flight time [min]")
    axes[1].set_title("What it buys — dashed is the urgency's cap",
                      fontsize=9, fontweight="bold")

    axes[2].bar(x, [r["soc_min_pct"] for r in rows], color=colours)
    for i, r in enumerate(rows):
        floor = r.get("soc_floor_pct")
        if floor:
            axes[2].plot([i - 0.42, i + 0.42], [floor, floor], color=C_BUST,
                         lw=1.8, ls="--")
    axes[2].set_ylabel("lowest SOC [%]")
    axes[2].set_title("What it leaves — dashed is the reserve",
                      fontsize=9, fontweight="bold")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(y=0.16)
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def plot_phase_comparison(runs, title: str = None, figsize=(12.5, 4.2)):
    """
    Energy share against time share, by phase, for several aircraft.

    One panel per aircraft so the phases stay side by side rather than
    stacked. The number to read is the ratio printed above each pair:
    above 1 the phase costs more than its share of the clock, and on a
    VTOL the two rotor-borne phases are always well above 1.

    Comparing aircraft here answers a question the totals cannot: whether
    a heavier machine is worse *everywhere* or only in the hover, which
    decides whether a short-hop mission and a long one want the same
    aircraft.
    """
    plt = _mpl()
    fig, axes = plt.subplots(1, len(runs), figsize=figsize, sharey=True)
    if len(runs) == 1:
        axes = [axes]
    for ax, r in zip(axes, runs):
        plot_energy_breakdown(r["energy"], ax=ax, title=r["label"])
        ax.set_ylabel("share of mission [%]" if ax is axes[0] else "")
        for t in ax.texts:
            if "energy share divided" in t.get_text():
                t.remove()
    axes[0].annotate("×n is energy share divided by time share — above 1 the "
                     "phase costs more than its share of the clock",
                     xy=(0.5, -0.34), xycoords="axes fraction", ha="center",
                     fontsize=7, color="0.4")
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    return fig


def plot_wind_exposure(runs, uav_configs, dem, title: str = None,
                       figsize=(12.0, 4.6), n_bearings: int = 24):
    """
    How much this route's schedule depends on a wind nobody has measured.

    Left, a polar rose of time penalty against wind direction, one curve
    per aircraft, with both of Quito's candidate directions marked. They
    are 250° apart, so a route whose rose is lobed has a schedule that
    depends on which source is right; a route whose rose is round does
    not.

    Right, penalty against wind speed at each aircraft's own worst
    direction, with each airframe's Beaufort limit drawn. The aircraft
    differ here for two reasons at once — a faster cruise resists wind
    better, and a higher wind rating lets the aircraft keep flying — and
    the panel separates them: where a curve stops mattering is set by
    speed, where it stops being legal by the rating.
    """
    plt = _mpl()
    from .wind import (BEAUFORT_MS, QUITO_ANNUAL_MEAN, QUITO_REANALYSIS,
                       QUITO_WINDIEST_MONTH, ROUGHNESS, WindField,
                       wind_penalty_report)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.30)
    ax_polar = fig.add_subplot(gs[0, 0], projection="polar")
    ax_speed = fig.add_subplot(gs[0, 1])
    cmap = plt.get_cmap("tab10")

    def field(speed, direction):
        return WindField(speed_ref=speed, direction_deg=direction,
                         reference_height_m=10.0,
                         reference_roughness_m=ROUGHNESS["open country"],
                         roughness_length_m=ROUGHNESS["dense urban"],
                         terrain_speedup=1.3)

    bearings = np.linspace(0, 360, n_bearings + 1)
    speed = QUITO_WINDIEST_MONTH.speed_ref
    curves, worst_dirs = [], []
    for r, u in zip(runs, uav_configs):
        pen = np.array([
            wind_penalty_report(r["path"], field(speed, float(b)), u,
                                dem=dem).get("time_penalty_s", 0.0) / 60.0
            for b in bearings])
        curves.append(pen)
        worst_dirs.append(float(bearings[int(np.argmax(pen))]))

    base = min(min(float(c.min()) for c in curves) * 1.25, -0.05)
    for i, (r, pen) in enumerate(zip(runs, curves)):
        ax_polar.plot(np.radians(bearings), pen - base, lw=1.7,
                      color=cmap(i % 10), label=r["label"])
    ax_polar.plot(np.radians(np.arange(0, 361, 5)),
                  np.full(73, -base), lw=1.2, color="0.3",
                  label="zero (inside = wind helps)")
    for w, tag, c in ((QUITO_ANNUAL_MEAN, "station NNW", "#C62828"),
                      (QUITO_REANALYSIS, "reanalysis E", "#EF6C00")):
        ax_polar.axvline(np.radians(w.direction_deg), color=c, lw=1.5,
                         ls="--", alpha=0.85)
        ax_polar.annotate(tag, xy=(np.radians(w.direction_deg), 1.03),
                          xycoords=("data", "axes fraction"), fontsize=6.5,
                          color=c, ha="center")
    top = max(float(c.max()) for c in curves) - base
    ax_polar.set_ylim(0, top * 1.08)
    ticks = np.linspace(base, top + base, 5)
    ax_polar.set_yticks(ticks - base)
    ax_polar.set_yticklabels([f"{t:+.1f}" for t in ticks])
    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)
    ax_polar.tick_params(labelsize=6.5)
    ax_polar.set_title(f"Penalty against wind direction\nat {speed:.1f} m/s",
                       fontsize=9, fontweight="bold", pad=18)
    ax_polar.legend(fontsize=6, loc="upper left",
                    bbox_to_anchor=(-0.34, 1.10), framealpha=0.9)

    speeds = np.linspace(0, 14, 22)
    for i, (r, u, brg) in enumerate(zip(runs, uav_configs, worst_dirs)):
        pen = [wind_penalty_report(r["path"], field(float(s), brg), u,
                                   dem=dem).get("time_penalty_s", 0.0) / 60.0
               for s in speeds]
        c = cmap(i % 10)
        ax_speed.plot(speeds, pen, lw=1.7, color=c,
                      label=f"{r['label']}  (worst from {brg:.0f}°)")
        ac = r.get("aircraft")
        if ac is not None and ac.wind_rating_beaufort:
            lo, _ = BEAUFORT_MS[ac.wind_rating_beaufort]
            ax_speed.axvline(lo, color=c, ls="--", lw=1.2, alpha=0.7)
    ax_speed.axvspan(QUITO_REANALYSIS.speed_ref, QUITO_WINDIEST_MONTH.speed_ref,
                     color="#90A4AE", alpha=0.30, lw=0)
    ax_speed.annotate("what Quito\nactually has",
                      xy=(QUITO_WINDIEST_MONTH.speed_ref + 0.25, 0.62),
                      xycoords=("data", "axes fraction"), fontsize=7,
                      color="#37474F")
    ax_speed.set_xlabel("wind at the 10 m mast [m/s]")
    ax_speed.set_ylabel("time penalty [min]")
    ax_speed.set_title("Penalty against wind speed\n(dashed: each airframe's "
                       "own limit)", fontsize=9, fontweight="bold")
    ax_speed.legend(fontsize=6.5, loc="upper left")
    ax_speed.spines[["top", "right"]].set_visible(False)

    fig.subplots_adjust(left=0.10, right=0.965, top=0.78, bottom=0.17,
                        wspace=0.38)
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    return fig
