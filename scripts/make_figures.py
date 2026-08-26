#!/usr/bin/env python3
"""
Generate the package's figures.

Produces the corridor figures the regulatory argument rests on, the
permission-coloured airspace map, the drag build-up, and a waiver map —
none of which existed before, and the first of which is the paper's
central figure.

Usage:
    python -m scripts.make_figures                 # all, into figures/
    python -m scripts.make_figures --only corridor
    python -m scripts.make_figures --out figures/paper --dpi 300
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from raptor.aero import get_polar
from raptor.airspace import build_airspace
from raptor.compliance import assess_compliance
from raptor.config import MissionConstraints, UAVConfig
from raptor.corridor import survey_corridor
from raptor.dem import DEMInterface, find_dem, void_report
from raptor.drag import quadplane_buildup
from raptor.energy import AircraftEnergyParams, analyze_path_energy
from raptor.regulations import MEDICAL_DELIVERY_CONTEXT, RDAC_101
from raptor.routed_path import RoutedPath
from raptor.builder import FacilityNode
from raptor.vehicles import get_vehicle
from raptor.terrain import TerrainAnalyzer
from raptor.viz_corridor import (
    C_BUST,
    plot_corridor, plot_corridor_comparison, plot_drag_breakdown,
    plot_permission_map, plot_waiver_map,
)
from raptor.wind import WindField

CORRIDORS = [
    ("H.Espejo → CS.Guamaní",   (-0.2000, -78.4930), (-0.3242, -78.5493)),
    ("H.Espejo → H.Calderón",   (-0.2000, -78.4930), (-0.0978, -78.4239)),
    ("H.Espejo → CS.Tumbaco",   (-0.2000, -78.4930), (-0.2158, -78.4085)),
    ("H.Garcés → CS.Lloa",      (-0.2641, -78.5504), (-0.2358, -78.5886)),
    ("H.Garcés → CS.Píntag",    (-0.2641, -78.5504), (-0.3833, -78.3889)),
    ("H.P.A.Suárez → IESS Sur", (-0.1273, -78.4977), (-0.2578, -78.5253)),
]


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
        "axes.linewidth": 0.6, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 7, "legend.framealpha": 0.92,
        "figure.dpi": 120, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    })


def _node(dem, name, lat, lon):
    return FacilityNode(name, lat, lon, float(dem.elevation(lat, lon)))


def _path(dem, uav, con, a, b, mode="agl", n=2):
    o = _node(dem, "O", *a)
    d = _node(dem, "D", *b)
    return RoutedPath(o, d, dem, uav, con, n_intermediate=n,
                      corridor_mode=mode).flight_path


def save(fig, out, name, dpi):
    path = os.path.join(out, name)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  {path}")


# ── Figures ───────────────────────────────────────────────────────────────

def fig_corridor(dem, uav, con, out, dpi):
    """The central figure: constant altitude against terrain-following."""
    label, a, b = CORRIDORS[2]          # Tumbaco: the most demanding relief
    amsl = _path(dem, uav, con, a, b, mode="amsl")
    agl = _path(dem, uav, con, a, b, mode="agl")

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.6))
    plot_corridor(amsl, dem, con, ax=axes[0],
                  title=f"{label} — constant-altitude cruise")
    plot_corridor(agl, dem, con, ax=axes[1],
                  title=f"{label} — terrain-following corridor")
    fig.tight_layout()
    save(fig, out, "corridor_amsl_vs_agl.png", dpi)

    fig = plot_corridor_comparison(
        [amsl, agl], ["constant altitude", "terrain-following"], dem, con,
        title=f"{label} — height above ground")
    save(fig, out, "corridor_agl_profile.png", dpi)


def fig_all_corridors(dem, uav, con, out, dpi):
    """Every study corridor in the frame the rule is written in."""
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 10.5))

    # Collect legend entries across *all* panels, not from the first.
    # The first corridor is the one with no ceiling exceedance, so it
    # never creates the "above ceiling" series — and taking the legend
    # from it left five panels showing red dots that nothing explained.
    seen, handles, names = set(), [], []
    for ax, (label, a, b) in zip(axes.ravel(), CORRIDORS):
        plot_corridor(_path(dem, uav, con, a, b), dem, con, ax=ax,
                      title=label)
        for h, n in zip(*ax.get_legend_handles_labels()):
            # Per-panel percentages differ, so the shared entry carries
            # the meaning and each panel's title carries the amount.
            key = n.split(" (")[0]
            if key not in seen:
                seen.add(key)
                handles.append(h)
                names.append(key)
        ax.legend().set_visible(False)
    fig.legend(handles, names, loc="lower center", ncol=5, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Terrain-following corridors, DMQ medical network",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "corridors_all.png", dpi)


def fig_permission_map(dem, uav, con, airspace, out, dpi):
    paths, labels = [], []
    for label, a, b in CORRIDORS[:4]:
        paths.append(_path(dem, uav, con, a, b))
        labels.append(label)
    fig, ax = plt.subplots(figsize=(8.0, 8.0))
    plot_permission_map(
        airspace, dem, paths=paths, labels=labels, ax=ax,
        bbox=(-0.40, 0.02, -78.62, -78.36),
        title="Airspace by permission class (RDAC 101.190)\ndirect corridors shown; the optimizer routes around the red zones",
        context=MEDICAL_DELIVERY_CONTEXT)
    save(fig, out, "airspace_permission_map.png", dpi)


def fig_waiver_map(dem, uav, con, airspace, out, dpi):
    ta = TerrainAnalyzer(dem, con)
    assessments = []
    for label, a, b in CORRIDORS:
        fp = _path(dem, uav, con, a, b)
        assessments.append(assess_compliance(
            fp, ta.analyze(fp), airspace.check_path(fp),
            regulation=RDAC_101, context=MEDICAL_DELIVERY_CONTEXT,
            constraints=con))
    fig, ax = plt.subplots(figsize=(8.0, 8.0))
    plot_waiver_map(assessments, [c[0] for c in CORRIDORS], dem, ax=ax,
                    bbox=(-0.42, 0.02, -78.62, -78.36),
                    title="Where height relief is needed (RDAC 101.035)")
    save(fig, out, "waiver_map.png", dpi)


def fig_drag(ac, out, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.4))
    plot_drag_breakdown(ac, V=30.0, altitude=2800.0, ax=axes[0])
    parkings = ["folded", "aligned", "unaligned"]
    speeds = np.linspace(20, 42, 40)
    for p in parkings:
        b = quadplane_buildup(rotor_parking=p)
        axes[1].plot(speeds, [b.C_D0(v, 2800.0) for v in speeds], lw=1.8,
                     label=f"rotors {p}")
    axes[1].axhline(0.013, color="#C62828", ls="--", lw=1.0,
                    label="previously assumed C_D0")
    axes[1].set_xlabel("airspeed [m/s]")
    axes[1].set_ylabel("non-wing $C_{D0}$")
    axes[1].set_title("Parasite drag vs speed and rotor parking",
                      fontsize=10, fontweight="bold")
    axes[1].legend(fontsize=7)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, out, "drag_buildup.png", dpi)


def fig_polar(ac, out, dpi):
    polar = get_polar(ac.wing_geometry, verbose=False)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6))
    cl = np.linspace(-0.2, 1.25, 60)
    for Re in polar.reynolds:
        axes[0].plot([polar.C_D(c, Re) for c in cl], cl, lw=1.6,
                     label=f"Re = {Re:.1e}")
    k = 1.0 / (np.pi * 0.78 * ac.AR)
    axes[0].plot(0.013 + k * cl ** 2, cl, "--", color="#C62828", lw=1.3,
                 label="previously assumed")
    axes[0].set_xlabel("$C_D$ (wing only)")
    axes[0].set_ylabel("$C_L$")
    axes[0].set_title("Computed drag polar", fontsize=10, fontweight="bold")
    axes[0].legend(fontsize=7)

    for Re in polar.reynolds:
        row = polar._row(Re, polar.CL)
        axes[1].plot(polar.alpha_deg, row, lw=1.6, label=f"Re = {Re:.1e}")
    axes[1].axhline(1.782, color="#C62828", ls="--", lw=1.3,
                    label="previously assumed $C_{L,max}$")
    axes[1].axhline(polar.C_L_max(3.2e5), color="#2E7D32", ls=":", lw=1.3,
                    label="computed $C_{L,max}$")
    axes[1].set_xlabel(r"$\alpha$ [deg]")
    axes[1].set_ylabel("$C_L$")
    axes[1].set_title("Lift curve and stall", fontsize=10, fontweight="bold")
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, out, "aero_polar.png", dpi)


def fig_dem_comparison(dem, out, dpi):
    """
    What DEM resolution does to the answer.

    The coarse grids here are the shipped 30 m DEM subsampled, not a
    separate product. That matters for the comparison: decimation
    changes the spacing and nothing else, so any difference in the bars
    is resolution alone, with no vertical datum, void-filling policy or
    acquisition date confounding it.
    """
    uav, con = UAVConfig(), MissionConstraints()
    grids = [(1, "30 m"), (3, "90 m"), (6, "180 m")]

    labels = [c[0] for c in CORRIDORS]
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    width = 0.26
    for i, (factor, name) in enumerate(grids):
        d = dem.decimated(factor)
        excess = [survey_corridor(d, a, b, con, uav).ceiling_excess_m
                  for _, a, b in CORRIDORS]
        ax.bar(np.arange(len(labels)) + (i - 1) * width, excess,
               width=width, label=f"{name} grid")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels([l.replace(" → ", "\n→ ") for l in labels], fontsize=7)
    ax.set_ylabel("ceiling exceedance [m]")
    ax.set_title("A coarse DEM smooths away the ridges that bind the corridor",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, out, "dem_resolution_effect.png", dpi)


def fig_wind(dem, uav, con, out, dpi):
    """
    What wind does to *these routes*, rather than to a textbook.

    The version this replaces drew a boundary-layer profile, a
    round-trip penalty curve and a bar chart of two data sources. All
    three were correct and none of them said anything about a corridor:
    a reader could not tell from any of them which route to worry about,
    or on which day.

    These three do. Each is built on the fact that makes Quito's wind
    interesting — **nobody knows which way it blows.** The station
    record says north-north-west, the reanalysis says east, and the two
    differ by 250°. Rather than pick one, sweep the direction and show
    what the answer depends on.
    """
    from raptor.wind import (BEAUFORT_MS, QUITO_ANNUAL_MEAN,
                             QUITO_REANALYSIS, QUITO_WINDIEST_MONTH,
                             ROUGHNESS, WindField, wind_penalty_report)

    ac = get_vehicle("va23").build_aero(verbose=False)
    v_cruise = uav.fw_cruise_airspeed

    fig = plt.figure(figsize=(15.0, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.05, 1.15], wspace=0.30)
    ax_polar = fig.add_subplot(gs[0, 0], projection="polar")
    ax_speed = fig.add_subplot(gs[0, 1])
    ax_track = fig.add_subplot(gs[0, 2])

    subset = CORRIDORS[:4]
    paths = [(lbl, _path(dem, uav, con, a, b)) for lbl, a, b in subset]
    cmap = plt.get_cmap("tab10")

    def field(speed, direction):
        return WindField(speed_ref=speed, direction_deg=direction,
                         reference_height_m=10.0,
                         reference_roughness_m=ROUGHNESS["open country"],
                         roughness_length_m=ROUGHNESS["dense urban"],
                         terrain_speedup=1.3)

    # ── 1. Cost against wind direction ────────────────────────────────
    # A wind rose of *penalty* rather than of frequency. A corridor that
    # is a circle on this plot does not care which way the wind blows; a
    # corridor that is a figure-of-eight is a scheduling problem.
    bearings = np.arange(0, 361, 15)
    speed = QUITO_WINDIEST_MONTH.speed_ref
    curves = []
    for lbl, fp in paths:
        pen = np.array([
            wind_penalty_report(fp, field(speed, float(b)), uav,
                                dem=dem).get("time_penalty_s", 0.0) / 60.0
            for b in bearings])
        curves.append(pen)

    # A penalty can be negative — a tailwind on a one-way leg makes the
    # flight quicker — and a polar axis cannot show a negative radius.
    # Shift every curve onto a common baseline and draw the zero as a
    # circle, so inside it means "the wind helps".
    floor = min(float(c.min()) for c in curves)
    base = min(floor * 1.25, -0.2)
    for i, ((lbl, _), pen) in enumerate(zip(paths, curves)):
        ax_polar.plot(np.radians(bearings), pen - base, lw=1.6,
                      color=cmap(i % 10), label=lbl.replace(" → ", "→"))
    ax_polar.plot(np.radians(np.arange(0, 361, 5)),
                  np.full(73, -base), lw=1.3, color="0.3", ls="-",
                  label="zero (inside = wind helps)")
    top = max(float(c.max()) for c in curves) - base
    ax_polar.set_ylim(0, top * 1.08)
    ticks = np.linspace(base, top + base, 5)
    ax_polar.set_yticks(ticks - base)
    ax_polar.set_yticklabels([f"{t:+.1f}" for t in ticks])
    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)
    for w, tag, c in ((QUITO_ANNUAL_MEAN, "station\nNNW", "#C62828"),
                      (QUITO_REANALYSIS, "reanalysis\nE", "#EF6C00")):
        ax_polar.axvline(np.radians(w.direction_deg), color=c, lw=1.6,
                         ls="--", alpha=0.85)
        ax_polar.annotate(tag, xy=(np.radians(w.direction_deg), 1.02),
                          xycoords=("data", "axes fraction"), fontsize=6.5,
                          color=c, ha="center")
    ax_polar.set_title(f"Time penalty against wind direction\n"
                       f"at {speed:.1f} m/s — the two sources disagree by 250°",
                       fontsize=9, fontweight="bold", pad=18)
    ax_polar.tick_params(labelsize=6.5)
    ax_polar.legend(fontsize=6, loc="upper left",
                    bbox_to_anchor=(-0.32, 1.10), framealpha=0.9)

    # ── 2. Cost against wind speed, at the worst direction ────────────
    speeds = np.linspace(0, 14, 24)
    for i, (lbl, fp) in enumerate(paths):
        worst_brg, worst = 0.0, -1e9
        for brg in np.arange(0, 360, 30):
            r = wind_penalty_report(fp, field(speed, float(brg)), uav, dem=dem)
            if r.get("time_penalty_s", 0.0) > worst:
                worst, worst_brg = r["time_penalty_s"], float(brg)
        pen = []
        for sp in speeds:
            r = wind_penalty_report(fp, field(float(sp), worst_brg), uav,
                                    dem=dem)
            pen.append(r.get("time_penalty_s", 0.0) / 60.0)
        ax_speed.plot(speeds, pen, lw=1.7, color=cmap(i % 10),
                      label=f"{lbl.replace(' → ', '→')}  (worst from "
                            f"{worst_brg:.0f}°)")
    lo, _ = BEAUFORT_MS[ac.wind_rating_beaufort]
    ax_speed.axvline(lo, color=C_BUST, lw=1.6, ls="--")
    ax_speed.annotate(f"{ac.name} limit\n(Beaufort "
                      f"{ac.wind_rating_beaufort}, {lo:.0f} m/s)",
                      xy=(lo, 0.93), xycoords=("data", "axes fraction"),
                      ha="right", va="top", fontsize=7, color=C_BUST)
    ax_speed.axvspan(QUITO_REANALYSIS.speed_ref, QUITO_WINDIEST_MONTH.speed_ref,
                     color="#90A4AE", alpha=0.30, lw=0)
    ax_speed.annotate("what Quito\nactually has",
                      xy=(QUITO_WINDIEST_MONTH.speed_ref + 0.3, 0.55),
                      xycoords=("data", "axes fraction"), fontsize=7,
                      color="#37474F")
    ax_speed.set_xlabel("wind at the 10 m mast [m/s]")
    ax_speed.set_ylabel("time penalty [min]")
    ax_speed.set_title("Cost against wind speed,\nat each corridor's worst "
                       "direction", fontsize=9, fontweight="bold")
    ax_speed.legend(fontsize=6.5, loc="upper left")
    ax_speed.spines[["top", "right"]].set_visible(False)

    # ── 3. The round trip, which loses both ways ──────────────────────
    # The result that is genuinely counterintuitive, and the one the
    # abstract penalty curve could not make concrete: a one-way leg can
    # *gain* from wind, and a round trip over the same ground never
    # does. The slow leg lasts longer than the fast leg saves, so the
    # two do not cancel — and the net is positive on every corridor and
    # from every direction.
    x = np.arange(len(paths))
    out_pen, back_pen = [], []
    brg_wind = QUITO_ANNUAL_MEAN.direction_deg
    w = field(QUITO_WINDIEST_MONTH.speed_ref, brg_wind)
    for lbl, fp in paths:
        arr = np.asarray(fp.get_waypoints_array(), float)
        d = float(arr[-1, 4])
        brg = DEMInterface.bearing(arr[0, 0], arr[0, 1], arr[-1, 0], arr[-1, 1])
        gs_out = w.ground_speed(v_cruise, brg, 90.0)
        gs_back = w.ground_speed(v_cruise, (brg + 180.0) % 360.0, 90.0)
        still = d / v_cruise
        out_pen.append((d / gs_out - still) / 60.0 if gs_out > 0 else np.nan)
        back_pen.append((d / gs_back - still) / 60.0 if gs_back > 0 else np.nan)

    out_pen = np.array(out_pen)
    back_pen = np.array(back_pen)
    net = out_pen + back_pen
    ax_track.bar(x - 0.26, out_pen, width=0.24, label="outbound",
                 color="#1565C0")
    ax_track.bar(x, back_pen, width=0.24, label="return", color="#EF6C00")
    ax_track.bar(x + 0.26, net, width=0.24, label="round trip, net",
                 color="#C62828")
    ax_track.axhline(0, color="0.35", lw=1.0)
    for xi, v in enumerate(net):
        ax_track.annotate(f"{v:+.2f}", (xi + 0.26, v), ha="center",
                          va="bottom" if v >= 0 else "top", fontsize=6.5,
                          color="#C62828")
    ax_track.set_xticks(x)
    ax_track.set_xticklabels([l.replace(" → ", "\n→ ") for l, _ in paths],
                             fontsize=6.5)
    ax_track.set_ylabel("time penalty [min]")
    ax_track.set_title(f"A one-way leg can gain; a round trip cannot\n"
                       f"{QUITO_WINDIEST_MONTH.speed_ref:.1f} m/s from "
                       f"{brg_wind:.0f}°", fontsize=9, fontweight="bold")
    ax_track.legend(fontsize=7)
    ax_track.spines[["top", "right"]].set_visible(False)

    # A polar axis is not compatible with tight_layout, so the spacing
    # is set explicitly instead.
    fig.subplots_adjust(left=0.09, right=0.965, top=0.80, bottom=0.16,
                        wspace=0.38)
    save(fig, out, "wind_sensitivity.png", dpi)


def fig_methods(dem, uav, con, out, dpi, effort=None):
    """
    Every planning method on one corridor, seen three ways.

    This is the comparison the package exists to make, and it needs all
    three views to be honest. The profile alone says the optimised route
    is better; the plan view says what that cost in lateral distance;
    the deviation trace says where the cost was paid.
    """
    from raptor.airspace import build_airspace
    from raptor.astar_baseline import AStarGridPlanner
    from raptor.optimizer import OptMode, PathOptimizer
    from raptor.vehicles import get_vehicle
    from raptor.viz_corridor import plot_plan_and_profile

    effort = effort or dict(maxiter=25, popsize=10)
    ac = get_vehicle("va23").build_aero(verbose=False)
    airspace = build_airspace(dem=dem)

    a, b = CORRIDORS[4][1], CORRIDORS[4][2]
    name = CORRIDORS[4][0]
    origin = FacilityNode("origin", a[0], a[1], dem.elevation(*a))
    dest = FacilityNode("destination", b[0], b[1], dem.elevation(*b))

    paths, labels, wpts = [], [], []

    def add(path, label, waypoints=None):
        paths.append(path)
        labels.append(label)
        wpts.append(waypoints)

    # 1. What you would fly with no rules at all.
    rp = RoutedPath(origin, dest, dem, uav, con, n_intermediate=0,
                    corridor_mode="amsl")
    add(rp.flight_path, "straight line, constant altitude",
        rp._waypoint_coords)

    # 2. Terrain-following, but blind to airspace.
    rp = RoutedPath(origin, dest, dem, uav, con, n_intermediate=0,
                    corridor_mode="agl")
    add(rp.flight_path, "terrain-following, DGAC off", rp._waypoint_coords)

    # 3. A* on a grid — a different search, same constraints.
    astar = AStarGridPlanner(dem, ac=ac, airspace=airspace,
                             grid_resolution_m=400.0)
    res = astar.plan((origin.lat, origin.lon, origin.ground_elev),
                     (dest.lat, dest.lon, dest.ground_elev))
    if res.path_found:
        add(res, f"A* grid ({res.grid_resolution_m:.0f} m cells)")
    else:
        print("  (A* found no path; omitted)")

    # 4-5. Differential evolution with the regulations on, at two
    #      waypoint counts — more freedom is not automatically better,
    #      and the figure should be able to show that.
    for n_wp in (1, 3):
        rp = RoutedPath(origin, dest, dem, uav, con, n_intermediate=n_wp,
                        corridor_mode="agl")
        opt = PathOptimizer(dem, uav, con, ac)
        r = opt.optimize_routed(rp, mode=OptMode.ENERGY, payload_kg=1.5,
                                airspace=airspace, seed=42, verbose=False,
                                **effort)
        add(r.optimized_path,
            f"optimised, DGAC on, N={n_wp}  ({r.energy_wh:.0f} Wh)",
            getattr(rp, "_waypoint_coords", None))

    o_name, d_name = [t.strip() for t in name.split("→")]
    fig = plot_plan_and_profile(paths, labels, dem, con, airspace=airspace,
                                waypoints=wpts, suptitle=name,
                                endpoint_names=(o_name, d_name))
    save(fig, out, "method_comparison.png", dpi)


def fig_planview(dem, uav, con, out, dpi):
    """
    Plan views of every study corridor, on one sheet.

    The repository's figures were all profiles, which cannot show where
    a route goes. Six panels, one per corridor, terrain-following with
    the regulations off — the shape of the problem before any
    optimisation is applied to it.
    """
    from raptor.airspace import build_airspace
    from raptor.viz_corridor import plot_plan_view

    airspace = build_airspace(dem=dem)
    n = len(CORRIDORS)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 5.4 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, (name, a, b) in zip(axes, CORRIDORS):
        origin = FacilityNode("origin", a[0], a[1], dem.elevation(*a))
        dest = FacilityNode("destination", b[0], b[1], dem.elevation(*b))
        rp = RoutedPath(origin, dest, dem, uav, con, n_intermediate=0,
                        corridor_mode="agl")
        # Name the ends. "origin" and "destination" say nothing on a
        # sheet where every panel has one of each.
        o_name, d_name = [t.strip() for t in name.split("→")]
        plot_plan_view([rp.flight_path], ["terrain-following"], dem, ax=ax,
                       airspace=airspace, title=name, show_waypoints=False,
                       endpoint_names=(o_name, d_name))
        ax.legend().remove()
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.tick_params(labelsize=6)
    for ax in axes[n:]:
        ax.axis("off")

    # One legend for the sheet. Each panel is at its own scale — the
    # corridors differ by a factor of three in length and forcing a
    # common extent would make the short ones unreadable — so the scale
    # bar in each panel is the thing to read, not the axis limits.
    from raptor.regulations import PermissionClass
    from raptor.viz_corridor import C_PERMIT, C_PROHIBITED
    handles = [
        plt.Line2D([], [], color="#1565C0", lw=2.0,
                   label="terrain-following corridor"),
        plt.Line2D([], [], marker="s", ls="", markersize=9, alpha=0.5,
                   markerfacecolor=C_PROHIBITED, markeredgecolor=C_PROHIBITED,
                   label=PermissionClass.PROHIBITED.value),
        plt.Line2D([], [], marker="s", ls="", markersize=9, alpha=0.5,
                   markerfacecolor=C_PERMIT, markeredgecolor=C_PERMIT,
                   label=PermissionClass.PERMIT.value),
    ]
    fig.legend(handles=handles, fontsize=9, ncol=3, loc="lower center",
               frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("The six study corridors, seen from above",
                 fontsize=13, fontweight="bold")
    # Equal-aspect panels come out at different heights, so the default
    # packing drops one panel's title onto its neighbour's axis.
    fig.tight_layout(rect=(0, 0.04, 1, 0.965), h_pad=3.2)
    save(fig, out, "corridors_plan_view.png", dpi)


FIGURES = {
    "corridor": fig_corridor,
    "corridors_all": fig_all_corridors,
    "permission": fig_permission_map,
    "waiver": fig_waiver_map,
    "drag": fig_drag,
    "polar": fig_polar,
    "dem": fig_dem_comparison,
    "wind": fig_wind,
    "planview": fig_planview,
    "methods": fig_methods,
}

#: Figures that run the optimizer and therefore take minutes, not
#: seconds. Excluded from a bare run so the common case stays fast.
SLOW = {"methods"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="figures/current")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--maxiter", type=int, default=25,
                    help="optimizer generations for the 'methods' figure")
    ap.add_argument("--popsize", type=int, default=10,
                    help="optimizer population for the 'methods' figure")
    ap.add_argument("--only", nargs="+", choices=sorted(FIGURES),
                    help="generate only these")
    args = ap.parse_args(argv)

    style()
    os.makedirs(args.out, exist_ok=True)

    dem = DEMInterface(find_dem())
    con = MissionConstraints()
    ac = AircraftEnergyParams().build_aero()
    uav = UAVConfig.for_vehicle(ac, altitude=2800.0)
    airspace = build_airspace(dem=dem)

    print(f"DEM      {find_dem()}")
    print(f"         {dem.metadata.provenance()}")
    print(f"aero     {ac.aero_source}")
    print(f"corridor {con.describe_corridor()}")
    print(f"airspace {len(airspace.zones)} zones")
    print(f"writing to {args.out}/\n")

    wanted = args.only or [n for n in sorted(FIGURES) if n not in SLOW]
    if not args.only and SLOW:
        print(f"(skipping {', '.join(sorted(SLOW))} — runs the optimizer; "
              f"add --only {' '.join(sorted(SLOW))} to include)\n")
    for name in wanted:
        print(f"{name}:")
        fn = FIGURES[name]
        try:
            if name in ("permission", "waiver"):
                fn(dem, uav, con, airspace, args.out, args.dpi)
            elif name in ("drag", "polar"):
                fn(ac, args.out, args.dpi)
            elif name == "dem":
                fn(dem, args.out, args.dpi)
            elif name == "methods":
                fn(dem, uav, con, args.out, args.dpi,
                   effort=dict(maxiter=args.maxiter, popsize=args.popsize))
            else:
                fn(dem, uav, con, args.out, args.dpi)
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
