#!/usr/bin/env python3
"""
Demo: Where the Energy Goes
============================

A mission's energy is not one number, and the interesting part is the
split. This walks a single delivery and shows:

1. **Density altitude.** The same aircraft at sea level and at Quito's
   2 800 m — stall speed, hover power and cruise power.
2. **Phase by phase.** Which segments cost what. On a VTOL the answer
   is rarely the one people expect.
3. **The pack.** Terminal voltage, current and the I²R heat that never
   reaches the propellers.

Usage:
    python -m examples.demo_energy_analysis
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raptor.atmosphere import isa_density
from raptor.builder import FacilityNode
from raptor.config import MissionConstraints, UAVConfig
from raptor.dem import DEMInterface, find_dem
from raptor.energy import (
    analyze_path_energy, power_fw_cruise, power_hover,
)
from raptor.routed_path import RoutedPath
from raptor.scenarios import (
    HOSPITAL_EUGENIO_ESPEJO, HOSPITAL_PABLO_ARTURO_SUAREZ,
)
from raptor.vehicles import get_vehicle


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vehicle", default="va23")
    ap.add_argument("--payload", type=float, default=1.5, help="kg")
    args = ap.parse_args(argv)

    dem = DEMInterface(find_dem())
    ac = get_vehicle(args.vehicle).build_aero(verbose=False)
    loaded = ac.with_payload(args.payload)
    con = MissionConstraints()
    uav = UAVConfig.for_vehicle(loaded, altitude=2800.0)

    print(f"{ac.name}, {args.payload} kg payload "
          f"({loaded.m_tow:.1f} kg all-up of {ac.m_tow_max:.1f} kg MTOW)")
    print(f"aero: {ac.aero_source}")
    print(f"terrain: {dem.metadata.provenance()}\n")

    # ── 1. What altitude costs ────────────────────────────────────────
    print("The price of altitude — same aircraft, thinner air\n")
    print(f"{'alt [m]':>8} {'rho':>7} {'stall':>7} {'1.3x':>7} "
          f"{'hover [W]':>10} {'cruise [W]':>11}")
    print("-" * 55)
    for alt in (0, 1500, 2800, 3500, 4200):
        rho = isa_density(alt)
        vs = loaded.stall_speed_at(alt)
        p_h = power_hover(loaded, alt) / loaded.eta_vtol_effective
        p_c = (power_fw_cruise(loaded, max(1.5 * vs, 1.0), alt)
               / loaded.eta_cruise_effective)
        print(f"{alt:8d} {rho:7.3f} {vs:7.1f} {vs * 1.3:7.1f} "
              f"{p_h:10.0f} {p_c:11.0f}")
    print("\nHover power rises as the square root of density, cruise power")
    print("less steeply — which is why a VTOL's worst phase gets worse")
    print("faster than its best one as the ground rises under it.\n")

    # ── 2. Phase by phase on a real leg ───────────────────────────────
    a, b = HOSPITAL_EUGENIO_ESPEJO, HOSPITAL_PABLO_ARTURO_SUAREZ
    origin = FacilityNode(a.name, a.lat, a.lon, dem.elevation(a.lat, a.lon))
    dest = FacilityNode(b.name, b.lat, b.lon, dem.elevation(b.lat, b.lon))
    path = RoutedPath(origin, dest, dem, uav, con,
                      n_intermediate=1, corridor_mode="agl").flight_path

    result = analyze_path_energy(path, loaded)
    print(f"{a.name} to {b.name}: "
          f"{path.metrics.total_ground_distance / 1000:.1f} km, "
          f"{result.total_time / 60:.1f} min, "
          f"{result.total_energy_wh:.0f} Wh\n")

    by_phase = {}
    for seg in result.segments:
        e = by_phase.setdefault(seg.segment_type, [0.0, 0.0, 0])
        e[0] += seg.energy_wh
        e[1] += seg.duration
        e[2] += 1

    print(f"{'phase':14s} {'n':>4} {'time':>8} {'energy':>9} {'share':>7} "
          f"{'mean P':>9}")
    print("-" * 56)
    for phase, (wh, t, n) in sorted(by_phase.items(),
                                    key=lambda kv: -kv[1][0]):
        print(f"{phase:14s} {n:4d} {t / 60:6.1f}min {wh:7.1f}Wh "
              f"{100 * wh / result.total_energy_wh:6.1f}% "
              f"{wh * 3600 / max(t, 1e-9):8.0f}W")

    # ── 3. The pack ───────────────────────────────────────────────────
    print(f"\nPack: {ac.battery_cells_series}S"
          f"{ac.battery_strings_parallel:.1f}P, "
          f"{ac.battery_energy_wh:.0f} Wh nameplate, "
          f"{ac.battery_usable_wh:.0f} Wh usable, "
          f"{1000 * ac.battery_resistance:.0f} mOhm, {ac.cell_chemistry}")
    print(f"  SOC 100% -> {100 * result.SOC_final:.1f}% "
          f"(lowest {100 * result.min_SOC:.1f}%)")
    print(f"  peak discharge {result.max_c_rate:.1f} C "
          f"(limit {ac.battery_max_c_rate:.0f} C)"
          + ("   ! exceeded" if result.c_rate_exceeded else ""))
    print(f"  I2R heat {result.ohmic_loss_wh:.1f} Wh — "
          f"{100 * result.ohmic_loss_wh / max(result.energy_from_cells_wh, 1e-9):.1f}% "
          f"of what left the cells never reached a propeller")
    if result.power_limited:
        print("  ! the pack could not meet the demanded power at some point")

    print("\nThe hover and transition phases are a small part of the")
    print("distance and a large part of the energy. That ratio is what")
    print("makes a multi-stop mission expensive: every extra stop pays")
    print("for another descent, hover and climb-out.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
