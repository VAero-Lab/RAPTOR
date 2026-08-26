#!/usr/bin/env python3
"""
Demo: The Three Aircraft, and Where Their Numbers Come From
============================================================

RAPTOR ships three vehicles, each derived from a real aircraft's
published data. This shows what they are, how they differ, and — the
part that matters for a paper — which of their forty-odd parameters can
be checked against a datasheet and which are engineering judgement.

Usage:
    python -m examples.demo_vehicle
    python -m examples.demo_vehicle --provenance va23
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raptor.config import UAVConfig
from raptor.energy import power_fw_cruise, power_hover
from raptor.vehicles import VEHICLE_CONFIGS, get_vehicle

ALT = 2800.0     # Quito basin floor, where these aircraft work


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provenance", metavar="NAME",
                    help="print where every parameter of one vehicle came from")
    ap.add_argument("--no-aero", dest="aero", action="store_false",
                    default=True,
                    help="skip the lifting-line polar. Fast, and wrong in a "
                         "specific way: without the airframe build-up the "
                         "wing alone has about twice the real L/D, so every "
                         "derived climb angle comes out far too shallow.")
    args = ap.parse_args(argv)

    names = [n for n in VEHICLE_CONFIGS if n != "default"]

    if args.provenance:
        ac = get_vehicle(args.provenance)
        print(f"{ac.name}\n{'=' * len(ac.name)}\n")
        prov = ac.provenance or {}
        print(f"source: {prov.get('_source_url', 'unrecorded')}")
        print(f"layout: {prov.get('_configuration', 'unrecorded')}\n")
        for kind in ("referenced", "derived", "assumed"):
            entries = {k: v for k, v in prov.items()
                       if not k.startswith("_") and v.startswith(kind)}
            print(f"{kind.upper()}  ({len(entries)})")
            for k, v in entries.items():
                print(f"  {k}\n      {v.split('— ', 1)[-1]}")
            print()
        return 0

    # ── Side by side ──────────────────────────────────────────────────
    print("The three aircraft, at maximum take-off mass\n")
    hdr = (f"{'':16s} {'MTOW':>6s} {'payload':>8s} {'span':>6s} {'wing ld':>8s} "
           f"{'disk ld':>8s} {'pack':>9s} {'wind':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for n in names:
        ac = get_vehicle(n)
        print(f"{ac.name:16s} {ac.m_tow_max:5.1f}kg {ac.payload_capacity_kg:7.1f}kg "
              f"{(ac.S_ref * ac.AR) ** 0.5:5.2f}m "
              f"{ac.m_tow_max / ac.S_ref:6.1f}kg/m2 "
              f"{ac.m_tow_max / ac.rotor_area_total:6.1f}kg/m2 "
              f"{ac.battery_energy_wh:6.0f}Wh {ac.wind_limit_ms:5.1f}m/s")

    # ── What each can do over Quito ───────────────────────────────────
    src = ("lifting line + airframe build-up" if args.aero
           else "declared constants — see --no-aero")
    print(f"\nSpeed and power at {ALT:.0f} m, loaded to MTOW  ({src})\n")
    hdr = (f"{'':16s} {'stall':>6s} {'min':>6s} {'cruise':>7s} {'max':>6s} "
           f"{'climb':>7s} {'turn':>6s} {'hover':>7s} {'cruise P':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for n in names:
        ac = get_vehicle(n)
        if args.aero:
            ac = ac.build_aero(verbose=False)
        loaded = ac.with_payload(ac.payload_capacity_kg)
        uav = UAVConfig.for_vehicle(loaded, altitude=ALT)
        p_h = power_hover(loaded, ALT) / loaded.eta_vtol_effective
        p_c = (power_fw_cruise(loaded, uav.fw_cruise_airspeed, ALT)
               / loaded.eta_cruise_effective)
        print(f"{ac.name:16s} {loaded.stall_speed_at(ALT):5.1f} "
              f"{uav.fw_min_airspeed:5.1f} {uav.fw_cruise_airspeed:6.1f} "
              f"{uav.fw_max_airspeed:5.1f} {uav.fw_max_climb_angle:6.1f}d "
              f"{uav.min_turn_radius:5.0f}m {p_h:6.0f}W {p_c:8.0f}W")

    # ── The check that can fail ───────────────────────────────────────
    print("\nDo the definitions contradict themselves?\n")
    for n in names:
        ac = get_vehicle(n)
        loaded = ac.with_payload(ac.payload_capacity_kg)
        problems = ac.validate() + UAVConfig.for_vehicle(
            loaded, altitude=ALT).manoeuvre_problems()
        if problems:
            print(f"  {ac.name}:")
            for p in problems:
                print(f"    ! {p}")
        else:
            print(f"  {ac.name:16s} no warnings")

    # And the same check on the hand-typed defaults, which do contradict
    # themselves — turn radius, bank angle and cruise speed are three
    # names for two independent facts, and typed in separately they drift.
    stale = UAVConfig().manoeuvre_problems()
    print(f"\nFor contrast, UAVConfig() with its historical hand-typed "
          f"constants:\n")
    for p in stale:
        print(f"    ! {p}")

    print("\nRun with --provenance va23 to see where each number came from.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
