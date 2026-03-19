#!/usr/bin/env python3
"""
Demo: Energy Model & Stall Analysis
======================================

Demonstrates: power models, stall speed, altitude effects, battery SOC.

Usage:
    python -m examples.demo_energy_analysis
    python -m examples.demo_energy_analysis --plot
"""
import sys, os, argparse, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from raptor.energy import (
    AircraftEnergyParams, analyze_path_energy,
    power_fw_cruise, power_hover, power_vertical_ascent,
)
from raptor.atmosphere import isa_density

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true', default=True)
    parser.add_argument('--no-plot', dest='plot', action='store_false')
    args = parser.parse_args()

    print("=== Demo: Energy Model ===\n")
    ac = AircraftEnergyParams()
    mass = ac.W / ac.g
    print(f"Aircraft: {mass:.0f}kg MTOW, S={ac.S_ref:.3f}m², battery={ac.battery_energy_wh:.0f}Wh\n")

    print(f"{'Alt[m]':>7s} | {'ρ[kg/m³]':>9s} | {'V_stall':>7s} | {'V_safe':>7s} | {'P_cruise':>8s} | {'P_hover':>8s}")
    print("-" * 60)
    for alt in [0, 1000, 2000, 2850, 3500, 4500]:
        rho = isa_density(alt)
        vs = ac.stall_speed_at(alt)
        pc = power_fw_cruise(ac, 25.0, alt) / ac.eta_prop
        ph = power_hover(ac, alt)
        print(f"{alt:7d} | {rho:9.3f} | {vs:7.1f} | {vs*1.3:7.1f} | {pc:8.0f} | {ph:8.0f}")

    if args.plot:
        import matplotlib; matplotlib.use('Agg')
        from raptor.visualization import plot_stall_envelope
        fig = plot_stall_envelope(ac, save_path='demo_stall_envelope.png')
        print("\nSaved demo_stall_envelope.png")

if __name__ == '__main__':
    main()
