"""
Parametric Vehicle Configurations
==================================

Pre-defined eVTOL configurations for comparative studies:
    - Baseline: the default medical delivery quad-plane (~12 kg)
    - Heavy cargo: larger platform (~25 kg)
    - Long range: optimized for endurance
    - High altitude: adapted for >4000m operations

Each config returns an AircraftEnergyParams object with correct
derived quantities (constructed via dataclass parameters).

Author: Victor (EPN / LUAS-EPN)
"""

from __future__ import annotations
from typing import Dict
from .energy import AircraftEnergyParams


def baseline_config() -> AircraftEnergyParams:
    """Default medical delivery quad-plane (~12 kg MTOW)."""
    return AircraftEnergyParams()


def heavy_cargo_config() -> AircraftEnergyParams:
    """Larger platform for heavy payloads (~25 kg MTOW)."""
    return AircraftEnergyParams(
        m_tow=25.0, S_ref=0.85, C_D0=0.035, AR=7.5,
        A_rotor=0.40, V_tip=90.0, eta_prop=0.58,
        battery_mass=6.0, specific_energy=200.0, battery_voltage=44.4,
    )


def long_range_config() -> AircraftEnergyParams:
    """Endurance-optimized platform (~10 kg, high L/D)."""
    return AircraftEnergyParams(
        m_tow=10.0, S_ref=0.55, C_D0=0.022, AR=10.0,
        e_oswald=0.82, C_L_max=1.5, A_rotor=0.18, eta_prop=0.72,
        battery_mass=3.0, specific_energy=220.0,
    )


def high_altitude_config() -> AircraftEnergyParams:
    """Platform adapted for >4000m operations."""
    return AircraftEnergyParams(
        m_tow=14.0, S_ref=0.70, C_D0=0.028, AR=8.5,
        C_L_max=1.4, A_rotor=0.28, V_tip=85.0, eta_prop=0.65,
        battery_mass=3.5, specific_energy=210.0,
    )


VEHICLE_CONFIGS: Dict[str, callable] = {
    'baseline': baseline_config,
    'heavy_cargo': heavy_cargo_config,
    'long_range': long_range_config,
    'high_altitude': high_altitude_config,
}


def get_vehicle(name: str) -> AircraftEnergyParams:
    """Get a vehicle configuration by name."""
    if name not in VEHICLE_CONFIGS:
        raise KeyError(f"Unknown vehicle '{name}'. Available: {list(VEHICLE_CONFIGS.keys())}")
    return VEHICLE_CONFIGS[name]()


def compare_vehicles_at_altitude(altitude: float = 2850.0, airspeed: float = 25.0) -> Dict:
    """Compare all vehicle configs at a given altitude."""
    from .energy import power_fw_cruise, power_hover
    results = {}
    for name, factory in VEHICLE_CONFIGS.items():
        ac = factory()
        mass = ac.W / ac.g
        vs = ac.stall_speed_at(altitude)
        p_cruise = power_fw_cruise(ac, airspeed, altitude) / ac.eta_prop
        p_hover = power_hover(ac, altitude)
        results[name] = {
            'mass_kg': mass, 'battery_wh': ac.battery_energy_wh,
            'V_stall_ms': vs, 'V_safe_ms': vs * 1.3,
            'P_cruise_W': p_cruise, 'P_hover_W': p_hover,
            'endurance_hover_min': ac.battery_energy_wh / p_hover * 60,
            'range_cruise_km': (ac.battery_energy_wh / p_cruise) * airspeed / 1000 * 3600,
        }
    return results
