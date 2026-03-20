"""
Vehicle Definitions — Physical, Geometrical & Battery Parameters
================================================================

Central module for all vehicle-related definitions:
    - AircraftEnergyParams  — dataclass holding every vehicle parameter
    - load_vehicle_from_json — construct a vehicle from a JSON file
    - Factory functions      — pre-defined configurations (also backed by JSON)
    - compare_vehicles_at_altitude — parametric comparison utility

JSON format (see data/default_vehicle.json and data/vehicles/):
    {
      "name": "...",
      "description": "...",
      "airframe":   { m_tow, g, S_ref, C_L, C_L_max, C_D0, e_oswald, AR },
      "rotors":     { A_rotor, V_tip, sigma_rotor, C_d_blade, k_i },
      "propulsion": { eta_prop },
      "battery":    { battery_mass, specific_energy, battery_voltage,
                      battery_capacity_mah }
    }

Author: Victor (LUAS-EPN / KU Leuven)
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np

from .atmosphere import isa_density


# ─── locate the package data directory ───────────────────────────────────────
_PKG_DIR = os.path.dirname(__file__)          # raptor/raptor/
_DATA_DIR = os.path.join(_PKG_DIR, '..', 'data')   # raptor/data/
_VEHICLES_DIR = os.path.join(_DATA_DIR, 'vehicles')


# ═════════════════════════════════════════════════════════════════════════════
# AIRCRAFT PARAMETER DATACLASS
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class AircraftEnergyParams:
    """
    Complete physical, geometrical, propulsion and battery parameters
    of an eVTOL lift+cruise aircraft.


    Instances can be created:
        - Directly:           AircraftEnergyParams(m_tow=12.0, ...)
        - From a JSON file:   AircraftEnergyParams.from_json('data/default_vehicle.json')
        - Via factory:        get_vehicle('baseline')
    """

    # --- Airframe ---
    m_tow: float = 12.0              # Total take-off mass [kg]
    g: float = 9.80665               # Gravitational acceleration [m/s²]
    S_ref: float = 0.50              # Wing reference area [m²]
    C_L: float = 1.159               # Lift coefficient (cruise)
    C_L_max: float = 1.782           # Maximum lift coefficient
    C_D0: float = 0.013              # Parasitic drag coefficient (zero-lift)
    e_oswald: float = 0.78           # Oswald efficiency factor
    AR: float = 8.0                  # Wing aspect ratio

    # --- VTOL Rotors ---
    A_rotor: float = 0.20            # Total rotor disk area (all lift rotors) [m²]
    V_tip: float = 80.0              # Rotor tip speed [m/s]
    sigma_rotor: float = 0.06        # Rotor solidity (Nc/πR)
    C_d_blade: float = 0.012         # Mean blade drag coefficient
    k_i: float = 1.15                # Induced power correction factor

    # --- Propulsion ---
    eta_prop: float = 0.6            # Overall propulsion efficiency

    # --- Battery ---
    battery_mass: float = 3.0        # Battery mass [kg]
    specific_energy: float = 200.0   # Battery specific energy [Wh/kg]
    battery_voltage: float = 22.2    # Nominal battery voltage [V] (6S LiPo)
    battery_capacity_mah: float = 0.0  # If 0, auto-computed from mass & specific energy

    # --- Metadata (set by from_json / factory functions) ---
    name: str = "Baseline Medical Delivery"
    description: str = ""

    def __post_init__(self):
        """Compute derived aerodynamic and battery quantities."""
        self.W = self.m_tow * self.g
        self.k_drag = 1.0 / (np.pi * self.e_oswald * self.AR)
        self.C_D = self.C_D0 + self.k_drag * self.C_L ** 2
        if self.battery_capacity_mah <= 0:
            energy_wh = self.battery_mass * self.specific_energy
            self.battery_capacity_mah = energy_wh / self.battery_voltage * 1000.0
        self.battery_energy_wh = self.battery_mass * self.specific_energy
        self.battery_energy_j = self.battery_energy_wh * 3600.0

    # ── Aerodynamic helpers ───────────────────────────────────────────────

    @property
    def stall_speed(self) -> float:
        """Stall speed at reference altitude 3000 m [m/s]."""
        return self.stall_speed_at(3000.0)

    def stall_speed_at(self, altitude: float, gamma_deg: float = 0.0) -> float:
        """
        Stall speed at a given altitude and flight-path angle [m/s].

        V_stall = sqrt(2·W·cos(γ) / (ρ·S·C_L_max))
        """
        rho = isa_density(altitude)
        gamma = np.radians(abs(gamma_deg))
        return np.sqrt(2 * self.W * np.cos(gamma) / (rho * self.S_ref * self.C_L_max))

    def C_L_required(self, V: float, altitude: float,
                     gamma_deg: float = 0.0) -> float:
        """Required lift coefficient for steady flight at V and altitude."""
        rho = isa_density(altitude)
        gamma = np.radians(abs(gamma_deg))
        q = 0.5 * rho * V ** 2
        if q * self.S_ref < 1e-6:
            return float('inf')
        return self.W * np.cos(gamma) / (q * self.S_ref)

    def check_aero_feasibility(self, V: float, altitude: float,
                               gamma_deg: float = 0.0,
                               safety_margin: float = 1.3) -> dict:
        """Check stall and speed-margin feasibility for a flight condition."""
        V_stall = self.stall_speed_at(altitude, gamma_deg)
        V_min_safe = safety_margin * V_stall
        C_L_req = self.C_L_required(V, altitude, gamma_deg)
        stall_margin = V / V_stall if V_stall > 0 else float('inf')

        feasible = True
        messages = []
        if C_L_req > self.C_L_max:
            feasible = False
            messages.append(
                f"STALL: C_L_req={C_L_req:.3f} > C_L_max={self.C_L_max:.3f} "
                f"at V={V:.1f} m/s, alt={altitude:.0f} m, γ={gamma_deg:.1f}°"
            )
        if V < V_min_safe:
            feasible = False
            messages.append(
                f"Below safe speed: V={V:.1f} m/s < V_min={V_min_safe:.1f} m/s "
                f"(V_stall={V_stall:.1f} m/s × margin={safety_margin:.1f})"
            )
        return {
            'feasible': feasible,
            'C_L_required': C_L_req,
            'C_L_max': self.C_L_max,
            'stall_margin': stall_margin,
            'V_stall': V_stall,
            'V_min_safe': V_min_safe,
            'message': '; '.join(messages) if messages else 'OK',
        }

    def C_D_at_CL(self, C_L: float) -> float:
        """Total drag coefficient at a given lift coefficient."""
        return self.C_D0 + self.k_drag * C_L ** 2

    def thrust_required(self, V: float, altitude: float,
                        gamma_deg: float = 0.0) -> float:
        """Thrust required for steady flight at V, altitude, and γ."""
        rho = isa_density(altitude)
        gamma = np.radians(gamma_deg)
        q = 0.5 * rho * V ** 2
        D = q * self.S_ref * self.C_D
        return D + self.W * np.sin(gamma)

    def to_dict(self) -> dict:
        """Serialize to the canonical JSON-compatible dict."""
        return {
            "name": self.name,
            "description": self.description,
            "airframe": {
                "m_tow": self.m_tow, "g": self.g,
                "S_ref": self.S_ref, "C_L": self.C_L,
                "C_L_max": self.C_L_max, "C_D0": self.C_D0,
                "e_oswald": self.e_oswald, "AR": self.AR,
            },
            "rotors": {
                "A_rotor": self.A_rotor, "V_tip": self.V_tip,
                "sigma_rotor": self.sigma_rotor, "C_d_blade": self.C_d_blade,
                "k_i": self.k_i,
            },
            "propulsion": {"eta_prop": self.eta_prop},
            "battery": {
                "battery_mass": self.battery_mass,
                "specific_energy": self.specific_energy,
                "battery_voltage": self.battery_voltage,
                "battery_capacity_mah": self.battery_capacity_mah,
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'AircraftEnergyParams':
        """Construct from the canonical dict format (as stored in JSON files)."""
        af = d.get("airframe", {})
        ro = d.get("rotors", {})
        pr = d.get("propulsion", {})
        ba = d.get("battery", {})
        return cls(
            name=d.get("name", ""),
            description=d.get("description", ""),
            m_tow=af.get("m_tow", 12.0),
            g=af.get("g", 9.80665),
            S_ref=af.get("S_ref", 0.50),
            C_L=af.get("C_L", 1.159),
            C_L_max=af.get("C_L_max", 1.782),
            C_D0=af.get("C_D0", 0.013),
            e_oswald=af.get("e_oswald", 0.78),
            AR=af.get("AR", 8.0),
            A_rotor=ro.get("A_rotor", 0.20),
            V_tip=ro.get("V_tip", 80.0),
            sigma_rotor=ro.get("sigma_rotor", 0.06),
            C_d_blade=ro.get("C_d_blade", 0.012),
            k_i=ro.get("k_i", 1.15),
            eta_prop=pr.get("eta_prop", 0.6),
            battery_mass=ba.get("battery_mass", 3.0),
            specific_energy=ba.get("specific_energy", 200.0),
            battery_voltage=ba.get("battery_voltage", 22.2),
            battery_capacity_mah=ba.get("battery_capacity_mah", 0.0),
        )

    @classmethod
    def from_json(cls, path: str) -> 'AircraftEnergyParams':
        """Load vehicle parameters from a JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


# ═════════════════════════════════════════════════════════════════════════════
# JSON LOADING HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def load_vehicle_from_json(path: str) -> AircraftEnergyParams:
    """
    Load a vehicle configuration from a JSON file.

    Parameters
    ----------
    path : str
        Absolute or relative path to a vehicle JSON file.

    Returns
    -------
    AircraftEnergyParams
    """
    return AircraftEnergyParams.from_json(path)


def load_default_vehicle() -> AircraftEnergyParams:
    """
    Load the default vehicle from data/default_vehicle.json.

    Falls back to hardcoded defaults if the file is not found.
    """
    json_path = os.path.normpath(os.path.join(_DATA_DIR, 'default_vehicle.json'))
    if os.path.exists(json_path):
        return AircraftEnergyParams.from_json(json_path)
    return AircraftEnergyParams()


def list_vehicle_configs() -> Dict[str, str]:
    """
    Return a dict of {name: json_path} for all vehicle JSON files
    found in data/vehicles/.
    """
    configs = {}
    if os.path.isdir(_VEHICLES_DIR):
        for fname in sorted(os.listdir(_VEHICLES_DIR)):
            if fname.endswith('.json'):
                key = fname[:-5]   # strip .json
                configs[key] = os.path.normpath(os.path.join(_VEHICLES_DIR, fname))
    return configs


# ═════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS  (backed by JSON files; fall back to hard-coded values)
# ═════════════════════════════════════════════════════════════════════════════

def _load_or_default(filename: str, fallback: AircraftEnergyParams) -> AircraftEnergyParams:
    """Try to load from data/vehicles/<filename>; return fallback on failure."""
    path = os.path.normpath(os.path.join(_VEHICLES_DIR, filename))
    if os.path.exists(path):
        return AircraftEnergyParams.from_json(path)
    return fallback


def baseline_config() -> AircraftEnergyParams:
    """Default medical delivery quad-plane (~12 kg MTOW)."""
    return _load_or_default('baseline.json', AircraftEnergyParams())


def heavy_cargo_config() -> AircraftEnergyParams:
    """Larger platform for heavy payloads (~25 kg MTOW)."""
    fallback = AircraftEnergyParams(
        name="Heavy Cargo",
        m_tow=25.0, S_ref=0.85, C_D0=0.035, AR=7.5,
        A_rotor=0.40, V_tip=90.0, eta_prop=0.58,
        battery_mass=6.0, specific_energy=200.0, battery_voltage=44.4,
    )
    return _load_or_default('heavy_cargo.json', fallback)


def long_range_config() -> AircraftEnergyParams:
    """Endurance-optimized platform (~10 kg, high L/D)."""
    fallback = AircraftEnergyParams(
        name="Long Range",
        m_tow=10.0, S_ref=0.55, C_D0=0.022, AR=10.0,
        e_oswald=0.82, C_L_max=1.5, A_rotor=0.18, eta_prop=0.72,
        battery_mass=3.0, specific_energy=220.0,
    )
    return _load_or_default('long_range.json', fallback)


def high_altitude_config() -> AircraftEnergyParams:
    """Platform adapted for >4000 m operations."""
    fallback = AircraftEnergyParams(
        name="High Altitude",
        m_tow=14.0, S_ref=0.70, C_D0=0.028, AR=8.5,
        C_L_max=1.4, A_rotor=0.28, V_tip=85.0, eta_prop=0.65,
        battery_mass=3.5, specific_energy=210.0,
    )
    return _load_or_default('high_altitude.json', fallback)


VEHICLE_CONFIGS: Dict[str, callable] = {
    'baseline':     baseline_config,
    'heavy_cargo':  heavy_cargo_config,
    'long_range':   long_range_config,
    'high_altitude': high_altitude_config,
}


def get_vehicle(name: str) -> AircraftEnergyParams:
    """
    Get a vehicle configuration by name or JSON file path.

    Parameters
    ----------
    name : str
        Either a key in VEHICLE_CONFIGS ('baseline', 'heavy_cargo', …)
        or a path to a JSON file.
    """
    if os.path.exists(name):
        return load_vehicle_from_json(name)
    if name not in VEHICLE_CONFIGS:
        available = list(VEHICLE_CONFIGS.keys())
        raise KeyError(f"Unknown vehicle '{name}'. Available: {available}")
    return VEHICLE_CONFIGS[name]()


def compare_vehicles_at_altitude(altitude: float = 2850.0,
                                 airspeed: float = 25.0) -> Dict:
    """Compare all built-in vehicle configs at a given altitude."""
    from .energy import power_fw_cruise, power_hover  # local import — avoids circular dep
    results = {}
    for name, factory in VEHICLE_CONFIGS.items():
        ac = factory()
        mass = ac.W / ac.g
        vs = ac.stall_speed_at(altitude)
        p_cruise = power_fw_cruise(ac, airspeed, altitude) / ac.eta_prop
        p_hover = power_hover(ac, altitude)
        results[name] = {
            'name': ac.name,
            'mass_kg': mass,
            'battery_wh': ac.battery_energy_wh,
            'V_stall_ms': vs,
            'V_safe_ms': vs * 1.3,
            'P_cruise_W': p_cruise,
            'P_hover_W': p_hover,
            'endurance_hover_min': ac.battery_energy_wh / p_hover * 60,
            'range_cruise_km': (ac.battery_energy_wh / p_cruise) * airspeed / 1000 * 3600,
        }
    return results
