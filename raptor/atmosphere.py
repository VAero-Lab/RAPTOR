"""
Atmosphere — ISA Standard Atmosphere Model
============================================

Computes air density, temperature, and pressure as functions of
altitude. Essential for accurate power calculations at Quito's
operating altitudes (2,300 – 5,500 m AMSL) where air density is
approximately 60–75% of sea level.
"""

import numpy as np


# ISA constants
_T0 = 288.15      # sea-level temperature [K]
_P0 = 101325.0     # sea-level pressure [Pa]
_RHO0 = 1.225      # sea-level density [kg/m³]
_g = 9.80665       # gravitational acceleration [m/s²]
_R = 287.05287     # specific gas constant for dry air [J/(kg·K)]
_LAPSE = -0.0065   # temperature lapse rate in troposphere [K/m]


def isa_temperature(altitude_m: float) -> float:
    """ISA temperature at altitude [K]."""
    return _T0 + _LAPSE * altitude_m


def isa_pressure(altitude_m: float) -> float:
    """ISA pressure at altitude [Pa]."""
    T = isa_temperature(altitude_m)
    return _P0 * (T / _T0) ** (-_g / (_LAPSE * _R))


def isa_density(altitude_m: float) -> float:
    """ISA density at altitude [kg/m³]."""
    T = isa_temperature(altitude_m)
    return _RHO0 * (T / _T0) ** (-(_g / (_LAPSE * _R)) - 1.0)


def isa_density_batch(altitudes_m: np.ndarray) -> np.ndarray:
    """Vectorized ISA density for an array of altitudes."""
    altitudes_m = np.asarray(altitudes_m)
    T = _T0 + _LAPSE * altitudes_m
    exponent = -(_g / (_LAPSE * _R)) - 1.0
    return _RHO0 * (T / _T0) ** exponent
