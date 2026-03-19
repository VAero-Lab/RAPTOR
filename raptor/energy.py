"""
Energy Model — Power, Energy & Battery for eVTOL Lift+Cruise
==============================================================

Implements the 7 parametric power equations (Table 10 of the reference
document) for each flight phase of a Quadplane (Lift+Cruise) eVTOL:

    P1: Vertical ascent (takeoff)       — VTOL rotors
    P2: Hover (stationary flight)       — VTOL rotors
    P3: Transition (VTOL → fixed-wing)  — VTOL + FW props
    P4: Climb with angle of inclination — FW mode
    P5: Cruise (level flight)           — FW mode
    P6: Descent with angle              — FW mode
    P7: Vertical descent (landing)      — VTOL rotors

Plus:
    - P_elec = P_mec / η_prop
    - E_phase = P_elec × Δt   (constant power per segment)
    - E_total = Σ E_phase,i
    - Battery SOC tracking (Coulomb counting → mAh, %)

Dependencies: NumPy only. No OpenMDAO, no OpenConcept.

References
----------
Adapted from Table 10: "Ecuaciones físicas para calcular la potencia
necesaria para cada fase de vuelo" in the energy_modeling document.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import numpy as np

from .atmosphere import isa_density


# ═════════════════════════════════════════════════════════════════════════════
# AIRCRAFT PHYSICAL PARAMETERS
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class AircraftEnergyParams:
    """
    Physical parameters of the eVTOL aircraft for energy calculations.

    Groups:
        Airframe:  mass, wing area, aerodynamic coefficients
        Rotors:    disk area, tip speed, solidity, blade drag
        Battery:   capacity, voltage, specific energy, efficiency
        Propulsion: overall propulsion efficiency

    The default values correspond to the Pugllohuma case study UAV
    (Table 11 of the reference), adapted for the medical delivery
    mission in Quito.
    """

    # --- Airframe ---
    m_tow: float = 12.0              # Total takeoff mass [kg]
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
    sigma_rotor: float = 0.06        # Rotor solidity (Nc/(πR))
    C_d_blade: float = 0.012         # Mean blade drag coefficient
    k_i: float = 1.15                # Induced power correction factor

    # --- Propulsion ---
    eta_prop: float = 0.6            # Overall propulsion efficiency

    # --- Battery ---
    battery_mass: float = 3.0        # Battery mass [kg]
    specific_energy: float = 200.0   # Battery specific energy [Wh/kg]
    battery_voltage: float = 22.2    # Nominal battery voltage [V] (6S LiPo)
    battery_capacity_mah: float = 0.0  # If 0, auto-computed from mass & spec energy

    def __post_init__(self):
        """Compute derived quantities."""
        self.W = self.m_tow * self.g  # Weight [N]
        # Induced drag coefficient: C_D_i = C_L² / (π e AR)
        self.k_drag = 1.0 / (np.pi * self.e_oswald * self.AR)
        # Total drag coefficient at cruise C_L
        self.C_D = self.C_D0 + self.k_drag * self.C_L**2
        # Battery capacity
        if self.battery_capacity_mah <= 0:
            energy_wh = self.battery_mass * self.specific_energy
            self.battery_capacity_mah = energy_wh / self.battery_voltage * 1000.0
        self.battery_energy_wh = self.battery_mass * self.specific_energy
        self.battery_energy_j = self.battery_energy_wh * 3600.0

    @property
    def stall_speed(self) -> float:
        """Estimate stall speed at a reference altitude (3000 m) [m/s]."""
        return self.stall_speed_at(3000.0)

    def stall_speed_at(self, altitude: float, gamma_deg: float = 0.0) -> float:
        """
        Compute stall speed at a given altitude and flight-path angle.

        In climbing flight, the lift must support only the component of
        weight perpendicular to the flight path:
            L = W·cos(γ)
        At stall: L = C_L_max · q · S_ref  →  V_stall = √(2·W·cos(γ) / (ρ·S·C_L_max))

        At high altitude (low ρ), stall speed increases significantly.

        Parameters
        ----------
        altitude : float
            Altitude AMSL [m].
        gamma_deg : float
            Flight-path angle [deg]. Positive = climb.

        Returns
        -------
        float
            Stall speed [m/s] at the given conditions.
        """
        rho = isa_density(altitude)
        gamma = np.radians(abs(gamma_deg))
        return np.sqrt(2 * self.W * np.cos(gamma) / (rho * self.S_ref * self.C_L_max))

    def C_L_required(self, V: float, altitude: float,
                     gamma_deg: float = 0.0) -> float:
        """
        Compute the lift coefficient required for steady flight.

        In level flight (γ=0): C_L = W / (q·S)
        In climbing flight:     C_L = W·cos(γ) / (q·S)

        Parameters
        ----------
        V : float
            True airspeed [m/s].
        altitude : float
            Altitude AMSL [m].
        gamma_deg : float
            Flight-path angle [deg].

        Returns
        -------
        float
            Required C_L. If > C_L_max, the aircraft will stall.
        """
        rho = isa_density(altitude)
        gamma = np.radians(abs(gamma_deg))
        q = 0.5 * rho * V**2
        if q * self.S_ref < 1e-6:
            return float('inf')
        return self.W * np.cos(gamma) / (q * self.S_ref)

    def check_aero_feasibility(self, V: float, altitude: float,
                               gamma_deg: float = 0.0,
                               safety_margin: float = 1.3) -> dict:
        """
        Check if a flight condition is aerodynamically feasible.

        Verifies:
        1. Required C_L does not exceed C_L_max (stall check)
        2. Speed is above stall speed with a safety margin (V ≥ margin × V_stall)
        3. Climb angle does not produce excessive C_L

        Parameters
        ----------
        V : float
            True airspeed [m/s].
        altitude : float
            Altitude AMSL [m].
        gamma_deg : float
            Flight-path angle [deg].
        safety_margin : float
            Speed must be at least this factor above stall (FAR standard: 1.3).

        Returns
        -------
        dict with keys:
            feasible : bool
            C_L_required : float
            C_L_max : float
            stall_margin : float  (V / V_stall; must be ≥ safety_margin)
            V_stall : float
            V_min_safe : float (= safety_margin × V_stall)
            message : str
        """
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
        return self.C_D0 + self.k_drag * C_L**2

    def thrust_required(self, V: float, altitude: float,
                        gamma_deg: float = 0.0) -> float:
        """
        Thrust required for steady flight at airspeed V, altitude, and
        flight path angle γ.
        """
        rho = isa_density(altitude)
        gamma = np.radians(gamma_deg)
        # In steady flight: T = D + W sin(γ)
        q = 0.5 * rho * V**2
        D = q * self.S_ref * self.C_D
        return D + self.W * np.sin(gamma)


# ═════════════════════════════════════════════════════════════════════════════
# POWER MODELS — THE 7 FLIGHT PHASES
# ═════════════════════════════════════════════════════════════════════════════

def power_vertical_ascent(ac: AircraftEnergyParams,
                          V_y: float,
                          altitude: float) -> float:
    """
    P1: Vertical ascent (takeoff) — VTOL rotors.

    P1 = P_c + P_i + P_o
       = (T₁·V_y) - (k_i·T₁·V_y/2)·√(V_y² + 2T₁/(ρ·A))
         + ρ·A·V_TIP³·(σ·C_d/8)

    where T₁ = W (thrust equals weight for vertical climb at
    constant rate, plus extra for acceleration — here we use T = W
    for steady vertical climb approximation, which is conservative).
    """
    rho = isa_density(altitude)
    W = ac.W
    T = W  # Steady vertical ascent: T ≈ W (neglecting acceleration)

    # Climb power
    P_c = T * V_y

    # Induced power (momentum theory with correction)
    v_h = np.sqrt(T / (2 * rho * ac.A_rotor))  # Hover induced velocity
    # For climbing: v_i from momentum theory
    # P_i = k_i * T * (V_y/2 + sqrt((V_y/2)² + v_h²))  ... simplified
    # Using the exact form from Table 10:
    P_i = (ac.k_i * T * V_y / 2) + \
          (ac.k_i * T / 2) * np.sqrt(V_y**2 + 2 * T / (rho * ac.A_rotor))

    # Profile power (blade drag)
    P_o = rho * ac.A_rotor * ac.V_tip**3 * (ac.sigma_rotor * ac.C_d_blade / 8)

    return P_c + P_i + P_o


def power_hover(ac: AircraftEnergyParams,
                altitude: float) -> float:
    """
    P2: Hover (stationary flight) — VTOL rotors.

    P2 = P_i,ideal + P_o
       = k_i · T² / √(2·ρ·A) + ρ·A·V_TIP³·(σ·C_d/8)

    In hover, T = W exactly.
    """
    rho = isa_density(altitude)
    T = ac.W

    # Ideal induced power + correction
    P_i = ac.k_i * T * np.sqrt(T / (2 * rho * ac.A_rotor))

    # Profile power
    P_o = rho * ac.A_rotor * ac.V_tip**3 * (ac.sigma_rotor * ac.C_d_blade / 8)

    return P_i + P_o


def power_transition(ac: AircraftEnergyParams,
                     V_inf: float,
                     altitude: float) -> float:
    """
    P3: Transition (VTOL → Fixed Wing) — combined propulsion.

    P3 = P_i + P_o + P_p
       = k_i·T³ / √((-V∞²/2 + √((V∞²/2)² + (T₃/(2ρA))²))
         + ρ·A·V_TIP³·(σ·C_D/8)·(1 + 4.6μ²)
         + 0.5·ρ·V∞³·C_d·S_ref

    where μ = V∞·cos(α) / V_TIP is the advance ratio.
    During transition, thrust ≈ weight (vehicle still partially supported
    by rotors).
    """
    rho = isa_density(altitude)
    T = ac.W  # Approximate: rotors still bearing most weight

    # Advance ratio (assuming small α, cos(α) ≈ 1)
    mu = V_inf / ac.V_tip

    # Induced power
    # Using momentum theory in forward flight
    v_h2 = T / (2 * rho * ac.A_rotor)
    v_term = np.sqrt((-V_inf**2 / 2)**2 + v_h2**2)
    v_i = np.sqrt(-V_inf**2 / 2 + v_term)
    P_i = ac.k_i * T * v_i

    # Profile power (increases with advance ratio)
    P_o = rho * ac.A_rotor * ac.V_tip**3 * \
          (ac.sigma_rotor * ac.C_d_blade / 8) * (1 + 4.6 * mu**2)

    # Parasitic power (fuselage drag)
    P_p = 0.5 * rho * V_inf**3 * ac.C_D * ac.S_ref

    return P_i + P_o + P_p


def power_fw_climb(ac: AircraftEnergyParams,
                   V_inf: float,
                   gamma_deg: float,
                   altitude: float) -> float:
    """
    P4: Fixed-wing climb at angle γ.

    P4 = P_p + P_c
       = V₄ · (½ρV₄²·S_ref·C_D + W·sin|γ|)
       = V₄ · D₄ + W₄·sin|γ|

    If C_L_required > C_L_max (stall condition), the function returns a
    very high penalty power to signal aerodynamic infeasibility to the
    optimizer. This prevents the optimizer from exploiting physically
    impossible flight states at high altitude where low density raises
    the required C_L.
    """
    rho = isa_density(altitude)
    gamma = np.radians(abs(gamma_deg))

    # Drag at climb speed (using parabolic drag polar)
    # In climb, C_L = W·cos(γ) / (0.5·ρ·V²·S_ref)
    q = 0.5 * rho * V_inf**2
    C_L_climb = ac.W * np.cos(gamma) / (q * ac.S_ref) if q * ac.S_ref > 0 else ac.C_L

    # ── Stall check ──────────────────────────────────────────────
    if C_L_climb > ac.C_L_max:
        # Return a steep penalty: nominal power × (C_L / C_L_max)²
        # This creates a smooth gradient pushing the optimizer away from
        # the stall boundary rather than a hard discontinuity.
        stall_ratio = C_L_climb / ac.C_L_max
        # Estimate a nominal power and scale by stall ratio squared
        P_nominal = ac.W * V_inf  # rough upper bound
        return P_nominal * stall_ratio**2

    C_D_climb = ac.C_D0 + ac.k_drag * C_L_climb**2
    D = q * ac.S_ref * C_D_climb

    # Total power = drag power + climb power
    P_p = V_inf * D
    P_c = ac.W * V_inf * np.sin(gamma)

    return P_p + P_c


def power_fw_cruise(ac: AircraftEnergyParams,
                    V_inf: float,
                    altitude: float) -> float:
    """
    P5: Fixed-wing cruise (level flight).

    P5 = D₅ · V₅ = (½ρV₅²·S_ref·C_D) · V₅

    In level cruise, C_L = W / (0.5·ρ·V²·S_ref).

    Includes stall detection: if C_L_cruise > C_L_max, returns a
    penalty power.
    """
    rho = isa_density(altitude)
    q = 0.5 * rho * V_inf**2

    # Lift coefficient from level flight condition
    C_L_cruise = ac.W / (q * ac.S_ref) if q * ac.S_ref > 0 else ac.C_L

    # ── Stall check ──────────────────────────────────────────────
    if C_L_cruise > ac.C_L_max:
        stall_ratio = C_L_cruise / ac.C_L_max
        P_nominal = ac.W * V_inf
        return P_nominal * stall_ratio**2

    C_D_cruise = ac.C_D0 + ac.k_drag * C_L_cruise**2
    D = q * ac.S_ref * C_D_cruise

    return D * V_inf


def power_fw_descent(ac: AircraftEnergyParams,
                     V_inf: float,
                     gamma_deg: float,
                     altitude: float) -> float:
    """
    P6: Fixed-wing descent at angle γ.

    P6 = P_p - P_c
       = V₆ · D₆ - W · V₆ · |sin(γ)|

    Power is reduced because gravity assists the motion.
    The result can be negative (regenerative), but we clamp at a
    small positive value for the electrical system.

    Includes stall detection for the descent condition.
    """
    rho = isa_density(altitude)
    gamma = np.radians(abs(gamma_deg))

    q = 0.5 * rho * V_inf**2
    C_L_desc = ac.W * np.cos(gamma) / (q * ac.S_ref) if q * ac.S_ref > 0 else ac.C_L

    # ── Stall check ──────────────────────────────────────────────
    if C_L_desc > ac.C_L_max:
        stall_ratio = C_L_desc / ac.C_L_max
        P_nominal = ac.W * V_inf
        return P_nominal * stall_ratio**2

    C_D_desc = ac.C_D0 + ac.k_drag * C_L_desc**2
    D = q * ac.S_ref * C_D_desc

    P_p = V_inf * D
    P_c = ac.W * V_inf * np.sin(gamma)

    P_mec = P_p - P_c
    # Minimum power: even in descent, avionics and control surfaces draw power
    return max(P_mec, 20.0)  # 20 W floor for avionics/servos


def power_vertical_descent(ac: AircraftEnergyParams,
                           V_y: float,
                           altitude: float) -> float:
    """
    P7: Vertical descent (landing) — VTOL rotors.

    Same form as P1 but with descent velocity.
    In vertical descent the rotor operates in vortex-ring / windmill
    state; we use the same momentum-theory model with |V_y| as a
    conservative estimate.
    """
    rho = isa_density(altitude)
    T = ac.W  # Controlled descent: T ≈ W

    V_y_abs = abs(V_y)

    # Induced power in descent (using climb formula with |V_y|)
    P_i = (ac.k_i * T * V_y_abs / 2) + \
          (ac.k_i * T / 2) * np.sqrt(V_y_abs**2 + 2 * T / (rho * ac.A_rotor))

    # Profile power
    P_o = rho * ac.A_rotor * ac.V_tip**3 * (ac.sigma_rotor * ac.C_d_blade / 8)

    # Climb power is negative (gravity assists), but rotor still works
    # In controlled descent, net power = P_i + P_o - T*V_y
    P_net = P_i + P_o - T * V_y_abs

    return max(P_net, P_o)  # At least profile drag power


# ═════════════════════════════════════════════════════════════════════════════
# SEGMENT → POWER MAPPING
# ═════════════════════════════════════════════════════════════════════════════

def compute_segment_power(ac: AircraftEnergyParams,
                          segment_type: str,
                          kinematics: dict) -> float:
    """
    Compute mechanical power for a flight segment.

    Parameters
    ----------
    ac : AircraftEnergyParams
        Aircraft configuration.
    segment_type : str
        One of: VTOL_ASCEND, VTOL_DESCEND, FW_CLIMB, FW_DESCEND,
        FW_CRUISE, TRANSITION
    kinematics : dict
        Must contain keys depending on segment type:
        - 'vertical_speed' [m/s]
        - 'airspeed' [m/s]
        - 'flight_path_angle' [deg]
        - 'altitude' [m AMSL] — mean altitude of the segment

    Returns
    -------
    P_mec : float
        Mechanical power [W]
    """
    alt = kinematics.get('altitude', 3000.0)

    if segment_type == "VTOL_ASCEND":
        V_y = abs(kinematics.get('vertical_speed', 2.0))
        return power_vertical_ascent(ac, V_y, alt)

    elif segment_type == "VTOL_DESCEND":
        V_y = abs(kinematics.get('vertical_speed', 2.0))
        return power_vertical_descent(ac, V_y, alt)

    elif segment_type == "TRANSITION":
        V_inf = kinematics.get('airspeed', 10.0)
        return power_transition(ac, V_inf, alt)

    elif segment_type == "FW_CLIMB":
        V_inf = kinematics.get('airspeed', 20.0)
        gamma = abs(kinematics.get('flight_path_angle', 5.0))
        return power_fw_climb(ac, V_inf, gamma, alt)

    elif segment_type == "FW_CRUISE":
        V_inf = kinematics.get('airspeed', 25.0)
        return power_fw_cruise(ac, V_inf, alt)

    elif segment_type == "FW_DESCEND":
        V_inf = kinematics.get('airspeed', 25.0)
        gamma = abs(kinematics.get('flight_path_angle', 5.0))
        return power_fw_descent(ac, V_inf, gamma, alt)

    else:
        # Default: hover
        return power_hover(ac, alt)


# ═════════════════════════════════════════════════════════════════════════════
# BATTERY MODEL (replaces OpenConcept SOCBattery + IntegratorGroup)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class BatteryState:
    """State of the battery at a point in time."""
    time: float           # s from mission start
    SOC: float            # State of charge [0, 1]
    voltage: float        # Terminal voltage [V]
    current_draw: float   # Instantaneous current [A]
    power_elec: float     # Electrical power [W]
    energy_consumed_wh: float  # Cumulative energy consumed [Wh]
    mah_consumed: float   # Cumulative mAh consumed
    mah_remaining: float  # Remaining capacity [mAh]
    percent_remaining: float  # Battery % remaining


@dataclass
class SegmentEnergyResult:
    """Energy analysis result for a single flight segment."""
    segment_type: str
    duration: float         # s
    P_mec: float            # Mechanical power [W]
    P_elec: float           # Electrical power [W]
    energy_j: float         # Energy consumed [J]
    energy_wh: float        # Energy consumed [Wh]
    mah_consumed: float     # mAh consumed in this segment
    current_draw: float     # Average current draw [A]
    SOC_start: float
    SOC_end: float


@dataclass
class MissionEnergyResult:
    """Complete energy analysis for a full flight path."""
    segments: List[SegmentEnergyResult]
    total_energy_wh: float
    total_energy_j: float
    total_mah_consumed: float
    total_time: float
    SOC_final: float
    percent_remaining: float
    mah_remaining: float
    feasible: bool          # True if SOC never goes below threshold
    min_SOC: float
    battery_timeline: List[BatteryState]

    @property
    def power_profile(self) -> np.ndarray:
        """Power profile as (N, 3) array: [time, P_mec, P_elec]."""
        rows = []
        t = 0.0
        for seg in self.segments:
            rows.append([t, seg.P_mec, seg.P_elec])
            t += seg.duration
            rows.append([t, seg.P_mec, seg.P_elec])
        return np.array(rows)


class BatteryModel:
    """
    Simple battery model with Coulomb-counting SOC tracking.

    This replaces OpenConcept's SOCBattery + IntegratorGroup with
    direct calculations. For each segment of constant power draw,
    we compute energy consumed and update SOC analytically.

    No OpenMDAO. No numerical integration needed for constant-power
    segments — E = P × Δt exactly.

    Parameters
    ----------
    ac : AircraftEnergyParams
        Aircraft with battery specifications.
    SOC_initial : float
        Starting state of charge [0, 1]. Default 1.0.
    SOC_min : float
        Minimum allowable SOC (safety reserve). Default 0.2.

    Example
    -------
    >>> ac = AircraftEnergyParams(m_tow=12, battery_mass=3.0)
    >>> battery = BatteryModel(ac)
    >>> battery.discharge(power_w=300, duration_s=120)
    >>> print(f"SOC: {battery.SOC:.2%}, mAh used: {battery.mah_consumed:.0f}")
    """

    def __init__(self, ac: AircraftEnergyParams,
                 SOC_initial: float = 1.0,
                 SOC_min: float = 0.2):
        self.ac = ac
        self.SOC = SOC_initial
        self.SOC_initial = SOC_initial
        self.SOC_min = SOC_min

        # Battery specs
        self.E_max_wh = ac.battery_energy_wh
        self.E_max_j = ac.battery_energy_j
        self.V_nom = ac.battery_voltage
        self.capacity_mah = ac.battery_capacity_mah

        # Tracking
        self.energy_consumed_wh = 0.0
        self.mah_consumed = 0.0
        self.time_elapsed = 0.0
        self.timeline: List[BatteryState] = []

        # Record initial state
        self._record_state(0.0, 0.0)

    def reset(self, SOC_initial: float = 1.0):
        """Reset battery to a given SOC."""
        self.SOC = SOC_initial
        self.SOC_initial = SOC_initial
        self.energy_consumed_wh = 0.0
        self.mah_consumed = 0.0
        self.time_elapsed = 0.0
        self.timeline.clear()
        self._record_state(0.0, 0.0)

    @property
    def voltage(self) -> float:
        """
        Approximate terminal voltage as a function of SOC.

        Simple linear model: V = V_nom × (0.85 + 0.15 × SOC)
        This captures the voltage sag at low SOC.
        """
        return self.V_nom * (0.85 + 0.15 * self.SOC)

    @property
    def mah_remaining(self) -> float:
        return self.capacity_mah - self.mah_consumed

    @property
    def percent_remaining(self) -> float:
        return self.SOC * 100.0

    def discharge(self, power_w: float, duration_s: float) -> Dict:
        """
        Discharge the battery at constant power for a duration.

        Parameters
        ----------
        power_w : float
            Electrical power draw [W].
        duration_s : float
            Duration of discharge [s].

        Returns
        -------
        dict with energy_wh, mah_consumed, SOC_end, current_A
        """
        # Energy consumed
        energy_j = power_w * duration_s
        energy_wh = energy_j / 3600.0

        # Current draw (using average voltage during segment)
        V_avg = self.voltage
        current_A = power_w / V_avg if V_avg > 0 else 0

        # mAh consumed: I [A] × t [h] × 1000
        mah = current_A * (duration_s / 3600.0) * 1000.0

        # Update SOC: ΔSOC = -E_consumed / E_max
        delta_SOC = energy_wh / self.E_max_wh if self.E_max_wh > 0 else 0
        SOC_start = self.SOC

        self.SOC = max(0.0, self.SOC - delta_SOC)
        self.energy_consumed_wh += energy_wh
        self.mah_consumed += mah
        self.time_elapsed += duration_s

        self._record_state(power_w, current_A)

        return {
            'energy_wh': energy_wh,
            'energy_j': energy_j,
            'mah_consumed': mah,
            'SOC_start': SOC_start,
            'SOC_end': self.SOC,
            'current_A': current_A,
        }

    def _record_state(self, power_w: float, current_A: float):
        """Record current battery state to timeline."""
        self.timeline.append(BatteryState(
            time=self.time_elapsed,
            SOC=self.SOC,
            voltage=self.voltage,
            current_draw=current_A,
            power_elec=power_w,
            energy_consumed_wh=self.energy_consumed_wh,
            mah_consumed=self.mah_consumed,
            mah_remaining=self.mah_remaining,
            percent_remaining=self.percent_remaining,
        ))

    @property
    def is_feasible(self) -> bool:
        """Check if SOC has stayed above minimum threshold."""
        return all(s.SOC >= self.SOC_min for s in self.timeline)


# ═════════════════════════════════════════════════════════════════════════════
# PATH → ENERGY ANALYSIS (integrates with FlightPath)
# ═════════════════════════════════════════════════════════════════════════════

def analyze_path_energy(path, ac: AircraftEnergyParams,
                        SOC_initial: float = 1.0,
                        SOC_min: float = 0.2) -> MissionEnergyResult:
    """
    Compute the full energy budget for a FlightPath.

    This is the main entry point connecting the path planning
    framework with the energy model.

    Parameters
    ----------
    path : FlightPath
        A complete flight path with segments.
    ac : AircraftEnergyParams
        Aircraft energy parameters.
    SOC_initial : float
        Starting battery SOC.
    SOC_min : float
        Minimum allowable SOC.

    Returns
    -------
    MissionEnergyResult with per-segment and total energy data.
    """
    battery = BatteryModel(ac, SOC_initial, SOC_min)
    segment_results = []

    for seg in path.segments:
        k = seg.kinematics
        s = seg.start_state
        e = seg.end_state

        # Mean altitude for density calculation
        mean_alt = (s.alt + e.alt) / 2.0

        # Build kinematics dict for the power model
        kin_dict = {
            'vertical_speed': abs(k.vertical_speed),
            'airspeed': k.airspeed,
            'flight_path_angle': k.flight_path_angle,
            'altitude': mean_alt,
        }

        # Compute mechanical power
        P_mec = compute_segment_power(ac, seg.segment_type.value, kin_dict)

        # Electrical power (accounting for propulsion efficiency)
        P_elec = P_mec / ac.eta_prop

        # Discharge battery for this segment's duration
        discharge = battery.discharge(P_elec, k.duration)

        segment_results.append(SegmentEnergyResult(
            segment_type=seg.segment_type.value,
            duration=k.duration,
            P_mec=P_mec,
            P_elec=P_elec,
            energy_j=discharge['energy_j'],
            energy_wh=discharge['energy_wh'],
            mah_consumed=discharge['mah_consumed'],
            current_draw=discharge['current_A'],
            SOC_start=discharge['SOC_start'],
            SOC_end=discharge['SOC_end'],
        ))

    total_wh = sum(s.energy_wh for s in segment_results)
    total_j = sum(s.energy_j for s in segment_results)
    total_mah = sum(s.mah_consumed for s in segment_results)
    total_time = sum(s.duration for s in segment_results)

    return MissionEnergyResult(
        segments=segment_results,
        total_energy_wh=total_wh,
        total_energy_j=total_j,
        total_mah_consumed=total_mah,
        total_time=total_time,
        SOC_final=battery.SOC,
        percent_remaining=battery.percent_remaining,
        mah_remaining=battery.mah_remaining,
        feasible=battery.is_feasible,
        min_SOC=min(s.SOC for s in battery.timeline),
        battery_timeline=battery.timeline,
    )
