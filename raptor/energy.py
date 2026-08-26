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
from dataclasses import dataclass
from typing import List, Dict
import warnings

import numpy as np

from .atmosphere import isa_density
# AircraftEnergyParams lives in vehicles.py; re-exported here so that
# existing code using `from raptor.energy import AircraftEnergyParams`
# continues to work without changes.
from .vehicles import AircraftEnergyParams  # noqa: F401


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
    T = ac.W  # Steady vertical ascent: T ≈ W (neglecting acceleration)

    # Climb power — the work done raising the weight
    P_c = T * V_y

    # Induced power. Momentum theory in axial climb gives
    #     v_i = -V_y/2 + sqrt((V_y/2)² + v_h²),      v_h² = T/(2ρA)
    # The induced velocity *falls* as the rotor climbs into undisturbed
    # air: the climb velocity already supplies part of the momentum flux.
    # Writing +V_y/2 instead — which is the correct form for *descent* —
    # overstates ascent induced power by 18 % at 3 m/s on this airframe,
    # and take-off is where a VTOL spends its worst power.
    v_h_sq = T / (2 * rho * ac.rotor_area_total)
    v_i = -V_y / 2 + np.sqrt((V_y / 2)**2 + v_h_sq)
    P_i = ac.k_i * T * v_i

    # Profile power (blade drag)
    P_o = rho * ac.rotor_area_total * ac.V_tip**3 * (ac.sigma_rotor * ac.C_d_blade / 8)

    return P_c + P_i + P_o


def power_hover(ac: AircraftEnergyParams,
                altitude: float) -> float:
    """
    P2: Hover (stationary flight) — VTOL rotors.

    P2 = P_i + P_o
       = k_i · T · √(T / (2·ρ·A)) + ρ·A·V_TIP³·(σ·C_d/8)

    In hover, T = W exactly, and the induced velocity is
    v_h = √(T / 2ρA).
    """
    rho = isa_density(altitude)
    T = ac.W

    # Ideal induced power + correction
    P_i = ac.k_i * T * np.sqrt(T / (2 * rho * ac.rotor_area_total))

    # Profile power
    P_o = rho * ac.rotor_area_total * ac.V_tip**3 * (ac.sigma_rotor * ac.C_d_blade / 8)

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
    v_h2 = T / (2 * rho * ac.rotor_area_total)
    v_term = np.sqrt((-V_inf**2 / 2)**2 + v_h2**2)
    v_i = np.sqrt(-V_inf**2 / 2 + v_term)
    P_i = ac.k_i * T * v_i

    # Profile power (increases with advance ratio)
    P_o = rho * ac.rotor_area_total * ac.V_tip**3 * \
          (ac.sigma_rotor * ac.C_d_blade / 8) * (1 + 4.6 * mu**2)

    # Parasitic power. C_D0, not the full cruise drag polar: during
    # transition the rotors still carry the weight, so the wing is barely
    # loaded and its induced drag has not appeared yet. Using ac.C_D here
    # charges the aircraft for lift it is not producing.
    P_p = 0.5 * rho * V_inf**3 * ac.C_D0 * ac.S_ref

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
    if C_L_climb > ac.C_L_max_at(V_inf, altitude):
        # Return a steep penalty: nominal power × (C_L / C_L_max)²
        # This creates a smooth gradient pushing the optimizer away from
        # the stall boundary rather than a hard discontinuity.
        stall_ratio = C_L_climb / ac.C_L_max_at(V_inf, altitude)
        # Estimate a nominal power and scale by stall ratio squared
        P_nominal = ac.W * V_inf  # rough upper bound
        return P_nominal * stall_ratio**2

    C_D_climb = ac.drag_coefficient(C_L_climb, V_inf, altitude)
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
    if C_L_cruise > ac.C_L_max_at(V_inf, altitude):
        stall_ratio = C_L_cruise / ac.C_L_max_at(V_inf, altitude)
        P_nominal = ac.W * V_inf
        return P_nominal * stall_ratio**2

    C_D_cruise = ac.drag_coefficient(C_L_cruise, V_inf, altitude)
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
    if C_L_desc > ac.C_L_max_at(V_inf, altitude):
        stall_ratio = C_L_desc / ac.C_L_max_at(V_inf, altitude)
        P_nominal = ac.W * V_inf
        return P_nominal * stall_ratio**2

    C_D_desc = ac.drag_coefficient(C_L_desc, V_inf, altitude)
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

    Momentum theory with the descent sign, which is valid while the
    descent rate stays well below the hover induced velocity.

    Between roughly 0.25·v_h and 1.5·v_h the rotor enters the vortex-ring
    state, where momentum theory does not merely lose accuracy but breaks
    down — real power *increases* there, and the model below would
    under-predict it. ``vortex_ring_margin`` reports how close a given
    descent rate is to that band, so a planner can keep out of it.
    """
    rho = isa_density(altitude)
    T = ac.W  # Controlled descent: T ≈ W

    V_y_abs = abs(V_y)

    # Same momentum-theory expression as the climb, with the sign the
    # other way round: descending, the rotor re-ingests its own wake and
    # the induced velocity *rises*.
    v_h_sq = T / (2 * rho * ac.rotor_area_total)
    v_i = V_y_abs / 2 + np.sqrt((V_y_abs / 2)**2 + v_h_sq)
    P_i = ac.k_i * T * v_i

    # Profile power
    P_o = rho * ac.rotor_area_total * ac.V_tip**3 * (ac.sigma_rotor * ac.C_d_blade / 8)

    # Gravity does part of the work, so the rotor supplies less.
    P_net = P_i + P_o - T * V_y_abs

    return max(P_net, P_o)  # At least profile drag power


# ═════════════════════════════════════════════════════════════════════════════
# SEGMENT → POWER MAPPING
# ═════════════════════════════════════════════════════════════════════════════

def vortex_ring_margin(ac: AircraftEnergyParams, V_descent: float,
                       altitude: float) -> dict:
    """
    How close a vertical descent is to the vortex-ring state.

    Between roughly 0.25 and 1.5 times the hover induced velocity, a
    descending rotor re-ingests its own wake. Momentum theory does not
    merely lose accuracy there — it inverts: real power *rises* while the
    model says it falls, and the rotor can lose thrust entirely. It is a
    genuine loss-of-control mechanism, not a modelling nicety.

    :func:`power_vertical_descent` is momentum theory, so it is only
    valid *below* that band. This reports where a given descent rate sits.

    Returns
    -------
    dict with the ratio ``V_descent / v_h``, the band edges, and whether
    the descent is safe, marginal or inside the vortex-ring region.
    """
    rho = isa_density(altitude)
    v_h = float(np.sqrt(ac.W / (2 * rho * ac.rotor_area_total)))
    ratio = abs(float(V_descent)) / max(v_h, 1e-9)

    if ratio < 0.25:
        state = "safe"
    elif ratio <= 1.5:
        state = "VORTEX RING — power model invalid, thrust may collapse"
    else:
        state = "windmill brake — beyond the vortex-ring band"

    return {
        "descent_rate": abs(float(V_descent)),
        "hover_induced_velocity": v_h,
        "ratio": ratio,
        "vrs_lower": 0.25 * v_h,
        "vrs_upper": 1.5 * v_h,
        "state": state,
        "safe": ratio < 0.25,
        # How much faster the aircraft could descend before entering the
        # band — the number a planner actually needs.
        "margin_ms": max(0.25 * v_h - abs(float(V_descent)), 0.0),
    }


def check_descent_rates(path, ac: AircraftEnergyParams) -> list:
    """
    Report any VTOL descent on a path that enters the vortex-ring band.

    Cheap to run and worth running: descent rates are design variables,
    and nothing else in the package stops the optimizer choosing one that
    is aerodynamically dangerous but energetically attractive.
    """
    out = []
    for i, seg in enumerate(path.segments):
        if seg.segment_type.value != "VTOL_DESCEND":
            continue
        k = seg.kinematics
        s, e = seg.start_state, seg.end_state
        m = vortex_ring_margin(ac, k.vertical_speed, 0.5 * (s.alt + e.alt))
        if not m["safe"]:
            out.append(
                f"segment {i}: descending at {m['descent_rate']:.1f} m/s, "
                f"{m['ratio']:.2f}x hover induced velocity — {m['state']}. "
                f"Stay below {m['vrs_lower']:.1f} m/s."
            )
    return out


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
    c_rate: float = 0.0        # Instantaneous discharge rate [C]
    ohmic_loss_w: float = 0.0  # I²R heat at this instant [W]
    power_limited: bool = False  # Pack could not supply the demanded power


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

    # ── Pack diagnostics ──────────────────────────────────────────────
    #: Energy drawn from the cells, which exceeds the energy delivered to
    #: the drivetrain by the I²R heat.
    energy_from_cells_wh: float = 0.0
    ohmic_loss_wh: float = 0.0
    #: Highest discharge rate seen [C], and whether it breached the
    #: vehicle's continuous rating.
    max_c_rate: float = 0.0
    c_rate_exceeded: bool = False
    #: True if the pack could not supply the demanded power at some point,
    #: which is a different failure from running out of energy.
    power_limited: bool = False

    def pack_summary(self) -> str:
        """Battery-side view of the mission."""
        lines = [
            f"Delivered to drivetrain: {self.total_energy_wh:7.1f} Wh",
            f"Drawn from cells:        {self.energy_from_cells_wh:7.1f} Wh",
            f"Ohmic loss (I²R):        {self.ohmic_loss_wh:7.1f} Wh"
            f"  ({100*self.ohmic_loss_wh/max(self.energy_from_cells_wh,1e-9):.1f} %)",
            f"Peak discharge rate:     {self.max_c_rate:7.1f} C"
            + ("   EXCEEDS continuous rating" if self.c_rate_exceeded else ""),
            f"SOC at arrival:          {self.SOC_final*100:7.1f} %",
        ]
        if self.power_limited:
            lines.append("  ! pack could not meet the demanded power at some point")
        return "\n".join(lines)

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

        # Usable energy, not nameplate: pack overhead, cell mismatch and
        # the unreachable tail below the cut-off voltage all sit between
        # the cell datasheet and what the aircraft can actually draw.
        self.E_usable_wh = ac.battery_usable_wh

        # Tracking
        self.energy_consumed_wh = 0.0      # delivered to the drivetrain
        self.energy_from_cells_wh = 0.0    # drawn from the chemistry
        self.ohmic_loss_wh = 0.0
        self.mah_consumed = 0.0
        self.time_elapsed = 0.0
        self.max_c_rate = 0.0
        self.power_limited = False
        self._v_terminal = self.ocv
        self.timeline: List[BatteryState] = []

        # Record initial state
        self._record_state(0.0, 0.0)

    def reset(self, SOC_initial: float = 1.0):
        """Reset battery to a given SOC."""
        self.SOC = SOC_initial
        self.SOC_initial = SOC_initial
        self.energy_consumed_wh = 0.0
        self.energy_from_cells_wh = 0.0
        self.ohmic_loss_wh = 0.0
        self.mah_consumed = 0.0
        self.time_elapsed = 0.0
        self.max_c_rate = 0.0
        self.power_limited = False
        self._v_terminal = self.ocv
        self.timeline.clear()
        self._record_state(0.0, 0.0)

    @property
    def ocv(self) -> float:
        """
        Open-circuit pack voltage at the present SOC [V].

        Three sources, in order of preference: a curve fitted from a
        measured discharge log, the published curve for the pack's
        chemistry, and — only if the chemistry is unrecognised — the
        generic LiPo table, with a warning.

        The model this replaced was ``V_nom × (0.85 + 0.15·SOC)``, which
        tops out at the *nominal* 22.2 V for a 6S pack that actually
        leaves the ground at 25.2 V and is empty near 19.8 V. Currents
        inferred from it were high at full charge and low when empty —
        exactly backwards.
        """
        soc = self.charge_fraction
        fit = getattr(self.ac, 'battery_fit', None)
        if fit is not None:
            cell = float(np.interp(soc, fit.ocv_soc, fit.ocv_cell_v))
        else:
            soc_grid, v_grid = self._chemistry_curve()
            cell = float(np.interp(soc, soc_grid, v_grid))
        return cell * self.ac.battery_cells_series

    @property
    def charge_fraction(self) -> float:
        """
        Fraction of the pack's *total* charge still in it [0, 1].

        Not the same as :attr:`SOC`, and the difference matters. ``SOC``
        runs from 1 at full to 0 when the **usable** energy is gone —
        that is what the mission planner reasons about, and what the
        reserve is expressed against. The OCV curve, though, is a
        property of the cell across its whole range, and looking it up
        at the mission SOC puts the pack at a fully flat 2.5 V/cell at
        the moment the reserve is reached, which is a voltage no flight
        ever sees.

        With a usable fraction ``f``, exhausting the usable energy still
        leaves ``1 − f`` of the pack, so the charge fraction is
        ``1 − (1 − SOC)·f``.
        """
        f = float(np.clip(self.ac.battery_usable_fraction, 0.0, 1.0))
        return float(np.clip(1.0 - (1.0 - self.SOC) * f, 0.0, 1.0))

    def _chemistry_curve(self):
        """The pack's OCV table, cached after the first lookup."""
        cached = getattr(self, '_ocv_table', None)
        if cached is not None:
            return cached
        from .battery import cell_ocv_curve
        chem = getattr(self.ac, 'cell_chemistry', 'li-ion')
        try:
            table = cell_ocv_curve(chem)
        except KeyError:
            warnings.warn(
                f"unknown cell chemistry {chem!r}; falling back to the "
                f"generic LiPo curve, which is the wrong shape for any "
                f"Li-ion pack. Set cell_chemistry to one of the known "
                f"values.",
                stacklevel=3,
            )
            table = cell_ocv_curve('lipo')
        self._ocv_table = table
        return table

    @property
    def internal_resistance(self) -> float:
        """
        Pack internal resistance at the present SOC [ohm].

        Nominal resistance scaled by the SOC dependence measured in
        Chen & Rincón-Mora (2006): nearly flat above 20 % and climbing
        steeply below it, roughly threefold by empty. A constant
        resistance understates voltage sag exactly where a mission has
        least margin, which is at the end of it — and voltage sag is
        what turns "enough energy left" into "cannot supply the hover
        power to land".
        """
        from .battery import chen_rincon_mora_resistance_factor
        R = self.ac.battery_resistance
        return float(R * chen_rincon_mora_resistance_factor(self.charge_fraction))

    @property
    def voltage(self) -> float:
        """Terminal voltage at the last recorded load [V]."""
        return self._v_terminal

    @property
    def max_power_w(self) -> float:
        """
        Greatest power the pack can deliver at this SOC [W].

        A source with internal resistance R and open-circuit voltage V
        cannot exceed V²/4R however hard it is loaded — beyond that the
        demand is simply unmeetable, which is a different failure from
        running out of energy and worth reporting as such.
        """
        R = self.internal_resistance
        if R <= 0:
            return float('inf')
        return self.ocv ** 2 / (4.0 * R)

    @property
    def mah_remaining(self) -> float:
        return self.capacity_mah - self.mah_consumed

    @property
    def percent_remaining(self) -> float:
        return self.SOC * 100.0

    def discharge(self, power_w: float, duration_s: float) -> Dict:
        """
        Draw constant power from the pack for a duration.

        The current is solved from the load rather than assumed, because
        the pack has resistance: delivering ``power_w`` at the terminals
        means the chemistry gives up ``I·V_oc``, of which ``I²R`` becomes
        heat and never reaches the propulsion system. At the currents a
        heavy lift+cruise VTOL pulls in hover that gap is percent-level,
        and it compounds over a mission.

        Solving ``I·V_oc − I²R = P`` for the smaller root:

            I = (V_oc − √(V_oc² − 4RP)) / 2R

        A negative discriminant means the pack simply cannot supply the
        demand at this state of charge; the draw is capped and flagged
        rather than silently producing an imaginary current.

        Parameters
        ----------
        power_w : float
            Electrical power required at the pack terminals [W].
        duration_s : float
            Duration of discharge [s].

        Returns
        -------
        dict with energy_wh, mah_consumed, SOC_end, current_A and the
        loss and limit diagnostics.
        """
        SOC_start = self.SOC
        dt_h = duration_s / 3600.0
        V_oc = self.ocv
        R = self.internal_resistance
        power_w = max(float(power_w), 0.0)

        limited = False
        if R > 0:
            disc = V_oc ** 2 - 4.0 * R * power_w
            if disc <= 0.0:
                # Beyond the matched-load maximum: cap at V_oc/2R.
                limited = True
                current_A = V_oc / (2.0 * R)
            else:
                current_A = (V_oc - np.sqrt(disc)) / (2.0 * R)
        else:
            current_A = power_w / V_oc if V_oc > 0 else 0.0

        V_term = V_oc - current_A * R
        ohmic_w = current_A ** 2 * R
        delivered_w = current_A * V_term

        # Energy leaving the cells, which is what depletes the pack.
        energy_cells_wh = current_A * V_oc * dt_h
        energy_wh = delivered_w * dt_h            # reaching the drivetrain
        energy_j = energy_wh * 3600.0
        mah = current_A * dt_h * 1000.0

        delta_SOC = (energy_cells_wh / self.E_usable_wh
                     if self.E_usable_wh > 0 else 0.0)

        self.SOC = max(0.0, self.SOC - delta_SOC)
        self.energy_consumed_wh += energy_wh
        self.energy_from_cells_wh += energy_cells_wh
        self.ohmic_loss_wh += ohmic_w * dt_h
        self.mah_consumed += mah
        self.time_elapsed += duration_s
        self._v_terminal = V_term
        self.power_limited = self.power_limited or limited

        c_rate = (current_A / self.ac.battery_capacity_ah
                  if self.ac.battery_capacity_ah > 0 else 0.0)
        self.max_c_rate = max(self.max_c_rate, c_rate)

        self._record_state(power_w, current_A, c_rate, ohmic_w, limited)

        return {
            'energy_wh': energy_wh,
            'energy_j': energy_j,
            'energy_from_cells_wh': energy_cells_wh,
            'ohmic_loss_wh': ohmic_w * dt_h,
            'mah_consumed': mah,
            'SOC_start': SOC_start,
            'SOC_end': self.SOC,
            'current_A': current_A,
            'voltage_V': V_term,
            'c_rate': c_rate,
            'power_limited': limited,
        }

    def _record_state(self, power_w: float, current_A: float,
                      c_rate: float = 0.0, ohmic_w: float = 0.0,
                      limited: bool = False):
        """Record current battery state to timeline."""
        self.timeline.append(BatteryState(
            time=self.time_elapsed,
            SOC=self.SOC,
            voltage=self._v_terminal,
            current_draw=current_A,
            power_elec=power_w,
            energy_consumed_wh=self.energy_consumed_wh,
            mah_consumed=self.mah_consumed,
            mah_remaining=self.mah_remaining,
            percent_remaining=self.percent_remaining,
            c_rate=c_rate,
            ohmic_loss_w=ohmic_w,
            power_limited=limited,
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
        seg_name = seg.segment_type.value
        P_mec = compute_segment_power(ac, seg_name, kin_dict)

        # Electrical power. Lift rotors and the cruise propulsor are
        # different machines with different efficiencies; charging hover
        # at the cruise figure (or the reverse) skews whichever phase
        # dominates the mission, and for short medical hops the VTOL
        # phases dominate.
        eta = (ac.eta_vtol_effective
               if seg_name in ("VTOL_ASCEND", "VTOL_DESCEND", "TRANSITION")
               else ac.eta_cruise_effective)
        P_elec = P_mec / eta

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
        energy_from_cells_wh=battery.energy_from_cells_wh,
        ohmic_loss_wh=battery.ohmic_loss_wh,
        max_c_rate=battery.max_c_rate,
        c_rate_exceeded=battery.max_c_rate > ac.battery_max_c_rate,
        power_limited=battery.power_limited,
    )
