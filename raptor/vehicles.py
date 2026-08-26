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
import warnings
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

def _sig(x: float, digits: int = 6) -> float:
    """Round to *digits* significant figures."""
    x = float(x)
    if x == 0.0 or not np.isfinite(x):
        return x
    from math import floor, log10
    return round(x, -int(floor(log10(abs(x)))) + (digits - 1))


@dataclass
class AircraftEnergyParams:
    """
    Complete physical, geometrical, propulsion and battery parameters
    of an eVTOL lift+cruise aircraft.


    Instances can be created:
        - Directly:           AircraftEnergyParams(m_tow=12.0, ...)
        - From a JSON file:   AircraftEnergyParams.from_json('data/default_vehicle.json')
        - Via factory:        get_vehicle('baseline')

    The field defaults below are kept identical to
    ``data/default_vehicle.json``. Two different answers to "what is the
    baseline aircraft" is how a test ends up passing against a machine
    nobody flies — ``test_dataclass_defaults_match_the_default_vehicle``
    pins them together.
    """

    # --- Airframe ---
    m_tow: float = 10.0
    """All-up mass *as this object is configured* [kg].

    A vehicle file carries the aircraft with **no payload**: operating
    empty mass, battery included. :meth:`with_payload` returns a copy
    with the payload added on top, so a definition whose ``m_tow`` was
    the datasheet MTOW would exceed MTOW the moment a payload was
    loaded, and every hover and stall number would be computed for an
    aircraft heavier than any that can legally fly.

    :attr:`m_tow_max` records the certificated ceiling so the two cannot
    be confused again."""

    m_tow_max: Optional[float] = 12.5
    """Maximum take-off mass the airframe is rated for [kg], or ``None``
    if unknown. Checked by :meth:`with_payload` and :meth:`validate`."""

    payload_capacity_kg: Optional[float] = 2.5
    """Maximum payload the manufacturer quotes [kg], or ``None``."""

    wind_rating_beaufort: Optional[int] = 5
    """Wind the airframe is rated to operate in, as a Beaufort force.

    Manufacturers quote "wind resistance level 5", which is a *band*
    (8.0-10.7 m/s), not a number. :attr:`wind_limit_ms` reads it at the
    bottom, because an aircraft rated to force 5 has been shown to
    handle 8.0 m/s, not 10.7."""

    g: float = 9.80665               # Gravitational acceleration [m/s²]
    S_ref: float = 0.6612              # Wing reference area [m²]
    C_L: float = 0.80                # Lift coefficient (cruise)
    C_L_max: float = 1.25932         # Max lift coefficient (from the polar)
    C_D0: float = 0.00713147        # Wing zero-lift drag; the rest of the
                                     # airframe comes from raptor.drag
    e_oswald: float = 0.818553        # Oswald efficiency factor
    AR: float = 8.0                  # Wing aspect ratio

    # --- VTOL Rotors ---
    A_rotor: float = 0.9817          # Total rotor disk area [m²] (legacy;
                                     # derived from diameter when given)
    V_tip: float = 110.0             # Rotor tip speed [m/s]
    sigma_rotor: float = 0.08        # Rotor solidity (Nc/πR)
    C_d_blade: float = 0.013         # Mean blade drag coefficient
    k_i: float = 1.15                # Induced power correction factor

    # --- Lift rotors ---
    # Specify the geometry, not the area. `A_rotor` used to be documented
    # as the total disk area but was populated with values four times too
    # small for the mass they lift, which is exactly the ambiguity that a
    # diameter and a count cannot have.
    n_rotors: int = 4                # Number of lift rotors
    rotor_diameter_m: float = 0.559   # Per-rotor diameter [m]; 0 = use A_rotor
    #: How the lift rotors come to rest in cruise: 'folded', 'aligned' or
    #: 'unaligned'. Moves total parasite drag by a factor of two.
    rotor_parking: str = "aligned"

    # --- Propulsion ---
    eta_prop: float = 0.67           # Overall propulsion efficiency (fallback)
    #: Efficiency of the lift-rotor drivetrain (motor × ESC × rotor). Hover
    #: figure-of-merit and cruise propeller efficiency are different numbers
    #: on a lift+cruise airframe; a single value biases whichever phase
    #: dominates the mission. ``None`` falls back to ``eta_prop``.
    eta_vtol: Optional[float] = 0.62
    #: Efficiency of the cruise propulsor. ``None`` falls back to ``eta_prop``.
    eta_cruise: Optional[float] = 0.72

    # --- Battery ---
    battery_mass: float = 5.058        # Battery mass [kg]
    specific_energy: float = 194.78
    """Pack-level specific energy [Wh/kg] — cells, wiring, BMS and case.

    Cell-level figures are 15-25 % higher, and quoting one here is the
    commonest way a UAV energy budget comes out optimistic. A high-power
    21700 reaches about 230 Wh/kg on its own chemistry, so a *pack*
    above roughly 200 Wh/kg is claiming to weigh less than its own
    cells; :meth:`validate` says so."""
    battery_voltage: float = 43.2    # Nominal pack voltage [V] (12S Li-ion)
    battery_capacity_mah: float = 22000.0  # If 0, auto-computed from mass & specific energy
    #: Fraction of nameplate energy actually deliverable. Pack overhead,
    #: the unusable tail below the cut-off voltage and cell mismatch all
    #: sit between the cell datasheet and what the aircraft can draw.
    battery_usable_fraction: float = 0.88
    #: Continuous discharge limit [C]. Exceeding it is not modelled as a
    #: hard failure, but it is reported — a design that only closes by
    #: pulling 15 C continuously has not really closed.
    battery_max_c_rate: float = 10.0
    #: Cells in series. Sets the open-circuit voltage curve; 6S LiPo is
    #: 25.2 V full, 22.2 V nominal, ~19.8 V empty.
    cell_chemistry: str = "li-ion"
    """Which open-circuit-voltage curve the pack model uses.

    One of the keys of :data:`raptor.battery.CELL_CHEMISTRY`. It matters
    more than it looks: a LiPo falls steadily from 4.2 V while a
    high-nickel Li-ion holds a long plateau and then drops in the last
    fifth, and the plateau is what decides how much of the pack sits
    above the cut-off."""

    battery_cells_series: int = 12
    #: Nominal capacity of one cell [Ah], used to infer how many strings
    #: are in parallel and hence the pack's internal resistance.
    cell_capacity_ah: float = 4.2
    """Capacity of **one cell**, not of the pack [Ah].

    The pack's capacity is ``battery_capacity_mah``; this is what one
    cell holds, and the ratio is how many parallel strings the pack has,
    which is what divides its internal resistance. Putting the pack
    figure here collapses the parallel count to one and multiplies the
    modelled pack resistance by that count."""
    #: Internal resistance of one cell [Ω] at room temperature.
    cell_internal_resistance_ohm: float = 0.025
    """DC internal resistance of **one cell** [ohm].

    Around 20-25 mOhm for a high-power 21700 such as the Molicel P42A
    (which quotes <15 mOhm at 1 kHz AC; the DC figure is higher). Series
    cells add resistance and parallel strings divide it, so the pack
    value follows from this, the series count and the parallel count —
    see :attr:`battery_resistance`."""
    #: Pack internal resistance [Ω]. ``None`` derives it from the cell
    #: figures and the series/parallel layout.
    battery_internal_resistance_ohm: Optional[float] = None

    # --- Wing section (used to build a computed drag polar) ---
    airfoil: str = "naca4412"        # section name for the aero model
    taper_ratio: float = 1.0
    twist_tip_deg: float = 0.0       # washout is negative

    #: Airframe dimensions for the drag build-up: fuselage, booms, tail.
    #: Empty means the build-up falls back to its own defaults.
    geometry: Dict = field(default_factory=dict)

    #: Where each number came from. Free-form, but the convention used by
    #: the shipped vehicles is one entry per parameter reading
    #: "referenced" (published by the manufacturer), "derived"
    #: (computed from referenced values by a stated relation) or
    #: "assumed" (engineering judgement, with the basis given). A
    #: parameter with no entry has no justification on record.
    provenance: Dict = field(default_factory=dict)

    # --- Metadata (set by from_json / factory functions) ---
    name: str = "Baseline Medical Delivery"
    description: str = ""

    def __post_init__(self):
        """Fill in the battery capacity if it was not given explicitly."""
        if self.battery_capacity_mah <= 0:
            energy_wh = self.battery_mass * self.specific_energy
            self.battery_capacity_mah = energy_wh / self.battery_voltage * 1000.0

    # ── Derived quantities ────────────────────────────────────────────────
    #
    # These are properties, not attributes computed once in __post_init__.
    # The optimizer models payload by raising ``m_tow`` on a copy of the
    # vehicle; with a cached ``W`` that assignment changed nothing at all,
    # so every payload the package has ever flown weighed the same as an
    # empty aircraft. Deriving on read makes that class of bug impossible.

    @property
    def W(self) -> float:
        """Weight [N]. Tracks ``m_tow``."""
        return self.m_tow * self.g

    @property
    def k_drag(self) -> float:
        """Induced-drag factor 1/(π·e·AR)."""
        return 1.0 / (np.pi * self.e_oswald * self.AR)

    @property
    def C_D(self) -> float:
        """Drag coefficient at the reference cruise ``C_L``."""
        return self.C_D0 + self.k_drag * self.C_L ** 2

    @property
    def battery_energy_wh(self) -> float:
        """
        Energy the chemistry holds [Wh].

        Taken from the pack's own voltage curve — mean open-circuit
        voltage over state of charge, times the cells in series, times
        the capacity — whenever those are known, and from
        ``battery_mass × specific_energy`` otherwise.

        The order matters, and it did not use to be this way round. With
        energy coming from mass and specific energy, the OCV curve
        affected only the current drawn and the resistive loss, so
        replacing an assumed curve with a measured one moved the
        endurance by **nothing at all** — the pack held two independent
        statements of its own energy and only the one with no chemistry
        in it reached the answer.

        Note what this figure is: the energy *in the cells*, at open
        circuit. It is not what arrives at the propellers. The pack
        model takes the resistive loss out downstream, from
        ``energy_from_cells_wh``, so subtracting it here as well would
        count it twice.

        :meth:`validate` checks the two routes agree.
        """
        curve = self.battery_energy_from_curve_wh
        if curve is not None:
            return curve
        return self.battery_mass * self.specific_energy

    @property
    def battery_energy_from_curve_wh(self) -> Optional[float]:
        """Pack energy implied by the OCV curve, or ``None`` if unknown."""
        if self.battery_cells_series <= 0 or self.battery_capacity_ah <= 0:
            return None
        try:
            from .battery import mean_cell_voltage
            v_mean = mean_cell_voltage(self.cell_chemistry)
        except Exception:
            return None
        return float(v_mean * self.battery_cells_series
                     * self.battery_capacity_ah)

    @property
    def battery_energy_from_mass_wh(self) -> float:
        """Pack energy implied by mass and specific energy [Wh]."""
        return float(self.battery_mass * self.specific_energy)

    @property
    def battery_energy_j(self) -> float:
        return self.battery_energy_wh * 3600.0

    @property
    def battery_capacity_ah(self) -> float:
        return self.battery_capacity_mah / 1000.0

    @property
    def battery_strings_parallel(self) -> float:
        """Parallel strings implied by pack capacity and cell capacity."""
        if self.cell_capacity_ah <= 0:
            return 1.0
        return max(self.battery_capacity_ah / self.cell_capacity_ah, 1.0)

    @property
    def battery_resistance(self) -> float:
        """
        Pack internal resistance [Ω].

        Series cells add resistance, parallel strings divide it. At the
        currents a heavy VTOL pulls in hover this is not a rounding term:
        the I²R heat comes out of the same pack that has to fly the
        mission, and ignoring it makes endurance look better than it is.
        """
        if self.battery_internal_resistance_ohm is not None:
            return float(self.battery_internal_resistance_ohm)
        return (self.battery_cells_series * self.cell_internal_resistance_ohm
                / self.battery_strings_parallel)

    @property
    def wind_limit_ms(self) -> float:
        """Operating wind limit [m/s], or infinity if unrated."""
        if self.wind_rating_beaufort is None:
            return float("inf")
        from .wind import beaufort_limit_ms
        return beaufort_limit_ms(self.wind_rating_beaufort)

    @property
    def battery_usable_wh(self) -> float:
        """Energy the aircraft can actually draw [Wh]."""
        return self.battery_energy_wh * self.battery_usable_fraction

    @property
    def eta_vtol_effective(self) -> float:
        return self.eta_vtol if self.eta_vtol is not None else self.eta_prop

    @property
    def eta_cruise_effective(self) -> float:
        return self.eta_cruise if self.eta_cruise is not None else self.eta_prop

    @property
    def rotor_diameter(self) -> float:
        """Diameter of one lift rotor [m]."""
        if self.rotor_diameter_m > 0:
            return float(self.rotor_diameter_m)
        if self.n_rotors <= 0 or self.A_rotor <= 0:
            return 0.0
        return 2.0 * np.sqrt(self.A_rotor / self.n_rotors / np.pi)

    @property
    def rotor_area_total(self) -> float:
        """
        Total lift-rotor disk area [m²].

        Derived from diameter and count when a diameter is given, so the
        area cannot silently disagree with the hardware. Falls back to the
        legacy ``A_rotor`` field otherwise.
        """
        if self.rotor_diameter_m > 0 and self.n_rotors > 0:
            return float(self.n_rotors * np.pi * (self.rotor_diameter_m / 2) ** 2)
        return float(self.A_rotor)

    @property
    def disk_loading(self) -> float:
        """Rotor disk loading [N/m²] over the total lift-rotor area."""
        A = self.rotor_area_total
        return self.W / A if A > 0 else float('inf')

    @property
    def wing_chord(self) -> float:
        """Mean aerodynamic chord implied by S_ref and AR [m]."""
        return np.sqrt(self.S_ref / self.AR) if self.AR > 0 else 0.0

    # ── Aerodynamics ──────────────────────────────────────────────────────

    @property
    def wing_geometry(self):
        """Wing geometry implied by this vehicle, for the aero model."""
        from .aero import WingGeometry
        return WingGeometry(
            S_ref=self.S_ref, AR=self.AR,
            taper_ratio=self.taper_ratio,
            twist_tip_deg=self.twist_tip_deg,
            airfoil=self.airfoil,
        )

    @property
    def aero_polar(self):
        """Attached drag polar, or ``None`` if the vehicle has no aero model."""
        return getattr(self, '_aero_polar', None)

    @property
    def drag_buildup(self):
        """Attached non-wing drag build-up, or ``None``."""
        return getattr(self, '_drag_buildup', None)

    def attach_drag_buildup(self, buildup) -> "AircraftEnergyParams":
        """
        Account for the fuselage, booms, gear and stopped rotors explicitly.

        With a build-up attached, ``C_D0`` on this vehicle is interpreted
        as the **wing alone** — the build-up supplies everything else. The
        two are summed, never overlapped: the lifting-line polar already
        contains the wing's profile and induced drag, and the build-up
        contains no wing.
        """
        self._drag_buildup = buildup
        return self

    def build_drag(self, **kwargs) -> "AircraftEnergyParams":
        """Construct a default quadplane build-up for this vehicle."""
        from .drag import quadplane_buildup
        for k, v in (self.geometry or {}).items():
            kwargs.setdefault(k, v)
        kwargs.setdefault("S_ref", self.S_ref)
        kwargs.setdefault("m_tow", self.m_tow)
        kwargs.setdefault("n_rotors", self.n_rotors)
        kwargs.setdefault("rotor_diameter", self.rotor_diameter)
        kwargs.setdefault("rotor_parking", self.rotor_parking)
        return self.attach_drag_buildup(quadplane_buildup(**kwargs))

    def build_aero(self, verbose: bool = False, **kwargs) -> "AircraftEnergyParams":
        """
        Attach both halves of the aerodynamic model in one call.

        The wing polar from the lifting line, and the rest of the airframe
        from the component build-up. Together they are the whole aircraft;
        either alone is not.
        """
        return self.build_polar(verbose=verbose, **kwargs).build_drag()

    def attach_polar(self, polar) -> "AircraftEnergyParams":
        """
        Use a computed drag polar instead of the constant-coefficient one.

        Returns self, so it chains. The polar is held by reference and
        deliberately not deep-copied by :meth:`with_payload`: it describes
        the wing, which payload does not change, and copying a table per
        payload variant would be waste.
        """
        self._aero_polar = polar
        return self

    def build_polar(self, refresh_scalars: bool = True,
                    **kwargs) -> "AircraftEnergyParams":
        """
        Compute (or load from cache) this vehicle's polar and attach it.

        By default the scalar ``C_L_max``, ``C_D0`` and ``e_oswald``
        fields are then refreshed from the computed table. They have to
        be: ``stall_speed_at`` reads the scalar, not the table, so a
        vehicle that attached a polar but kept the class-default
        ``C_L_max`` had its stall speed — and therefore its whole speed
        envelope — computed for a different wing than the one it flies.
        Pass ``refresh_scalars=False`` to keep hand-set values.
        """
        from .aero import get_polar
        self.attach_polar(get_polar(self.wing_geometry, **kwargs))
        if refresh_scalars:
            self.refresh_scalars_from_polar()
        return self

    def refresh_scalars_from_polar(self, altitude: float = 2800.0,
                                   speed_margin: float = 1.3,
                                   iterations: int = 4
                                   ) -> "AircraftEnergyParams":
        """
        Set ``C_L_max``, ``C_D0`` and ``e_oswald`` from the attached polar.

        Evaluated at the Reynolds number the aircraft actually cruises
        at, which depends on the speed, which depends on the stall speed,
        which depends on ``C_L_max`` — so it is iterated. Three passes
        are enough: Reynolds number moves the coefficients weakly, so the
        loop converges long before it runs out.
        """
        polar = self.aero_polar
        if polar is None:
            return self

        for _ in range(max(iterations, 1)):
            v_ref = speed_margin * self.stall_speed_at(altitude)
            Re = self.reynolds_at(v_ref, altitude)
            cl_max = polar.C_L_max(Re)
            if not np.isfinite(cl_max) or cl_max <= 0:
                break
            if abs(cl_max - self.C_L_max) < 1e-4:
                self.C_L_max = float(cl_max)
                break
            self.C_L_max = float(cl_max)

        cd0, k = polar.equivalent_parabolic(
            self.reynolds_at(speed_margin * self.stall_speed_at(altitude),
                             altitude))
        # Rounded to six significant figures. A lifting-line solve is not
        # accurate to seventeen, and carrying the full float makes the
        # vehicle files churn on every rebuild for no information.
        self.C_L_max = _sig(self.C_L_max)
        self.C_D0 = _sig(cd0)
        if k > 0:
            self.e_oswald = _sig(1.0 / (np.pi * k * self.AR))
        return self

    def reynolds_at(self, V: float, altitude: float) -> float:
        """Chord Reynolds number at a flight condition."""
        from .aero import reynolds
        return reynolds(V, self.wing_chord, altitude)

    def drag_coefficient(self, C_L: float, V: float = None,
                         altitude: float = None) -> float:
        """
        Drag coefficient at a trimmed lift coefficient.

        Uses the attached polar when there is one, and the analytic
        parabolic form otherwise. Both branches take the same arguments,
        so the caller never has to know which is in use — but
        ``aero_source`` reports it, because a result built on assumed
        constants should not be quoted as computed aerodynamics.
        """
        polar = self.aero_polar
        if polar is None or V is None or altitude is None:
            wing = self.C_D0 + self.k_drag * C_L ** 2
        else:
            wing = polar.C_D(C_L, self.reynolds_at(V, altitude))

        buildup = self.drag_buildup
        if buildup is None or V is None or altitude is None:
            return wing
        return wing + buildup.C_D0(V, altitude)

    def C_L_max_at(self, V: float = None, altitude: float = None) -> float:
        """Maximum lift coefficient, from the polar when one is attached."""
        polar = self.aero_polar
        if polar is None or V is None or altitude is None:
            return self.C_L_max
        return polar.C_L_max(self.reynolds_at(V, altitude))

    @property
    def aero_is_complete(self) -> bool:
        """True when both halves of the aerodynamic model are attached.

        The wing polar and the airframe build-up are both needed; either
        alone is not the aircraft. A vehicle with only the wing has
        roughly twice the lift-to-drag ratio of the real thing, which
        propagates straight into the derived climb angle — and a
        *shallower* one, because climb angle goes as 1/(L/D). So a
        forgotten ``build_aero()`` does not fail, it quietly makes the
        aircraft look worse at climbing and better at cruising."""
        return self.aero_polar is not None and self.drag_buildup is not None

    @property
    def aero_source(self) -> str:
        polar = self.aero_polar
        wing = polar.source if polar is not None else "constant-coefficient"
        rest = "build-up" if self.drag_buildup is not None else "lumped in C_D0"
        return f"wing: {wing}, airframe: {rest}"

    def drag_report(self, V: float = 25.0, altitude: float = 2800.0,
                    C_L: float = None) -> str:
        """Where this vehicle's drag comes from at a flight condition."""
        rho = isa_density(altitude)
        if C_L is None:
            C_L = self.W / (0.5 * rho * V ** 2 * self.S_ref)

        polar = self.aero_polar
        wing = (polar.C_D(C_L, self.reynolds_at(V, altitude))
                if polar is not None else self.C_D0 + self.k_drag * C_L ** 2)
        rest = (self.drag_buildup.C_D0(V, altitude)
                if self.drag_buildup is not None else 0.0)
        total = wing + rest

        lines = [
            f"Drag at {V:.0f} m/s, {altitude:.0f} m, C_L = {C_L:.3f}",
            f"  wing (profile + induced)  {wing:.5f}  {100*wing/total:5.1f} %",
            f"  rest of airframe          {rest:.5f}  {100*rest/total:5.1f} %",
            f"  total C_D                 {total:.5f}",
            f"  L/D                       {C_L/total:.1f}",
            f"  source: {self.aero_source}",
        ]
        if self.drag_buildup is not None:
            lines.append("")
            lines.append(self.drag_buildup.summary(V, altitude))
        return "\n".join(lines)

    # ── Battery characterisation ──────────────────────────────────────────

    @property
    def battery_fit(self):
        """Fitted pack characterisation, or ``None`` if none was measured."""
        return getattr(self, '_battery_fit', None)

    def fit_battery_from_log(self, log, **kwargs) -> "AircraftEnergyParams":
        """
        Replace the assumed pack model with a fit to measured discharge data.

        Sets the open-circuit voltage curve, internal resistance and
        capacity from the log, and records the provenance so a result
        computed from measurement is distinguishable from one computed
        from a textbook table.
        """
        from .battery import fit_battery
        kwargs.setdefault("resistance_prior_ohm", self.battery_resistance)
        fit = fit_battery(log, **kwargs)
        self._battery_fit = fit
        self.battery_capacity_mah = fit.capacity_ah * 1000.0
        if fit.resistance_fitted:
            self.battery_internal_resistance_ohm = fit.resistance_ohm
        return self

    def battery_provenance(self) -> str:
        """Which battery numbers are measured and which are assumed."""
        from .battery import battery_provenance
        return battery_provenance(self)

    def with_payload(self, payload_kg: float) -> "AircraftEnergyParams":
        """
        A copy of this vehicle carrying ``payload_kg`` of payload.

        Prefer this to mutating ``m_tow``: it states the intent, and it
        leaves the original untouched so a sweep cannot accumulate mass
        across iterations.
        """
        import copy as _copy
        # Held by reference, not deep-copied: they describe the aircraft,
        # which payload does not change, and duplicating a lookup table
        # per payload variant is waste.
        shared = {}
        for attr in ("_aero_polar", "_drag_buildup", "_battery_fit"):
            if hasattr(self, attr):
                shared[attr] = getattr(self, attr)
                setattr(self, attr, None)
        try:
            out = _copy.deepcopy(self)
        finally:
            for attr, value in shared.items():
                setattr(self, attr, value)

        if payload_kg < 0:
            raise ValueError(f"payload must be >= 0, got {payload_kg}")
        out.m_tow = self.m_tow + payload_kg
        for attr, value in shared.items():
            setattr(out, attr, value)

        # Loud rather than silent: exceeding MTOW does not make the
        # physics wrong, it makes the flight illegal, and the energy
        # numbers that come back will look perfectly reasonable.
        if self.m_tow_max is not None and out.m_tow > self.m_tow_max + 1e-9:
            warnings.warn(
                f"{self.name or 'vehicle'}: {payload_kg:.2f} kg of payload "
                f"puts all-up mass at {out.m_tow:.2f} kg, over the "
                f"{self.m_tow_max:.2f} kg maximum. The result is a mission "
                f"no regulator would approve.",
                stacklevel=2,
            )
        return out

    @property
    def payload_headroom_kg(self) -> float:
        """Payload this aircraft can still take before hitting MTOW [kg]."""
        if self.m_tow_max is None:
            return float("inf")
        return float(self.m_tow_max - self.m_tow)

    def validate(self) -> list:
        """
        Report physically implausible parameter combinations.

        Returns a list of human-readable warnings; empty means nothing
        obviously wrong. These are sanity bounds, not hard limits — a
        deliberate design may sit outside them, but it should do so
        knowingly.
        """
        out = []

        # Judged at maximum take-off mass, not empty: disk loading and
        # stall margin are worst when the aircraft is loaded, and that is
        # the case a limit has to cover.
        m_worst = self.m_tow_max if self.m_tow_max is not None else self.m_tow
        A = self.rotor_area_total
        dl_kg = m_worst / A if A > 0 else float('inf')
        if dl_kg > 25.0:
            out.append(
                f"Disk loading is {dl_kg:.0f} kg/m² over {A:.2f} m² of total "
                f"rotor area — multirotors of this class run 5–15 kg/m². That "
                f"implies {self.n_rotors} rotors of only "
                f"{self.rotor_diameter*100:.0f} cm diameter. Hover power scales "
                f"with the square root of disk loading, so this drives every "
                f"VTOL number in the mission."
            )
        if self.rotor_diameter_m > 0 and self.A_rotor > 0:
            implied = self.n_rotors * np.pi * (self.rotor_diameter_m / 2) ** 2
            if abs(implied - self.A_rotor) > 0.02 * max(implied, 1e-9):
                out.append(
                    f"A_rotor ({self.A_rotor:.3f} m²) disagrees with "
                    f"{self.n_rotors} × {self.rotor_diameter_m:.3f} m rotors "
                    f"({implied:.3f} m²). The diameter wins, so A_rotor is "
                    f"dead weight that will mislead the next reader — set it "
                    f"to {implied:.4f} or remove it."
                )
        if self.rotor_diameter_m <= 0:
            out.append(
                "Rotor geometry given as a total area rather than a diameter "
                "and count. Set rotor_diameter_m so the disk area cannot "
                "disagree with the hardware."
            )
        elif dl_kg < 3.0:
            out.append(f"Disk loading is only {dl_kg:.1f} kg/m² — unusually low.")

        if not 0.2 <= self.eta_vtol_effective <= 0.85:
            out.append(f"eta_vtol {self.eta_vtol_effective:.2f} is outside 0.20–0.85.")
        if not 0.2 <= self.eta_cruise_effective <= 0.90:
            out.append(f"eta_cruise {self.eta_cruise_effective:.2f} is outside 0.20–0.90.")

        # Nominal cell voltage varies by chemistry — 3.7 V for LiPo,
        # 3.6 V for cylindrical Li-ion, 3.2 V for LFP — so the check is
        # that the implied per-cell voltage is a real chemistry, not that
        # it equals any one of them. Assuming LiPo here used to flag
        # every correctly specified Li-ion pack.
        if self.battery_cells_series > 0:
            v_cell = self.battery_voltage / self.battery_cells_series
            if not 3.1 <= v_cell <= 3.9:
                out.append(
                    f"battery_voltage {self.battery_voltage:.1f} V over "
                    f"{self.battery_cells_series}S implies {v_cell:.2f} V per "
                    f"cell, which is no lithium chemistry in use (LFP 3.2, "
                    f"Li-ion 3.6, LiPo 3.7)."
                )

        from .energy import vortex_ring_margin
        vrs = vortex_ring_margin(self, 0.0, 2800.0)
        v_safe = vrs["vrs_lower"]
        if v_safe < 1.0:
            out.append(
                f"Vortex-ring onset at only {v_safe:.1f} m/s of descent "
                f"(hover induced velocity {vrs['hover_induced_velocity']:.1f} "
                f"m/s). Very lightly loaded rotors enter the vortex-ring "
                f"state at low descent rates, which constrains the approach."
            )

        if self.m_tow_max is not None:
            if self.m_tow > self.m_tow_max:
                out.append(
                    f"Empty mass {self.m_tow:.2f} kg already exceeds the "
                    f"{self.m_tow_max:.2f} kg maximum take-off mass — there "
                    f"is no payload this aircraft can carry."
                )
            elif (self.payload_capacity_kg is not None
                  and abs(self.payload_headroom_kg
                          - self.payload_capacity_kg) > 0.05):
                out.append(
                    f"Mass budget does not close: {self.m_tow_max:.2f} kg "
                    f"MTOW minus {self.m_tow:.2f} kg empty leaves "
                    f"{self.payload_headroom_kg:.2f} kg, but the quoted "
                    f"payload capacity is {self.payload_capacity_kg:.2f} kg."
                )

        # Two independent statements of how much energy the pack holds:
        # mass x specific energy, and the OCV curve integrated over the
        # charge it carries. Nothing forces them to agree, and when they
        # do not, every endurance figure is wrong by the difference.
        curve_wh = self.battery_energy_from_curve_wh
        mass_wh = self.battery_energy_from_mass_wh
        if curve_wh and mass_wh > 0:
            gap = abs(curve_wh - mass_wh) / mass_wh
            if gap > 0.10:
                out.append(
                    f"Pack energy disagrees with itself: "
                    f"{self.battery_mass:.2f} kg at "
                    f"{self.specific_energy:.0f} Wh/kg gives {mass_wh:.0f} Wh, "
                    f"the {self.cell_chemistry} curve over "
                    f"{self.battery_capacity_ah:.1f} Ah at "
                    f"{self.battery_cells_series}S gives {curve_wh:.0f} Wh "
                    f"({100*gap:.0f}% apart). The curve is what the model "
                    f"uses; the mass figure is then wrong, or the capacity is."
                )
            # Pack-level specific energy above the cells' own is not a
            # thing. Wiring, BMS, interconnects and case are real mass.
            implied = curve_wh / max(self.battery_mass, 1e-9)
            if implied > 240.0:
                out.append(
                    f"{implied:.0f} Wh/kg at pack level. Cells of this class "
                    f"reach about 230 Wh/kg on their own, so this pack "
                    f"weighs less than its cells — battery_mass is probably "
                    f"cell mass with no allowance for wiring, BMS and case, "
                    f"which is 10-20 % on a real pack."
                )

        n_parallel = self.battery_strings_parallel
        if n_parallel <= 1.0001 and self.battery_capacity_ah > 10.0:
            out.append(
                f"cell_capacity_ah {self.cell_capacity_ah:.1f} Ah looks like "
                f"a pack figure, not a cell one: no {self.battery_capacity_ah:.0f} Ah "
                f"pack is a single string of cells. The implied parallel "
                f"count of {n_parallel:.1f} is what divides the pack's "
                f"internal resistance, so this inflates it by roughly the "
                f"real parallel count."
            )

        if self.C_L_max <= self.C_L:
            out.append(
                f"C_L_max ({self.C_L_max:.2f}) is not above the reference cruise "
                f"C_L ({self.C_L:.2f}) — the aircraft cruises at or beyond stall."
            )

        return out

    # ── Available power ───────────────────────────────────────────────────

    #: Fraction of the pack's maximum discharge rate treated as available
    #: for a sustained climb. The C-rate on a pack's label is a burst
    #: figure; holding it for the minutes a climb takes would heat the
    #: cells and is not what a flight controller will allow. Half is the
    #: conventional continuous derating and is an assumption, not a
    #: measurement — it is the first number to replace if the motor and
    #: ESC ratings are known.
    CONTINUOUS_POWER_FRACTION: float = 0.5

    @property
    def electrical_power_max_w(self) -> float:
        """
        Peak electrical power the pack can deliver [W].

        C-rate multiplies the **pack's** capacity, not a cell's. Reading
        ``cell_capacity_ah`` here sized a 30 Ah pack as though it were a
        single 4.2 Ah string, giving a seventh of the real power — which
        then became the binding constraint on the derived climb angle
        and halved it.
        """
        return float(self.battery_max_c_rate * self.battery_capacity_ah
                     * self.battery_voltage)

    def shaft_power_available_w(self, continuous: bool = True) -> float:
        """
        Propulsive power available at the airscrew [W].

        Pack electrical power through the cruise propulsive efficiency.
        Derated to a continuous rating by default, because a climb is
        held for minutes and the pack's C-rate is a burst number.
        """
        frac = self.CONTINUOUS_POWER_FRACTION if continuous else 1.0
        return float(self.electrical_power_max_w * frac
                     * self.eta_cruise_effective)

    #: Continuous shaft power the cruise motor can hold, as a multiple of
    #: the power needed for level cruise. A pusher on a delivery quadplane
    #: is sized with enough margin to climb out and to hold speed against
    #: a gust, not with the several-fold margin a sport airframe carries.
    #: 2.5 is the assumption; it is the single number that sets how steep
    #: a terrain-following corridor may be, so it is worth revisiting
    #: against a real motor curve.
    CLIMB_POWER_RATIO: float = 2.5

    def max_climb_angle_at(self, altitude: float, V: float = None,
                           structural_limit_deg: float = 15.0,
                           power_ratio: float = None) -> float:
        """
        Steepest sustained climb the propulsion supports [deg].

        Steady climb balances power: ``P_avail = D·V + W·V·sin γ``, so

            sin γ = (P_avail − D·V) / (W·V)

        and with ``P_avail = ratio · D · V`` this collapses to

            sin γ = (ratio − 1) · D / W = (ratio − 1) / (L/D)

        — the climb angle is set by lift-to-drag and installed power
        margin, and by nothing else. That is worth stating because the
        number it replaces was typed in: 15 degrees, independent of the
        aircraft, the payload and the air density, at a point in the code
        where the corridor envelope uses it to decide whether terrain can
        be followed at all. A climb angle assumed twice what the
        propulsion supports makes corridors look flyable that are not.

        The pack's own discharge limit is applied as a ceiling on top,
        for the case of a small pack behind a large motor.
        """
        ratio = power_ratio if power_ratio is not None else self.CLIMB_POWER_RATIO
        V = V if V is not None else 1.3 * self.stall_speed_at(altitude)
        rho = isa_density(altitude)
        q = 0.5 * rho * V * V
        if q <= 0 or self.S_ref <= 0:
            return 0.0
        C_L = self.W / (q * self.S_ref)
        C_D = self.drag_coefficient(C_L, V=V, altitude=altitude)
        D = q * self.S_ref * C_D
        if D <= 0:
            return float(structural_limit_deg)

        p_motor = ratio * D * V
        p_pack = self.shaft_power_available_w()
        excess = min(p_motor, p_pack) - D * V
        if excess <= 0:
            return 0.0
        sin_gamma = excess / (self.W * V)
        if sin_gamma >= 1.0:
            return float(structural_limit_deg)
        return float(min(np.degrees(np.arcsin(sin_gamma)),
                         structural_limit_deg))

    def max_descent_angle_at(self, altitude: float, V_max: float = None,
                             structural_limit_deg: float = 15.0) -> float:
        """
        Steepest sustained descent the airframe can hold [deg].

        A clean airframe with no spoilers cannot descend more steeply
        than it glides at idle: past that it accelerates instead. The
        glide angle is ``atan(C_D / C_L)``, which is steepest at the
        fastest speed the envelope allows, since ``C_L`` falls with the
        square of speed while ``C_D`` falls far more slowly.

        This is usually the binding constraint on a terrain-following
        corridor over falling ground, and it is normally *shallower*
        than the climb limit — the opposite of what a hand-typed pair of
        angles tends to assume.
        """
        V_max = V_max if V_max is not None else 2.1 * self.stall_speed_at(altitude)
        rho = isa_density(altitude)
        q = 0.5 * rho * V_max * V_max
        C_L = self.W / (q * self.S_ref) if q > 0 else float('inf')
        C_D = self.drag_coefficient(C_L, V=V_max, altitude=altitude)
        if C_L <= 0:
            return float(structural_limit_deg)
        return float(min(np.degrees(np.arctan(C_D / C_L)),
                         structural_limit_deg))

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
        """
        Serialize to the canonical JSON-compatible dict.

        Every field ``from_dict`` reads is written here. That symmetry is
        the whole point: an incomplete ``to_dict`` means saving a vehicle
        and loading it back returns a *different* aircraft, silently —
        the version that dropped the efficiencies and the airframe
        geometry came back with the fallback propulsive efficiency and
        no fuselage to compute drag from.
        """
        return {
            "name": self.name,
            "description": self.description,
            "provenance": dict(getattr(self, "provenance", {}) or {}),
            "airframe": {
                "m_tow": self.m_tow,
                "m_tow_max": self.m_tow_max,
                "payload_capacity_kg": self.payload_capacity_kg,
                "wind_rating_beaufort": self.wind_rating_beaufort,
                "g": self.g,
                "S_ref": self.S_ref, "AR": self.AR,
                "airfoil": self.airfoil,
                "taper_ratio": self.taper_ratio,
                "twist_tip_deg": self.twist_tip_deg,
                "C_L": self.C_L, "C_L_max": self.C_L_max,
                "C_D0": self.C_D0, "e_oswald": self.e_oswald,
            },
            "rotors": {
                "n_rotors": self.n_rotors,
                "rotor_diameter_m": self.rotor_diameter_m,
                "rotor_parking": self.rotor_parking,
                "A_rotor": self.A_rotor,
                "V_tip": self.V_tip,
                "sigma_rotor": self.sigma_rotor,
                "C_d_blade": self.C_d_blade,
                "k_i": self.k_i,
            },
            "geometry": dict(self.geometry or {}),
            "propulsion": {
                "eta_prop": self.eta_prop,
                "eta_vtol": self.eta_vtol,
                "eta_cruise": self.eta_cruise,
            },
            "battery": {
                "battery_mass": self.battery_mass,
                "specific_energy": self.specific_energy,
                "battery_voltage": self.battery_voltage,
                "battery_capacity_mah": self.battery_capacity_mah,
                "battery_cells_series": self.battery_cells_series,
                "cell_chemistry": self.cell_chemistry,
                "cell_capacity_ah": self.cell_capacity_ah,
                "cell_internal_resistance_ohm": self.cell_internal_resistance_ohm,
                "battery_internal_resistance_ohm":
                    self.battery_internal_resistance_ohm,
                "battery_usable_fraction": self.battery_usable_fraction,
                "battery_max_c_rate": self.battery_max_c_rate,
            },
        }

    def to_json(self, path: str, indent: int = 2) -> str:
        """Write this vehicle to a JSON file and return the path."""
        import os as _os
        d = _os.path.dirname(_os.path.abspath(path))
        if d:
            _os.makedirs(d, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=indent, ensure_ascii=False)
            f.write("\n")
        return path

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
            m_tow_max=af.get("m_tow_max"),
            payload_capacity_kg=af.get("payload_capacity_kg"),
            wind_rating_beaufort=af.get("wind_rating_beaufort"),
            g=af.get("g", 9.80665),
            S_ref=af.get("S_ref", 0.50),
            C_L=af.get("C_L", 1.159),
            C_L_max=af.get("C_L_max", 1.782),
            C_D0=af.get("C_D0", 0.013),
            e_oswald=af.get("e_oswald", 0.78),
            AR=af.get("AR", 8.0),
            airfoil=af.get("airfoil", "naca4412"),
            taper_ratio=af.get("taper_ratio", 1.0),
            twist_tip_deg=af.get("twist_tip_deg", 0.0),
            A_rotor=ro.get("A_rotor", 0.20),
            n_rotors=ro.get("n_rotors", 4),
            rotor_diameter_m=ro.get("rotor_diameter_m", 0.0),
            rotor_parking=ro.get("rotor_parking", "aligned"),
            V_tip=ro.get("V_tip", 80.0),
            sigma_rotor=ro.get("sigma_rotor", 0.06),
            C_d_blade=ro.get("C_d_blade", 0.012),
            k_i=ro.get("k_i", 1.15),
            eta_prop=pr.get("eta_prop", 0.6),
            eta_vtol=pr.get("eta_vtol"),
            eta_cruise=pr.get("eta_cruise"),
            battery_mass=ba.get("battery_mass", 3.0),
            specific_energy=ba.get("specific_energy", 200.0),
            battery_voltage=ba.get("battery_voltage", 22.2),
            battery_capacity_mah=ba.get("battery_capacity_mah", 0.0),
            battery_cells_series=ba.get("battery_cells_series", 6),
            cell_chemistry=ba.get("cell_chemistry", "li-ion"),
            cell_capacity_ah=ba.get("cell_capacity_ah", 5.0),
            cell_internal_resistance_ohm=ba.get(
                "cell_internal_resistance_ohm", 0.004),
            battery_internal_resistance_ohm=ba.get(
                "battery_internal_resistance_ohm"),
            battery_usable_fraction=ba.get("battery_usable_fraction", 0.88),
            battery_max_c_rate=ba.get("battery_max_c_rate", 10.0),
            geometry=d.get("geometry", {}) or {},
            provenance=d.get("provenance", {}) or {},
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

def _load_vehicle_file(filename: str) -> AircraftEnergyParams:
    """
    Load ``data/vehicles/<filename>``, or say how to make it.

    Deliberately without a hard-coded fallback. The versions of these
    factories that had one returned an invented aircraft whenever the
    file was missing — plausible numbers, no provenance, and no sign to
    the caller that anything had gone wrong.
    """
    path = os.path.normpath(os.path.join(_VEHICLES_DIR, filename))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. The shipped vehicles are generated from "
            f"published aircraft data; rebuild them with:\n"
            f"    python -m scripts.build_vehicles"
        )
    return AircraftEnergyParams.from_json(path)


def va17_config() -> AircraftEnergyParams:
    """
    T-Drones VA17 — 4.3 kg tilt-rotor, 0.9 kg payload.

    The small end of the class: one blood cooler or a vaccine box, short
    hops. Its rotors tilt, so it carries no parked-rotor drag but pays
    for it in cruise propulsive efficiency.
    """
    return _load_vehicle_file('va17.json')


def va23_config() -> AircraftEnergyParams:
    """
    T-Drones VA23 — 12.5 kg lift+cruise quadplane, 2.5 kg payload.

    The workhorse, and RAPTOR's default: the largest payload of the
    three, which is what a multi-stop delivery mission is limited by.
    """
    return _load_vehicle_file('va23.json')


def va25_config() -> AircraftEnergyParams:
    """
    Alafija VA25 — 13 kg lift+cruise quadplane, 2 kg payload.

    The range end: a faster cruise, a higher wind rating (Beaufort 6
    against the other two's 5) and the largest pack, at the cost of
    half a kilogram of payload against the VA23.
    """
    return _load_vehicle_file('va25.json')


#: The vehicles RAPTOR ships. Every one is derived from a real
#: aircraft's published data — see ``ac.provenance`` for which numbers
#: are quoted, which are computed and which are assumed.
VEHICLE_CONFIGS: Dict[str, callable] = {
    'va17': va17_config,
    'va23': va23_config,
    'va25': va25_config,
}

#: Alias for the default vehicle, so callers can ask for it by role.
VEHICLE_CONFIGS['default'] = va23_config


def get_vehicle(name: str) -> AircraftEnergyParams:
    """
    Get a vehicle configuration by name or JSON file path.

    Parameters
    ----------
    name : str
        Either a key in VEHICLE_CONFIGS ('va17', 'va23', 'va25',
        'default') or a path to a JSON file.
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
