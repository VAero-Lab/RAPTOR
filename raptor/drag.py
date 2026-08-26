"""
Drag Build-Up — Where the Airframe's Drag Actually Comes From
==============================================================

The wing is computed (:mod:`raptor.aero`, nonlinear lifting line). The
rest of the aircraft was one number: ``C_D0``, typed into a JSON file.

For a lift+cruise quadplane that is the wrong place to guess, because
the non-wing drag is not a small correction. The airframe carries a
fuselage, two booms, four *stopped* lift rotors and their nacelles all
the way through cruise, and a stopped rotor is close to a flat plate. On
the bundled airframe those parts turn out to be roughly as draggy as the
wing itself at cruise lift.

This module estimates them the standard way — component build-up:

    C_D0 = (1/S_ref) · Σ  C_f,i · FF_i · Q_i · S_wet,i

``C_f`` is flat-plate skin friction at the component's own Reynolds
number, ``FF`` a form factor for thickness and pressure drag, ``Q`` an
interference factor, and ``S_wet`` the wetted area. It is textbook
(Raymer, Hoerner, Gudmundsson), it is transparent, and every term can be
pointed at a physical part — which a single tuned constant cannot.

Everything here is an estimate, and says so. The point is not that these
numbers are exact; it is that they are *traceable*, they scale correctly
with speed and altitude, and a reviewer can see which component drives
the answer.

Author: Victor (LUAS-EPN / KU Leuven)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .aero import NU_AIR
from .atmosphere import isa_density


# ═══════════════════════════════════════════════════════════════════════════
# SKIN FRICTION AND FORM FACTORS
# ═══════════════════════════════════════════════════════════════════════════

def skin_friction(Re: float, laminar_fraction: float = 0.0) -> float:
    """
    Flat-plate skin-friction coefficient.

    Turbulent: the Prandtl–Schlichting fit ``0.455 / (log10 Re)^2.58``.
    Laminar: Blasius ``1.328 / √Re``.

    At the Reynolds numbers a small delivery aircraft flies — 10⁵ to
    10⁶ — the two differ by a factor of two or more, so how much of a
    surface stays laminar matters. On a smooth moulded fuselage a third
    is plausible; behind a rotor wake, nothing is.

    Parameters
    ----------
    Re : float
        Reynolds number on the component's reference length.
    laminar_fraction : float
        Fraction of the surface in laminar flow, 0 to 1.
    """
    Re = max(float(Re), 1e3)
    cf_turb = 0.455 / (np.log10(Re) ** 2.58)
    cf_lam = 1.328 / np.sqrt(Re)
    f = float(np.clip(laminar_fraction, 0.0, 1.0))
    return float(f * cf_lam + (1 - f) * cf_turb)


def form_factor_body(fineness_ratio: float) -> float:
    """
    Form factor for a body of revolution (fuselage, boom, nacelle).

    ``FF = 1 + 60/f³ + f/400`` with ``f`` the length-to-diameter ratio.
    Stubby bodies pay heavily: at f = 3 the factor is 3.2, at f = 8 it is
    1.14. Delivery-aircraft fuselages are stubby because they carry a
    box, which is exactly why this term should not be folded into a
    guessed constant.
    """
    f = max(float(fineness_ratio), 1.5)
    return float(1.0 + 60.0 / f ** 3 + f / 400.0)


def form_factor_surface(thickness_ratio: float,
                        sweep_deg: float = 0.0,
                        x_c_max: float = 0.3,
                        mach: float = 0.1) -> float:
    """
    Form factor for a lifting surface (tail, pylon, exposed strut).

    Raymer's expression. At the Mach numbers here the compressibility
    term is inert, but it is kept so the function stays valid if the
    package is ever pointed at something faster.
    """
    t_c = max(float(thickness_ratio), 0.02)
    sweep = np.radians(sweep_deg)
    return float(
        (1 + 0.6 / max(x_c_max, 0.05) * t_c + 100 * t_c ** 4)
        * (1.34 * mach ** 0.18 * np.cos(sweep) ** 0.28)
    )


# ═══════════════════════════════════════════════════════════════════════════
# COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DragComponent:
    """
    One physical part and the drag it contributes.

    Attributes
    ----------
    name : str
    wetted_area : float
        Surface area exposed to the flow [m²].
    reference_length : float
        Length used for this component's Reynolds number [m].
    form_factor : float
        Thickness and pressure drag multiplier. 1.0 for a flat plate.
    interference_factor : float
        Penalty for sitting near another surface. 1.0 clean, 1.1–1.5 for
        a nacelle or boom junction.
    laminar_fraction : float
        Fraction of the surface in laminar flow.
    drag_area : float or None
        Bypass the build-up and specify an equivalent flat-plate area
        ``C_D·A`` [m²] directly. Used for parts where a build-up is
        meaningless — a stopped rotor blade, an exposed leg, an antenna.
    note : str
        Where the number came from, so a reader can judge it.
    """
    name: str
    wetted_area: float = 0.0
    reference_length: float = 1.0
    form_factor: float = 1.0
    interference_factor: float = 1.0
    laminar_fraction: float = 0.0
    drag_area: Optional[float] = None
    note: str = ""

    def reynolds(self, V: float, altitude: float) -> float:
        rho = isa_density(altitude)
        nu = NU_AIR * (1.225 / max(rho, 1e-6))
        return float(V * self.reference_length / max(nu, 1e-12))

    def drag_area_at(self, V: float, altitude: float) -> float:
        """Equivalent flat-plate area ``C_D·A`` of this component [m²]."""
        if self.drag_area is not None:
            return float(self.drag_area)
        cf = skin_friction(self.reynolds(V, altitude), self.laminar_fraction)
        return float(cf * self.form_factor * self.interference_factor
                     * self.wetted_area)


# ── Constructors for the parts a quadplane actually has ───────────────────

def fuselage(length: float, diameter: float,
             laminar_fraction: float = 0.2,
             interference: float = 1.0) -> DragComponent:
    """
    A body of revolution. Wetted area from a cylinder with rounded ends,
    which is within a few per cent of a real pod for these fineness
    ratios.
    """
    f = length / max(diameter, 1e-6)
    s_wet = np.pi * diameter * length * 0.85 + 0.5 * np.pi * diameter ** 2
    return DragComponent(
        name="fuselage", wetted_area=float(s_wet),
        reference_length=float(length),
        form_factor=form_factor_body(f),
        interference_factor=interference,
        laminar_fraction=laminar_fraction,
        note=f"body of revolution, fineness {f:.1f}",
    )


def boom(length: float, diameter: float, count: int = 2,
         interference: float = 1.2) -> DragComponent:
    """
    Tail or motor booms. Interference is charged because a boom meets
    the wing at a junction, and junctions separate.
    """
    f = length / max(diameter, 1e-6)
    s_wet = np.pi * diameter * length * count
    return DragComponent(
        name=f"booms ×{count}", wetted_area=float(s_wet),
        reference_length=float(length),
        form_factor=form_factor_body(f),
        interference_factor=interference,
        laminar_fraction=0.1,
        note=f"{count} booms, fineness {f:.1f}",
    )


def tail_surface(area: float, chord: float, thickness_ratio: float = 0.10,
                 name: str = "tail", interference: float = 1.05
                 ) -> DragComponent:
    """Horizontal or vertical tail. Wetted area ≈ 2.05 × planform."""
    return DragComponent(
        name=name, wetted_area=float(2.05 * area),
        reference_length=float(chord),
        form_factor=form_factor_surface(thickness_ratio),
        interference_factor=interference,
        laminar_fraction=0.15,
        note=f"planform {area:.3f} m², t/c {thickness_ratio:.2f}",
    )


#: How much of a stopped blade's planform stays exposed to the flow.
#: This is the most uncertain number in the build-up and the one that
#: most changes the answer, so the choices are named rather than tuned.
#:
#: ``folded``    blades fold back along the boom — a small residual only
#: ``aligned``   blades park fore-and-aft, presenting thickness (~10 % of
#:               chord) plus the hub and imperfect alignment
#: ``unaligned`` blades stop wherever they stop; assume a third of the
#:               planform is broadside on average
#: ``tilted``    a tilt-rotor turns its lift rotors into cruise
#:               propellers, so nothing is parked at all
ROTOR_PARK_EXPOSURE = {
    "tilted": 0.0,
    "folded": 0.04,
    "aligned": 0.15,
    "unaligned": 0.35,
}


def stopped_rotors(n_rotors: int, blade_chord: float, blade_length: float,
                   n_blades: int = 2, cd_blade: float = 1.1,
                   parking: str = "aligned",
                   parked_fraction: float = None) -> DragComponent:
    """
    Lift rotors carried, stopped, through cruise.

    The term most often left out of a lift+cruise drag estimate, and on
    this class of airframe usually the largest single non-wing
    contributor. A blade broadside to the flow is a bluff plate
    (``C_D ≈ 1.1`` on frontal area); parked fore-and-aft it presents
    roughly its thickness instead, which is an order of magnitude less.

    Because the answer swings by a factor of eight between a folded rotor
    and one that stops wherever it likes, the exposure is chosen from
    :data:`ROTOR_PARK_EXPOSURE` by name — so a drag estimate always
    records *which assumption* it was built on.

    Parameters
    ----------
    parking : {'tilted', 'folded', 'aligned', 'unaligned'}
        How the blades come to rest. ``aligned`` is the sensible default
        for a lift+cruise machine with rotor-position control.
        ``tilted`` is for a tilt-rotor, which has no parked rotors
        because the lift rotors become the cruise propellers — the
        component is then zero and the propulsive efficiency carries
        the cost instead.
    parked_fraction : float
        Override the named value with a measured one.
    """
    if parked_fraction is None:
        if parking not in ROTOR_PARK_EXPOSURE:
            raise ValueError(
                f"parking must be one of {sorted(ROTOR_PARK_EXPOSURE)}, "
                f"got {parking!r}"
            )
        parked_fraction = ROTOR_PARK_EXPOSURE[parking]
        basis = parking
    else:
        basis = "measured"

    frontal = n_rotors * n_blades * blade_chord * blade_length * parked_fraction
    return DragComponent(
        name=f"stopped rotors ×{n_rotors}",
        drag_area=float(cd_blade * frontal),
        note=(f"{n_rotors}×{n_blades} blades, {basis}: "
              f"{parked_fraction*100:.0f}% of blade area exposed, "
              f"C_D {cd_blade:.1f}"),
    )


def nacelles(n: int, length: float, diameter: float,
             interference: float = 1.3) -> DragComponent:
    """Motor pods. Heavy interference: they sit in the wing's flow."""
    f = length / max(diameter, 1e-6)
    s_wet = np.pi * diameter * length * n
    return DragComponent(
        name=f"nacelles ×{n}", wetted_area=float(s_wet),
        reference_length=float(length),
        form_factor=form_factor_body(f),
        interference_factor=interference,
        laminar_fraction=0.0,
        note=f"{n} motor pods, fineness {f:.1f}",
    )


def landing_gear(drag_area: float = 0.004, name: str = "landing gear",
                 note: str = "fixed skids, estimated C_D·A") -> DragComponent:
    """Fixed gear or skids, as an equivalent flat-plate area."""
    return DragComponent(name=name, drag_area=float(drag_area), note=note)


def miscellaneous(drag_area: float = 0.002,
                  note: str = "antennas, sensors, gaps, surface roughness"
                  ) -> DragComponent:
    """
    The parts nobody models.

    Every real airframe has more drag than its parts list. A small
    explicit allowance is more honest than a quietly inflated form
    factor somewhere else.
    """
    return DragComponent(name="miscellaneous", drag_area=float(drag_area),
                         note=note)


# ═══════════════════════════════════════════════════════════════════════════
# BUILD-UP
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DragBuildup:
    """
    The non-wing drag of an airframe, part by part.

    The wing is deliberately excluded: it comes from the lifting-line
    polar, which already contains its profile and induced drag. Adding a
    build-up estimate for it as well would count it twice.
    """
    components: List[DragComponent] = field(default_factory=list)
    S_ref: float = 0.50
    description: str = ""

    def add(self, component: DragComponent) -> "DragBuildup":
        self.components.append(component)
        return self

    def drag_area(self, V: float, altitude: float) -> float:
        """Total equivalent flat-plate area of everything but the wing [m²]."""
        return float(sum(c.drag_area_at(V, altitude) for c in self.components))

    def C_D0(self, V: float, altitude: float) -> float:
        """
        Non-wing zero-lift drag coefficient, on ``S_ref``.

        Depends on speed and altitude through Reynolds number — which a
        constant cannot, and which matters when the same airframe cruises
        at 20 m/s near the ground and 32 m/s at 3 000 m.
        """
        return float(self.drag_area(V, altitude) / max(self.S_ref, 1e-9))

    def breakdown(self, V: float, altitude: float) -> Dict[str, float]:
        """Per-component drag area [m²], largest first."""
        d = {c.name: c.drag_area_at(V, altitude) for c in self.components}
        return dict(sorted(d.items(), key=lambda kv: -kv[1]))

    def parasite_L_over_D(self, W: float, V: float, altitude: float) -> float:
        """
        Lift-to-drag ratio counting only the non-wing drag [-].

        A quick smell test: this is an upper bound on the airframe's real
        L/D, since the wing has not been charged yet. A clean sailplane
        would be in the hundreds; a delivery quadplane carrying stopped
        rotors and fixed gear is usually 8–15, and anything above 30 means
        a component has been left out.
        """
        rho = isa_density(altitude)
        D = 0.5 * rho * V ** 2 * self.drag_area(V, altitude)
        return float(W / D) if D > 0 else float('inf')

    def summary(self, V: float = 25.0, altitude: float = 2800.0) -> str:
        total = self.drag_area(V, altitude)
        lines = [
            f"Non-wing drag build-up at {V:.0f} m/s, {altitude:.0f} m",
            f"  {'component':<24s} {'C_D·A [m²]':>11s} {'share':>7s}  basis",
        ]
        for c in self.components:
            da = c.drag_area_at(V, altitude)
            share = 100 * da / total if total > 0 else 0.0
            lines.append(f"  {c.name:<24s} {da:11.5f} {share:6.1f}%  {c.note}")
        lines += [
            f"  {'':<24s} {'-'*11}",
            f"  {'TOTAL':<24s} {total:11.5f}",
            f"  C_D0 on S_ref = {self.S_ref:.3f} m²:  {total/self.S_ref:.5f}",
        ]
        return "\n".join(lines)

    def validate(self, W: float = 117.7, V: float = 25.0,
                 altitude: float = 2800.0) -> list:
        """Report a build-up that looks physically implausible."""
        out = []
        cd0 = self.C_D0(V, altitude)
        if cd0 < 0.010:
            out.append(
                f"Non-wing C_D0 of {cd0:.4f} is sailplane-clean. A lift+cruise "
                f"airframe carries stopped rotors, booms and gear through "
                f"cruise; check nothing has been omitted."
            )
        if cd0 > 0.12:
            out.append(
                f"Non-wing C_D0 of {cd0:.4f} is very high — the aircraft would "
                f"spend most of its power on parasite drag. Check the "
                f"stopped-rotor parking assumption and the wetted areas."
            )
        ld = self.parasite_L_over_D(W, V, altitude)
        if ld > 30:
            out.append(f"Parasite-only L/D of {ld:.0f} is implausibly good; "
                       f"a component is probably missing.")
        return out


def quadplane_buildup(
    S_ref: float = 0.50,
    m_tow: float = 12.0,
    n_rotors: int = 4,
    rotor_diameter: float = 0.51,
    fuselage_length: float = 1.10,
    fuselage_diameter: float = 0.18,
    boom_length: float = 1.00,
    boom_diameter: float = 0.035,
    n_booms: int = 2,
    tail_area: float = 0.09,
    tail_chord: float = 0.18,
    rotor_parking: str = "aligned",
    **kwargs,
) -> DragBuildup:
    """
    A default build-up for a lift+cruise delivery quadplane.

    Dimensions default to a coherent set for a ~12 kg airframe with a
    2 m wing: a fuselage big enough for a few kilograms of cold-chain
    payload, booms carrying four lift rotors, and a modest tail. Override
    any of them with your own measurements — the point of a build-up is
    that each number is a dimension you can go and measure.

    Parameters
    ----------
    rotor_parking : {'folded', 'aligned', 'unaligned'}
        How the lift rotors come to rest in cruise. This one choice moves
        the total by a factor of two, so it is worth knowing which one
        your airframe does.
    """
    blade_length = rotor_diameter / 2 * 0.85
    blade_chord = rotor_diameter * 0.09

    b = DragBuildup(S_ref=S_ref,
                    description=f"lift+cruise quadplane, {m_tow:.0f} kg MTOW")
    b.add(fuselage(fuselage_length, fuselage_diameter))
    b.add(boom(boom_length, boom_diameter, count=n_booms))
    b.add(tail_surface(tail_area, tail_chord, name="tail (H+V)"))
    b.add(nacelles(n_rotors, length=0.12, diameter=0.05))
    b.add(stopped_rotors(n_rotors, blade_chord, blade_length,
                         parking=rotor_parking))
    b.add(landing_gear())
    b.add(miscellaneous())
    return b
