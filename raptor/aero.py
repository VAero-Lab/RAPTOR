"""
Aerodynamics — A Real Drag Polar Instead of a Guessed One
==========================================================

The package used a fixed parabolic polar, ``C_D = C_D0 + k·C_L²`` with
``C_D0`` a constant and ``k = 1/(π·e·AR)``. That is a reasonable
book-keeping form, but the two numbers in it were assumptions, and at the
Reynolds numbers a 2 m-span delivery aircraft actually flies — 2×10⁵ to
5×10⁵, where laminar separation dominates — a constant ``C_D0`` is not
close to true. Drag there depends strongly on both incidence and Reynolds
number, and the errors run in both directions: at the airframe in
``data/default_vehicle.json`` the fixed polar overstates cruise drag by
roughly a third at moderate lift and understates it near stall.

This module builds the polar from geometry with AeroSandbox's nonlinear
lifting line, which couples a vortex-lattice spanwise solution to real
2-D section data.

Why a surrogate and not a direct call
-------------------------------------
A nonlinear lifting-line solve takes about 3.5 s. A differential-evolution
run evaluates the objective a few thousand times, and each evaluation
prices twenty-odd flight segments — call it 75 000 solves, or three days
per corridor. So the solver runs once over a grid of (α, Re), the result
is cached to disk, and the optimizer interpolates. Interpolation costs
microseconds and is smooth, which the optimizer also needs.

Everything degrades gracefully: with no AeroSandbox installed, or no
cached table, the analytic parabolic polar is used and says so.

Author: Victor (LUAS-EPN / KU Leuven)
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence, Tuple

import numpy as np

from .atmosphere import isa_density

#: Kinematic viscosity of air at ~15 °C [m²/s]. Reynolds number is not
#: sensitive enough to temperature here to justify carrying a model.
NU_AIR = 1.46e-5


def reynolds(V: float, chord: float, altitude: float) -> float:
    """Chord Reynolds number at a flight condition."""
    rho = isa_density(altitude)
    nu = NU_AIR * (1.225 / max(rho, 1e-6))     # ν scales as 1/ρ at fixed μ
    return float(V * chord / max(nu, 1e-12))


# ═══════════════════════════════════════════════════════════════════════════
# GEOMETRY
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class WingGeometry:
    """
    The wing the polar is computed for.

    Defaults are derived from the reference area and aspect ratio that
    the vehicle definitions already carry, so an existing vehicle gains a
    real polar without new inputs. Supply taper, twist or a different
    airfoil to refine it.
    """
    S_ref: float = 0.50            # m²
    AR: float = 8.0
    taper_ratio: float = 1.0
    twist_tip_deg: float = 0.0     # washout is negative
    airfoil: str = "naca4412"
    n_spanwise: int = 8

    @property
    def span(self) -> float:
        return float(np.sqrt(self.S_ref * self.AR))

    @property
    def chord_root(self) -> float:
        # S = b·c_mean, c_mean = c_root·(1+λ)/2
        return float(2 * self.S_ref / (self.span * (1 + self.taper_ratio)))

    @property
    def chord_tip(self) -> float:
        return self.chord_root * self.taper_ratio

    @property
    def chord_mean(self) -> float:
        return float(self.S_ref / self.span)

    def fingerprint(self) -> str:
        """Stable hash of the geometry, used to key the cached table."""
        blob = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════════
# POLAR
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AeroPolar:
    """
    A drag polar the optimizer can afford to evaluate.

    Holds ``CL`` and ``CD`` on a grid of angle of attack and Reynolds
    number, and interpolates ``C_D(C_L, Re)`` from it. Interpolation is
    linear in ``C_L`` and in ``log Re`` — the latter because drag varies
    roughly with a power of Reynolds number, so the log scale keeps the
    surrogate smooth across the flight envelope.

    Attributes
    ----------
    alpha_deg : ndarray, shape (n_alpha,)
    reynolds : ndarray, shape (n_re,)
    CL, CD : ndarray, shape (n_re, n_alpha)
    CDi, CDp : ndarray
        Induced and profile components, kept so the polar can be reported
        and sanity-checked against the analytic form.
    source : str
        ``'aerosandbox'`` or ``'parabolic'`` — never silently ambiguous.
    """
    alpha_deg: np.ndarray
    reynolds: np.ndarray
    CL: np.ndarray
    CD: np.ndarray
    CDi: Optional[np.ndarray] = None
    CDp: Optional[np.ndarray] = None
    geometry: Optional[WingGeometry] = None
    source: str = "aerosandbox"

    # ── Queries ───────────────────────────────────────────────────────

    def C_L_max(self, Re: float = None) -> float:
        """
        Maximum lift coefficient reached on the tabulated α range.

        If :meth:`reached_stall` is False the table stopped while lift
        was still rising, and this is a lower bound rather than a stall
        value.
        """
        if Re is None:
            return float(np.max(self.CL))
        row = self._row(Re, self.CL)
        return float(np.max(row))

    def reached_stall(self, Re: float = None) -> bool:
        """
        True if the lift peak lies inside the tabulated α range.

        When it does not, the stall check downstream is comparing against
        the edge of the table rather than against stall, which errs
        towards declaring flight conditions safe that are not.
        """
        row = self._row(Re, self.CL) if Re is not None else np.max(self.CL, axis=0)
        return int(np.argmax(row)) < len(row) - 1

    def stall_alpha_deg(self, Re: float = None) -> float:
        row = self.CL if Re is None else self._row(Re, self.CL)
        row = row if row.ndim == 1 else np.max(row, axis=0)
        return float(self.alpha_deg[int(np.argmax(row))])

    def alpha_for_CL(self, C_L: float, Re: float) -> float:
        """Incidence that trims to ``C_L`` at this Reynolds number [deg]."""
        cl = self._row(Re, self.CL)
        i_peak = int(np.argmax(cl))
        cl_lin, a_lin = cl[:i_peak + 1], self.alpha_deg[:i_peak + 1]
        return float(np.interp(C_L, cl_lin, a_lin))

    def C_D(self, C_L: float, Re: float) -> float:
        """
        Drag coefficient at a trimmed lift coefficient [-].

        Only the pre-stall branch is interpolated: past the lift peak the
        relationship stops being a function of ``C_L`` at all, and the
        energy model handles stall separately by penalising it.
        """
        cl = self._row(Re, self.CL)
        cd = self._row(Re, self.CD)
        i_peak = int(np.argmax(cl))
        cl_lin, cd_lin = cl[:i_peak + 1], cd[:i_peak + 1]

        if C_L <= cl_lin[0]:
            return float(cd_lin[0])
        if C_L >= cl_lin[-1]:
            # Beyond the table: extend quadratically in C_L from the last
            # point, which is the right shape for induced drag and keeps
            # the surrogate monotone for the optimizer.
            k = (cd_lin[-1] - cd_lin[-2]) / max(
                cl_lin[-1] ** 2 - cl_lin[-2] ** 2, 1e-9)
            return float(cd_lin[-1] + k * (C_L ** 2 - cl_lin[-1] ** 2))
        return float(np.interp(C_L, cl_lin, cd_lin))

    def C_D_array(self, C_L: np.ndarray, Re: np.ndarray) -> np.ndarray:
        """Vectorised :meth:`C_D` for profiling and plotting."""
        C_L = np.atleast_1d(C_L)
        Re = np.broadcast_to(np.atleast_1d(Re), C_L.shape)
        return np.array([self.C_D(float(a), float(b)) for a, b in zip(C_L, Re)])

    def _row(self, Re: float, table: np.ndarray) -> np.ndarray:
        """Interpolate a table row at a Reynolds number, linear in log Re."""
        if table.ndim == 1:
            return table
        logs = np.log(self.reynolds)
        x = np.log(max(Re, 1.0))
        if x <= logs[0]:
            return table[0]
        if x >= logs[-1]:
            return table[-1]
        j = int(np.searchsorted(logs, x)) - 1
        j = int(np.clip(j, 0, len(logs) - 2))
        w = (x - logs[j]) / (logs[j + 1] - logs[j])
        return (1 - w) * table[j] + w * table[j + 1]

    # ── Reporting ─────────────────────────────────────────────────────

    def equivalent_parabolic(self, Re: float) -> Tuple[float, float]:
        """
        Least-squares fit of ``C_D = C_D0 + k·C_L²`` to this polar.

        Useful for comparing against whatever constants a vehicle file
        currently declares, and for handing a two-number polar to code
        that cannot take a table.
        """
        cl = self._row(Re, self.CL)
        cd = self._row(Re, self.CD)
        i_peak = int(np.argmax(cl))
        cl, cd = cl[:i_peak + 1], cd[:i_peak + 1]
        A = np.column_stack([np.ones_like(cl), cl ** 2])
        coef, *_ = np.linalg.lstsq(A, cd, rcond=None)
        return float(coef[0]), float(coef[1])

    def oswald_efficiency(self, Re: float) -> float:
        """Oswald factor implied by the fitted induced-drag slope."""
        _, k = self.equivalent_parabolic(Re)
        AR = self.geometry.AR if self.geometry else 8.0
        return float(1.0 / (np.pi * k * AR)) if k > 0 else float('nan')

    def summary(self, Re: float = None) -> str:
        Re = Re if Re is not None else float(np.median(self.reynolds))
        cd0, k = self.equivalent_parabolic(Re)
        lines = [
            f"Drag polar ({self.source})",
            f"  Re grid:            "
            + ", ".join(f"{r:.1e}" for r in self.reynolds),
            f"  alpha grid:         {self.alpha_deg[0]:.0f}° to "
            f"{self.alpha_deg[-1]:.0f}° in {len(self.alpha_deg)} steps",
            f"  at Re = {Re:.1e}:",
            f"    C_L,max           {self.C_L_max(Re):.3f} "
            f"at alpha {self.stall_alpha_deg(Re):.1f}°"
            + ("" if self.reached_stall(Re)
               else "   (lower bound — grid ends before stall)"),
            f"    equivalent C_D0   {cd0:.4f}",
            f"    equivalent k      {k:.4f}  (Oswald e = "
            f"{self.oswald_efficiency(Re):.2f})",
        ]
        return "\n".join(lines)

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload = dict(
            alpha_deg=self.alpha_deg, reynolds=self.reynolds,
            CL=self.CL, CD=self.CD,
            source=np.array(self.source),
        )
        if self.CDi is not None:
            payload["CDi"] = self.CDi
        if self.CDp is not None:
            payload["CDp"] = self.CDp
        if self.geometry is not None:
            payload["geometry"] = np.array(json.dumps(asdict(self.geometry)))
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str) -> "AeroPolar":
        d = np.load(path, allow_pickle=False)
        geom = None
        if "geometry" in d:
            geom = WingGeometry(**json.loads(str(d["geometry"])))
        return cls(
            alpha_deg=d["alpha_deg"], reynolds=d["reynolds"],
            CL=d["CL"], CD=d["CD"],
            CDi=d["CDi"] if "CDi" in d else None,
            CDp=d["CDp"] if "CDp" in d else None,
            geometry=geom,
            source=str(d["source"]) if "source" in d else "aerosandbox",
        )


# ═══════════════════════════════════════════════════════════════════════════
# ANALYTIC FALLBACK
# ═══════════════════════════════════════════════════════════════════════════

def parabolic_polar(
    geometry: WingGeometry,
    C_D0: float = 0.013,
    e_oswald: float = 0.78,
    C_L_max: float = 1.5,
    alpha_deg: Sequence[float] = None,
    reynolds_grid: Sequence[float] = None,
) -> AeroPolar:
    """
    The classical polar, wrapped in the same interface.

    Used when AeroSandbox is unavailable, so calling code never has to
    branch — but ``source`` says ``'parabolic'`` so results built on it
    are never mistaken for computed aerodynamics.
    """
    alpha_deg = np.asarray(alpha_deg if alpha_deg is not None
                           else np.linspace(-4, 14, 19), dtype=float)
    re_grid = np.asarray(reynolds_grid if reynolds_grid is not None
                         else [1e5, 3e5, 5e5, 1e6], dtype=float)

    a0 = 2 * np.pi / (1 + 2 / geometry.AR)          # 3-D lift slope [1/rad]
    CL_lin = a0 * np.radians(alpha_deg)
    CL = np.clip(CL_lin, -C_L_max, C_L_max)
    k = 1.0 / (np.pi * e_oswald * geometry.AR)
    CD = C_D0 + k * CL ** 2

    return AeroPolar(
        alpha_deg=alpha_deg,
        reynolds=re_grid,
        CL=np.tile(CL, (len(re_grid), 1)),
        CD=np.tile(CD, (len(re_grid), 1)),
        CDi=np.tile(k * CL ** 2, (len(re_grid), 1)),
        CDp=np.tile(np.full_like(CL, C_D0), (len(re_grid), 1)),
        geometry=geometry,
        source="parabolic",
    )


# ═══════════════════════════════════════════════════════════════════════════
# BUILDING THE POLAR WITH AEROSANDBOX
# ═══════════════════════════════════════════════════════════════════════════

def aerosandbox_available() -> bool:
    try:
        import aerosandbox  # noqa: F401
        return True
    except Exception:
        return False


def build_polar(
    geometry: WingGeometry = None,
    alpha_deg: Sequence[float] = None,
    reynolds_grid: Sequence[float] = None,
    altitude_m: float = 2800.0,
    verbose: bool = True,
) -> AeroPolar:
    """
    Compute a drag polar with AeroSandbox's nonlinear lifting line.

    One solve per grid point at roughly 3.5 s each, so a 13 × 3 grid is
    about two and a half minutes. Call this once and cache the result —
    :func:`get_polar` does that for you.

    Parameters
    ----------
    geometry : WingGeometry
        Wing to analyse. Defaults to the baseline delivery wing.
    alpha_deg : sequence
        Incidence grid. Should extend past stall so ``C_L,max`` is
        captured rather than extrapolated.
    reynolds_grid : sequence
        Chord Reynolds numbers to cover. Should bracket the aircraft's
        actual speed and altitude range.
    altitude_m : float
        Altitude used to set air density. Reynolds number is imposed
        separately by adjusting the velocity, so this only affects the
        (negligible) Mach number.
    verbose : bool
        Report progress; the solve is slow enough to warrant it.

    Returns
    -------
    AeroPolar with ``source='aerosandbox'``, or the parabolic fallback if
    AeroSandbox is missing or every solve fails.
    """
    geometry = geometry or WingGeometry()
    # Out to 20°, not 14°: a grid that stops while C_L is still rising
    # never sees the lift peak, so C_L,max comes back as "the last point
    # I looked at" — which is the number the stall check depends on.
    alpha_deg = np.asarray(alpha_deg if alpha_deg is not None
                           else np.arange(-4.0, 20.1, 1.5), dtype=float)
    re_grid = np.asarray(reynolds_grid if reynolds_grid is not None
                         else [1.5e5, 3.0e5, 5.0e5], dtype=float)

    if not aerosandbox_available():
        warnings.warn(
            "AeroSandbox is not installed; falling back to the analytic "
            "parabolic polar. Install aerosandbox for computed aerodynamics.",
            stacklevel=2,
        )
        return parabolic_polar(geometry, alpha_deg=alpha_deg,
                               reynolds_grid=re_grid)

    import aerosandbox as asb
    from aerosandbox.aerodynamics.aero_3D.nonlinear_lifting_line import (
        NonlinearLiftingLine,
    )

    af = asb.Airfoil(geometry.airfoil)
    b, c_r, c_t = geometry.span, geometry.chord_root, geometry.chord_tip
    wing = asb.Wing(name="wing", symmetric=True, xsecs=[
        asb.WingXSec(xyz_le=[0, 0, 0], chord=c_r, twist=0.0, airfoil=af),
        asb.WingXSec(xyz_le=[0, b / 2, 0], chord=c_t,
                     twist=geometry.twist_tip_deg, airfoil=af),
    ])
    plane = asb.Airplane(name="raptor-wing", wings=[wing],
                         s_ref=geometry.S_ref, c_ref=geometry.chord_mean,
                         b_ref=b)
    atm = asb.Atmosphere(altitude=altitude_m)
    rho = float(atm.density())
    nu = NU_AIR * (1.225 / rho)

    CL = np.full((len(re_grid), len(alpha_deg)), np.nan)
    CD = np.full_like(CL, np.nan)
    CDi = np.full_like(CL, np.nan)
    CDp = np.full_like(CL, np.nan)

    n_fail = 0
    for i, Re in enumerate(re_grid):
        # Impose Reynolds number through velocity at the chosen density.
        V = float(Re * nu / geometry.chord_mean)
        for j, a in enumerate(alpha_deg):
            try:
                op = asb.OperatingPoint(atmosphere=atm, velocity=V, alpha=float(a))
                r = NonlinearLiftingLine(
                    airplane=plane, op_point=op,
                    spanwise_resolution=geometry.n_spanwise,
                    verbose=False,
                ).run()
                CL[i, j] = float(r["CL"])
                CD[i, j] = float(r["CD"])
                CDi[i, j] = float(r["CDi"])
                CDp[i, j] = float(r["CDp"])
            except Exception as exc:      # a single point failing is normal
                n_fail += 1
                if verbose:
                    print(f"    Re={Re:.1e} alpha={a:+.1f}: {type(exc).__name__}")
        if verbose:
            ok = int(np.sum(~np.isnan(CL[i])))
            print(f"  Re = {Re:.1e} (V = {V:.1f} m/s): "
                  f"{ok}/{len(alpha_deg)} points")

    if np.all(np.isnan(CL)):
        warnings.warn(
            "Every lifting-line solve failed; falling back to the parabolic "
            "polar. Check the airfoil name and wing geometry.",
            stacklevel=2,
        )
        return parabolic_polar(geometry, alpha_deg=alpha_deg,
                               reynolds_grid=re_grid)

    # Fill isolated failures by interpolating along alpha, so one bad
    # point does not put a hole in the middle of the surrogate.
    for table in (CL, CD, CDi, CDp):
        for i in range(table.shape[0]):
            row = table[i]
            bad = np.isnan(row)
            if bad.any() and (~bad).sum() >= 2:
                row[bad] = np.interp(alpha_deg[bad], alpha_deg[~bad], row[~bad])

    if verbose and n_fail:
        print(f"  {n_fail} point(s) interpolated after solver failures")

    return AeroPolar(alpha_deg=alpha_deg, reynolds=re_grid,
                     CL=CL, CD=CD, CDi=CDi, CDp=CDp,
                     geometry=geometry, source="aerosandbox")


# ═══════════════════════════════════════════════════════════════════════════
# CACHE
# ═══════════════════════════════════════════════════════════════════════════

def default_cache_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "..", "data", "polars")


def _cache_key(geometry: WingGeometry, build_kwargs: dict) -> str:
    """
    Hash covering both the wing and the grid it was sampled on.

    Keying on geometry alone would let a polar computed over a narrow
    incidence range be served for a request that needs it out to stall —
    the arrays would load fine and quietly answer the wrong question.
    """
    payload = {"geometry": asdict(geometry)}
    for k in ("alpha_deg", "reynolds_grid", "altitude_m"):
        v = build_kwargs.get(k)
        if v is not None:
            payload[k] = [float(x) for x in np.atleast_1d(v)]
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def get_polar(
    geometry: WingGeometry = None,
    cache_dir: str = None,
    rebuild: bool = False,
    verbose: bool = True,
    **build_kwargs,
) -> AeroPolar:
    """
    Load a cached polar for this geometry and grid, computing it if absent.

    The cache key covers the wing *and* the sampling grid, so changing
    either invalidates it automatically — a stale polar silently
    describing a different aircraft is exactly the failure this avoids.
    """
    geometry = geometry or WingGeometry()
    cache_dir = cache_dir or default_cache_dir()
    path = os.path.join(cache_dir, f"polar_{_cache_key(geometry, build_kwargs)}.npz")

    if not rebuild and os.path.exists(path):
        try:
            polar = AeroPolar.load(path)
            if verbose:
                print(f"Loaded cached polar: {path}")
            return polar
        except Exception as exc:
            warnings.warn(f"Cached polar unreadable ({exc}); rebuilding.",
                          stacklevel=2)

    if verbose:
        print(f"Building drag polar for {geometry.airfoil}, "
              f"S={geometry.S_ref} m², AR={geometry.AR} "
              f"(this takes a few minutes)...")
    polar = build_polar(geometry, verbose=verbose, **build_kwargs)
    try:
        polar.save(path)
        if verbose:
            print(f"Cached to {path}")
    except Exception as exc:
        warnings.warn(f"Could not cache polar: {exc}", stacklevel=2)
    return polar


# ═══════════════════════════════════════════════════════════════════════════
# ENVELOPE CROSS-CHECK
# ═══════════════════════════════════════════════════════════════════════════

def check_flight_envelope(uav, ac, altitude: float = 2800.0,
                          safety_margin: float = None) -> list:
    """
    Check the configured speed envelope against the aircraft's real lift.

    ``UAVConfig`` declares the speeds the planner is allowed to fly and
    ``AircraftEnergyParams`` declares the aerodynamics; nothing forces
    them to agree, and they are edited in different files. When the
    declared ``C_L_max`` is optimistic, the resulting stall speed is too
    low, and a cruise speed that looks comfortably above it is not.

    The error runs in the unsafe direction, which is why this is worth a
    check rather than a comment: over-estimating drag makes a mission
    look expensive, but over-estimating lift makes a stall look like
    cruise.

    Parameters
    ----------
    uav : UAVConfig
        Speed envelope and stall safety margin.
    ac : AircraftEnergyParams
        Aerodynamics, with a polar attached if there is one.
    altitude : float
        Altitude to evaluate at [m AMSL]. Density falls with altitude, so
        stall speed rises — Quito is the demanding case, not sea level.
    safety_margin : float
        Multiple of stall speed required. Defaults to the UAV's own
        ``stall_safety_margin`` (1.3, the FAR 23 convention).

    Returns
    -------
    List of human-readable problems; empty means the envelope is sound.
    """
    out = []
    margin = (safety_margin if safety_margin is not None
              else getattr(uav, 'stall_safety_margin', 1.3))
    rho = isa_density(altitude)

    polar = ac.aero_polar
    if polar is not None:
        Re = reynolds(uav.fw_cruise_airspeed, ac.wing_chord, altitude)
        cl_max = polar.C_L_max(Re)
        source = f"computed polar ({polar.source})"
        if not polar.reached_stall(Re):
            out.append(
                "The drag polar's incidence grid ends before the lift peak, "
                "so C_L,max is a lower bound. Rebuild it over a wider alpha "
                "range before trusting the stall speeds below."
            )
    else:
        cl_max = ac.C_L_max
        source = "declared C_L_max (no aero model attached)"

    v_stall = float(np.sqrt(2 * ac.W / (rho * ac.S_ref * cl_max)))
    v_safe = margin * v_stall

    if ac.aero_polar is not None and abs(cl_max - ac.C_L_max) > 0.05:
        out.append(
            f"C_L_max: vehicle file says {ac.C_L_max:.3f}, the {source} gives "
            f"{cl_max:.3f}. Stall speed at {altitude:.0f} m is therefore "
            f"{v_stall:.1f} m/s, not "
            f"{np.sqrt(2*ac.W/(rho*ac.S_ref*ac.C_L_max)):.1f} m/s."
        )

    if uav.fw_min_airspeed < v_stall:
        out.append(
            f"fw_min_airspeed {uav.fw_min_airspeed:.1f} m/s is BELOW the stall "
            f"speed {v_stall:.1f} m/s at {altitude:.0f} m — the optimizer is "
            f"free to select speeds the aircraft cannot fly."
        )
    elif uav.fw_min_airspeed < v_safe:
        out.append(
            f"fw_min_airspeed {uav.fw_min_airspeed:.1f} m/s is below the "
            f"{margin:.2f}x stall margin ({v_safe:.1f} m/s) at "
            f"{altitude:.0f} m."
        )

    if uav.fw_cruise_airspeed < v_safe:
        out.append(
            f"fw_cruise_airspeed {uav.fw_cruise_airspeed:.1f} m/s is below the "
            f"{margin:.2f}x stall margin ({v_safe:.1f} m/s) at "
            f"{altitude:.0f} m. Raise the cruise speed, enlarge the wing, or "
            f"accept a smaller margin explicitly."
        )

    if uav.fw_max_airspeed <= v_safe:
        out.append(
            f"fw_max_airspeed {uav.fw_max_airspeed:.1f} m/s leaves no usable "
            f"speed band above the stall margin at {altitude:.0f} m."
        )

    return out
