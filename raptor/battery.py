"""
Battery Characterisation — Measured Numbers Instead of Assumed Ones
====================================================================

:mod:`raptor.energy` models the pack as an open-circuit voltage curve
behind an internal resistance. Both of those are properties of a
particular pack at a particular age and temperature, and the defaults in
the vehicle files are textbook values for "a 6S LiPo" — good enough to
plan with, not good enough to publish an endurance figure against.

This module replaces them with a fit to a real discharge log, and — just
as important — records which numbers came from measurement and which are
still assumptions, so a result never quietly implies more than it knows.

How the fit works
-----------------
A discharge log gives time, terminal voltage and current. State of charge
follows from coulomb-counting the current. The pack model says

    V_terminal(t) = V_oc(SOC(t)) − I(t)·R

Parameterise ``V_oc`` as piecewise-linear on a SOC grid and the whole
thing is **linear** in the unknowns — the grid node voltages and ``R``
together — so one least-squares solve recovers both, with no iteration
and no initial guess. Current variation is what separates them: a log
flown at constant current cannot distinguish a sagging cell from a
resistive one, and :meth:`DischargeLog.identifiability` says so rather
than returning a confident number from an ill-posed fit.

Author: Victor (LUAS-EPN / KU Leuven)
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# PUBLISHED CELL CURVES
# ═══════════════════════════════════════════════════════════════════════════
#
# Until a pack is on a bench, the open-circuit-voltage curve has to come
# from somewhere. "A generic LiPo table" is somewhere, but not a place
# that can be cited, and it is the wrong shape for the Li-ion packs that
# long-endurance VTOLs actually fly: a LiPo falls steadily from 4.2 V,
# while a high-nickel Li-ion holds a long plateau and then drops sharply
# in the last fifth. That plateau is what decides how much of the pack
# sits above the cut-off, which is to say how much of it is real.
#
# So the curves here are published ones, named and cited, with the
# rescaling to a particular cell's endpoints stated rather than folded
# in silently.


def chen_rincon_mora_ocv(soc, v_min: float = 3.0, v_max: float = 4.2):
    """
    Per-cell open-circuit voltage from Chen & Rincón-Mora (2006), Eq. (2).

    .. math::
        V_{OC}(s) = -1.031 e^{-35 s} + 3.685 + 0.2156 s
                    - 0.1178 s^2 + 0.3201 s^3

    Characterised on a TCL PL-383562 850 mAh polymer Li-ion cell charged
    to 4.1 V and discharged to a 3.0 V cut-off, and reported to predict
    runtime within 0.12 % and terminal voltage within 21 mV across
    discharge currents from 80 mA to 640 mA.

    M. Chen and G. A. Rincón-Mora, "Accurate Electrical Battery Model
    Capable of Predicting Runtime and I–V Performance", *IEEE
    Transactions on Energy Conversion* **21**(2), 504–511, 2006.

    Parameters
    ----------
    soc : array_like
        State of charge in [0, 1].
    v_min, v_max : float
        Endpoints of the cell being modelled. The published curve spans
        2.654–4.103 V, which is its own cell charged to 4.1 V; a modern
        high-nickel 21700 charged to 4.2 V sits about 0.1 V above it at
        the top. The curve is affinely rescaled onto ``[v_min, v_max]``.

        **This rescaling is an assumption**, and the reason the function
        takes the endpoints explicitly rather than hiding them: the
        published *shape* is transferable between Li-ion cells, the
        absolute voltages are not.
    """
    s = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
    raw = (-1.031 * np.exp(-35.0 * s) + 3.685 + 0.2156 * s
           - 0.1178 * s ** 2 + 0.3201 * s ** 3)
    raw_min = float(-1.031 + 3.685)                       # s = 0
    raw_max = float(3.685 + 0.2156 - 0.1178 + 0.3201)     # s = 1
    span = raw_max - raw_min
    return v_min + (raw - raw_min) * (v_max - v_min) / span


def chen_rincon_mora_resistance_factor(soc):
    """
    Internal resistance against SOC, normalised to its mid-SOC value.

    From Eq. (3) of the same paper,
    ``R_series(s) = 0.1562 e^{-24.37 s} + 0.07446`` ohm, divided by its
    value at 50 % so it can multiply any cell's nominal resistance.

    Resistance is nearly flat above 20 % SOC and climbs steeply below
    it — roughly threefold by the time the pack is empty. A constant
    resistance therefore *understates* voltage sag exactly where a
    mission has least margin, which is the end of it.
    """
    s = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
    r = 0.1562 * np.exp(-24.37 * s) + 0.07446
    r_mid = 0.1562 * np.exp(-24.37 * 0.5) + 0.07446
    return r / r_mid


#: The LiPo curve the package shipped before chemistries were named.
#: Kept because it is what a LiPo pack looks like, and labelled for what
#: it is: a plausible shape with no measurement behind it.
_LIPO_SOC = np.array([0.00, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.95, 1.00])
_LIPO_CELL_V = np.array([3.30, 3.45, 3.55, 3.65, 3.75, 3.85, 4.00, 4.15, 4.20])

#: High-nickel Li-ion (NMC) 21700 — **measured**, not assumed.
#:
#: Capacity-weighted open-circuit voltage of a Samsung INR21700-30T,
#: extracted from a C/30 charge-discharge characterisation and averaged
#: over the two branches to remove hysteresis (which is 131 mV at its
#: worst, so the branch matters and the average is the equilibrium
#: curve the pack model wants — the load drop is applied separately
#: through the internal resistance).
#:
#: P. Pillai, S. Sundaresan, P. Kumar, K. Pattipati and B. Balasingam,
#: "Open-Circuit Voltage Models for Battery Management Systems: A
#: Review", *Energies* 15(18), 2022. Dataset: "OCV test Voltage and
#: Current Data — Samsung-30T INR21700 Battery", Mendeley Data,
#: doi:10.17632/fywnpsjfpc.1, CC BY 4.0.
#:
#: Averaged over the three cells in the dataset whose charge and
#: discharge capacities balance. The fourth returns twice as much charge
#: as it delivers, so its two branches are not halves of one cycle;
#: averaging it in pulls the curve up by 100 mV, and the extraction
#: script excludes it and says why. The three that remain agree to
#: **1.3 mV in mean OCV**, and recovered 3.020–3.030 Ah against the
#: cell's 3.000 Ah nominal.
#:
#: Re-extract with ``python -m scripts.extract_ocv --all --download``.
#:
#: The table this replaced was an NMC-typical *shape* anchored on
#: datasheet endpoints. It was systematically 0.10-0.16 V low between
#: 50 % and 80 % state of charge — the part of the curve a mission
#: actually spends its time on.
_NMC_SOC = np.array([0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                     0.60, 0.70, 0.80, 0.90, 0.95, 1.00])
_NMC_CELL_V = np.array([2.5962, 3.3277, 3.4030, 3.5058, 3.5929, 3.6546,
                        3.7368, 3.8418, 3.9258, 4.0342, 4.0834, 4.1056,
                        4.1932])

#: The measured cell's own particulars, for the datasheet check.
MEASURED_CELL = dict(
    name="Samsung INR21700-30T",
    capacity_ah=3.000,
    measured_capacity_ah=3.030,
    nominal_v=3.6,
    v_max=4.2,
    v_cutoff=2.5,
    datasheet_current_a=0.6,
    dc_resistance_ohm=0.020,
)

#: Per-cell facts that are *not* in the curve: the nominal voltage the
#: cell is sold under, and the cut-off below which it should not be
#: taken. The endpoints are deliberately absent — they are properties of
#: the curve, and declaring them here as well would be a second source
#: of truth for the same two numbers. :func:`cell_voltage_window` reads
#: them off the curve instead.
CELL_CHEMISTRY = {
    "lipo": dict(
        v_nominal=3.70, v_cutoff=3.40,
        source="generic LiPo table — plausible shape, no measurement "
               "behind it",
    ),
    "li-ion": dict(
        v_nominal=3.60, v_cutoff=3.00,
        source="measured — Samsung INR21700-30T at C/30, "
               "doi:10.17632/fywnpsjfpc.1 (CC BY 4.0)",
    ),
    "li-ion-semisolid": dict(
        v_nominal=3.65, v_cutoff=3.00, charge_shift=0.05,
        source="the measured NMC curve shifted 0.05 V at the top for the "
               "higher charge voltage semi-solid packs specify — the "
               "shift is assumed, the shape is measured",
    ),
}


def cell_ocv_curve(chemistry: str = "li-ion"):
    """
    Tabulated ``(soc, cell volts)`` for a named chemistry.

    Returns the arrays the pack model interpolates. Raises rather than
    falling back, because a silent default here is a pack whose voltage
    is the wrong shape and whose currents are therefore wrong in a way
    nothing downstream can detect.
    """
    if chemistry not in CELL_CHEMISTRY:
        raise KeyError(
            f"unknown cell chemistry {chemistry!r}; known: "
            f"{sorted(CELL_CHEMISTRY)}"
        )
    if chemistry == "lipo":
        return _LIPO_SOC.copy(), _LIPO_CELL_V.copy()
    spec = CELL_CHEMISTRY[chemistry]
    v = _NMC_CELL_V.copy()
    # The semi-solid variant differs by charge voltage, so the curve is
    # lifted at the top and pinned at the bottom.
    shift = spec.get("charge_shift", 0.0)
    if shift:
        v = v + shift * _NMC_SOC ** 2
    return _NMC_SOC.copy(), v


def cell_voltage_window(chemistry: str = "li-ion") -> Tuple[float, float]:
    """
    Resting voltage of the cell empty and fully charged [V].

    Read off the curve rather than declared beside it. The measured
    NMC cell relaxes to 2.596 V after being discharged to a 2.5 V
    cut-off — the cut-off is a *load* voltage and the empty-cell open
    circuit sits above it, so the two are different numbers and
    declaring both invites them to disagree.
    """
    _, v = cell_ocv_curve(chemistry)
    return float(v[0]), float(v[-1])


def mean_cell_voltage(chemistry: str = "li-ion",
                      above_cutoff: bool = False) -> float:
    """
    Capacity-weighted mean open-circuit voltage of a cell [V].

    This is the number that has to match a datasheet: a cell's quoted
    watt-hours divided by its quoted amp-hours is its mean voltage under
    discharge, and a curve that misses it is delivering the wrong energy
    however plausible its shape.
    """
    soc, v = cell_ocv_curve(chemistry)
    if not above_cutoff:
        return float(np.trapezoid(v, soc))
    cut = CELL_CHEMISTRY[chemistry]["v_cutoff"]
    s_cut = float(np.interp(cut, v, soc))
    keep = soc >= s_cut
    ss = np.concatenate([[s_cut], soc[keep]])
    vv = np.concatenate([[cut], v[keep]])
    return float(np.trapezoid(vv, ss) / max(1.0 - s_cut, 1e-9))


def reachable_fraction(chemistry: str = "li-ion") -> float:
    """
    Fraction of the cell's charge that sits above its cut-off voltage.

    Part — but only part — of why a pack's usable energy is less than
    its nameplate. The rest is pack overhead, cell mismatch and the
    reserve an operator keeps.
    """
    soc, v = cell_ocv_curve(chemistry)
    cut = CELL_CHEMISTRY[chemistry]["v_cutoff"]
    return float(1.0 - np.interp(cut, v, soc))


def check_pack_against_datasheet(chemistry: str, cell_capacity_ah: float,
                                 datasheet_wh: float,
                                 current_a: float = None,
                                 resistance_ohm: float = None,
                                 tolerance: float = 0.05) -> dict:
    """
    Does this OCV curve deliver the energy the cell is sold as holding?

    A datasheet states capacity in amp-hours *and* energy in watt-hours,
    and the ratio is a mean voltage. But it is a mean **terminal**
    voltage, measured while the cell was delivering the datasheet's
    stated current — and the curve here is the **open-circuit** voltage,
    which is higher by the drop across the internal resistance. Comparing
    them directly asks the curve to be wrong by exactly one ``I·R``.

    That is not hypothetical. The version of this check that compared
    OCV against the datasheet ratio passed a curve that was 0.1 V too
    low across the middle of its range, because the error happened to be
    about the size of the drop the check was ignoring. **A validation
    that omits a term can be satisfied by an error of the same size**,
    and will then certify the thing it was meant to catch.

    Parameters
    ----------
    current_a, resistance_ohm : float, optional
        The datasheet's discharge current and the cell's internal
        resistance. Given both, the comparison is made at the terminals.
        Omit them and it is made open-circuit, with the result flagged.

    Returns
    -------
    dict with the modelled and datasheet mean voltages, the relative
    error, and whether it is inside *tolerance*.
    """
    v_oc = mean_cell_voltage(chemistry)
    v_datasheet = datasheet_wh / cell_capacity_ah

    if current_a and resistance_ohm:
        v_model = v_oc - current_a * resistance_ohm
        basis = (f"terminal at {current_a:.2f} A through "
                 f"{1000*resistance_ohm:.0f} mOhm")
    else:
        v_model = v_oc
        basis = "open-circuit (no discharge current given — see the docstring)"

    err = (v_model - v_datasheet) / v_datasheet
    return {
        "chemistry": chemistry,
        "modelled_mean_v": v_model,
        "open_circuit_mean_v": v_oc,
        "datasheet_mean_v": v_datasheet,
        "basis": basis,
        "relative_error": err,
        "within_tolerance": bool(abs(err) <= tolerance),
        "modelled_wh": v_model * cell_capacity_ah,
        "datasheet_wh": datasheet_wh,
        "note": (
            f"curve gives {v_model:.3f} V {basis}, datasheet "
            f"{v_datasheet:.3f} V ({100*err:+.1f} %)"
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# DISCHARGE LOG
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DischargeLog:
    """
    A measured discharge: time, terminal voltage, current.

    Attributes
    ----------
    time_s : ndarray
        Seconds from the start of discharge.
    voltage_v : ndarray
        Pack terminal voltage [V].
    current_a : ndarray
        Pack current [A], positive when discharging.
    cells_series : int
        Cells in series, so per-cell voltages can be reported.
    capacity_ah : float or None
        Nameplate capacity, for comparison with what was delivered.
    temperature_c : float or None
        Cell temperature during the test. Capacity and resistance both
        move with it, so a fit without one is only valid near whatever
        it happened to be.
    source : str
        Where the data came from. Synthetic logs say so.
    """
    time_s: np.ndarray
    voltage_v: np.ndarray
    current_a: np.ndarray
    cells_series: int = 6
    capacity_ah: Optional[float] = None
    temperature_c: Optional[float] = None
    source: str = "unknown"

    def __post_init__(self):
        self.time_s = np.asarray(self.time_s, dtype=float)
        self.voltage_v = np.asarray(self.voltage_v, dtype=float)
        self.current_a = np.asarray(self.current_a, dtype=float)
        n = len(self.time_s)
        if not (len(self.voltage_v) == len(self.current_a) == n):
            raise ValueError("time, voltage and current must be the same length")
        if n < 10:
            raise ValueError("a discharge log needs at least 10 samples")

    @property
    def charge_delivered_ah(self) -> float:
        """Ampere-hours actually taken out of the pack."""
        return float(np.trapezoid(self.current_a, self.time_s) / 3600.0)

    @property
    def energy_delivered_wh(self) -> float:
        """Watt-hours delivered to the load."""
        return float(np.trapezoid(self.current_a * self.voltage_v,
                                  self.time_s) / 3600.0)

    def soc(self, capacity_ah: float = None) -> np.ndarray:
        """
        State of charge through the discharge, by coulomb counting.

        Referenced to the charge the pack actually delivered rather than
        its nameplate, so a tired pack reads 0 % when it is empty rather
        than stopping at 20 %.
        """
        cap = capacity_ah or self.capacity_ah or self.charge_delivered_ah
        ah = np.concatenate(([0.0], np.cumsum(
            0.5 * (self.current_a[1:] + self.current_a[:-1])
            * np.diff(self.time_s) / 3600.0)))
        return np.clip(1.0 - ah / max(cap, 1e-9), 0.0, 1.0)

    def identifiability(self) -> Dict[str, float]:
        """
        Whether this log can separate open-circuit voltage from resistance.

        The two are only distinguishable where the current *changes*: at
        constant current, any resistance can be absorbed into a shifted
        voltage curve. Returns the current spread and a verdict.
        """
        i = self.current_a
        spread = float(np.max(i) - np.min(i))
        rel = spread / max(float(np.mean(i)), 1e-9)
        return {
            "current_min_a": float(np.min(i)),
            "current_max_a": float(np.max(i)),
            "current_spread_a": spread,
            "relative_spread": rel,
            # Below about 20 % variation the split is poorly conditioned;
            # the fit will still return a number, and it should not be
            # trusted.
            "separable": rel >= 0.20,
        }

    @classmethod
    def from_csv(cls, path: str, time_col: str = "time_s",
                 voltage_col: str = "voltage_v", current_col: str = "current_a",
                 **kwargs) -> "DischargeLog":
        """
        Read a log from CSV.

        Expects a header row. Column names are configurable because every
        logger names them differently.
        """
        t, v, i = [], [], []
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            missing = {time_col, voltage_col, current_col} - set(
                reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"{path}: missing column(s) {sorted(missing)}; found "
                    f"{reader.fieldnames}"
                )
            for row in reader:
                t.append(float(row[time_col]))
                v.append(float(row[voltage_col]))
                i.append(float(row[current_col]))
        kwargs.setdefault("source", os.path.basename(path))
        return cls(np.array(t), np.array(v), np.array(i), **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# FIT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BatteryFit:
    """What a discharge log says about a pack."""
    ocv_soc: np.ndarray
    ocv_cell_v: np.ndarray
    resistance_ohm: float
    capacity_ah: float
    energy_wh: float
    cells_series: int
    rms_error_v: float
    max_error_v: float
    separable: bool
    n_samples: int
    source: str = ""
    temperature_c: Optional[float] = None
    #: SOC range the log actually covered. Outside it the curve is an
    #: extrapolation, and a discharge that stopped at 20 % says nothing
    #: about the last fifth of the pack.
    soc_range: Tuple[float, float] = (0.0, 1.0)
    #: True when the resistance was fitted; False when it was held at a
    #: prior because the log could not identify it.
    resistance_fitted: bool = True

    @property
    def resistance_per_cell_ohm(self) -> float:
        return self.resistance_ohm / max(self.cells_series, 1)

    def summary(self) -> str:
        lines = [
            f"Battery fit from {self.source or 'log'} "
            f"({self.n_samples} samples)",
            f"  capacity delivered   {self.capacity_ah:6.2f} Ah",
            f"  energy delivered     {self.energy_wh:6.1f} Wh",
            f"  internal resistance  {self.resistance_ohm*1000:6.1f} mΩ "
            f"({self.resistance_per_cell_ohm*1000:.1f} mΩ/cell)",
            f"  fit residual         {self.rms_error_v*1000:6.1f} mV rms, "
            f"{self.max_error_v*1000:.0f} mV max",
        ]
        if self.temperature_c is not None:
            lines.append(f"  test temperature     {self.temperature_c:6.1f} °C")
        if not self.separable:
            lines.append(
                "  ! current barely varied in this log, so open-circuit "
                "voltage and resistance are not separately identifiable. "
                "The resistance above is not trustworthy — re-run the test "
                "with a stepped load."
            )
        lo, hi = self.soc_range
        lines.append(f"  SOC covered          {lo*100:5.0f} – {hi*100:.0f} %")
        if lo > 0.02 or hi < 0.98:
            lines.append(
                f"  ! the log did not reach the ends of the pack; outside "
                f"{lo*100:.0f}–{hi*100:.0f} % the curve is extrapolated."
            )
        lines.append("  OCV per cell: " + ", ".join(
            f"{s:.2f}→{v:.2f}V" for s, v in
            zip(self.ocv_soc[::2], self.ocv_cell_v[::2])))
        return "\n".join(lines)


def fit_battery(log: DischargeLog, n_nodes: int = 9,
                capacity_ah: float = None,
                resistance_prior_ohm: float = None) -> BatteryFit:
    """
    Recover the OCV curve and internal resistance from a discharge log.

    Solves ``V_terminal = V_oc(SOC) − I·R`` as one linear least-squares
    problem: ``V_oc`` is piecewise-linear on a SOC grid, so the model is
    linear in the node voltages and in ``R`` together.

    Parameters
    ----------
    log : DischargeLog
    n_nodes : int
        Nodes in the SOC grid. More resolves the knee better and fits
        noise sooner; nine is a reasonable compromise for a full
        discharge.
    capacity_ah : float
        Capacity to reference SOC against. Defaults to the charge the
        log actually delivered.
    resistance_prior_ohm : float
        Resistance to hold fixed when the log cannot identify it. Without
        one, a constant-current log is fitted with ``R = 0``.

    Returns
    -------
    BatteryFit
    """
    cap = capacity_ah or log.capacity_ah or log.charge_delivered_ah
    soc = log.soc(cap)
    ident = log.identifiability()
    separable = bool(ident["separable"])

    # The grid spans the SOC the log actually visited, not [0, 1]. Nodes
    # outside the measured range are unconstrained by any observation,
    # and least squares would leave them at whatever minimises the
    # coefficient norm — which is zero, i.e. a pack that reads 0 V when
    # full.
    s_lo, s_hi = float(np.min(soc)), float(np.max(soc))
    if s_hi - s_lo < 0.05:
        raise ValueError(
            f"the log spans only {100*(s_hi-s_lo):.1f} % of state of charge; "
            f"a fit needs a substantial discharge."
        )
    grid = np.linspace(s_lo, s_hi, n_nodes)

    n = len(soc)
    idx = np.clip(np.searchsorted(grid, soc) - 1, 0, n_nodes - 2)
    lo, hi = grid[idx], grid[idx + 1]
    w = np.clip((soc - lo) / np.maximum(hi - lo, 1e-12), 0.0, 1.0)

    W = np.zeros((n, n_nodes))
    W[np.arange(n), idx] = 1.0 - w
    W[np.arange(n), idx + 1] = w

    if separable:
        # Fit node voltages and R together — linear in both.
        A = np.hstack([W, -log.current_a[:, None]])
        coef, *_ = np.linalg.lstsq(A, log.voltage_v, rcond=None)
        ocv_pack, R = coef[:n_nodes], float(coef[-1])
        R = max(R, 0.0)
        resistance_fitted = True
    else:
        # Rank-deficient: with constant current the ``-I`` column is a
        # multiple of the interpolation weights' row sum, so any
        # resistance can be traded against a shifted voltage curve. Least
        # squares will still return a number — the minimum-norm one,
        # which quietly assigns most of the offset to R because its
        # column is large. Hold R at the prior and fit the curve alone.
        R = float(resistance_prior_ohm or 0.0)
        ocv_pack, *_ = np.linalg.lstsq(
            W, log.voltage_v + R * log.current_a, rcond=None)
        resistance_fitted = False

    # Monotone: open-circuit voltage cannot rise as charge is removed,
    # and interpolating a non-monotone curve downstream is nonsense.
    ocv_pack = np.maximum.accumulate(ocv_pack)

    predicted = W @ ocv_pack - R * log.current_a
    residual = log.voltage_v - predicted

    return BatteryFit(
        ocv_soc=grid,
        ocv_cell_v=ocv_pack / max(log.cells_series, 1),
        resistance_ohm=R,
        capacity_ah=float(cap),
        energy_wh=log.energy_delivered_wh,
        cells_series=log.cells_series,
        rms_error_v=float(np.sqrt(np.mean(residual ** 2))),
        max_error_v=float(np.max(np.abs(residual))),
        separable=separable,
        n_samples=n,
        source=log.source,
        temperature_c=log.temperature_c,
        soc_range=(s_lo, s_hi),
        resistance_fitted=resistance_fitted,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PROVENANCE
# ═══════════════════════════════════════════════════════════════════════════

def battery_provenance(ac) -> str:
    """
    Which of a vehicle's battery numbers are measured and which assumed.

    An endurance figure computed from a fitted pack and one computed from
    textbook defaults look identical on the page. This is what separates
    them.
    """
    fitted = getattr(ac, "_battery_fit", None)
    lines = ["Battery model provenance"]

    if fitted is not None:
        lines += [
            f"  OCV curve            MEASURED — {fitted.source}",
            (f"  internal resistance  MEASURED — "
             f"{fitted.resistance_ohm*1000:.1f} mΩ" if fitted.resistance_fitted
             else f"  internal resistance  NOT IDENTIFIED by this log — held "
                  f"at {fitted.resistance_ohm*1000:.1f} mΩ"),
            f"  capacity             MEASURED — {fitted.capacity_ah:.2f} Ah "
            f"({fitted.energy_wh:.0f} Wh delivered)",
        ]
        if fitted.temperature_c is not None:
            lines.append(f"  valid near           {fitted.temperature_c:.0f} °C")
    else:
        lines += [
            "  OCV curve            ASSUMED — generic LiPo table",
            f"  internal resistance  ASSUMED — {ac.battery_resistance*1000:.1f} mΩ "
            f"from {ac.cell_internal_resistance_ohm*1000:.0f} mΩ/cell × "
            f"{ac.battery_cells_series}S / "
            f"{ac.battery_strings_parallel:.1f}P",
            f"  capacity             ASSUMED — {ac.battery_capacity_ah:.1f} Ah "
            f"from {ac.battery_mass:.1f} kg × "
            f"{ac.specific_energy:.0f} Wh/kg",
        ]

    lines += [
        f"  usable fraction      ASSUMED — {ac.battery_usable_fraction:.2f}",
        f"  temperature effects  NOT MODELLED",
        f"  ageing               NOT MODELLED",
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# SYNTHETIC LOG — for testing the machinery without hardware
# ═══════════════════════════════════════════════════════════════════════════

def synthetic_log(capacity_ah: float = 30.0, cells_series: int = 12,
                  resistance_ohm: float = 0.0035,
                  chemistry: str = "li-ion",
                  current_profile: str = "stepped",
                  duration_s: float = 900.0, n: int = 400,
                  noise_v: float = 0.01, seed: int = 0) -> DischargeLog:
    """
    Generate a discharge log from a known pack, for testing the fit.

    Clearly labelled synthetic: it exists so the fitting machinery can be
    exercised and its recovery of known parameters verified, not to stand
    in for a measurement.

    Parameters
    ----------
    current_profile : {'stepped', 'constant'}
        ``stepped`` varies the load so OCV and resistance are separable;
        ``constant`` does not, and exists to check that the fit *says so*
        instead of returning a confident wrong number.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, duration_s, n)

    if current_profile == "stepped":
        i = np.where((t % 240.0) < 120.0, 120.0, 60.0)
    elif current_profile == "constant":
        i = np.full_like(t, 90.0)
    else:
        raise ValueError("current_profile must be 'stepped' or 'constant'")

    ah = np.concatenate(([0.0], np.cumsum(
        0.5 * (i[1:] + i[:-1]) * np.diff(t) / 3600.0)))
    soc = np.clip(1.0 - ah / capacity_ah, 0.0, 1.0)

    grid, curve = cell_ocv_curve(chemistry)
    cell = np.interp(soc, grid, curve)
    v_oc = cell * cells_series
    v_term = v_oc - i * resistance_ohm + rng.normal(0.0, noise_v, n)

    return DischargeLog(
        time_s=t, voltage_v=v_term, current_a=i,
        cells_series=cells_series, capacity_ah=capacity_ah,
        temperature_c=25.0,
        source=f"synthetic ({current_profile} load)",
    )
