#!/usr/bin/env python3
"""
Extract a measured OCV curve from the Samsung INR21700-30T dataset.

Why this exists
---------------
The pack model needs an open-circuit-voltage curve, and until this
script there was not a measured one to give it. The two published curves
I could find were for the wrong chemistries — Chen & Rincón-Mora's is a
LiCoO2 polymer cell, and the unified-OCV model of Weng et al. is
LiFePO4 — and transferring either onto a high-nickel 21700 puts the mean
voltage several per cent out.

This one is the right cell class, it is measured, and it is CC BY 4.0:

    P. Pillai, S. Sundaresan, P. Kumar, K. Pattipati, B. Balasingam,
    "Open-Circuit Voltage Models for Battery Management Systems: A
    Review", Energies 15(18), 2022.
    Dataset: doi:10.17632/fywnpsjfpc.1

The test is a C/30 charge and discharge, which is slow enough that the
terminal voltage is the open-circuit voltage to within a couple of
millivolts. Charge and discharge branches are averaged, because at C/30
they still differ by up to 131 mV of hysteresis and the equilibrium
curve lies between them — the load drop is applied separately by the
pack model, through the internal resistance, and folding it into the
curve as well would count it twice.

Usage
-----
    python -m scripts.extract_ocv --download
    python -m scripts.extract_ocv --mat C1204_OCV.mat
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

#: The dataset's own files. Column meanings come from the ``demoVI.m``
#: script shipped alongside them, not from guesswork.
FILES = {
    "C1202_OCV.mat": "535ad06f-07ec-4374-8b1b-f44be1caaf2c",
    "C1203_OCV.mat": "8f5aee0c-19f8-442b-9fe2-fab605785fc7",
    "C1204_OCV.mat": "e936bf95-a1f9-4b92-805a-0e97d8bce6e6",
    "C1205_OCV.mat": "a7df8752-830e-4737-a197-dc32f550ce38",
}
BASE_URL = "https://data.mendeley.com/public-files/datasets/fywnpsjfpc/files"

#: Column indices, zero-based, per ``demoVI.m``: time in seconds,
#: current in amps (positive charging), terminal voltage in volts.
COL_TIME, COL_CURRENT, COL_VOLTAGE = 2, 6, 7

#: The row window ``demoVI.m`` uses, trimming the rest and settling rows.
ROW_SLICE = slice(101, 3894)

SOC_GRID = np.array([0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                     0.60, 0.70, 0.80, 0.90, 0.95, 1.00])


def download(name: str, dest: str) -> str:
    """Fetch one dataset file, if it is not already here."""
    import urllib.request

    path = os.path.join(dest, name)
    if os.path.exists(path):
        return path
    os.makedirs(dest, exist_ok=True)
    url = f"{BASE_URL}/{FILES[name]}/file_downloaded"
    print(f"  downloading {name} ...", end="", flush=True)
    # Mendeley refuses urllib's default User-Agent with a 403.
    request = urllib.request.Request(
        url, headers={"User-Agent": "raptor/0.8 (+research use)"})
    with urllib.request.urlopen(request, timeout=120) as response, \
            open(path, "wb") as out:
        out.write(response.read())
    print(f" {os.path.getsize(path)} bytes")
    return path


def coulomb_count(t_h, current_a):
    """
    State of charge by integrating current, as the dataset's own script
    does: charge and discharge capacities are measured separately from
    the test itself rather than taken from the cell's nameplate.
    """
    charging = np.where(current_a > 0)[0]
    discharging = np.where(current_a < 0)[0]
    if not len(charging) or not len(discharging):
        raise ValueError("log contains only one direction of current")

    q_chg = ((t_h[charging[-1]] - t_h[charging[0]])
             * np.mean(np.abs(current_a[charging[0]:charging[-1] + 1])))
    q_dis = ((t_h[discharging[-1]] - t_h[discharging[0]])
             * np.mean(np.abs(current_a[discharging[0]:discharging[-1] + 1])))

    soc = np.zeros(len(current_a))
    soc[0] = 1.0
    for k in range(1, len(current_a)):
        dt = t_h[k] - t_h[k - 1]
        soc[k] = soc[k - 1] + dt * current_a[k] / (
            q_dis if current_a[k] < 0 else q_chg)
    return soc, q_chg, q_dis


def extract(mat_path: str, verbose: bool = True):
    """Return ``(soc_grid, ocv_volts, diagnostics)`` from one .mat file."""
    from scipy.io import loadmat

    raw = loadmat(mat_path)
    key = [k for k in raw if not k.startswith("__")][0]
    a = raw[key]

    t_h = a[ROW_SLICE, COL_TIME] / 3600.0
    current = a[ROW_SLICE, COL_CURRENT]
    voltage = a[ROW_SLICE, COL_VOLTAGE]

    soc, q_chg, q_dis = coulomb_count(t_h, current)

    branches = {}
    for name, mask in (("discharge", current < 0), ("charge", current > 0)):
        s, v = soc[mask], voltage[mask]
        order = np.argsort(s)
        s, v = s[order], v[order]
        s_unique, idx = np.unique(np.round(s, 5), return_index=True)
        branches[name] = np.interp(SOC_GRID, s_unique, v[idx])

    ocv = 0.5 * (branches["discharge"] + branches["charge"])
    hysteresis = float(np.max(branches["charge"] - branches["discharge"]))

    # A coulombic-efficiency check on the test itself. Over one slow
    # cycle a healthy cell puts back what it gave out to within a couple
    # of per cent; a large mismatch means the row window has caught more
    # than one charge, or the log is not the cycle it appears to be.
    # Either way the branches are not two halves of the same cycle, and
    # averaging them shifts the curve — in this dataset one of the four
    # files fails this by 95 % and pulls the mean OCV up by 100 mV.
    imbalance = abs(q_chg - q_dis) / max(q_dis, 1e-9)
    usable = imbalance <= 0.10

    diagnostics = dict(
        file=os.path.basename(mat_path),
        charge_capacity_ah=float(q_chg),
        discharge_capacity_ah=float(q_dis),
        current_range_a=(float(current.min()), float(current.max())),
        voltage_range_v=(float(voltage.min()), float(voltage.max())),
        duration_h=float(t_h[-1] - t_h[0]),
        hysteresis_v=hysteresis,
        mean_ocv_v=float(np.trapezoid(ocv, SOC_GRID)),
        capacity_imbalance=float(imbalance),
        usable=bool(usable),
        branches=branches,
    )

    if verbose:
        d = diagnostics
        print(f"  {d['file']}: {d['duration_h']:.1f} h at "
              f"{d['current_range_a'][0]:+.2f} to {d['current_range_a'][1]:+.2f} A")
        print(f"    capacity  {d['discharge_capacity_ah']:.4f} Ah discharge, "
              f"{d['charge_capacity_ah']:.4f} Ah charge")
        print(f"    voltage   {d['voltage_range_v'][0]:.3f} .. "
              f"{d['voltage_range_v'][1]:.3f} V")
        print(f"    hysteresis {1000*d['hysteresis_v']:.0f} mV worst case "
              f"(branches averaged)")
        print(f"    mean OCV  {d['mean_ocv_v']:.4f} V")
        if not d["usable"]:
            print(f"    ! charge and discharge capacities differ by "
                  f"{100*d['capacity_imbalance']:.0f} % — these are not two "
                  f"halves of one cycle, and averaging them would shift the "
                  f"curve. EXCLUDED.")

    return SOC_GRID, ocv, diagnostics


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mat", default="C1204_OCV.mat",
                    help="which cell's file to use")
    ap.add_argument("--dir", default="data/ocv",
                    help="where the dataset files live")
    ap.add_argument("--download", action="store_true",
                    help="fetch the file if it is not present")
    ap.add_argument("--all", action="store_true",
                    help="extract every cell and compare them")
    args = ap.parse_args(argv)

    names = list(FILES) if args.all else [args.mat]
    curves, excluded = [], []
    for name in names:
        path = os.path.join(args.dir, name)
        if not os.path.exists(path):
            if not args.download:
                print(f"{path} not found. Re-run with --download.",
                      file=sys.stderr)
                return 1
            path = download(name, args.dir)
        grid, ocv, diag = extract(path)
        (curves if diag["usable"] else excluded).append((name, ocv))
        print()

    if not curves:
        print("no usable cells", file=sys.stderr)
        return 1

    stack = np.array([c for _, c in curves])
    ocv = stack.mean(axis=0)
    if len(curves) > 1:
        spread = stack.max(axis=0) - stack.min(axis=0)
        print(f"{len(curves)} usable cell(s); spread between them "
              f"{1000*spread.max():.1f} mV worst case, "
              f"{1000*spread.mean():.1f} mV mean")
        means = [float(np.trapezoid(c, grid)) for _, c in curves]
        print(f"mean OCV per cell: "
              + ", ".join(f"{m:.4f}" for m in means)
              + f"  (range {1000*(max(means)-min(means)):.1f} mV)")
    if excluded:
        print(f"excluded: {', '.join(n for n, _ in excluded)}")

    print("\nTable for raptor/battery.py:\n")
    print("_NMC_SOC = np.array([" + ", ".join(f"{x:.2f}" for x in grid) + "])")
    print("_NMC_CELL_V = np.array([" + ", ".join(f"{x:.4f}" for x in ocv) + "])")

    from raptor.battery import _NMC_CELL_V
    delta = ocv - _NMC_CELL_V
    print(f"\nAgainst what the package currently ships: worst "
          f"{1000*np.abs(delta).max():.1f} mV, mean "
          f"{1000*np.abs(delta).mean():.1f} mV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
