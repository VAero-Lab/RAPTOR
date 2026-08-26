"""
DEM Interface — Terrain Elevation Queries
==========================================

Wraps the NumPy meshgrid DEM to provide:
- Point elevation queries (with interpolation)
- Profile extraction along arbitrary lines
- Terrain statistics within regions
- Coordinate transforms (lat/lon ↔ grid indices)

Designed to work with the DMQ SRTM data produced by the
DEM preparation script.
"""

import json

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from typing import Tuple, Optional, List
from dataclasses import dataclass, field


@dataclass
class DEMMetadata:
    """Metadata for a loaded DEM."""
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    n_lat: int
    n_lon: int
    dlat_m: float    # grid spacing in meters (latitude direction)
    dlon_m: float    # grid spacing in meters (longitude direction)
    elev_min: float
    elev_max: float
    elev_mean: float
    source: dict = field(default_factory=dict)
    """Provenance recorded when the DEM was built: which product, which
    tiles, how many voids were filled. Written by ``save_dem`` and until
    now read by nothing — a DEM whose origin cannot be recovered cannot
    be cited in a paper or reproduced by a reader."""

    def provenance(self) -> str:
        """One line naming where this terrain came from."""
        src = self.source or {}
        product = src.get("product") or src.get("source") or "unknown source"
        voids = src.get("voids_filled")
        cell = f"{min(self.dlat_m, self.dlon_m):.0f} m"
        parts = [product, f"{self.n_lat}x{self.n_lon} at ~{cell}"]
        if voids is not None:
            parts.append("void-free" if voids == 0
                         else f"{voids} cells interpolated")
        return "; ".join(parts)


class DEMInterface:
    """
    Interface to a gridded Digital Elevation Model.

    Provides fast elevation queries, terrain profiles, and
    spatial analysis over the DEM domain.

    Parameters
    ----------
    npz_path : str
        Path to the .npz file containing 'lat_grid', 'lon_grid',
        'elev_grid', and optionally 'lat_1d', 'lon_1d', 'slope_deg',
        'hillshade'.

    Example
    -------
    >>> dem = DEMInterface("dmq_dem.npz")
    >>> dem.elevation(-0.22, -78.50)
    2830.5
    >>> profile = dem.terrain_profile((-0.25, -78.55), (-0.18, -78.49), n=200)
    """

    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)

        self.elev_grid = data['elev_grid']

        # 1D coordinate axes are the source of truth. The full coordinate
        # meshes are optional: on a 30 m grid over a city they are three
        # million points each, and they hold no information the axes do
        # not — so they are rebuilt on demand instead of stored.
        if 'lat_1d' in data:
            self.lat_1d = data['lat_1d']
            self.lon_1d = data['lon_1d']
        else:
            self.lat_1d = data['lat_grid'][:, 0]
            self.lon_1d = data['lon_grid'][0, :]

        self._lat_grid = data['lat_grid'] if 'lat_grid' in data else None
        self._lon_grid = data['lon_grid'] if 'lon_grid' in data else None

        # Cells filled by interpolation rather than measured. Stored
        # bit-packed because it is one bit per cell over millions of them.
        self.void_mask = None
        if 'void_mask' in data and 'void_shape' in data:
            shape = tuple(int(x) for x in data['void_shape'])
            bits = np.unpackbits(data['void_mask'])[:int(np.prod(shape))]
            self.void_mask = bits.astype(bool).reshape(shape)

        self.metadata_json = {}
        if 'metadata' in data:
            try:
                self.metadata_json = json.loads(str(data['metadata']))
            except Exception:
                self.metadata_json = {}

        # Optional derived fields
        self.slope_deg = data['slope_deg'] if 'slope_deg' in data else None
        self.hillshade = data['hillshade'] if 'hillshade' in data else None

        # Build interpolator for fast queries
        # Ensure lat is ascending (should be from linspace).
        # Everything keyed by row has to flip together: a void mask left
        # in the old order would mark the wrong cells as interpolated,
        # and a void report is only useful if it points at the right
        # ground.
        if self.lat_1d[0] > self.lat_1d[-1]:
            self.lat_1d = self.lat_1d[::-1]
            self.elev_grid = self.elev_grid[::-1]
            if self.void_mask is not None:
                self.void_mask = self.void_mask[::-1]
            if self.slope_deg is not None:
                self.slope_deg = self.slope_deg[::-1]
            if self.hillshade is not None:
                self.hillshade = self.hillshade[::-1]
            if self._lat_grid is not None:
                self._lat_grid = self._lat_grid[::-1]
            if self._lon_grid is not None:
                self._lon_grid = self._lon_grid[::-1]

        self._interpolator = RegularGridInterpolator(
            (self.lat_1d, self.lon_1d),
            self.elev_grid,
            method='linear',
            bounds_error=False,
            fill_value=np.nan
        )

        # Build slope interpolator if available
        self._slope_interpolator = None
        if self.slope_deg is not None:
            self._slope_interpolator = RegularGridInterpolator(
                (self.lat_1d, self.lon_1d),
                self.slope_deg,
                method='linear',
                bounds_error=False,
                fill_value=np.nan
            )

        # Compute metadata
        mean_lat = np.mean(self.lat_1d)
        dlat = self.lat_1d[1] - self.lat_1d[0]
        dlon = self.lon_1d[1] - self.lon_1d[0]
        self.metadata = DEMMetadata(
            lat_min=float(self.lat_1d[0]),
            lat_max=float(self.lat_1d[-1]),
            lon_min=float(self.lon_1d[0]),
            lon_max=float(self.lon_1d[-1]),
            n_lat=len(self.lat_1d),
            n_lon=len(self.lon_1d),
            dlat_m=float(dlat * 111_320),
            dlon_m=float(dlon * 111_320 * np.cos(np.radians(mean_lat))),
            elev_min=float(np.nanmin(self.elev_grid)),
            elev_max=float(np.nanmax(self.elev_grid)),
            elev_mean=float(np.nanmean(self.elev_grid)),
            source=dict(self.metadata_json),
        )

    @classmethod
    def from_arrays(cls, lat_1d, lon_1d, elev_grid, metadata=None,
                    void_mask=None) -> "DEMInterface":
        """
        Build a DEM from arrays already in memory, without a file.

        Used for decimation studies (how much does grid spacing change
        the answer?) and for synthetic terrain in tests, both of which
        otherwise have to round-trip through a temporary ``.npz``.
        """
        import io
        from .dem_build import save_dem

        buf = io.BytesIO()
        save_dem_kwargs = dict(metadata=metadata or {})
        if void_mask is not None:
            save_dem_kwargs["void_mask"] = void_mask
        # save_dem also derives the slope grid, so going through it keeps
        # an in-memory DEM identical to one loaded from disk rather than
        # subtly missing a field.
        import tempfile, os as _os
        with tempfile.TemporaryDirectory() as td:
            path = _os.path.join(td, "dem.npz")
            save_dem(path, np.asarray(lat_1d, float), np.asarray(lon_1d, float),
                     np.asarray(elev_grid, float), **save_dem_kwargs)
            return cls(path)

    def decimated(self, factor: int) -> "DEMInterface":
        """
        The same terrain on a grid *factor* times coarser.

        Answers "how much of this result is the terrain and how much is
        the resolution?" without needing a second data product: the
        coarse grid is this one subsampled, so the two differ in spacing
        and nothing else.
        """
        if factor < 1:
            raise ValueError(f"decimation factor must be >= 1, got {factor}")
        if factor == 1:
            return self
        meta = dict(self.metadata_json)
        meta["decimated_by"] = factor
        meta["product"] = (meta.get("product", "unknown")
                           + f" decimated {factor}x")
        return DEMInterface.from_arrays(
            self.lat_1d[::factor], self.lon_1d[::factor],
            self.elev_grid[::factor, ::factor], metadata=meta,
        )

    # ── Point queries ────────────────────────────────────────────────────

    # ── Coordinate meshes ─────────────────────────────────────────────
    #
    # Kept lazy so a fine DEM does not carry two extra full-size arrays
    # for the sake of callers that mostly want the 1-D axes anyway.

    @property
    def lat_grid(self) -> np.ndarray:
        """Latitude of every cell, shape (n_lat, n_lon)."""
        if self._lat_grid is None:
            self._lon_grid, self._lat_grid = np.meshgrid(self.lon_1d, self.lat_1d)
        return self._lat_grid

    @property
    def lon_grid(self) -> np.ndarray:
        """Longitude of every cell, shape (n_lat, n_lon)."""
        if self._lon_grid is None:
            self._lon_grid, self._lat_grid = np.meshgrid(self.lon_1d, self.lat_1d)
        return self._lon_grid

    def elevation(self, lat: float, lon: float) -> float:
        """
        Query terrain elevation at a single (lat, lon) point.

        Returns NaN if outside the DEM domain.
        """
        return float(self._interpolator([[lat, lon]])[0])

    def elevation_batch(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """
        Query terrain elevation at multiple points (vectorized).

        Parameters
        ----------
        lats, lons : array-like, shape (N,)

        Returns
        -------
        elevations : ndarray, shape (N,)
        """
        points = np.column_stack([np.asarray(lats), np.asarray(lons)])
        return self._interpolator(points)

    def slope_at(self, lat: float, lon: float) -> float:
        """Query terrain slope [degrees] at a point."""
        if self._slope_interpolator is None:
            raise ValueError("Slope data not available in this DEM.")
        return float(self._slope_interpolator([[lat, lon]])[0])

    def in_domain(self, lat: float, lon: float) -> bool:
        """Check if a point is within the DEM domain."""
        return (self.metadata.lat_min <= lat <= self.metadata.lat_max and
                self.metadata.lon_min <= lon <= self.metadata.lon_max)

    # ── Terrain profiles ─────────────────────────────────────────────────

    def terrain_profile(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        n: int = 200
    ) -> dict:
        """
        Extract terrain elevation profile along a straight line.

        Parameters
        ----------
        start : (lat, lon)
            Start point.
        end : (lat, lon)
            End point.
        n : int
            Number of sample points along the profile.

        Returns
        -------
        dict with keys:
            'lats', 'lons' : ndarray (n,)
            'elevations' : ndarray (n,) — terrain elevation [m AMSL]
            'distances' : ndarray (n,) — cumulative ground distance [m]
            'total_distance' : float — total profile length [m]
            'max_elevation' : float
            'min_elevation' : float
        """
        lats = np.linspace(start[0], end[0], n)
        lons = np.linspace(start[1], end[1], n)
        elevs = self.elevation_batch(lats, lons)

        # Compute cumulative ground distance
        dlat_m = np.diff(lats) * 111_320
        dlon_m = np.diff(lons) * 111_320 * np.cos(np.radians(lats[:-1]))
        seg_dist = np.sqrt(dlat_m**2 + dlon_m**2)
        distances = np.concatenate([[0], np.cumsum(seg_dist)])

        return {
            'lats': lats,
            'lons': lons,
            'elevations': elevs,
            'distances': distances,
            'total_distance': float(distances[-1]),
            'max_elevation': float(np.nanmax(elevs)),
            'min_elevation': float(np.nanmin(elevs)),
        }

    def max_terrain_between(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        n: int = 300
    ) -> Tuple[float, float, Tuple[float, float]]:
        """
        Find the maximum terrain elevation along a straight line.

        Returns
        -------
        (max_elev, distance_at_max, (lat_at_max, lon_at_max))
        """
        profile = self.terrain_profile(start, end, n)
        idx = np.nanargmax(profile['elevations'])
        return (
            float(profile['elevations'][idx]),
            float(profile['distances'][idx]),
            (float(profile['lats'][idx]), float(profile['lons'][idx]))
        )

    # ── Spatial utilities ────────────────────────────────────────────────

    @staticmethod
    def haversine(lat1: float, lon1: float,
                  lat2: float, lon2: float) -> float:
        """
        Great-circle distance between two points [meters].
        """
        R = 6_371_000  # Earth radius [m]
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlam = np.radians(lon2 - lon1)
        a = (np.sin(dphi / 2)**2 +
             np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2)**2)
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    @staticmethod
    def bearing(lat1: float, lon1: float,
                lat2: float, lon2: float) -> float:
        """
        Initial bearing from point 1 to point 2 [degrees, 0=N, 90=E].
        """
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dlam = np.radians(lon2 - lon1)
        x = np.sin(dlam) * np.cos(phi2)
        y = (np.cos(phi1) * np.sin(phi2) -
             np.sin(phi1) * np.cos(phi2) * np.cos(dlam))
        return np.degrees(np.arctan2(x, y)) % 360

    @staticmethod
    def destination_point(lat: float, lon: float,
                          bearing_deg: float, distance_m: float
                          ) -> Tuple[float, float]:
        """
        Compute destination point given start, bearing, and distance.

        Returns (lat, lon) in degrees.
        """
        R = 6_371_000
        d = distance_m / R
        brng = np.radians(bearing_deg)
        phi1 = np.radians(lat)
        lam1 = np.radians(lon)

        phi2 = np.arcsin(
            np.sin(phi1) * np.cos(d) +
            np.cos(phi1) * np.sin(d) * np.cos(brng)
        )
        lam2 = lam1 + np.arctan2(
            np.sin(brng) * np.sin(d) * np.cos(phi1),
            np.cos(d) - np.sin(phi1) * np.sin(phi2)
        )
        return float(np.degrees(phi2)), float(np.degrees(lam2))

    def __repr__(self):
        m = self.metadata
        return (f"DEMInterface(lat=[{m.lat_min:.3f}, {m.lat_max:.3f}], "
                f"lon=[{m.lon_min:.3f}, {m.lon_max:.3f}], "
                f"shape=({m.n_lat}, {m.n_lon}), "
                f"elev=[{m.elev_min:.0f}, {m.elev_max:.0f}] m)")


# ═════════════════════════════════════════════════════════════════════════════
# LOCATING A DEM
# ═════════════════════════════════════════════════════════════════════════════

#: Candidate DEM filenames, finest first. Scripts resolve through this so
#: rebuilding the terrain at a better resolution takes effect everywhere
#: without editing every entry point — and so nobody quietly keeps
#: planning on a coarse grid because that is what a hard-coded path said.
DEM_CANDIDATES = (
    "dmq_dem_30m.npz",
    "dmq_dem_90m.npz",
    "dmq_dem.npz",
)

DEM_SEARCH_DIRS = ("data", ".", "../data", "..")


def find_dem(candidates=None, search_dirs=None, verbose: bool = False) -> str:
    """
    Path to the best available DEM.

    Prefers the finest grid present. Raises with a usable instruction
    rather than a bare FileNotFoundError, because the fix — build one —
    is a single command.
    """
    import os

    candidates = candidates or DEM_CANDIDATES
    search_dirs = search_dirs or DEM_SEARCH_DIRS

    for name in candidates:
        for d in search_dirs:
            p = os.path.join(d, name)
            if os.path.exists(p):
                if verbose:
                    print(f"DEM: {p}")
                return p

    raise FileNotFoundError(
        "No DEM found. Looked for "
        + ", ".join(candidates)
        + " under " + ", ".join(search_dirs) + ".\n"
        "Build one with:\n"
        "  python -m scripts.build_dem tiles --tiff 'data/TIF_files/*.tif' "
        "--bbox LAT_MIN LAT_MAX LON_MIN LON_MAX --spacing 30 --out data/dem.npz\n"
        "or fetch terrain over the network:\n"
        "  python -m scripts.build_dem corridor --from LAT LON --to LAT LON "
        "--out data/corridor.npz"
    )


# ═════════════════════════════════════════════════════════════════════════════
# DATA QUALITY
# ═════════════════════════════════════════════════════════════════════════════

def void_report(dem, waypoints=None, half_width_m: float = 1000.0) -> dict:
    """
    How much of a DEM — or of one corridor through it — is invented.

    SRTM returns no signal from steep slopes and open water, and the
    holes are filled by interpolating neighbours. That is the right thing
    to do, but it means some of the terrain a planner clears was never
    measured, and over the Andes the voids cluster on exactly the steep
    ground that threatens a low corridor.

    Parameters
    ----------
    dem : DEMInterface
    waypoints : list of (lat, lon), optional
        Restrict the report to a swath along this route.
    half_width_m : float
        Swath half-width when ``waypoints`` is given.

    Returns
    -------
    dict with the void fraction overall and, if a route was given, along
    it — plus a verdict on whether it is worth worrying about.
    """
    if getattr(dem, "void_mask", None) is None:
        return {"available": False,
                "note": "This DEM carries no void record. Rebuild it with "
                        "scripts/build_dem.py to capture one."}

    mask = dem.void_mask
    out = {
        "available": True,
        "total_cells": int(mask.size),
        "void_cells": int(mask.sum()),
        "void_fraction": float(mask.mean()),
    }

    if waypoints:
        from .dem_build import route_mask
        lon_grid, lat_grid = np.meshgrid(dem.lon_1d, dem.lat_1d)
        swath = route_mask(lat_grid, lon_grid, waypoints, half_width_m)
        n = int(swath.sum())
        v = int((swath & mask).sum())
        out.update({
            "corridor_cells": n,
            "corridor_void_cells": v,
            "corridor_void_fraction": float(v / n) if n else 0.0,
        })
        frac = out["corridor_void_fraction"]
    else:
        frac = out["void_fraction"]

    out["concerning"] = frac > 0.10
    out["note"] = (
        f"{frac*100:.1f}% of the terrain here was interpolated, not measured."
        + (" That is enough to matter for clearance; consider a void-filled "
           "product such as NASADEM or SRTM v3." if frac > 0.10 else "")
    )
    return out
