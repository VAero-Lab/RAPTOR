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

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from typing import Tuple, Optional, List
from dataclasses import dataclass


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

        self.lat_grid = data['lat_grid']
        self.lon_grid = data['lon_grid']
        self.elev_grid = data['elev_grid']

        # 1D coordinate arrays
        if 'lat_1d' in data:
            self.lat_1d = data['lat_1d']
            self.lon_1d = data['lon_1d']
        else:
            self.lat_1d = self.lat_grid[:, 0]
            self.lon_1d = self.lon_grid[0, :]

        # Optional derived fields
        self.slope_deg = data.get('slope_deg', None)
        self.hillshade = data.get('hillshade', None)

        # Build interpolator for fast queries
        # Ensure lat is ascending (should be from linspace)
        if self.lat_1d[0] > self.lat_1d[-1]:
            self.lat_1d = self.lat_1d[::-1]
            self.elev_grid = self.elev_grid[::-1]

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
            slope_data = self.slope_deg
            if self.lat_1d[0] > self.lat_grid[0, 0]:
                slope_data = slope_data[::-1]
            self._slope_interpolator = RegularGridInterpolator(
                (self.lat_1d, self.lon_1d),
                slope_data,
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
        )

    # ── Point queries ────────────────────────────────────────────────────

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
