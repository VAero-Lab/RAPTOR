"""
Building DEMs — From Local Tiles or From a Terrain API
=======================================================

RAPTOR consumes a gridded DEM. This module builds one, by either route:

**From local GeoTIFF tiles.** SRTM/NASADEM/ASTER tiles as downloaded
from USGS. Free, offline, and at the source resolution — for the
1-arc-second tiles in ``data/TIF_files`` that is ~30 m, four times finer
than the ~186 m grid the bundled ``.npz`` was decimated to.

**From OpenTopoData.** A public HTTP terrain API (opentopodata.org)
serving SRTM, NASADEM, ASTER, EU-DEM, Mapzen and others. No files to
manage, works anywhere on Earth, and so makes the package genuinely
region-agnostic — but the public instance is rate-limited, and those
limits decide what is practical:

===============  ==========================================
Limit            Public instance
===============  ==========================================
locations/call   100
calls/second     1
calls/day        1000
===============  ==========================================

That is 100 000 points per day. A 30 m grid over the Quito metropolitan
district is about 3 million points — thirty days of quota. So for the
public API, fetch a **corridor**: a swath along the route rather than a
rectangle over the whole city. A 20 km corridor with a 2 km-wide swath at
30 m is roughly 45 000 points, or about 450 calls — one afternoon's
quota, cached forever after.

For a full raster, self-host: ``docker run -p 5000:5000
opentopodata/opentopodata`` has no limits. Pass its URL as ``api_url``.

Author: Victor (LUAS-EPN / KU Leuven)
"""

from __future__ import annotations

import json
import math
import os
import time
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

M_PER_DEG_LAT = 111_320.0


# ═══════════════════════════════════════════════════════════════════════════
# GRID HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _grid_axes(lat_min: float, lat_max: float,
               lon_min: float, lon_max: float,
               spacing_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """Latitude and longitude axes at an (approximately) uniform spacing."""
    mid_lat = 0.5 * (lat_min + lat_max)
    dlat = spacing_m / M_PER_DEG_LAT
    dlon = spacing_m / (M_PER_DEG_LAT * max(math.cos(math.radians(mid_lat)), 1e-6))
    n_lat = max(int(round((lat_max - lat_min) / dlat)) + 1, 2)
    n_lon = max(int(round((lon_max - lon_min) / dlon)) + 1, 2)
    return (np.linspace(lat_min, lat_max, n_lat),
            np.linspace(lon_min, lon_max, n_lon))


def fill_voids(elev: np.ndarray, nodata_mask: np.ndarray,
               max_passes: int = 64) -> np.ndarray:
    """
    Fill nodata cells by iterative averaging of valid neighbours.

    SRTM has genuine holes — steep slopes and water return no signal, and
    about 9 % of the tiles over the Andes are void. Leaving them as NaN
    puts blind spots in the terrain model exactly where the terrain is
    most demanding; a corridor planner that cannot see a ridge will
    happily route through it.
    """
    out = elev.astype(float).copy()
    out[nodata_mask] = np.nan
    if not np.isnan(out).any():
        return out

    for _ in range(max_passes):
        bad = np.isnan(out)
        if not bad.any():
            break
        padded = np.pad(out, 1, mode='edge')
        stack = np.stack([
            padded[:-2, 1:-1], padded[2:, 1:-1],
            padded[1:-1, :-2], padded[1:-1, 2:],
            padded[:-2, :-2], padded[:-2, 2:],
            padded[2:, :-2], padded[2:, 2:],
        ])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            neighbour_mean = np.nanmean(stack, axis=0)
        out[bad] = neighbour_mean[bad]

    if np.isnan(out).any():
        # Whole regions with no valid neighbour at all: fall back to the
        # global mean rather than shipping NaN into the planner.
        out[np.isnan(out)] = float(np.nanmean(out))
    return out


def save_dem(path: str, lat_1d: np.ndarray, lon_1d: np.ndarray,
             elev: np.ndarray, metadata: Dict = None):
    """
    Write a DEM in the ``.npz`` layout ``DEMInterface`` reads.

    Also stores terrain slope, which the planner uses to reason about
    where a corridor can descend fast enough to stay under its ceiling.
    """
    mid_lat = float(np.mean(lat_1d))
    dlat_m = float(np.abs(lat_1d[1] - lat_1d[0]) * M_PER_DEG_LAT)
    dlon_m = float(np.abs(lon_1d[1] - lon_1d[0]) * M_PER_DEG_LAT
                   * math.cos(math.radians(mid_lat)))
    gy, gx = np.gradient(elev, dlat_m, dlon_m)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    # float32 and no coordinate meshes: a 30 m grid over a metropolitan
    # area is a few million cells, and storing lat/lon per cell in float64
    # quadruples the file for information the axes already carry.
    np.savez_compressed(
        path,
        elev_grid=elev.astype(np.float32),
        lat_1d=lat_1d, lon_1d=lon_1d,
        slope_deg=slope_deg.astype(np.float32),
        metadata=np.array(json.dumps(metadata or {})),
    )


# ═══════════════════════════════════════════════════════════════════════════
# LOCAL GEOTIFF TILES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GeoTiffTile:
    """One georeferenced raster tile, read lazily."""
    path: str
    lat_max: float
    lon_min: float
    dlat: float
    dlon: float
    n_rows: int
    n_cols: int

    @property
    def lat_min(self) -> float:
        return self.lat_max - self.dlat * (self.n_rows - 1)

    @property
    def lon_max(self) -> float:
        return self.lon_min + self.dlon * (self.n_cols - 1)

    def covers(self, lat: float, lon: float) -> bool:
        return (self.lat_min <= lat <= self.lat_max
                and self.lon_min <= lon <= self.lon_max)


def read_geotiff_tile(path: str) -> Tuple[GeoTiffTile, np.ndarray]:
    """
    Read a GeoTIFF tile and its georeferencing.

    Uses ``tifffile``, which reads the raster and the GeoTIFF tags
    without needing GDAL — one fewer system dependency for a package
    whose users are aerospace researchers, not GIS administrators.
    """
    try:
        import tifffile
    except ImportError as exc:
        raise ImportError(
            "Reading GeoTIFF tiles needs tifffile: pip install 'raptor[terrain]'. "
            "Alternatively fetch terrain over the network with "
            "build_dem_from_opentopodata(), which needs nothing extra."
        ) from exc

    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        array = page.asarray()

        scale = tiepoint = None
        for tag in page.tags:
            name = getattr(tag, "name", "")
            if name == "ModelPixelScaleTag":
                scale = tag.value
            elif name == "ModelTiepointTag":
                tiepoint = tag.value

    if scale is None or tiepoint is None:
        raise ValueError(
            f"{path}: no ModelPixelScaleTag/ModelTiepointTag — not a "
            f"georeferenced GeoTIFF this reader can place."
        )

    dlon, dlat = float(scale[0]), float(scale[1])
    lon_min, lat_max = float(tiepoint[3]), float(tiepoint[4])
    n_rows, n_cols = array.shape[:2]

    tile = GeoTiffTile(path=path, lat_max=lat_max, lon_min=lon_min,
                       dlat=dlat, dlon=dlon, n_rows=n_rows, n_cols=n_cols)
    return tile, array


def build_dem_from_geotiff(
    tiff_paths: Sequence[str],
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    spacing_m: float = 30.0,
    nodata: int = -32767,
    out_path: str = None,
    verbose: bool = True,
) -> Dict:
    """
    Mosaic local GeoTIFF tiles into a RAPTOR DEM.

    Parameters
    ----------
    tiff_paths : sequence of str
        Tiles to draw from. Tiles that do not intersect the box are
        skipped, so a whole directory can be passed.
    lat_min, lat_max, lon_min, lon_max : float
        Bounding box to extract.
    spacing_m : float
        Output grid spacing. Requesting finer than the source resolution
        interpolates and buys nothing real — 1-arc-second tiles are
        ~30 m at the equator.
    nodata : int
        Source nodata value. SRTM uses -32767; voids are filled.
    out_path : str
        Where to write the ``.npz``. If ``None``, nothing is written and
        the arrays are returned for inspection.

    Returns
    -------
    dict with ``lat_1d``, ``lon_1d``, ``elev``, and coverage statistics.
    """
    tiles = []
    for p in tiff_paths:
        try:
            tile, array = read_geotiff_tile(p)
        except Exception as exc:
            if verbose:
                print(f"  skipped {os.path.basename(p)}: {exc}")
            continue
        if (tile.lat_max < lat_min or tile.lat_min > lat_max
                or tile.lon_max < lon_min or tile.lon_min > lon_max):
            if verbose:
                print(f"  {os.path.basename(p)}: outside the box, skipped")
            continue
        tiles.append((tile, array))
        if verbose:
            print(f"  {os.path.basename(p)}: lat "
                  f"{tile.lat_min:.2f}..{tile.lat_max:.2f}, lon "
                  f"{tile.lon_min:.2f}..{tile.lon_max:.2f}, "
                  f"{tile.n_rows}x{tile.n_cols}")

    if not tiles:
        raise ValueError("No tiles intersect the requested bounding box.")

    lat_1d, lon_1d = _grid_axes(lat_min, lat_max, lon_min, lon_max, spacing_m)
    elev = np.full((len(lat_1d), len(lon_1d)), np.nan)
    LON, LAT = np.meshgrid(lon_1d, lat_1d)

    for tile, array in tiles:
        # Nearest-neighbour sampling: the output grid is at or coarser
        # than the source, so interpolating would only smooth away the
        # ridges that matter for clearance.
        row = np.round((tile.lat_max - LAT) / tile.dlat).astype(int)
        col = np.round((LON - tile.lon_min) / tile.dlon).astype(int)
        inside = ((row >= 0) & (row < tile.n_rows)
                  & (col >= 0) & (col < tile.n_cols))
        if not inside.any():
            continue
        vals = np.full(LAT.shape, np.nan)
        vals[inside] = array[row[inside], col[inside]].astype(float)
        vals[vals == nodata] = np.nan
        take = np.isnan(elev) & ~np.isnan(vals)
        elev[take] = vals[take]

    n_void = int(np.isnan(elev).sum())
    elev = fill_voids(elev, np.isnan(elev))

    meta = {
        "source": "geotiff",
        "tiles": [os.path.basename(t.path) for t, _ in tiles],
        "spacing_m": spacing_m,
        "bbox": [lat_min, lat_max, lon_min, lon_max],
        "voids_filled": n_void,
    }
    if verbose:
        print(f"  grid {len(lat_1d)} x {len(lon_1d)} at ~{spacing_m:.0f} m")
        print(f"  elevation {np.nanmin(elev):.0f}..{np.nanmax(elev):.0f} m")
        print(f"  voids filled: {n_void} "
              f"({100*n_void/elev.size:.1f}% of cells)")

    if out_path:
        save_dem(out_path, lat_1d, lon_1d, elev, meta)
        if verbose:
            print(f"  wrote {out_path}")

    return {"lat_1d": lat_1d, "lon_1d": lon_1d, "elev": elev,
            "metadata": meta}


# ═══════════════════════════════════════════════════════════════════════════
# OPENTOPODATA
# ═══════════════════════════════════════════════════════════════════════════

PUBLIC_API = "https://api.opentopodata.org/v1"

#: Limits of the public instance. A self-hosted instance has none of
#: these; pass your own values when you run one.
PUBLIC_LIMITS = dict(locations_per_call=100, calls_per_second=1.0,
                     calls_per_day=1000)


class OpenTopoDataClient:
    """
    Minimal OpenTopoData client with the rate limits built in.

    Deliberately not a thin wrapper around ``requests.get``: the public
    instance will refuse service if the limits are exceeded, and a
    half-fetched DEM is worse than none. Every request is spaced, every
    response is cached, and the quota is checked *before* starting so a
    request too large to complete is refused up front rather than
    discovered 900 calls in.
    """

    def __init__(self, api_url: str = PUBLIC_API,
                 dataset: str = "srtm30m",
                 locations_per_call: int = None,
                 calls_per_second: float = None,
                 calls_per_day: int = None,
                 cache_path: str = None,
                 timeout_s: float = 30.0):
        self.api_url = api_url.rstrip("/")
        self.dataset = dataset
        is_public = self.api_url.startswith(PUBLIC_API.rsplit("/", 1)[0])

        self.locations_per_call = locations_per_call or (
            PUBLIC_LIMITS["locations_per_call"] if is_public else 500)
        self.calls_per_second = calls_per_second if calls_per_second is not None else (
            PUBLIC_LIMITS["calls_per_second"] if is_public else 20.0)
        self.calls_per_day = calls_per_day or (
            PUBLIC_LIMITS["calls_per_day"] if is_public else 10 ** 9)

        self.timeout_s = timeout_s
        self.cache_path = cache_path
        self._cache: Dict[str, float] = {}
        self._calls_made = 0
        self._last_call = 0.0
        if cache_path and os.path.exists(cache_path):
            self._load_cache()

    # ── Cache ─────────────────────────────────────────────────────────

    @staticmethod
    def _key(lat: float, lon: float) -> str:
        return f"{lat:.6f},{lon:.6f}"

    def _load_cache(self):
        try:
            d = np.load(self.cache_path, allow_pickle=False)
            self._cache = dict(zip(
                (str(k) for k in d["keys"]),
                (float(v) for v in d["values"]),
            ))
        except Exception as exc:
            warnings.warn(f"Elevation cache unreadable ({exc}); ignoring.",
                          stacklevel=2)

    def save_cache(self):
        if not self.cache_path or not self._cache:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)) or ".",
                    exist_ok=True)
        keys = np.array(list(self._cache.keys()))
        vals = np.array(list(self._cache.values()), dtype=float)
        np.savez_compressed(self.cache_path, keys=keys, values=vals)

    # ── Quota ─────────────────────────────────────────────────────────

    def estimate(self, n_points: int) -> Dict:
        """Cost of fetching ``n_points``, before committing to it."""
        n_calls = math.ceil(n_points / self.locations_per_call)
        seconds = n_calls / max(self.calls_per_second, 1e-9)
        return {
            "points": n_points,
            "calls": n_calls,
            "seconds": seconds,
            "days_of_quota": n_calls / max(self.calls_per_day, 1),
            "within_daily_quota": n_calls <= self.calls_per_day,
        }

    # ── Fetch ─────────────────────────────────────────────────────────

    def elevations(self, lats: Sequence[float], lons: Sequence[float],
                   verbose: bool = True,
                   allow_over_quota: bool = False) -> np.ndarray:
        """
        Elevations at arbitrary points [m], in the order given.

        Cached points are served locally and never re-requested, so an
        interrupted fetch resumes where it stopped.
        """
        import urllib.parse
        import urllib.request

        lats = np.asarray(lats, dtype=float)
        lons = np.asarray(lons, dtype=float)
        out = np.full(len(lats), np.nan)

        todo = []
        for i, (la, lo) in enumerate(zip(lats, lons)):
            k = self._key(la, lo)
            if k in self._cache:
                out[i] = self._cache[k]
            else:
                todo.append(i)

        if not todo:
            return out

        est = self.estimate(len(todo))
        if verbose:
            print(f"  {len(todo)} point(s) to fetch — {est['calls']} call(s), "
                  f"~{est['seconds']/60:.1f} min")
        if not est["within_daily_quota"] and not allow_over_quota:
            raise RuntimeError(
                f"This request needs {est['calls']} API calls but the daily "
                f"limit is {self.calls_per_day} "
                f"({est['days_of_quota']:.1f} days of quota). Fetch a corridor "
                f"instead of a full grid, coarsen the spacing, or point "
                f"api_url at a self-hosted instance "
                f"(docker run -p 5000:5000 opentopodata/opentopodata). "
                f"Pass allow_over_quota=True to attempt it anyway."
            )

        chunk = self.locations_per_call
        min_gap = 1.0 / max(self.calls_per_second, 1e-9)

        for start in range(0, len(todo), chunk):
            idx = todo[start:start + chunk]
            locs = "|".join(f"{lats[i]:.6f},{lons[i]:.6f}" for i in idx)
            url = (f"{self.api_url}/{self.dataset}?"
                   + urllib.parse.urlencode({"locations": locs}))

            gap = time.time() - self._last_call
            if gap < min_gap:
                time.sleep(min_gap - gap)

            try:
                with urllib.request.urlopen(url, timeout=self.timeout_s) as resp:
                    payload = json.loads(resp.read().decode())
            except Exception as exc:
                warnings.warn(
                    f"OpenTopoData request failed ({exc}); "
                    f"{len(idx)} point(s) left unresolved.", stacklevel=2)
                self._last_call = time.time()
                continue

            self._last_call = time.time()
            self._calls_made += 1

            for i, res in zip(idx, payload.get("results", [])):
                e = res.get("elevation")
                if e is None:
                    continue
                out[i] = float(e)
                self._cache[self._key(lats[i], lons[i])] = float(e)

            if verbose and self._calls_made % 25 == 0:
                done = start + len(idx)
                print(f"    {done}/{len(todo)} points "
                      f"({self._calls_made} calls)")

        self.save_cache()
        return out


def build_dem_from_opentopodata(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    spacing_m: float = 90.0,
    dataset: str = "srtm30m",
    api_url: str = PUBLIC_API,
    out_path: str = None,
    cache_path: str = None,
    verbose: bool = True,
    allow_over_quota: bool = False,
    corridor: Sequence[Tuple[float, float]] = None,
    corridor_half_width_m: float = 3000.0,
    **client_kwargs,
) -> Dict:
    """
    Build a DEM over a bounding box by querying OpenTopoData.

    Check :meth:`OpenTopoDataClient.estimate` first for anything large:
    the public instance allows 100 000 points a day, and a 30 m grid over
    a city is tens of times that.
    """
    client = OpenTopoDataClient(api_url=api_url, dataset=dataset,
                                cache_path=cache_path, **client_kwargs)
    lat_1d, lon_1d = _grid_axes(lat_min, lat_max, lon_min, lon_max, spacing_m)
    LON, LAT = np.meshgrid(lon_1d, lat_1d)

    if corridor is not None:
        wanted = route_mask(LAT, LON, corridor, corridor_half_width_m)
    else:
        wanted = np.ones(LAT.shape, dtype=bool)

    if verbose:
        est = client.estimate(int(wanted.sum()))
        print(f"Grid {len(lat_1d)} x {len(lon_1d)} = {LAT.size} cells "
              f"at ~{spacing_m:.0f} m")
        if corridor is not None:
            print(f"  swath {corridor_half_width_m:.0f} m either side of the "
                  f"route: {int(wanted.sum()):,} cells "
                  f"({100*wanted.mean():.0f}% of the box)")
        print(f"  {est['calls']} call(s), ~{est['seconds']/60:.1f} min, "
              f"{est['days_of_quota']:.2f} days of quota")

    elev = np.full(LAT.shape, np.nan)
    elev[wanted] = client.elevations(
        LAT[wanted].ravel(), LON[wanted].ravel(),
        verbose=verbose, allow_over_quota=allow_over_quota,
    )

    # Fill only the voids *inside* the fetched area. Cells outside the
    # corridor were never requested and stay NaN — the planner already
    # treats unknown ground as unknown, and inventing terrain there would
    # let a route wander off the surveyed strip without anything noticing.
    inside_void = wanted & np.isnan(elev)
    n_void = int(inside_void.sum())
    if wanted.all():
        elev = fill_voids(elev, np.isnan(elev))
    elif n_void:
        patched = fill_voids(np.where(wanted, elev, np.nan), inside_void)
        elev = np.where(wanted, patched, np.nan)

    meta = {"source": f"opentopodata:{dataset}", "api_url": api_url,
            "spacing_m": spacing_m,
            "bbox": [lat_min, lat_max, lon_min, lon_max],
            "voids_filled": n_void,
            "corridor_half_width_m": (corridor_half_width_m
                                      if corridor is not None else None),
            "surveyed_fraction": float(wanted.mean())}

    if out_path:
        save_dem(out_path, lat_1d, lon_1d, elev, meta)
        if verbose:
            print(f"  wrote {out_path}")

    return {"lat_1d": lat_1d, "lon_1d": lon_1d, "elev": elev,
            "metadata": meta}


def route_mask(lat_grid: np.ndarray, lon_grid: np.ndarray,
               waypoints: Sequence[Tuple[float, float]],
               half_width_m: float) -> np.ndarray:
    """
    Cells within ``half_width_m`` of the route polyline.

    The bounding box around a *diagonal* route is far larger than the
    route: a 15 km leg running south-west fills a box whose corners are
    nowhere near it, and on a 30 m grid those corners are most of the
    points. Masking to the actual swath is what brings a corridor fetch
    inside the public API's daily quota.
    """
    mid_lat = float(np.mean(lat_grid))
    m_lat = M_PER_DEG_LAT
    m_lon = M_PER_DEG_LAT * max(math.cos(math.radians(mid_lat)), 1e-6)

    px = (lon_grid - waypoints[0][1]) * m_lon
    py = (lat_grid - waypoints[0][0]) * m_lat

    inside = np.zeros(lat_grid.shape, dtype=bool)
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        ax = (a[1] - waypoints[0][1]) * m_lon
        ay = (a[0] - waypoints[0][0]) * m_lat
        bx = (b[1] - waypoints[0][1]) * m_lon
        by = (b[0] - waypoints[0][0]) * m_lat

        abx, aby = bx - ax, by - ay
        ab_sq = abx * abx + aby * aby
        if ab_sq < 1e-9:
            d = np.hypot(px - ax, py - ay)
        else:
            t = np.clip(((px - ax) * abx + (py - ay) * aby) / ab_sq, 0.0, 1.0)
            d = np.hypot(px - (ax + t * abx), py - (ay + t * aby))
        inside |= d <= half_width_m

    return inside


def corridor_bbox(waypoints: Sequence[Tuple[float, float]],
                  half_width_m: float = 3000.0) -> Tuple[float, float, float, float]:
    """
    Bounding box around a route, widened by ``half_width_m``.

    The practical unit of terrain for this package: the optimizer needs
    to see far enough either side of the direct line to find a detour,
    but not the whole province. A 3 km swath is enough for the lateral
    offsets the router actually explores.
    """
    lats = [w[0] for w in waypoints]
    lons = [w[1] for w in waypoints]
    mid_lat = 0.5 * (min(lats) + max(lats))
    dlat = half_width_m / M_PER_DEG_LAT
    dlon = half_width_m / (M_PER_DEG_LAT
                           * max(math.cos(math.radians(mid_lat)), 1e-6))
    return (min(lats) - dlat, max(lats) + dlat,
            min(lons) - dlon, max(lons) + dlon)
