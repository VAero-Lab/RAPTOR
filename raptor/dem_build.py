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

#: Used when a tile declares no ``GDAL_NODATA`` tag. SRTM v3's value.
DEFAULT_NODATA = -32767.0

#: Any sample below this is treated as void whatever the tags say. The
#: lowest dry land on Earth is the Dead Sea shore at about -430 m, so
#: -1000 m is comfortably below anything real and comfortably above
#: every sentinel in common use.
VOID_FLOOR_M = -1000.0


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
        if not np.isfinite(out).any():
            # Nothing to interpolate from. Returning zeros here would ship
            # a sea-level terrain model into a planner that would then
            # cheerfully clear every ridge in the Andes.
            raise ValueError(
                "every cell is void — the requested area has no elevation "
                "data at all. Check the bounding box against the tiles' "
                "coverage."
            )
        # Regions with no valid neighbour within reach: fall back to the
        # mean of what was measured rather than shipping NaN onward.
        out[np.isnan(out)] = float(np.nanmean(out))
    return out


def save_dem(path: str, lat_1d: np.ndarray, lon_1d: np.ndarray,
             elev: np.ndarray, metadata: Dict = None,
             void_mask: np.ndarray = None):
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
    payload = dict(
        elev_grid=elev.astype(np.float32),
        lat_1d=lat_1d, lon_1d=lon_1d,
        slope_deg=slope_deg.astype(np.float32),
        metadata=np.array(json.dumps(metadata or {})),
    )
    # Which cells were interpolated rather than measured. SRTM returns
    # nothing on steep slopes and water, which over the Andes is exactly
    # where the terrain matters most — so a planner should be able to ask
    # whether its corridor crosses invented ground.
    if void_mask is not None:
        payload["void_mask"] = np.packbits(np.asarray(void_mask, dtype=bool))
        payload["void_shape"] = np.array(np.asarray(void_mask).shape)
    np.savez_compressed(path, **payload)


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
    nodata: Optional[float] = None
    """Void sentinel declared by the file itself (GDAL_NODATA tag).

    SRTM v3 tiles declare -32767, NASADEM declares -32768. Reading it
    from the tag rather than assuming one means a mosaic of both is
    voided correctly without the caller having to know."""

    @property
    def lat_min(self) -> float:
        return self.lat_max - self.dlat * (self.n_rows - 1)

    @property
    def lon_max(self) -> float:
        return self.lon_min + self.dlon * (self.n_cols - 1)

    def covers(self, lat: float, lon: float) -> bool:
        return (self.lat_min <= lat <= self.lat_max
                and self.lon_min <= lon <= self.lon_max)


RASTER_SUFFIXES = (".tif", ".tiff")
ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar")


def expand_tile_archives(paths: Sequence[str],
                         verbose: bool = True) -> List[str]:
    """
    Replace any ``.tar.gz`` in *paths* with the rasters inside it.

    DEM tiles are large and compress well, so they are stored in the
    repository as archives. Members are extracted next to the archive
    and reused on later calls, so this costs the unpacking once.

    Paths that are already rasters pass through untouched, which means
    a caller can hand this a whole directory listing without caring
    which entries are archives.
    """
    import tarfile

    out: List[str] = []
    for path in paths:
        low = path.lower()
        if not low.endswith(ARCHIVE_SUFFIXES):
            out.append(path)
            continue
        dest = os.path.dirname(os.path.abspath(path)) or "."
        with tarfile.open(path) as tar:
            members = [m for m in tar.getmembers()
                       if m.isfile() and m.name.lower().endswith(RASTER_SUFFIXES)]
            if not members:
                if verbose:
                    print(f"  {os.path.basename(path)}: no rasters inside, skipped")
                continue
            for m in members:
                # Refuse absolute paths and parent traversal: an archive
                # is untrusted input and this writes to disk.
                name = os.path.normpath(m.name)
                if os.path.isabs(name) or name.startswith(".."):
                    raise ValueError(
                        f"{path}: member {m.name!r} escapes the destination "
                        f"directory; refusing to extract."
                    )
                target = os.path.join(dest, name)
                if not os.path.exists(target):
                    if verbose:
                        print(f"  extracting {name} from "
                              f"{os.path.basename(path)}")
                    tar.extract(m, dest)
                out.append(target)
    return out


def _geokey(geokeys, key_id: int) -> Optional[int]:
    """Look up a short GeoTIFF GeoKey in a GeoKeyDirectoryTag.

    The tag is a flat array: a four-value header whose last entry is the
    key count, then one four-value record per key. Only keys stored
    inline (``location == 0``) are returned; the others live in
    companion tags this reader has no use for.
    """
    if geokeys is None or len(geokeys) < 4:
        return None
    n_keys = int(geokeys[3])
    for i in range(n_keys):
        base = 4 + 4 * i
        if base + 3 >= len(geokeys) + 1:
            break
        if int(geokeys[base]) == key_id and int(geokeys[base + 1]) == 0:
            return int(geokeys[base + 3])
    return None


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
            "Reading GeoTIFF tiles needs tifffile (and imagecodecs for the "
            "LZW-compressed NASADEM tiles): pip install 'raptor[terrain]'. "
            "Alternatively fetch terrain over the network with "
            "build_dem_from_opentopodata(), which needs nothing extra."
        ) from exc

    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        array = page.asarray()

        scale = tiepoint = nodata = geokeys = None
        for tag in page.tags:
            name = getattr(tag, "name", "")
            if name == "ModelPixelScaleTag":
                scale = tag.value
            elif name == "ModelTiepointTag":
                tiepoint = tag.value
            elif name == "GeoKeyDirectoryTag":
                geokeys = tag.value
            elif name == "GDAL_NODATA":
                # Stored as an ASCII string, e.g. "-32768".
                try:
                    nodata = float(str(tag.value).strip().strip("\x00"))
                except (TypeError, ValueError):
                    nodata = None

    if scale is None or tiepoint is None:
        raise ValueError(
            f"{path}: no ModelPixelScaleTag/ModelTiepointTag — not a "
            f"georeferenced GeoTIFF this reader can place."
        )

    dlon, dlat = float(scale[0]), float(scale[1])
    lon_min, lat_max = float(tiepoint[3]), float(tiepoint[4])
    n_rows, n_cols = array.shape[:2]

    # GTRasterTypeGeoKey (1025) says what the tiepoint refers to:
    # 1 = RasterPixelIsArea, the tiepoint is the *corner* of the first
    # cell; 2 = RasterPixelIsPoint, it is the cell *centre*. Everything
    # downstream indexes by cell centre, so an Area raster needs half a
    # cell added. SRTM v3 ships as Point and NASADEM as Area — reading
    # both with one convention misplaces one of them by 15 m, which on
    # a 40° Andean slope is about 13 m of invented elevation error.
    if _geokey(geokeys, 1025) == 1:
        lon_min += 0.5 * dlon
        lat_max -= 0.5 * dlat

    tile = GeoTiffTile(path=path, lat_max=lat_max, lon_min=lon_min,
                       dlat=dlat, dlon=dlon, n_rows=n_rows, n_cols=n_cols,
                       nodata=nodata)
    return tile, array


def build_dem_from_geotiff(
    tiff_paths: Sequence[str],
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    spacing_m: float = 30.0,
    nodata: Optional[float] = None,
    product: Optional[str] = None,
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
    nodata : float, optional
        Override the void sentinel. By default each tile's own
        ``GDAL_NODATA`` tag is used (SRTM v3 declares -32767, NASADEM
        -32768), falling back to -32767 for tiles that declare none.
        Any value below ``VOID_FLOOR_M`` is treated as void regardless,
        so a mislabelled tile cannot inject a -32768 m "mountain".
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
        sentinel = nodata if nodata is not None else (
            tile.nodata if tile.nodata is not None else DEFAULT_NODATA)
        # Two independent guards. The declared sentinel catches the exact
        # value the producer used; the floor catches anything the tag got
        # wrong, because no point on Earth's land surface is 1 km below
        # the Dead Sea and a stray -32768 read as metres would carve a
        # canyon the planner would dive straight into.
        vals[(vals == sentinel) | (vals < VOID_FLOOR_M)] = np.nan
        take = np.isnan(elev) & ~np.isnan(vals)
        elev[take] = vals[take]

    void_mask = np.isnan(elev)
    n_void = int(void_mask.sum())
    elev = fill_voids(elev, void_mask)

    meta = {
        "source": "geotiff",
        # Which terrain product these tiles came from. Filenames alone
        # do not say — "output_be.tif" could be anything — and a DEM
        # whose provenance is unrecorded cannot be cited in a paper.
        "product": product or "unspecified",
        "tiles": [os.path.basename(t.path) for t, _ in tiles],
        "nodata": {os.path.basename(t.path):
                   (nodata if nodata is not None else
                    (t.nodata if t.nodata is not None else DEFAULT_NODATA))
                   for t, _ in tiles},
        "spacing_m": spacing_m,
        "bbox": [lat_min, lat_max, lon_min, lon_max],
        "voids_filled": n_void,
        "void_fraction": float(n_void) / float(elev.size),
    }
    if verbose:
        print(f"  grid {len(lat_1d)} x {len(lon_1d)} at ~{spacing_m:.0f} m")
        print(f"  elevation {np.nanmin(elev):.0f}..{np.nanmax(elev):.0f} m")
        print(f"  voids filled: {n_void} "
              f"({100*n_void/elev.size:.1f}% of cells)")

    if out_path:
        save_dem(out_path, lat_1d, lon_1d, elev, meta, void_mask=void_mask)
        if verbose:
            print(f"  wrote {out_path}")

    return {"lat_1d": lat_1d, "lon_1d": lon_1d, "elev": elev,
            "void_mask": void_mask, "metadata": meta}


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

    def _key(self, lat: float, lon: float) -> str:
        """
        Cache key for one point.

        Includes the dataset name. Keying on coordinates alone means a
        cross-check against a second source silently reads back the
        first one's values and reports perfect agreement — which is
        exactly the failure a cross-check exists to catch.
        """
        return f"{self.dataset}|{lat:.6f},{lon:.6f}"

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

    #: Datasets the public instance serves. NASADEM is *not* among them,
    #: which is worth knowing before planning a validation around it.
    PUBLIC_DATASETS = ("srtm30m", "srtm90m", "aster30m", "mapzen",
                       "eudem25m", "ned10m", "etopo1", "gebco2020")

    def available_datasets(self) -> List[str]:
        """Datasets this instance is configured with."""
        return list(self.PUBLIC_DATASETS)

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
        import urllib.error
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
            except urllib.error.HTTPError as exc:
                # The API explains itself in the body — an unknown dataset
                # name, a bad bounding box. Reporting only "HTTP 400" hides
                # the one piece of information that would fix it.
                detail = ""
                try:
                    detail = json.loads(exc.read().decode()).get("error", "")
                except Exception:
                    pass
                self._last_call = time.time()
                if exc.code == 400 and "not in config" in detail:
                    raise ValueError(
                        f"{detail} Datasets available on this instance: "
                        f"{', '.join(self.available_datasets())}."
                    ) from exc
                warnings.warn(
                    f"OpenTopoData request failed (HTTP {exc.code}"
                    + (f": {detail}" if detail else "")
                    + f"); {len(idx)} point(s) left unresolved.", stacklevel=2)
                continue
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


# ═══════════════════════════════════════════════════════════════════════════
# CROSS-CHECK
# ═══════════════════════════════════════════════════════════════════════════

def cross_check_corridor(
    dem,
    waypoints: Sequence[Tuple[float, float]],
    n_samples: int = 200,
    dataset: str = "aster30m",
    api_url: str = PUBLIC_API,
    cache_path: str = None,
    verbose: bool = True,
    **client_kwargs,
) -> Dict:
    """
    Compare a local DEM against an independent source along one route.

    Cheap by design: 200 points is two API calls, so the public quota is
    irrelevant. That is enough to detect the failures that matter — a
    vertical datum offset, a mis-registered tile, or a systematic bias
    from void filling — none of which a self-consistent mosaic can reveal
    on its own.

    ``aster30m`` is the default because it is genuinely independent:
    ASTER is optical stereo where SRTM is radar, so the two fail in
    different places and agreement between them means something.
    ``srtm30m`` compares against the same source as most local tiles — it
    cannot reveal a shared bias, but it does catch tile mis-registration
    and datum errors, and it should agree to within a metre.

    NASADEM would be the ideal reference — the same radar reprocessed
    with voids filled at source — but the public OpenTopoData instance
    does not serve it. Self-host to use it.

    Expect a metre or two of scatter against a same-source reference and
    ten or more against an independent one: ASTER's own vertical accuracy
    is around ±17 m, so a difference of that order says more about ASTER
    than about the local mosaic. The verdict below accounts for that.

    Returns
    -------
    dict of difference statistics (local minus reference) and a verdict.
    """
    import urllib.error  # noqa: F401  (surfaced by the client's error path)
    dists, lats, lons = [], [], []
    total = 0.0
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        seg = int(max(n_samples / max(len(waypoints) - 1, 1), 2))
        for f in np.linspace(0, 1, seg, endpoint=False):
            lats.append(a[0] + f * (b[0] - a[0]))
            lons.append(a[1] + f * (b[1] - a[1]))
    lats.append(waypoints[-1][0])
    lons.append(waypoints[-1][1])
    lats, lons = np.array(lats), np.array(lons)

    local = np.asarray(dem.elevation_batch(lats, lons), dtype=float)

    client = OpenTopoDataClient(api_url=api_url, dataset=dataset,
                                cache_path=cache_path, **client_kwargs)
    if verbose:
        est = client.estimate(len(lats))
        print(f"  cross-checking {len(lats)} points against {dataset}: "
              f"{est['calls']} call(s)")
    reference = client.elevations(lats, lons, verbose=False)

    good = ~(np.isnan(local) | np.isnan(reference))
    if good.sum() < 10:
        return {"available": False,
                "note": "too few overlapping samples to compare"}

    diff = local[good] - reference[good]
    bias = float(np.mean(diff))
    spread = float(np.std(diff))
    worst = float(np.max(np.abs(diff)))

    # A metre or two of scatter between products is normal. A consistent
    # offset is not — it points at a datum or registration problem, and
    # it shifts every clearance number by the same amount.
    # An independent optical product legitimately differs from radar by
    # more than a same-source one does. Judging both against the same
    # threshold reports a healthy cross-check as a fault.
    independent = not dataset.startswith("srtm")
    bias_limit = 20.0 if independent else 3.0
    spread_limit = 30.0 if independent else 10.0

    verdict = "consistent"
    if abs(bias) > bias_limit:
        verdict = "systematic offset — check the vertical datum"
    elif spread > spread_limit:
        verdict = "large scatter — check tile registration or void filling"

    return {
        "available": True,
        "dataset": dataset,
        "n_compared": int(good.sum()),
        "bias_m": bias,
        "std_m": spread,
        "max_abs_m": worst,
        "verdict": verdict,
        "independent_source": independent,
        "note": (f"local minus {dataset}: bias {bias:+.1f} m, "
                 f"scatter {spread:.1f} m, worst {worst:.0f} m — {verdict}"),
    }
