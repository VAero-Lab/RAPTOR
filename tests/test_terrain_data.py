"""
Regression tests for DEM construction, quality reporting and cross-checks.

Network-dependent tests are skipped when offline, so the suite stays
usable on a machine with no route to the internet.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from raptor.dem import DEMInterface, find_dem, void_report
from raptor.dem_build import (
    OpenTopoDataClient, _grid_axes, corridor_bbox, fill_voids, route_mask,
    save_dem,
)

DEM_PATH = "data/dmq_dem_30m.npz"
pytestmark = pytest.mark.skipif(
    not os.path.exists(DEM_PATH), reason="30 m DMQ DEM not built"
)


@pytest.fixture(scope="module")
def dem():
    return DEMInterface(DEM_PATH)


# ── Void handling ─────────────────────────────────────────────────────────

def test_fill_voids_leaves_no_holes():
    a = np.arange(100, dtype=float).reshape(10, 10)
    mask = np.zeros_like(a, dtype=bool)
    mask[3:6, 3:6] = True
    filled = fill_voids(a, mask)
    assert not np.isnan(filled).any()
    # Untouched cells stay exactly as they were.
    assert np.allclose(filled[~mask], a[~mask])


def test_fill_voids_refuses_an_all_void_grid():
    """
    Nothing to interpolate from. Silently returning zeros would ship a
    sea-level terrain model into a planner that would then clear every
    ridge in the Andes.
    """
    a = np.full((5, 5), np.nan)
    with pytest.raises(ValueError, match="no elevation data"):
        fill_voids(a, np.ones_like(a, dtype=bool))


def test_void_mask_round_trips(tmp_path):
    """
    The mask is bit-packed to store one bit per cell over millions of
    them; it has to come back the same shape and the same values.
    """
    lat = np.linspace(-0.1, 0.1, 12)
    lon = np.linspace(-78.6, -78.4, 15)
    elev = np.full((12, 15), 2800.0)
    mask = np.zeros((12, 15), dtype=bool)
    mask[2:5, 3:7] = True

    path = str(tmp_path / "d.npz")
    save_dem(path, lat, lon, elev, {"source": "test"}, void_mask=mask)
    back = DEMInterface(path)
    assert back.void_mask is not None
    assert back.void_mask.shape == mask.shape
    assert np.array_equal(back.void_mask, mask)


def test_void_report_is_honest_when_there_is_no_record(tmp_path):
    lat = np.linspace(-0.1, 0.1, 8)
    lon = np.linspace(-78.6, -78.4, 8)
    path = str(tmp_path / "d.npz")
    save_dem(path, lat, lon, np.full((8, 8), 2800.0), {})
    r = void_report(DEMInterface(path))
    assert r["available"] is False


def test_corridors_are_mostly_measured_ground(dem):
    """
    The DMQ mosaic is ~8 % void overall, but the voids sit on steep
    ground away from the study routes. If that ever stops being true the
    clearance numbers are resting on interpolation.
    """
    r = void_report(dem, [(-0.2000, -78.4930), (-0.3242, -78.5493)],
                    half_width_m=2000.0)
    assert r["available"]
    assert r["corridor_void_fraction"] < 0.10


# ── Geometry helpers ──────────────────────────────────────────────────────

def test_route_mask_selects_a_swath_not_a_box():
    """
    The box around a diagonal route is far larger than the route. Masking
    to the swath is what brings a corridor fetch inside the daily quota.
    """
    route = [(-0.20, -78.49), (-0.32, -78.55)]
    bbox = corridor_bbox(route, half_width_m=2000.0)
    lat, lon = _grid_axes(*bbox, 200.0)
    LON, LAT = np.meshgrid(lon, lat)
    mask = route_mask(LAT, LON, route, 2000.0)
    assert 0.1 < mask.mean() < 0.85       # a swath, not everything


def test_route_mask_includes_the_endpoints():
    route = [(-0.20, -78.49), (-0.32, -78.55)]
    lat = np.array([-0.20, -0.32])
    lon = np.array([-78.49, -78.55])
    assert route_mask(lat, lon, route, 100.0).all()


def test_grid_spacing_is_roughly_uniform_in_metres():
    lat, lon = _grid_axes(-0.42, 0.08, -78.72, -78.28, 30.0)
    dlat_m = (lat[1] - lat[0]) * 111_320.0
    dlon_m = (lon[1] - lon[0]) * 111_320.0 * np.cos(np.radians(-0.17))
    assert dlat_m == pytest.approx(30.0, rel=0.05)
    assert dlon_m == pytest.approx(30.0, rel=0.05)


# ── Quota arithmetic ──────────────────────────────────────────────────────

def test_quota_estimate_refuses_an_impossible_request():
    c = OpenTopoDataClient()
    est = c.estimate(3_000_000)
    assert not est["within_daily_quota"]
    assert est["days_of_quota"] > 10


def test_a_corridor_swath_fits_the_daily_quota():
    c = OpenTopoDataClient()
    assert c.estimate(80_000)["within_daily_quota"]


def test_self_hosted_instance_has_no_quota():
    c = OpenTopoDataClient(api_url="http://localhost:5000/v1")
    assert c.estimate(3_000_000)["within_daily_quota"]


def test_cache_key_includes_the_dataset():
    """
    Keying on coordinates alone makes a cross-check against a second
    source read back the first one's values and report perfect
    agreement — the exact failure a cross-check exists to catch.
    """
    a = OpenTopoDataClient(dataset="srtm30m")
    b = OpenTopoDataClient(dataset="aster30m")
    assert a._key(-0.25, -78.57) != b._key(-0.25, -78.57)


def test_unknown_dataset_lists_the_valid_ones():
    c = OpenTopoDataClient()
    assert "srtm30m" in c.available_datasets()
    assert "nasadem" not in c.available_datasets()


# ── DEM resolution ────────────────────────────────────────────────────────

def test_find_dem_prefers_the_finest_grid():
    assert find_dem().endswith("dmq_dem_30m.npz")


def test_dem_resolution_is_what_it_claims(dem):
    cell = min(dem.metadata.dlat_m, dem.metadata.dlon_m)
    assert cell == pytest.approx(30.0, rel=0.05)


def test_coordinate_meshes_are_lazy_but_correct(dem):
    """
    A 30 m grid over a city is millions of cells; storing lat and lon per
    cell triples the file for information the axes already carry.
    """
    assert dem.lat_grid.shape == dem.elev_grid.shape
    assert dem.lat_grid[:, 0] == pytest.approx(dem.lat_1d)
    assert dem.lon_grid[0, :] == pytest.approx(dem.lon_1d)


# ── Reporting semantics ───────────────────────────────────────────────────

def test_airspace_report_separates_hard_from_permits(dem):
    """
    `n_violations` counts permit-zone entries too, which are legal once
    authorised. Anything that reports feasibility must use
    `hard_violations`, or a flyable route is labelled a violation.
    """
    from raptor.airspace import (
        AirspaceManager, AirspaceZone, CircularZone, ZoneType,
    )
    from raptor.path import FlightPath
    from raptor.segments import FWCruise

    ground = float(dem.elevation(-0.2000, -78.4930))
    permit = AirspaceZone(
        zone_id="HOSP", zone_type=ZoneType.HOSPITAL,
        geometry=CircularZone(-0.2000, -78.4930, 5000.0),
        altitude_ceiling_m=9000.0,
    )
    fp = FlightPath(-0.2000, -78.4930, ground, -0.1950, -78.4930, ground)
    fp.add_segment(FWCruise(ground_distance=500.0, airspeed=30.0))
    for seg in fp.segments:
        st = seg.start_state
        seg.start_state = type(st)(
            lat=st.lat, lon=st.lon, alt=ground + 90.0,
            bearing=st.bearing, time=st.time, distance=st.distance,
        )

    report = AirspaceManager([permit], dem=dem).check_path(fp)
    assert report.n_violations > 0          # entries were recorded
    assert not report.hard_violations       # but none of them are fatal
    assert report.feasible                  # so the route is legal
    assert report.permits_required


# ── Pixel convention and void sentinels ───────────────────────────────────
#
# The DEM the package ships changed from SRTM v3 to NASADEM. The two
# products disagree about what a GeoTIFF tiepoint means and about which
# integer marks a void, and getting either wrong is silent: the planner
# receives a plausible-looking grid that is misplaced or contains a
# 32 km-deep canyon. These tests pin both conventions down.

SRTM_TILE = "data/TIF_files/n00_w079_1arc_v3.tif"
NASADEM_TILE = "data/TIF_files/output_be.tif"


def test_pixel_is_area_tiles_are_shifted_half_a_cell():
    """
    GTRasterTypeGeoKey 1025 says whether a tiepoint marks a pixel's
    corner (Area) or its centre (Point). SRTM v3 ships as Point and
    NASADEM as Area. Reading both with one convention misplaces one of
    them by half a cell — 15 m at 1 arc-second — which on a 40 degree
    Andean slope is about 13 m of invented elevation error, spread over
    the whole map and pointing consistently downhill.

    The check: after the shift, every tile's cell centres must land on
    the same whole-arc-second lattice, whatever convention it declared.
    """
    from raptor.dem_build import read_geotiff_tile

    for path in (SRTM_TILE, NASADEM_TILE):
        if not os.path.exists(path):
            pytest.skip(f"{path} not present")
        tile, _ = read_geotiff_tile(path)
        for value, step in ((tile.lat_max, tile.dlat),
                            (tile.lon_min, tile.dlon)):
            lattice = value / step
            assert lattice == pytest.approx(round(lattice), abs=1e-3), (
                f"{path}: cell centres are off the arc-second lattice by "
                f"{abs(lattice - round(lattice)):.3f} cells"
            )


def test_nodata_is_read_from_the_tile_not_assumed():
    """SRTM v3 declares -32767, NASADEM -32768. One hard-coded value
    cannot serve both, and the wrong one leaves the sentinel in the
    grid as though it were terrain."""
    from raptor.dem_build import read_geotiff_tile

    expected = {SRTM_TILE: -32767.0, NASADEM_TILE: -32768.0}
    for path, want in expected.items():
        if not os.path.exists(path):
            pytest.skip(f"{path} not present")
        tile, _ = read_geotiff_tile(path)
        assert tile.nodata == want


def test_absurd_elevations_are_voided_whatever_the_tag_says(tmp_path):
    """
    Belt and braces for a mislabelled tile. A sentinel read as metres
    would carve a canyon 32 km deep, and a planner minimising energy
    would dive straight into it.
    """
    from raptor.dem_build import VOID_FLOOR_M, build_dem_from_geotiff
    import raptor.dem_build as db

    grid = np.full((60, 60), 2800.0)
    grid[20:24, 20:24] = -32768.0          # voids, but tagged -32767
    tile = db.GeoTiffTile(path="fake.tif", lat_max=0.02, lon_min=-78.52,
                          dlat=1 / 3600, dlon=1 / 3600,
                          n_rows=60, n_cols=60, nodata=-32767.0)

    real = db.read_geotiff_tile
    db.read_geotiff_tile = lambda p: (tile, grid)
    try:
        out = build_dem_from_geotiff(
            ["fake.tif"], 0.005, 0.015, -78.515, -78.505,
            spacing_m=30.0, out_path=None, verbose=False,
        )
    finally:
        db.read_geotiff_tile = real

    assert out["elev"].min() > VOID_FLOOR_M
    assert out["elev"].min() > 0.0


def test_shipped_dem_has_no_voids(dem):
    """
    NASADEM is void-filled at source. The DEM RAPTOR ships should
    therefore contain no interpolated cells at all — every metre of
    terrain the planner clears was measured.
    """
    report = void_report(dem)
    assert report["available"]
    assert report["void_cells"] == 0, (
        f"{report['void_cells']} interpolated cells in the shipped DEM; "
        f"it was not built from a void-free product."
    )


def test_dem_records_which_product_it_came_from(dem):
    """A DEM whose provenance is unrecorded cannot be cited, and
    'output_be.tif' does not identify a product."""
    product = dem.metadata.source.get("product", "")
    assert product and product != "unspecified", (
        "rebuild the DEM passing --product so the source is recorded"
    )
    assert "NASADEM" in dem.metadata.provenance()


# ── Tile archives ─────────────────────────────────────────────────────────

def test_expand_tile_archives_passes_rasters_through():
    from raptor.dem_build import expand_tile_archives
    assert expand_tile_archives(["a.tif", "b.TIFF"], verbose=False) == \
        ["a.tif", "b.TIFF"]


def test_expand_tile_archives_refuses_path_traversal(tmp_path):
    """An archive is untrusted input and extraction writes to disk."""
    import tarfile
    from raptor.dem_build import expand_tile_archives

    payload = tmp_path / "evil.tif"
    payload.write_bytes(b"not really a tiff")
    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(payload, arcname="../escaped.tif")

    with pytest.raises(ValueError, match="escapes"):
        expand_tile_archives([str(archive)], verbose=False)


# ── Shaded relief ─────────────────────────────────────────────────────────

def test_hillshade_is_bounded_and_responds_to_slope():
    from raptor.viz_corridor import hillshade

    flat = np.full((20, 20), 2800.0)
    ramp = 2800.0 + 30.0 * np.arange(20)[None, :].repeat(20, axis=0)

    s_flat = hillshade(flat, 30.0, 30.0)
    s_ramp = hillshade(ramp, 30.0, 30.0)
    for s in (s_flat, s_ramp):
        assert s.min() >= 0.0 and s.max() <= 1.0
    # A slope facing away from the north-west sun is darker than flat
    # ground; if these came out equal the light direction is being
    # ignored and the "relief" is decoration.
    assert s_ramp.mean() != pytest.approx(s_flat.mean(), abs=1e-3)


def test_terrain_backdrop_rejects_a_bbox_off_the_dem(dem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from raptor.viz_corridor import terrain_backdrop

    _, ax = plt.subplots()
    with pytest.raises(ValueError, match="outside or below"):
        terrain_backdrop(ax, dem, bbox=(40.0, 41.0, 10.0, 11.0))
    plt.close("all")
