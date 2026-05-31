"""
Comprehensive test suite for the TerraTiff package.

Covers round-trip I/O, dtype casting, multi-band, binary masks,
CRS encoding, resampling, non-spatial array export, and band operations.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from terratiff import TerraTiff, parse_crs, utm_zone_from_latlon, utm_epsg, supported_dtypes
from terratiff.crs import CRSInfo
from terratiff.dtypes import resolve_dtype, cast_array, is_binary
from terratiff.utils import (
    build_tiepoint, parse_tiepoint,
    build_pixel_scale, parse_pixel_scale,
    compute_bounds, nearest_resample, ensure_3d,
    bilinear_resample, cubic_resample, average_resample,
    rasterize_polygon, coord_to_pixel, pixel_to_coord,
)


# ── Helpers ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory for test files."""
    return tmp_path


def _roundtrip(raster: TerraTiff, tmp_dir: Path, dtype: str | None = None) -> TerraTiff:
    """Save and re-open a TerraTiff for round-trip verification."""
    path = tmp_dir / "roundtrip.tif"
    raster.save(str(path), dtype=dtype)
    return TerraTiff.open(str(path))


# ═══════════════════════════════════════════════════════════════════════════
# Utils tests
# ═══════════════════════════════════════════════════════════════════════════

class TestUtils:
    def test_tiepoint_roundtrip(self):
        tp = build_tiepoint(500_000.0, 4_500_000.0)
        x, y = parse_tiepoint(tp)
        assert x == 500_000.0
        assert y == 4_500_000.0

    def test_pixel_scale_roundtrip(self):
        ps = build_pixel_scale(30.0, -30.0)
        w, h = parse_pixel_scale(ps)
        assert w == 30.0
        assert h == -30.0  # convention: negative for north-up

    def test_compute_bounds(self):
        xmin, ymin, xmax, ymax = compute_bounds(
            origin_x=100.0, origin_y=200.0,
            pixel_width=10.0, pixel_height=-10.0,
            cols=5, rows=4,
        )
        assert xmin == 100.0
        assert xmax == 150.0  # 100 + 10*5
        assert ymin == 160.0  # 200 + (-10)*4
        assert ymax == 200.0

    def test_ensure_3d_from_2d(self):
        arr = np.zeros((10, 20))
        out = ensure_3d(arr)
        assert out.shape == (1, 10, 20)

    def test_ensure_3d_already_3d(self):
        arr = np.zeros((3, 10, 20))
        out = ensure_3d(arr)
        assert out.shape == (3, 10, 20)

    def test_nearest_resample_2d(self):
        arr = np.arange(100).reshape(10, 10)
        out = nearest_resample(arr, 5, 5)
        assert out.shape == (5, 5)

    def test_nearest_resample_3d(self):
        arr = np.arange(300).reshape(3, 10, 10)
        out = nearest_resample(arr, 5, 5)
        assert out.shape == (3, 5, 5)

    def test_bilinear_resample_2d(self):
        arr = np.ones((10, 10), dtype=np.float32) * 5.0
        out = bilinear_resample(arr, 5, 5)
        assert out.shape == (5, 5)
        np.testing.assert_allclose(out, 5.0, atol=1e-5)

    def test_bilinear_resample_3d(self):
        arr = np.ones((3, 10, 10), dtype=np.float32) * 3.0
        out = bilinear_resample(arr, 20, 20)
        assert out.shape == (3, 20, 20)
        np.testing.assert_allclose(out, 3.0, atol=1e-5)

    def test_cubic_resample_2d(self):
        arr = np.ones((10, 10), dtype=np.float32) * 7.0
        out = cubic_resample(arr, 5, 5)
        assert out.shape == (5, 5)
        np.testing.assert_allclose(out, 7.0, atol=1e-3)

    def test_cubic_resample_3d(self):
        arr = np.ones((2, 10, 10), dtype=np.float32) * 2.0
        out = cubic_resample(arr, 20, 20)
        assert out.shape == (2, 20, 20)
        np.testing.assert_allclose(out, 2.0, atol=1e-3)

    def test_average_resample_2d(self):
        arr = np.ones((10, 10), dtype=np.float32) * 4.0
        out = average_resample(arr, 5, 5)
        assert out.shape == (5, 5)
        np.testing.assert_allclose(out, 4.0)

    def test_average_resample_3d(self):
        arr = np.ones((3, 10, 10), dtype=np.float32) * 6.0
        out = average_resample(arr, 2, 2)
        assert out.shape == (3, 2, 2)
        np.testing.assert_allclose(out, 6.0)

    def test_rasterize_polygon_square(self):
        """A square polygon covering the inner region of a 10×10 grid."""
        # Grid: origin (0,0), pixel 1×-1, so y goes from 0 (top) to -10 (bottom)
        # x goes from 0 (left) to 10 (right)
        polygon = [(2.0, -2.0), (8.0, -2.0), (8.0, -8.0), (2.0, -8.0)]
        mask = rasterize_polygon(polygon, 10, 10, 0.0, 0.0, 1.0, -1.0)
        assert mask.shape == (10, 10)
        # Centre pixels should be inside
        assert mask[5, 5] == True
        # Corner pixels should be outside
        assert mask[0, 0] == False

    def test_coord_to_pixel(self):
        row, col = coord_to_pixel(
            x=500015.0, y=4499985.0,
            origin_x=500000, origin_y=4500000,
            pixel_width=30, pixel_height=-30,
        )
        assert col == 0
        assert row == 0

    def test_pixel_to_coord(self):
        x, y = pixel_to_coord(
            row=0, col=0,
            origin_x=500000, origin_y=4500000,
            pixel_width=30, pixel_height=-30,
        )
        assert x == pytest.approx(500015.0)
        assert y == pytest.approx(4499985.0)


# ═══════════════════════════════════════════════════════════════════════════
# CRS tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCRS:
    def test_parse_wgs84(self):
        crs = parse_crs("WGS84")
        assert crs.epsg == 4326
        assert not crs.is_projected

    def test_parse_epsg_int(self):
        crs = parse_crs(4326)
        assert crs.epsg == 4326

    def test_parse_epsg_string(self):
        crs = parse_crs("EPSG:32633")
        assert crs.epsg == 32633
        assert crs.is_projected

    def test_parse_utm_north(self):
        crs = parse_crs("UTM:33N")
        assert crs.epsg == 32633
        assert crs.is_projected

    def test_parse_utm_south(self):
        crs = parse_crs("UTM:33S")
        assert crs.epsg == 32733

    def test_utm_zone_from_latlon(self):
        zone, hemi = utm_zone_from_latlon(42.0, -93.5)
        assert zone == 15
        assert hemi == "N"

    def test_utm_zone_south(self):
        zone, hemi = utm_zone_from_latlon(-33.9, 18.4)
        assert zone == 34
        assert hemi == "S"

    def test_utm_epsg(self):
        assert utm_epsg(33, "N") == 32633
        assert utm_epsg(33, "S") == 32733

    def test_invalid_crs_raises(self):
        with pytest.raises(ValueError):
            parse_crs("UNKNOWN_CRS")


# ═══════════════════════════════════════════════════════════════════════════
# Dtypes tests
# ═══════════════════════════════════════════════════════════════════════════

class TestDtypes:
    def test_resolve_known(self):
        dt = resolve_dtype("float32")
        assert dt == np.dtype(np.float32)

    def test_resolve_none_fallback(self):
        dt = resolve_dtype(None, np.dtype(np.int16))
        assert dt == np.dtype(np.int16)

    def test_resolve_invalid_raises(self):
        with pytest.raises(ValueError):
            resolve_dtype("float128")

    def test_is_binary(self):
        assert is_binary("binary") is True
        assert is_binary("float32") is False
        assert is_binary(None) is False

    def test_cast_array_binary(self):
        arr = np.array([0, 5, 0, 255], dtype=np.uint8)
        out = cast_array(arr, "binary")
        np.testing.assert_array_equal(out, [0, 1, 0, 1])

    def test_cast_array_type_change(self):
        arr = np.array([1.5, 2.7], dtype=np.float64)
        out = cast_array(arr, "int16")
        assert out.dtype == np.int16

    def test_supported_dtypes_list(self):
        names = supported_dtypes()
        assert "float32" in names
        assert "binary" in names


# ═══════════════════════════════════════════════════════════════════════════
# TerraTiff round-trip tests
# ═══════════════════════════════════════════════════════════════════════════

class TestTerraTiffRoundTrip:
    """Write → read → verify data, CRS, and transform survive the trip."""

    def test_single_band_float32(self, tmp_dir):
        arr = np.random.rand(50, 60).astype(np.float32)
        gt = TerraTiff.from_array(
            arr, origin_x=500_000, origin_y=4_500_000,
            pixel_width=30, pixel_height=-30, crs="UTM:33N",
        )
        loaded = _roundtrip(gt, tmp_dir, dtype="float32")

        assert loaded.shape == (1, 50, 60)
        assert loaded.crs == "EPSG:32633"
        np.testing.assert_allclose(loaded.data[0], arr, atol=1e-5)
        assert loaded.origin_x == pytest.approx(500_000)
        assert loaded.origin_y == pytest.approx(4_500_000)
        assert loaded.pixel_width == pytest.approx(30)
        assert loaded.pixel_height == pytest.approx(-30)

    def test_single_band_float64(self, tmp_dir):
        arr = np.random.rand(30, 40).astype(np.float64)
        gt = TerraTiff.from_array(
            arr, origin_x=-93.5, origin_y=42.0,
            pixel_width=0.001, pixel_height=-0.001, crs="WGS84",
        )
        loaded = _roundtrip(gt, tmp_dir, dtype="float64")
        assert loaded.dtype == np.float64
        np.testing.assert_allclose(loaded.data[0], arr)

    def test_single_band_int16(self, tmp_dir):
        arr = np.random.randint(-1000, 1000, (40, 50), dtype=np.int16)
        gt = TerraTiff.from_array(
            arr, origin_x=0, origin_y=0,
            pixel_width=10, pixel_height=-10, crs="UTM:15N",
        )
        loaded = _roundtrip(gt, tmp_dir, dtype="int16")
        assert loaded.dtype == np.int16
        np.testing.assert_array_equal(loaded.data[0], arr)

    def test_uint8(self, tmp_dir):
        arr = np.random.randint(0, 255, (20, 20), dtype=np.uint8)
        gt = TerraTiff.from_array(
            arr, origin_x=0, origin_y=100,
            pixel_width=1, pixel_height=-1, crs="WGS84",
        )
        loaded = _roundtrip(gt, tmp_dir, dtype="uint8")
        assert loaded.dtype == np.uint8
        np.testing.assert_array_equal(loaded.data[0], arr)

    def test_multi_band_3(self, tmp_dir):
        arr = np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8)
        gt = TerraTiff.from_array(
            arr, origin_x=0, origin_y=0,
            pixel_width=0.5, pixel_height=-0.5, crs="WGS84",
        )
        loaded = _roundtrip(gt, tmp_dir, dtype="uint8")
        assert loaded.shape == (3, 64, 64)
        np.testing.assert_array_equal(loaded.data, arr)

    def test_multi_band_7(self, tmp_dir):
        arr = np.random.rand(7, 32, 32).astype(np.float32)
        gt = TerraTiff.from_array(
            arr, origin_x=100, origin_y=200,
            pixel_width=10, pixel_height=-10, crs="UTM:33N",
        )
        loaded = _roundtrip(gt, tmp_dir, dtype="float32")
        assert loaded.shape == (7, 32, 32)
        np.testing.assert_allclose(loaded.data, arr, atol=1e-5)

    def test_binary_mask(self, tmp_dir):
        arr = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.uint8)
        gt = TerraTiff.from_array(
            arr, origin_x=0, origin_y=0,
            pixel_width=1, pixel_height=-1, crs="WGS84",
        )
        loaded = _roundtrip(gt, tmp_dir, dtype="binary")
        np.testing.assert_array_equal(loaded.data[0], arr)
        assert set(np.unique(loaded.data)) <= {0, 1}

    def test_nodata_roundtrip(self, tmp_dir):
        arr = np.full((10, 10), -9999.0, dtype=np.float32)
        gt = TerraTiff.from_array(
            arr, origin_x=0, origin_y=0,
            pixel_width=1, pixel_height=-1, crs="WGS84",
            nodata=-9999.0,
        )
        loaded = _roundtrip(gt, tmp_dir, dtype="float32")
        assert loaded.nodata == pytest.approx(-9999.0)


# ═══════════════════════════════════════════════════════════════════════════
# CRS round-trip (GeoKey encoding / decoding)
# ═══════════════════════════════════════════════════════════════════════════

class TestCRSRoundTrip:
    def test_wgs84_roundtrip(self, tmp_dir):
        gt = TerraTiff.from_array(
            np.zeros((5, 5), dtype=np.float32),
            origin_x=0, origin_y=0, pixel_width=1, pixel_height=-1,
            crs="WGS84",
        )
        loaded = _roundtrip(gt, tmp_dir)
        assert loaded.crs == "EPSG:4326"
        assert not loaded.crs_info.is_projected

    def test_utm_roundtrip(self, tmp_dir):
        gt = TerraTiff.from_array(
            np.zeros((5, 5), dtype=np.float32),
            origin_x=500_000, origin_y=4_500_000,
            pixel_width=30, pixel_height=-30, crs="UTM:33N",
        )
        loaded = _roundtrip(gt, tmp_dir)
        assert loaded.crs == "EPSG:32633"
        assert loaded.crs_info.is_projected


# ═══════════════════════════════════════════════════════════════════════════
# Resampling (all methods)
# ═══════════════════════════════════════════════════════════════════════════

class TestResample:
    def test_downsample_nearest(self, tmp_dir):
        arr = np.arange(10_000, dtype=np.float32).reshape(100, 100)
        gt = TerraTiff.from_array(
            arr, origin_x=0, origin_y=0,
            pixel_width=10, pixel_height=-10, crs="UTM:33N",
        )
        resampled = gt.resample(pixel_width=20, pixel_height=-20, method="nearest")
        assert resampled.shape == (1, 50, 50)
        assert resampled.pixel_width == pytest.approx(20)
        assert resampled.pixel_height == pytest.approx(-20)

    def test_upsample_nearest(self, tmp_dir):
        arr = np.ones((10, 10), dtype=np.float32)
        gt = TerraTiff.from_array(
            arr, origin_x=0, origin_y=0,
            pixel_width=100, pixel_height=-100, crs="UTM:33N",
        )
        resampled = gt.resample(pixel_width=50, pixel_height=-50)
        assert resampled.shape == (1, 20, 20)
        np.testing.assert_array_equal(resampled.data[0], 1.0)

    def test_resample_preserves_bounds(self, tmp_dir):
        arr = np.ones((100, 100), dtype=np.float32)
        gt = TerraTiff.from_array(
            arr, origin_x=500_000, origin_y=4_500_000,
            pixel_width=30, pixel_height=-30, crs="UTM:33N",
        )
        original_bounds = gt.get_bounds()
        resampled = gt.resample(pixel_width=60, pixel_height=-60)
        new_bounds = resampled.get_bounds()
        assert new_bounds[0] == pytest.approx(original_bounds[0])
        assert new_bounds[2] == pytest.approx(original_bounds[2])

    def test_resample_bilinear(self):
        arr = np.ones((20, 20), dtype=np.float32) * 10.0
        gt = TerraTiff.from_array(
            arr, origin_x=0, origin_y=0,
            pixel_width=10, pixel_height=-10, crs="UTM:33N",
        )
        resampled = gt.resample(pixel_width=20, pixel_height=-20, method="bilinear")
        assert resampled.shape == (1, 10, 10)
        np.testing.assert_allclose(resampled.data[0], 10.0, atol=1e-5)

    def test_resample_cubic(self):
        arr = np.ones((20, 20), dtype=np.float32) * 5.0
        gt = TerraTiff.from_array(
            arr, origin_x=0, origin_y=0,
            pixel_width=10, pixel_height=-10, crs="UTM:33N",
        )
        resampled = gt.resample(pixel_width=20, pixel_height=-20, method="cubic")
        assert resampled.shape == (1, 10, 10)
        np.testing.assert_allclose(resampled.data[0], 5.0, atol=1e-3)

    def test_resample_average(self):
        arr = np.ones((20, 20), dtype=np.float32) * 8.0
        gt = TerraTiff.from_array(
            arr, origin_x=0, origin_y=0,
            pixel_width=10, pixel_height=-10, crs="UTM:33N",
        )
        resampled = gt.resample(pixel_width=20, pixel_height=-20, method="average")
        assert resampled.shape == (1, 10, 10)
        np.testing.assert_allclose(resampled.data[0], 8.0)

    def test_resample_invalid_method(self):
        arr = np.ones((10, 10), dtype=np.float32)
        gt = TerraTiff.from_array(
            arr, origin_x=0, origin_y=0,
            pixel_width=10, pixel_height=-10, crs="UTM:33N",
        )
        with pytest.raises(ValueError, match="Unknown resampling"):
            gt.resample(pixel_width=20, pixel_height=-20, method="lanczos")

    @pytest.mark.parametrize("method", ["nearest", "bilinear", "cubic", "average"])
    def test_resample_roundtrip_all_methods(self, tmp_dir, method):
        """Each method produces valid output that can be saved & reloaded."""
        arr = np.random.rand(40, 40).astype(np.float32) * 100
        gt = TerraTiff.from_array(
            arr, origin_x=500_000, origin_y=4_500_000,
            pixel_width=30, pixel_height=-30, crs="UTM:33N",
        )
        resampled = gt.resample(pixel_width=60, pixel_height=-60, method=method)
        path = tmp_dir / f"resample_{method}.tif"
        resampled.save(str(path), dtype="float32")
        loaded = TerraTiff.open(str(path))
        assert loaded.shape == resampled.shape


# ═══════════════════════════════════════════════════════════════════════════
# Band operations
# ═══════════════════════════════════════════════════════════════════════════

class TestBandOps:
    def test_get_band(self):
        arr = np.random.rand(3, 10, 10).astype(np.float32)
        gt = TerraTiff.from_array(
            arr, origin_x=0, origin_y=0,
            pixel_width=1, pixel_height=-1, crs="WGS84",
        )
        band0 = gt.get_band(0)
        np.testing.assert_array_equal(band0, arr[0])

    def test_get_band_out_of_range(self):
        gt = TerraTiff.from_array(
            np.zeros((10, 10)), origin_x=0, origin_y=0,
            pixel_width=1, pixel_height=-1, crs="WGS84",
        )
        with pytest.raises(IndexError):
            gt.get_band(5)

    def test_add_band(self):
        gt = TerraTiff.from_array(
            np.zeros((10, 10)), origin_x=0, origin_y=0,
            pixel_width=1, pixel_height=-1, crs="WGS84",
        )
        assert gt.bands == 1
        gt.add_band(np.ones((10, 10)))
        assert gt.bands == 2

    def test_add_band_shape_mismatch(self):
        gt = TerraTiff.from_array(
            np.zeros((10, 10)), origin_x=0, origin_y=0,
            pixel_width=1, pixel_height=-1, crs="WGS84",
        )
        with pytest.raises(ValueError):
            gt.add_band(np.ones((5, 5)))


# ═══════════════════════════════════════════════════════════════════════════
# Dtype export variants
# ═══════════════════════════════════════════════════════════════════════════

class TestDtypeExport:
    """Ensure every supported dtype can be written and read back."""

    @pytest.mark.parametrize("dtype_name", [
        "uint8", "uint16", "uint32",
        "int8", "int16", "int32",
        "float32", "float64",
    ])
    def test_dtype_roundtrip(self, tmp_dir, dtype_name):
        arr = np.ones((10, 10), dtype=np.float64) * 42
        gt = TerraTiff.from_array(
            arr, origin_x=0, origin_y=0,
            pixel_width=1, pixel_height=-1, crs="WGS84",
        )
        loaded = _roundtrip(gt, tmp_dir, dtype=dtype_name)
        expected_np = resolve_dtype(dtype_name)
        assert loaded.dtype == expected_np

    def test_float16_write_read(self, tmp_dir):
        """float16 is a special case — tifffile may promote it."""
        arr = np.ones((10, 10), dtype=np.float32)
        gt = TerraTiff.from_array(
            arr, origin_x=0, origin_y=0,
            pixel_width=1, pixel_height=-1, crs="WGS84",
        )
        path = tmp_dir / "f16.tif"
        gt.save(str(path), dtype="float16")
        assert path.exists()
        loaded = TerraTiff.open(str(path))
        # tifffile may read it back as float16 or promote to float32
        assert loaded.dtype in (np.dtype(np.float16), np.dtype(np.float32))


# ═══════════════════════════════════════════════════════════════════════════
# Non-spatial array export
# ═══════════════════════════════════════════════════════════════════════════

class TestFromArray:
    def test_from_plain_array(self, tmp_dir):
        arr = np.random.rand(100, 200).astype(np.float32)
        gt = TerraTiff.from_array(
            arr, origin_x=500_000, origin_y=4_500_000,
            pixel_width=30, pixel_height=-30, crs="UTM:33N",
        )
        loaded = _roundtrip(gt, tmp_dir, dtype="float32")
        assert loaded.shape == (1, 100, 200)
        np.testing.assert_allclose(loaded.data[0], arr, atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════
# Clip (extent-based masking)
# ═══════════════════════════════════════════════════════════════════════════

class TestClip:
    def test_clip_centre(self):
        """Clip the centre region of a raster."""
        arr = np.arange(10_000, dtype=np.float32).reshape(100, 100)
        gt = TerraTiff.from_array(
            arr, origin_x=0, origin_y=0,
            pixel_width=10, pixel_height=-10, crs="UTM:33N",
        )
        # Full extent: x [0, 1000], y [-1000, 0]
        clipped = gt.clip(200, -800, 800, -200)
        assert clipped.cols == 60
        assert clipped.rows == 60
        assert clipped.origin_x == pytest.approx(200)

    def test_clip_preserves_data(self, tmp_dir):
        arr = np.ones((50, 50), dtype=np.float32) * 42
        gt = TerraTiff.from_array(
            arr, origin_x=0, origin_y=0,
            pixel_width=1, pixel_height=-1, crs="WGS84",
        )
        clipped = gt.clip(10, -40, 40, -10)
        np.testing.assert_allclose(clipped.data[0], 42.0)
        # Save and reload
        path = tmp_dir / "clipped.tif"
        clipped.save(str(path))
        loaded = TerraTiff.open(str(path))
        assert loaded.cols == 30
        assert loaded.rows == 30

    def test_clip_no_overlap_raises(self):
        gt = TerraTiff.from_array(
            np.zeros((10, 10)), origin_x=0, origin_y=0,
            pixel_width=1, pixel_height=-1, crs="WGS84",
        )
        with pytest.raises(ValueError, match="does not overlap"):
            gt.clip(100, 100, 200, 200)

    def test_clip_partial_overlap(self):
        """Clip with extent partially outside the raster — clamped to intersection."""
        arr = np.ones((100, 100), dtype=np.float32)
        gt = TerraTiff.from_array(
            arr, origin_x=0, origin_y=0,
            pixel_width=1, pixel_height=-1, crs="UTM:33N",
        )
        # Extent extends beyond the right edge
        clipped = gt.clip(50, -100, 200, 0)
        assert clipped.cols == 50  # clamped to 100 - 50
        assert clipped.rows == 100


# ═══════════════════════════════════════════════════════════════════════════
# Polygon masking
# ═══════════════════════════════════════════════════════════════════════════

class TestMaskWithPolygon:
    def test_polygon_mask_basic(self):
        """Pixels outside the polygon should be set to nodata."""
        arr = np.ones((20, 20), dtype=np.float32) * 100
        gt = TerraTiff.from_array(
            arr, origin_x=0, origin_y=0,
            pixel_width=1, pixel_height=-1, crs="UTM:33N",
        )
        # Polygon covers roughly the centre
        polygon = [(5, -5), (15, -5), (15, -15), (5, -15)]
        masked = gt.mask_with_polygon(polygon, nodata=-1)

        # Inside the polygon → 100
        assert masked.data[0, 10, 10] == 100
        # Outside the polygon → nodata
        assert masked.data[0, 0, 0] == -1
        assert masked.nodata == -1

    def test_polygon_mask_invert(self):
        """With invert=True, inside pixels should be masked."""
        arr = np.ones((20, 20), dtype=np.float32) * 50
        gt = TerraTiff.from_array(
            arr, origin_x=0, origin_y=0,
            pixel_width=1, pixel_height=-1, crs="UTM:33N",
        )
        polygon = [(5, -5), (15, -5), (15, -15), (5, -15)]
        masked = gt.mask_with_polygon(polygon, invert=True, nodata=0)

        # Inside the polygon → nodata (0)
        assert masked.data[0, 10, 10] == 0
        # Outside the polygon → 50
        assert masked.data[0, 0, 0] == 50

    def test_polygon_mask_multiband(self):
        arr = np.ones((3, 20, 20), dtype=np.float32) * 10
        gt = TerraTiff.from_array(
            arr, origin_x=0, origin_y=0,
            pixel_width=1, pixel_height=-1, crs="UTM:33N",
        )
        polygon = [(5, -5), (15, -5), (15, -15), (5, -15)]
        masked = gt.mask_with_polygon(polygon, nodata=-9999)
        # All bands should be masked outside
        for b in range(3):
            assert masked.data[b, 0, 0] == -9999
            assert masked.data[b, 10, 10] == 10

    def test_polygon_mask_roundtrip(self, tmp_dir):
        """Masked raster can be saved and reloaded."""
        arr = np.ones((20, 20), dtype=np.float32) * 42
        gt = TerraTiff.from_array(
            arr, origin_x=0, origin_y=0,
            pixel_width=1, pixel_height=-1, crs="UTM:33N",
        )
        polygon = [(5, -5), (15, -5), (15, -15), (5, -15)]
        masked = gt.mask_with_polygon(polygon, nodata=-9999)
        path = tmp_dir / "polygon_masked.tif"
        masked.save(str(path), dtype="float32")
        loaded = TerraTiff.open(str(path))
        assert loaded.nodata == pytest.approx(-9999)


# ═══════════════════════════════════════════════════════════════════════════
# Raster masking
# ═══════════════════════════════════════════════════════════════════════════

class TestMaskWithRaster:
    def test_raster_mask_basic(self):
        """Zero pixels in the mask should null the target."""
        data = np.ones((10, 10), dtype=np.float32) * 100
        mask_arr = np.ones((10, 10), dtype=np.uint8)
        mask_arr[0:3, 0:3] = 0  # mask the top-left corner

        gt = TerraTiff.from_array(
            data, origin_x=0, origin_y=0,
            pixel_width=1, pixel_height=-1, crs="UTM:33N",
        )
        mask_gt = TerraTiff.from_array(
            mask_arr, origin_x=0, origin_y=0,
            pixel_width=1, pixel_height=-1, crs="UTM:33N",
        )

        result = gt.mask_with_raster(mask_gt, nodata=-1)
        assert result.data[0, 0, 0] == -1    # masked
        assert result.data[0, 5, 5] == 100    # preserved

    def test_raster_mask_nodata_propagation(self):
        """Mask pixels equal to mask.nodata should also be masked."""
        data = np.full((10, 10), 42.0, dtype=np.float32)
        mask_arr = np.ones((10, 10), dtype=np.float32) * 5
        mask_arr[4:6, 4:6] = -9999  # mask nodata region

        gt = TerraTiff.from_array(
            data, origin_x=0, origin_y=0,
            pixel_width=1, pixel_height=-1, crs="UTM:33N",
        )
        mask_gt = TerraTiff.from_array(
            mask_arr, origin_x=0, origin_y=0,
            pixel_width=1, pixel_height=-1, crs="UTM:33N",
            nodata=-9999,
        )

        result = gt.mask_with_raster(mask_gt, nodata=0)
        assert result.data[0, 4, 4] == 0
        assert result.data[0, 0, 0] == 42

    def test_raster_mask_shape_mismatch(self):
        gt = TerraTiff.from_array(
            np.zeros((10, 10)), origin_x=0, origin_y=0,
            pixel_width=1, pixel_height=-1, crs="UTM:33N",
        )
        mask_gt = TerraTiff.from_array(
            np.zeros((5, 5)), origin_x=0, origin_y=0,
            pixel_width=1, pixel_height=-1, crs="UTM:33N",
        )
        with pytest.raises(ValueError, match="does not match"):
            gt.mask_with_raster(mask_gt)

    def test_raster_mask_multiband(self):
        data = np.ones((3, 10, 10), dtype=np.float32) * 7
        mask_arr = np.ones((10, 10), dtype=np.uint8)
        mask_arr[8:, 8:] = 0

        gt = TerraTiff.from_array(
            data, origin_x=0, origin_y=0,
            pixel_width=1, pixel_height=-1, crs="UTM:33N",
        )
        mask_gt = TerraTiff.from_array(
            mask_arr, origin_x=0, origin_y=0,
            pixel_width=1, pixel_height=-1, crs="UTM:33N",
        )
        result = gt.mask_with_raster(mask_gt, nodata=0)
        for b in range(3):
            assert result.data[b, 9, 9] == 0
            assert result.data[b, 0, 0] == 7


# ═══════════════════════════════════════════════════════════════════════════
# Repr & copy
# ═══════════════════════════════════════════════════════════════════════════

class TestMisc:
    def test_repr(self):
        gt = TerraTiff.from_array(
            np.zeros((5, 5)), origin_x=0, origin_y=0,
            pixel_width=1, pixel_height=-1, crs="WGS84",
        )
        r = repr(gt)
        assert "TerraTiff" in r
        assert "EPSG:4326" in r

    def test_copy(self):
        gt = TerraTiff.from_array(
            np.ones((5, 5)), origin_x=0, origin_y=0,
            pixel_width=1, pixel_height=-1, crs="WGS84",
        )
        gt2 = gt.copy()
        gt2.data[0, 0, 0] = 999
        assert gt.data[0, 0, 0] != 999  # independent copy

    def test_get_bounds(self):
        gt = TerraTiff.from_array(
            np.zeros((10, 20)), origin_x=100, origin_y=200,
            pixel_width=5, pixel_height=-5, crs="WGS84",
        )
        xmin, ymin, xmax, ymax = gt.get_bounds()
        assert xmin == 100
        assert xmax == 200  # 100 + 5*20
        assert ymin == 150  # 200 + (-5)*10
        assert ymax == 200

    def test_get_transform(self):
        gt = TerraTiff.from_array(
            np.zeros((10, 20)), origin_x=100, origin_y=200,
            pixel_width=5, pixel_height=-5, crs="WGS84",
        )
        ox, oy, pw, ph = gt.get_transform()
        assert ox == 100
        assert oy == 200
        assert pw == 5
        assert ph == -5
