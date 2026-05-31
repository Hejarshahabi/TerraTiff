"""
Shared utility functions for the TerraTiff package.

Handles ModelTiepointTag / ModelPixelScaleTag construction and parsing,
coordinate-bounds computation, and nearest-neighbour resampling.
"""

from __future__ import annotations

import numpy as np


# ── GeoTIFF tag IDs ─────────────────────────────────────────────────────────
TAG_MODEL_PIXEL_SCALE  = 33550
TAG_MODEL_TIEPOINT     = 33922
TAG_GEO_KEY_DIRECTORY  = 34735
TAG_GEO_DOUBLE_PARAMS  = 34736
TAG_GEO_ASCII_PARAMS   = 34737
TAG_GDAL_NODATA        = 42113


# ── ModelTiepointTag helpers ────────────────────────────────────────────────

def build_tiepoint(origin_x: float, origin_y: float) -> tuple[float, ...]:
    """
    Create a ``ModelTiepointTag`` value.

    The tiepoint maps pixel (0, 0) → model coordinate (origin_x, origin_y).
    Format: ``(I, J, K, X, Y, Z)``
    """
    return (0.0, 0.0, 0.0, float(origin_x), float(origin_y), 0.0)


def parse_tiepoint(tag_value: tuple | list) -> tuple[float, float]:
    """
    Extract the map origin ``(x, y)`` from a raw ``ModelTiepointTag`` value.

    Parameters
    ----------
    tag_value : sequence
        At least 6 elements: ``(I, J, K, X, Y, Z)``.

    Returns
    -------
    (origin_x, origin_y)
    """
    return (float(tag_value[3]), float(tag_value[4]))


# ── ModelPixelScaleTag helpers ──────────────────────────────────────────────

def build_pixel_scale(pixel_width: float, pixel_height: float) -> tuple[float, ...]:
    """
    Create a ``ModelPixelScaleTag`` value.

    ``pixel_height`` is stored as its absolute value (the tag always uses
    positive numbers; the sign convention comes from the tiepoint).

    Format: ``(ScaleX, ScaleY, ScaleZ)``
    """
    return (abs(float(pixel_width)), abs(float(pixel_height)), 0.0)


def parse_pixel_scale(tag_value: tuple | list) -> tuple[float, float]:
    """
    Extract ``(pixel_width, pixel_height)`` from a raw ``ModelPixelScaleTag``.

    By GeoTIFF convention the Y scale is returned as a **negative** value
    (north-up rasters).
    """
    px_w = float(tag_value[0])
    px_h = -abs(float(tag_value[1]))
    return (px_w, px_h)


# ── Bounds computation ──────────────────────────────────────────────────────

def compute_bounds(
    origin_x: float,
    origin_y: float,
    pixel_width: float,
    pixel_height: float,
    cols: int,
    rows: int,
) -> tuple[float, float, float, float]:
    """
    Return ``(xmin, ymin, xmax, ymax)`` for the raster extent.
    """
    x2 = origin_x + pixel_width * cols
    y2 = origin_y + pixel_height * rows  # pixel_height is typically negative

    xmin = min(origin_x, x2)
    xmax = max(origin_x, x2)
    ymin = min(origin_y, y2)
    ymax = max(origin_y, y2)
    return (xmin, ymin, xmax, ymax)


# ── Nearest-neighbour resampling ────────────────────────────────────────────

def nearest_resample(
    data: np.ndarray,
    dst_rows: int,
    dst_cols: int,
) -> np.ndarray:
    """
    Resample *data* to ``(dst_rows, dst_cols)`` using nearest-neighbour.

    Parameters
    ----------
    data : numpy.ndarray
        2-D ``(rows, cols)`` or 3-D ``(bands, rows, cols)``.
    dst_rows, dst_cols : int
        Target dimensions.

    Returns
    -------
    numpy.ndarray
        Resampled array with the same dtype and band count.
    """
    if data.ndim == 2:
        src_rows, src_cols = data.shape
        row_idx = (np.arange(dst_rows) * src_rows / dst_rows).astype(int)
        col_idx = (np.arange(dst_cols) * src_cols / dst_cols).astype(int)
        row_idx = np.clip(row_idx, 0, src_rows - 1)
        col_idx = np.clip(col_idx, 0, src_cols - 1)
        return data[np.ix_(row_idx, col_idx)]

    if data.ndim == 3:
        bands, src_rows, src_cols = data.shape
        row_idx = (np.arange(dst_rows) * src_rows / dst_rows).astype(int)
        col_idx = (np.arange(dst_cols) * src_cols / dst_cols).astype(int)
        row_idx = np.clip(row_idx, 0, src_rows - 1)
        col_idx = np.clip(col_idx, 0, src_cols - 1)
        return data[:, np.ix_(row_idx, col_idx)[0], np.ix_(row_idx, col_idx)[1]]

    raise ValueError(f"Expected 2-D or 3-D array, got {data.ndim}-D")


def ensure_3d(data: np.ndarray) -> np.ndarray:
    """
    Normalise *data* to 3-D ``(bands, rows, cols)``.

    If *data* is 2-D it is expanded along axis 0 (single band).
    """
    if data.ndim == 2:
        return data[np.newaxis, :, :]
    if data.ndim == 3:
        return data
    raise ValueError(f"Expected 2-D or 3-D array, got {data.ndim}-D")


# ── Bilinear resampling ─────────────────────────────────────────────────

def _bilinear_2d(data: np.ndarray, dst_rows: int, dst_cols: int) -> np.ndarray:
    """Bilinear interpolation for a single 2-D array."""
    src_rows, src_cols = data.shape
    # Map destination pixel centres to source coordinates
    row_coords = (np.arange(dst_rows) + 0.5) * src_rows / dst_rows - 0.5
    col_coords = (np.arange(dst_cols) + 0.5) * src_cols / dst_cols - 0.5

    r0 = np.floor(row_coords).astype(int)
    c0 = np.floor(col_coords).astype(int)
    dr = (row_coords - r0).astype(np.float64)
    dc = (col_coords - c0).astype(np.float64)

    r0 = np.clip(r0, 0, src_rows - 2)
    c0 = np.clip(c0, 0, src_cols - 2)
    r1 = r0 + 1
    c1 = c0 + 1

    # Bilinear weights: (1-dr)(1-dc), (1-dr)(dc), (dr)(1-dc), (dr)(dc)
    src = data.astype(np.float64)
    w00 = np.outer(1 - dr, 1 - dc)
    w01 = np.outer(1 - dr, dc)
    w10 = np.outer(dr, 1 - dc)
    w11 = np.outer(dr, dc)

    result = (
        src[np.ix_(r0, c0)] * w00 +
        src[np.ix_(r0, c1)] * w01 +
        src[np.ix_(r1, c0)] * w10 +
        src[np.ix_(r1, c1)] * w11
    )
    return result.astype(data.dtype)


def bilinear_resample(
    data: np.ndarray,
    dst_rows: int,
    dst_cols: int,
) -> np.ndarray:
    """
    Resample *data* using **bilinear** interpolation.

    Good for continuous surfaces (elevation, temperature) where smooth
    transitions between pixels are desired.

    Parameters
    ----------
    data : numpy.ndarray
        2-D ``(rows, cols)`` or 3-D ``(bands, rows, cols)``.
    dst_rows, dst_cols : int
        Target dimensions.
    """
    if data.ndim == 2:
        return _bilinear_2d(data, dst_rows, dst_cols)
    if data.ndim == 3:
        return np.stack(
            [_bilinear_2d(data[b], dst_rows, dst_cols) for b in range(data.shape[0])],
            axis=0,
        )
    raise ValueError(f"Expected 2-D or 3-D array, got {data.ndim}-D")


# ── Cubic (Catmull-Rom) resampling ──────────────────────────────────────

def _cubic_weight(t: np.ndarray) -> np.ndarray:
    """Catmull-Rom (Keys) cubic kernel, a = -0.5."""
    t = np.abs(t)
    out = np.zeros_like(t, dtype=np.float64)

    mask1 = t <= 1.0
    mask2 = (t > 1.0) & (t <= 2.0)

    out[mask1] = (1.5 * t[mask1] ** 3 - 2.5 * t[mask1] ** 2 + 1.0)
    out[mask2] = (-0.5 * t[mask2] ** 3 + 2.5 * t[mask2] ** 2
                  - 4.0 * t[mask2] + 2.0)
    return out


def _cubic_2d(data: np.ndarray, dst_rows: int, dst_cols: int) -> np.ndarray:
    """Cubic (Catmull-Rom) interpolation for a single 2-D array."""
    src_rows, src_cols = data.shape
    src = data.astype(np.float64)

    # Pad source by 2 pixels on each side to avoid boundary issues
    padded = np.pad(src, 2, mode="edge")

    row_coords = (np.arange(dst_rows) + 0.5) * src_rows / dst_rows - 0.5
    col_coords = (np.arange(dst_cols) + 0.5) * src_cols / dst_cols - 0.5

    # Integer part in padded coordinates (+2 for the padding offset)
    ri = np.floor(row_coords).astype(int) + 2
    ci = np.floor(col_coords).astype(int) + 2
    dr = row_coords - np.floor(row_coords)
    dc = col_coords - np.floor(col_coords)

    result = np.zeros((dst_rows, dst_cols), dtype=np.float64)

    for m in range(-1, 3):
        wr = _cubic_weight(dr - m)  # shape (dst_rows,)
        r_idx = np.clip(ri + m, 0, padded.shape[0] - 1)
        for n in range(-1, 3):
            wc = _cubic_weight(dc - n)  # shape (dst_cols,)
            c_idx = np.clip(ci + n, 0, padded.shape[1] - 1)
            result += np.outer(wr, wc) * padded[np.ix_(r_idx, c_idx)]

    return result.astype(data.dtype)


def cubic_resample(
    data: np.ndarray,
    dst_rows: int,
    dst_cols: int,
) -> np.ndarray:
    """
    Resample *data* using **cubic** (Catmull-Rom) interpolation.

    Produces sharper results than bilinear; best for photographic imagery
    and continuous surfaces where edge detail matters.

    Parameters
    ----------
    data : numpy.ndarray
        2-D ``(rows, cols)`` or 3-D ``(bands, rows, cols)``.
    dst_rows, dst_cols : int
        Target dimensions.
    """
    if data.ndim == 2:
        return _cubic_2d(data, dst_rows, dst_cols)
    if data.ndim == 3:
        return np.stack(
            [_cubic_2d(data[b], dst_rows, dst_cols) for b in range(data.shape[0])],
            axis=0,
        )
    raise ValueError(f"Expected 2-D or 3-D array, got {data.ndim}-D")


# ── Average (block-mean) resampling ─────────────────────────────────────

def _average_2d(data: np.ndarray, dst_rows: int, dst_cols: int) -> np.ndarray:
    """Block-mean downsampling / averaging for a single 2-D array."""
    src_rows, src_cols = data.shape
    src = data.astype(np.float64)

    # Row / col boundaries in source space for each destination pixel
    row_edges = np.linspace(0, src_rows, dst_rows + 1)
    col_edges = np.linspace(0, src_cols, dst_cols + 1)

    result = np.zeros((dst_rows, dst_cols), dtype=np.float64)

    for i in range(dst_rows):
        r_start = int(np.floor(row_edges[i]))
        r_end   = int(np.ceil(row_edges[i + 1]))
        r_start = max(0, r_start)
        r_end   = min(src_rows, max(r_end, r_start + 1))

        for j in range(dst_cols):
            c_start = int(np.floor(col_edges[j]))
            c_end   = int(np.ceil(col_edges[j + 1]))
            c_start = max(0, c_start)
            c_end   = min(src_cols, max(c_end, c_start + 1))

            result[i, j] = src[r_start:r_end, c_start:c_end].mean()

    return result.astype(data.dtype)


def average_resample(
    data: np.ndarray,
    dst_rows: int,
    dst_cols: int,
) -> np.ndarray:
    """
    Resample *data* using **average** (block-mean) aggregation.

    Ideal for **downsampling** — each output pixel is the mean of the
    source pixels it covers.  Avoids the aliasing artifacts that
    nearest-neighbour can produce when reducing resolution.

    Parameters
    ----------
    data : numpy.ndarray
        2-D ``(rows, cols)`` or 3-D ``(bands, rows, cols)``.
    dst_rows, dst_cols : int
        Target dimensions.
    """
    if data.ndim == 2:
        return _average_2d(data, dst_rows, dst_cols)
    if data.ndim == 3:
        return np.stack(
            [_average_2d(data[b], dst_rows, dst_cols) for b in range(data.shape[0])],
            axis=0,
        )
    raise ValueError(f"Expected 2-D or 3-D array, got {data.ndim}-D")


# ── Resample dispatcher ────────────────────────────────────────────────

RESAMPLE_METHODS: dict[str, callable] = {
    "nearest":  nearest_resample,
    "bilinear": bilinear_resample,
    "cubic":    cubic_resample,
    "average":  average_resample,
}


def resample(
    data: np.ndarray,
    dst_rows: int,
    dst_cols: int,
    method: str = "nearest",
) -> np.ndarray:
    """
    Resample *data* using the specified *method*.

    Parameters
    ----------
    data : numpy.ndarray
        2-D or 3-D array.
    dst_rows, dst_cols : int
        Target dimensions.
    method : str
        ``"nearest"`` | ``"bilinear"`` | ``"cubic"`` | ``"average"``.
    """
    key = method.strip().lower()
    func = RESAMPLE_METHODS.get(key)
    if func is None:
        raise ValueError(
            f"Unknown resampling method '{method}'. "
            f"Choose from: {', '.join(RESAMPLE_METHODS)}"
        )
    return func(data, dst_rows, dst_cols)


# ── Coordinate ↔ pixel helpers ──────────────────────────────────────────

def coord_to_pixel(
    x: float, y: float,
    origin_x: float, origin_y: float,
    pixel_width: float, pixel_height: float,
) -> tuple[int, int]:
    """
    Convert map coordinate ``(x, y)`` to pixel index ``(row, col)``.

    Returns the integer pixel that contains the coordinate.
    """
    col = int(np.floor((x - origin_x) / pixel_width))
    row = int(np.floor((y - origin_y) / pixel_height))
    return (row, col)


def pixel_to_coord(
    row: int, col: int,
    origin_x: float, origin_y: float,
    pixel_width: float, pixel_height: float,
) -> tuple[float, float]:
    """
    Convert pixel index ``(row, col)`` to the centre coordinate ``(x, y)``.
    """
    x = origin_x + (col + 0.5) * pixel_width
    y = origin_y + (row + 0.5) * pixel_height
    return (x, y)


# ── Polygon rasterisation (ray-casting) ─────────────────────────────────

def rasterize_polygon(
    polygon: list[tuple[float, float]],
    rows: int,
    cols: int,
    origin_x: float,
    origin_y: float,
    pixel_width: float,
    pixel_height: float,
) -> np.ndarray:
    """
    Rasterise a polygon into a boolean mask on the given grid.

    Uses the ray-casting (point-in-polygon) algorithm, vectorised with
    numpy for speed.

    Parameters
    ----------
    polygon : list of (x, y) tuples
        Vertices of the polygon in **map coordinates** (same CRS as the
        raster).  The polygon is auto-closed (last vertex connects back
        to the first).
    rows, cols : int
        Grid dimensions.
    origin_x, origin_y, pixel_width, pixel_height : float
        Spatial transform of the grid.

    Returns
    -------
    numpy.ndarray
        Boolean 2-D array of shape ``(rows, cols)``; ``True`` = inside.
    """
    # Pixel-centre coordinates for the full grid
    col_coords = origin_x + (np.arange(cols) + 0.5) * pixel_width
    row_coords = origin_y + (np.arange(rows) + 0.5) * pixel_height
    px, py = np.meshgrid(col_coords, row_coords)  # both (rows, cols)

    n = len(polygon)
    inside = np.zeros((rows, cols), dtype=bool)

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        # Does the ray from (px, py) → +x cross edge (i, j)?
        cond_y = (yi > py) != (yj > py)
        # X intercept of the edge at the scanline height py
        with np.errstate(divide="ignore", invalid="ignore"):
            x_intercept = (xj - xi) * (py - yi) / (yj - yi) + xi
        crossing = cond_y & (px < x_intercept)
        inside = inside ^ crossing

        j = i

    return inside
