"""
Core TerraTiff class — read, write, and export GeoTIFF rasters.

This module provides the main user-facing API.  It uses ``tifffile`` for
TIFF I/O and embeds geospatial metadata via standard GeoTIFF tags (no GDAL).
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from terratiff.crs import (
    CRSInfo,
    crs_to_terratiff_tags,
    geokeys_from_page,
    parse_crs,
    utm_zone_from_latlon,
)
from terratiff.dtypes import cast_array, dtype_name, is_binary, resolve_dtype
from terratiff.utils import (
    TAG_GDAL_NODATA,
    TAG_MODEL_PIXEL_SCALE,
    TAG_MODEL_TIEPOINT,
    build_pixel_scale,
    build_tiepoint,
    compute_bounds,
    coord_to_pixel,
    ensure_3d,
    parse_pixel_scale,
    parse_tiepoint,
    rasterize_polygon,
    resample as _resample_dispatch,
)


class TerraTiff:
    """
    A lightweight, GDAL-free GeoTIFF container.

    Attributes
    ----------
    data : numpy.ndarray
        Raster data in ``(bands, rows, cols)`` layout.
    crs_info : CRSInfo
        Coordinate reference system.
    origin_x, origin_y : float
        Map coordinates of the upper-left pixel corner.
    pixel_width : float
        Pixel size in the X (east) direction (positive).
    pixel_height : float
        Pixel size in the Y (north) direction (typically negative for
        north-up rasters).
    nodata : float or int or None
        NoData sentinel value.
    """

    # ── Construction ────────────────────────────────────────────────────

    def __init__(
        self,
        data: np.ndarray,
        crs: str | int | CRSInfo,
        origin_x: float,
        origin_y: float,
        pixel_width: float,
        pixel_height: float,
        nodata: float | int | None = None,
        first_band: bool = True,
    ) -> None:
        self.data: np.ndarray = ensure_3d(np.asarray(data), first_band=first_band)
        self.crs_info: CRSInfo = crs if isinstance(crs, CRSInfo) else parse_crs(crs)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.pixel_width = float(pixel_width)
        self.pixel_height = float(pixel_height)
        self.nodata = nodata

    # ── Convenience properties ──────────────────────────────────────────

    @property
    def crs(self) -> str:
        """CRS as ``'EPSG:NNNNN'`` string."""
        return f"EPSG:{self.crs_info.epsg}"

    @property
    def shape(self) -> tuple[int, int, int]:
        """``(bands, rows, cols)``."""
        return self.data.shape  # type: ignore[return-value]

    @property
    def bands(self) -> int:
        return self.data.shape[0]

    @property
    def rows(self) -> int:
        return self.data.shape[1]

    @property
    def cols(self) -> int:
        return self.data.shape[2]

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    # ── Class-method constructors ───────────────────────────────────────

    @classmethod
    def open(cls, filepath: str | Path, first_band: bool = True) -> "TerraTiff":
        """
        Read a GeoTIFF file and return a :class:`TerraTiff` instance.

        Parameters
        ----------
        filepath : str or Path
            Path to a ``.tif`` / ``.tiff`` file.

        Raises
        ------
        FileNotFoundError
            If *filepath* does not exist.
        ValueError
            If no spatial metadata (tiepoint / pixel-scale) is found.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(filepath)

        with tifffile.TiffFile(str(filepath)) as tif:
            page = tif.pages[0]

            # ── Data ────────────────────────────────────────────────────
            data = tif.asarray()

            # ── Pixel scale ─────────────────────────────────────────────
            ps_tag = page.tags.get(TAG_MODEL_PIXEL_SCALE)
            if ps_tag is not None:
                pixel_width, pixel_height = parse_pixel_scale(ps_tag.value)
            else:
                pixel_width, pixel_height = 1.0, -1.0

            # ── Tiepoint ────────────────────────────────────────────────
            tp_tag = page.tags.get(TAG_MODEL_TIEPOINT)
            if tp_tag is not None:
                origin_x, origin_y = parse_tiepoint(tp_tag.value)
            else:
                origin_x, origin_y = 0.0, 0.0

            # ── CRS from GeoKeys ────────────────────────────────────────
            crs = geokeys_from_page(page.tags)
            if crs is None:
                crs = parse_crs("WGS84")  # safe default

            # ── NoData (GDAL convention: tag 42113) ─────────────────────
            nodata_tag = page.tags.get(TAG_GDAL_NODATA)
            nodata: float | int | None = None
            if nodata_tag is not None:
                try:
                    raw = nodata_tag.value
                    if isinstance(raw, bytes):
                        raw = raw.decode()
                    nodata = float(str(raw).strip())
                except (ValueError, TypeError):
                    nodata = None

        return cls(
            data=data,
            crs=crs,
            origin_x=origin_x,
            origin_y=origin_y,
            pixel_width=pixel_width,
            pixel_height=pixel_height,
            nodata=nodata,
            first_band=first_band,
        )

    @classmethod
    def from_array(
        cls,
        array: np.ndarray,
        origin_x: float,
        origin_y: float,
        pixel_width: float,
        pixel_height: float,
        crs: str | int = "WGS84",
        nodata: float | int | None = None,
        first_band: bool = True,
    ) -> "TerraTiff":
        """
        Create a :class:`TerraTiff` from a plain numpy array.

        This is the primary entry point for exporting non-spatial data to a
        georeferenced TIFF.  The caller supplies the map origin, pixel size,
        and CRS; the array is treated as the raster grid.

        Parameters
        ----------
        array : numpy.ndarray
            2-D ``(rows, cols)`` for single-band, or 3-D ``(bands, rows, cols)``.
        origin_x, origin_y : float
            Map coordinates of the upper-left pixel corner.
        pixel_width : float
            Pixel size in X direction.
        pixel_height : float
            Pixel size in Y direction (use **negative** for north-up rasters).
        crs : str or int
            ``"WGS84"``, ``"EPSG:32633"``, ``"UTM:33N"``, or int EPSG code.
        nodata : float or int, optional
            NoData sentinel.
        first_band : bool, optional
            If True (default), the array is expected to be ``(bands, rows, cols)``.
            If False, it is transposed from ``(rows, cols, bands)`` to ``(bands, rows, cols)``.
        """
        return cls(
            data=array,
            crs=crs,
            origin_x=origin_x,
            origin_y=origin_y,
            pixel_width=pixel_width,
            pixel_height=pixel_height,
            nodata=nodata,
            first_band=first_band,
        )

    # ── Saving / exporting ──────────────────────────────────────────────

    def save(
        self,
        filepath: str | Path,
        dtype: str | None = None,
    ) -> None:
        """
        Write the raster to a GeoTIFF file.

        Parameters
        ----------
        filepath : str or Path
            Destination path (will be created / overwritten).
        dtype : str or None
            Target data type: ``"uint8"``, ``"int16"``, ``"float32"``,
            ``"float64"``, ``"binary"``, etc.  Defaults to the current
            array dtype.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        out_data = cast_array(self.data, dtype)

        # ── Build GeoTIFF extratags ─────────────────────────────────────
        extratags: list[tuple] = []

        # Pixel scale
        ps = build_pixel_scale(self.pixel_width, self.pixel_height)
        extratags.append((TAG_MODEL_PIXEL_SCALE, "d", 3, ps, True))

        # Tiepoint
        tp = build_tiepoint(self.origin_x, self.origin_y)
        extratags.append((TAG_MODEL_TIEPOINT, "d", 6, tp, True))

        # CRS GeoKeys
        extratags.extend(crs_to_terratiff_tags(self.crs_info))

        # NoData (GDAL-style ASCII tag)
        if self.nodata is not None:
            nd_str = str(self.nodata)
            extratags.append(
                (TAG_GDAL_NODATA, "s", 0, nd_str, True)
            )

        # ── Write ───────────────────────────────────────────────────────
        # tifffile expects (samples, rows, cols) for multi-band with
        # photometric='minisblack'.  We write each band as a separate page
        # when band count > 1 for maximum compatibility, or as a single
        # interleaved image.

        if out_data.shape[0] == 1:
            # Single band — squeeze to 2-D
            tifffile.imwrite(
                str(filepath),
                out_data[0],
                photometric="minisblack",
                extratags=extratags,
            )
        else:
            # Multi-band — write as (bands, rows, cols)
            tifffile.imwrite(
                str(filepath),
                out_data,
                photometric="minisblack",
                planarconfig="SEPARATE",
                extratags=extratags,
            )

    # ── Band access ─────────────────────────────────────────────────────

    def get_band(self, index: int) -> np.ndarray:
        """
        Return band *index* as a 2-D ``(rows, cols)`` array (0-based).
        """
        if index < 0 or index >= self.bands:
            raise IndexError(
                f"Band index {index} out of range [0, {self.bands})."
            )
        return self.data[index]

    def add_band(self, array: np.ndarray) -> None:
        """
        Append a 2-D ``(rows, cols)`` array as a new band.
        """
        arr = np.asarray(array)
        if arr.ndim != 2:
            raise ValueError("Expected a 2-D array for a single band.")
        if arr.shape != (self.rows, self.cols):
            raise ValueError(
                f"Shape mismatch: expected ({self.rows}, {self.cols}), "
                f"got {arr.shape}."
            )
        self.data = np.concatenate(
            [self.data, arr[np.newaxis, :, :]], axis=0
        )

    # ── Spatial queries ─────────────────────────────────────────────────

    def get_bounds(self) -> tuple[float, float, float, float]:
        """Return ``(xmin, ymin, xmax, ymax)`` in the raster's CRS units."""
        return compute_bounds(
            self.origin_x, self.origin_y,
            self.pixel_width, self.pixel_height,
            self.cols, self.rows,
        )

    def get_transform(self) -> tuple[float, float, float, float]:
        """Return ``(origin_x, origin_y, pixel_width, pixel_height)``."""
        return (self.origin_x, self.origin_y, self.pixel_width, self.pixel_height)

    # ── Resampling ──────────────────────────────────────────────────────

    def resample(
        self,
        pixel_width: float,
        pixel_height: float,
        method: str = "nearest",
    ) -> "TerraTiff":
        """
        Return a new :class:`TerraTiff` resampled to the given pixel size.

        The geographic extent (bounds) is preserved; the number of
        rows/cols changes.

        Parameters
        ----------
        pixel_width : float
            New pixel width (positive).
        pixel_height : float
            New pixel height (negative for north-up).
        method : str
            Resampling algorithm:

            * ``"nearest"``  — nearest-neighbour (fast, good for categorical)
            * ``"bilinear"`` — 2×2 weighted average (smooth, good for continuous)
            * ``"cubic"``    — 4×4 Catmull-Rom (sharp, good for imagery)
            * ``"average"``  — block-mean (best for downsampling)
        """
        # Use absolute values for size computation
        abs_pw = abs(pixel_width)
        abs_ph = abs(pixel_height)
        abs_cur_pw = abs(self.pixel_width)
        abs_cur_ph = abs(self.pixel_height)

        width_extent = abs_cur_pw * self.cols
        height_extent = abs_cur_ph * self.rows

        new_cols = max(1, int(round(width_extent / abs_pw)))
        new_rows = max(1, int(round(height_extent / abs_ph)))

        new_data = _resample_dispatch(self.data, new_rows, new_cols, method)

        # Preserve sign convention
        new_ph = -abs_ph if self.pixel_height < 0 else abs_ph

        return TerraTiff(
            data=new_data,
            crs=self.crs_info,
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            pixel_width=abs_pw,
            pixel_height=new_ph,
            nodata=self.nodata,
        )

    # ── CRS conversion (metadata-only) ──────────────────────────────────

    def to_crs(self, target_crs: str | int) -> "TerraTiff":
        """
        Return a new :class:`TerraTiff` with the origin reprojected to
        *target_crs*.

        .. note::

           This performs a **metadata-only** transformation — the pixel
           grid is not warped.  Use this when the data is already on the
           target grid and you only need to update the CRS tag, **or**
           when converting origin coordinates between WGS 84 ↔ UTM.

        Parameters
        ----------
        target_crs : str or int
            Target CRS in any format accepted by :func:`parse_crs`.
        """
        from pyproj import Transformer

        new_crs = parse_crs(target_crs)

        transformer = Transformer.from_crs(
            f"EPSG:{self.crs_info.epsg}",
            f"EPSG:{new_crs.epsg}",
            always_xy=True,
        )

        # Transform the origin
        new_x, new_y = transformer.transform(self.origin_x, self.origin_y)

        # Also transform a point one pixel away to compute new scale
        x1, y1 = transformer.transform(
            self.origin_x + self.pixel_width,
            self.origin_y + self.pixel_height,
        )
        new_pw = x1 - new_x
        new_ph = y1 - new_y

        return TerraTiff(
            data=self.data.copy(),
            crs=new_crs,
            origin_x=new_x,
            origin_y=new_y,
            pixel_width=new_pw,
            pixel_height=new_ph,
            nodata=self.nodata,
        )

    # ── Masking & clipping ───────────────────────────────────────────────

    def clip(self, xmin: float, ymin: float, xmax: float, ymax: float) -> "TerraTiff":
        """
        Crop the raster to a bounding-box extent.

        Returns a new :class:`TerraTiff` whose spatial extent and data array
        are trimmed to the intersection of the raster and the supplied box.

        Parameters
        ----------
        xmin, ymin, xmax, ymax : float
            Target extent in the raster's CRS units.

        Raises
        ------
        ValueError
            If the requested extent does not overlap the raster.
        """
        # Current bounds
        cur_xmin, cur_ymin, cur_xmax, cur_ymax = self.get_bounds()

        # Clamp to intersection
        ixmin = max(xmin, cur_xmin)
        iymin = max(ymin, cur_ymin)
        ixmax = min(xmax, cur_xmax)
        iymax = min(ymax, cur_ymax)

        if ixmin >= ixmax or iymin >= iymax:
            raise ValueError(
                f"Clip extent ({xmin}, {ymin}, {xmax}, {ymax}) does not "
                f"overlap raster extent ({cur_xmin}, {cur_ymin}, "
                f"{cur_xmax}, {cur_ymax})."
            )

        # Convert intersection bounds to pixel indices
        # For north-up rasters (pixel_height < 0):
        #   row 0 = top (origin_y), last row = bottom
        pw = self.pixel_width
        ph = self.pixel_height  # typically negative

        # Columns: origin_x is the left edge
        col_start = int(np.floor((ixmin - self.origin_x) / pw))
        col_end   = int(np.ceil((ixmax - self.origin_x) / pw))

        # Rows: origin_y is the top edge (for north-up)
        if ph < 0:
            row_start = int(np.floor((iymax - self.origin_y) / ph))
            row_end   = int(np.ceil((iymin - self.origin_y) / ph))
        else:
            row_start = int(np.floor((iymin - self.origin_y) / ph))
            row_end   = int(np.ceil((iymax - self.origin_y) / ph))

        # Clamp to valid pixel range
        col_start = max(0, col_start)
        col_end   = min(self.cols, col_end)
        row_start = max(0, row_start)
        row_end   = min(self.rows, row_end)

        if row_start >= row_end or col_start >= col_end:
            raise ValueError("Clip produced an empty raster.")

        new_data = self.data[:, row_start:row_end, col_start:col_end].copy()

        new_origin_x = self.origin_x + col_start * pw
        new_origin_y = self.origin_y + row_start * ph

        return TerraTiff(
            data=new_data,
            crs=self.crs_info,
            origin_x=new_origin_x,
            origin_y=new_origin_y,
            pixel_width=self.pixel_width,
            pixel_height=self.pixel_height,
            nodata=self.nodata,
        )

    def mask_with_polygon(
        self,
        polygon: list[tuple[float, float]],
        invert: bool = False,
        nodata: float | int | None = None,
    ) -> "TerraTiff":
        """
        Mask the raster using a polygon.

        Pixels **outside** the polygon are set to *nodata*.  If
        ``invert=True``, pixels **inside** are set to *nodata* instead.

        Parameters
        ----------
        polygon : list of (x, y) tuples
            Vertices of the polygon in the raster's CRS.  The polygon is
            automatically closed.
        invert : bool
            If ``True``, mask the *interior* of the polygon instead.
        nodata : float or int, optional
            Value to assign to masked pixels.  Defaults to the raster's
            existing NoData value, or ``0`` if none is set.

        Returns
        -------
        TerraTiff
            A new raster with masked pixels set to *nodata*.
        """
        nd = nodata if nodata is not None else (self.nodata if self.nodata is not None else 0)

        inside = rasterize_polygon(
            polygon,
            self.rows, self.cols,
            self.origin_x, self.origin_y,
            self.pixel_width, self.pixel_height,
        )

        keep = ~inside if invert else inside
        new_data = self.data.copy()
        for b in range(self.bands):
            new_data[b][~keep] = nd

        return TerraTiff(
            data=new_data,
            crs=self.crs_info,
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            pixel_width=self.pixel_width,
            pixel_height=self.pixel_height,
            nodata=nd,
        )

    def mask_with_raster(
        self,
        mask: "TerraTiff",
        nodata: float | int | None = None,
    ) -> "TerraTiff":
        """
        Mask the raster using another raster as a mask layer.

        Pixels where the *mask* raster equals ``0`` (or its own NoData
        value) are set to *nodata* in the output.

        Parameters
        ----------
        mask : TerraTiff
            A single-band raster aligned to the same grid.  Non-zero
            pixels indicate **valid** areas.
        nodata : float or int, optional
            Value for masked pixels.  Defaults to the raster's existing
            NoData, or ``0``.

        Returns
        -------
        TerraTiff

        Raises
        ------
        ValueError
            If the mask grid dimensions do not match.
        """
        if mask.rows != self.rows or mask.cols != self.cols:
            raise ValueError(
                f"Mask shape ({mask.rows}, {mask.cols}) does not match "
                f"raster shape ({self.rows}, {self.cols}).  Resample or "
                f"clip the mask to the same grid first."
            )

        nd = nodata if nodata is not None else (self.nodata if self.nodata is not None else 0)

        # Build a boolean mask: True = pixel is valid
        mask_band = mask.data[0]
        valid = mask_band != 0
        if mask.nodata is not None:
            valid = valid & (mask_band != mask.nodata)

        new_data = self.data.copy()
        for b in range(self.bands):
            new_data[b][~valid] = nd

        return TerraTiff(
            data=new_data,
            crs=self.crs_info,
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            pixel_width=self.pixel_width,
            pixel_height=self.pixel_height,
            nodata=nd,
        )

    # ── Dunder helpers ──────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"TerraTiff(shape={self.shape}, dtype={self.dtype}, "
            f"crs={self.crs}, origin=({self.origin_x}, {self.origin_y}), "
            f"pixel=({self.pixel_width}, {self.pixel_height}))"
        )

    def copy(self) -> "TerraTiff":
        """Return a deep copy."""
        return TerraTiff(
            data=self.data.copy(),
            crs=self.crs_info,
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            pixel_width=self.pixel_width,
            pixel_height=self.pixel_height,
            nodata=self.nodata,
        )
