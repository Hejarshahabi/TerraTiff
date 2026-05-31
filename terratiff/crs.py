"""
Coordinate Reference System (CRS) helpers for the TerraTiff package.

Translates between human-friendly CRS identifiers (``"WGS84"``, ``"UTM:33N"``,
``"EPSG:4326"``, or integer EPSG codes) and the low-level GeoTIFF GeoKey tag
entries required to write a valid GeoTIFF.

Also provides utilities for computing UTM zones from geographic coordinates.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any

from terratiff.utils import (
    TAG_GEO_KEY_DIRECTORY,
    TAG_GEO_DOUBLE_PARAMS,
    TAG_GEO_ASCII_PARAMS,
)

# ── GeoKey IDs (GeoTIFF spec §6) ───────────────────────────────────────────
_GT_MODEL_TYPE_GEOKEY       = 1024   # 1=Projected, 2=Geographic, 3=Geocentric
_GT_RASTER_TYPE_GEOKEY      = 1025   # 1=RasterPixelIsArea, 2=RasterPixelIsPoint
_GEODETIC_CRS_GEOKEY        = 2048   # GeographicTypeGeoKey (EPSG GCS code)
_PROJECTED_CRS_GEOKEY       = 3072   # ProjectedCSTypeGeoKey (EPSG PCS code)

_MODEL_TYPE_PROJECTED  = 1
_MODEL_TYPE_GEOGRAPHIC = 2

_RASTER_PIXEL_IS_AREA  = 1


# ── CRSInfo data class ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class CRSInfo:
    """Lightweight representation of a CRS for GeoTIFF purposes."""

    epsg: int
    """EPSG code (e.g. 4326, 32633)."""

    is_projected: bool
    """True for projected (e.g. UTM), False for geographic (e.g. WGS 84)."""

    name: str
    """Human-readable label, e.g. ``'WGS 84'`` or ``'WGS 84 / UTM zone 33N'``."""

    def __repr__(self) -> str:
        return f"CRSInfo(EPSG:{self.epsg}, {self.name})"


# ── Public helpers ──────────────────────────────────────────────────────────

def parse_crs(crs_input: str | int) -> CRSInfo:
    """
    Parse a user-supplied CRS identifier into a :class:`CRSInfo`.

    Accepted formats
    ----------------
    * ``"WGS84"`` or ``"wgs84"``  → EPSG:4326
    * ``"EPSG:4326"`` or ``4326`` → looked up
    * ``"UTM:33N"``               → EPSG:32633
    * ``"UTM:33S"``               → EPSG:32733

    Raises
    ------
    ValueError
        If the input cannot be resolved.
    """
    if isinstance(crs_input, int):
        return _from_epsg(crs_input)

    s = str(crs_input).strip()

    # ── WGS84 shortcut ──────────────────────────────────────────────────
    if s.upper() in ("WGS84", "WGS 84"):
        return CRSInfo(epsg=4326, is_projected=False, name="WGS 84")

    # ── EPSG:NNNNN ──────────────────────────────────────────────────────
    if s.upper().startswith("EPSG:"):
        code = int(s.split(":")[1])
        return _from_epsg(code)

    # ── UTM:ZoneHemisphere (e.g. UTM:33N, UTM:60S) ─────────────────────
    if s.upper().startswith("UTM:"):
        payload = s[4:].strip().upper()
        hemisphere = payload[-1]
        zone = int(payload[:-1])
        if hemisphere not in ("N", "S"):
            raise ValueError(
                f"UTM hemisphere must be 'N' or 'S', got '{hemisphere}'."
            )
        if not (1 <= zone <= 60):
            raise ValueError(f"UTM zone must be 1–60, got {zone}.")
        epsg = utm_epsg(zone, hemisphere)
        name = f"WGS 84 / UTM zone {zone}{hemisphere}"
        return CRSInfo(epsg=epsg, is_projected=True, name=name)

    raise ValueError(
        f"Cannot parse CRS '{crs_input}'. Use 'WGS84', 'EPSG:NNNNN', "
        f"or 'UTM:ZoneH' (e.g. 'UTM:33N')."
    )


def _from_epsg(code: int) -> CRSInfo:
    """Build a :class:`CRSInfo` from a raw EPSG code."""
    if code == 4326:
        return CRSInfo(epsg=4326, is_projected=False, name="WGS 84")

    # WGS 84 UTM North zones  32601 – 32660
    if 32601 <= code <= 32660:
        zone = code - 32600
        return CRSInfo(
            epsg=code, is_projected=True,
            name=f"WGS 84 / UTM zone {zone}N",
        )

    # WGS 84 UTM South zones  32701 – 32760
    if 32701 <= code <= 32760:
        zone = code - 32700
        return CRSInfo(
            epsg=code, is_projected=True,
            name=f"WGS 84 / UTM zone {zone}S",
        )

    # Generic fallback — mark as projected if code is in typical PCS range
    is_proj = code >= 10000
    return CRSInfo(epsg=code, is_projected=is_proj, name=f"EPSG:{code}")


# ── UTM utilities ───────────────────────────────────────────────────────────

def utm_zone_from_latlon(lat: float, lon: float) -> tuple[int, str]:
    """
    Return ``(zone_number, hemisphere)`` for a geographic coordinate.

    Parameters
    ----------
    lat, lon : float
        Latitude and longitude in degrees.

    Returns
    -------
    (zone, hemisphere) : (int, str)
        Zone 1–60, hemisphere ``'N'`` or ``'S'``.
    """
    # Standard 6° zone formula
    zone = int(math.floor((lon + 180.0) / 6.0)) % 60 + 1
    hemisphere = "N" if lat >= 0 else "S"
    return zone, hemisphere


def utm_epsg(zone: int, hemisphere: str) -> int:
    """
    Return the EPSG code for a WGS 84 UTM zone.

    Examples: zone 33 N → 32633, zone 33 S → 32733.
    """
    base = 32600 if hemisphere.upper() == "N" else 32700
    return base + zone


# ── GeoKey tag builders ─────────────────────────────────────────────────────

def crs_to_terratiff_tags(crs: CRSInfo) -> list[tuple[int, str, int, Any, bool]]:
    """
    Build the TIFF ``extratags`` list that encodes a CRS into a GeoTIFF.

    Returns a list of ``(tag_id, dtype_char, count, value, writeonce)`` tuples
    compatible with :func:`tifffile.imwrite`.
    """
    keys: list[int] = []

    # Key 1: GTModelTypeGeoKey
    model_type = _MODEL_TYPE_PROJECTED if crs.is_projected else _MODEL_TYPE_GEOGRAPHIC
    keys.extend([_GT_MODEL_TYPE_GEOKEY, 0, 1, model_type])

    # Key 2: GTRasterTypeGeoKey — always PixelIsArea
    keys.extend([_GT_RASTER_TYPE_GEOKEY, 0, 1, _RASTER_PIXEL_IS_AREA])

    if crs.is_projected:
        # Key 3: ProjectedCRSGeoKey
        keys.extend([_PROJECTED_CRS_GEOKEY, 0, 1, crs.epsg])
    else:
        # Key 3: GeodeticCRSGeoKey
        keys.extend([_GEODETIC_CRS_GEOKEY, 0, 1, crs.epsg])

    num_keys = len(keys) // 4

    # Header: version=1, revision=1, minor=0, numberOfKeys
    header = [1, 1, 0, num_keys]
    geo_key_directory = tuple(header + keys)

    return [
        (TAG_GEO_KEY_DIRECTORY, "H", len(geo_key_directory),
         geo_key_directory, True),
    ]


# ── GeoKey tag parsers ──────────────────────────────────────────────────────

def geokeys_from_page(tags: dict) -> CRSInfo | None:
    """
    Attempt to reconstruct a :class:`CRSInfo` from the TIFF-page tags.

    Parameters
    ----------
    tags : dict
        ``{tag_id: tag_object}`` mapping from a ``tifffile.TiffPage``.

    Returns
    -------
    CRSInfo or None
        ``None`` if no GeoKey directory is found.
    """
    dir_tag = tags.get(TAG_GEO_KEY_DIRECTORY)
    if dir_tag is None:
        return None

    values = dir_tag.value
    if len(values) < 4:
        return None

    num_keys = int(values[3])
    epsg: int | None = None
    is_projected: bool = False

    for i in range(num_keys):
        offset = 4 + i * 4
        if offset + 3 >= len(values):
            break
        key_id   = int(values[offset])
        _tag_loc = int(values[offset + 1])
        _count   = int(values[offset + 2])
        val      = int(values[offset + 3])

        if key_id == _GT_MODEL_TYPE_GEOKEY:
            is_projected = (val == _MODEL_TYPE_PROJECTED)
        elif key_id == _PROJECTED_CRS_GEOKEY:
            epsg = val
        elif key_id == _GEODETIC_CRS_GEOKEY:
            epsg = val

    if epsg is None:
        return None

    return _from_epsg(epsg)
