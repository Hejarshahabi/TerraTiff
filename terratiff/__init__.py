"""
TerraTiff — A lightweight, GDAL-free Python package for GeoTIFF rasters.

Quick start::

    from terratiff import TerraTiff

    # Read
    raster = TerraTiff.open("elevation.tif")

    # Create from array
    raster = TerraTiff.from_array(my_array, origin_x=0, origin_y=0,
                                 pixel_width=30, pixel_height=-30,
                                 crs="UTM:33N")

    # Save
    raster.save("output.tif", dtype="float32")
"""

from terratiff.terratiff import TerraTiff
from terratiff.crs import CRSInfo, parse_crs, utm_zone_from_latlon, utm_epsg
from terratiff.dtypes import supported_dtypes

__all__ = [
    "TerraTiff",
    "CRSInfo",
    "parse_crs",
    "utm_zone_from_latlon",
    "utm_epsg",
    "supported_dtypes",
]

__version__ = "0.1.0"
