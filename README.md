# 🌍 TerraTiff

[![PyPI version](https://img.shields.io/badge/PyPi%20Package-0.1.11-green)](https://pypi.org/project/TerraTiff/) [![Downloads](https://pepy.tech/badge/terratiff)](https://pepy.tech/project/terratiff) [![Github](https://img.shields.io/badge/Github-TerraTiff-blueviolet)](https://github.com/Hejarshahabi/TerraTiff) [![LinkedIn](https://img.shields.io/badge/LinkedIn-Hejar%20Shahabi-blue)](https://www.linkedin.com/in/hejarshahabi/) [![Twitter URL](https://img.shields.io/twitter/url?color=blue&label=Hejar%20Shahabi&style=social&url=https%3A%2F%2Ftwitter.com%2Fhejarshahabi)](https://twitter.com/hejarshahabi)

![TerraTiff Logo](https://raw.githubusercontent.com/Hejarshahabi/TerraTiff/main/logo.png)

A lightweight, **GDAL-free** Python package for reading, writing, and exporting GeoTIFF raster files.

Built on `tifffile` + `numpy` + `pyproj` — **no GDAL installation required**.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Tutorial](#tutorial)
  - [1. Creating a GeoTIFF from a Numpy Array](#1-creating-a-geotiff-from-a-numpy-array)
  - [2. Reading an Existing GeoTIFF](#2-reading-an-existing-geotiff)
  - [3. Choosing a Coordinate Reference System (CRS)](#3-choosing-a-coordinate-reference-system-crs)
  - [4. Exporting with Different Data Types](#4-exporting-with-different-data-types)
  - [5. Working with Multi-Band Rasters](#5-working-with-multi-band-rasters)
  - [6. Band Operations](#6-band-operations)
  - [7. Binary Mask Export](#7-binary-mask-export)
  - [8. Changing Spatial Resolution (Resampling)](#8-changing-spatial-resolution-resampling)
  - [9. Clipping by Extent](#9-clipping-by-extent)
  - [10. Masking with a Polygon](#10-masking-with-a-polygon)
  - [11. Masking with a Raster](#11-masking-with-a-raster)
  - [12. Converting Between Coordinate Systems](#12-converting-between-coordinate-systems)
  - [13. Querying Spatial Metadata](#13-querying-spatial-metadata)
  - [14. Working with NoData Values](#14-working-with-nodata-values)
  - [15. UTM Zone Utilities](#15-utm-zone-utilities)
- [API Reference](#api-reference)
- [Supported CRS Formats](#supported-crs-formats)
- [Supported Data Types](#supported-data-types)
- [License](#license)

---

## Features

| Feature | Description |
|---------|-------------|
| 📖 **Read** | Load existing GeoTIFF files (single-band and multi-band) |
| 💾 **Write** | Export georeferenced TIFF files with proper GeoKeys |
| 🗺️ **CRS** | WGS 84, all 120 UTM zones (N/S), any EPSG code |
| 🔢 **Data Types** | uint8/16/32, int8/16/32, float16/32/64, binary |
| 📐 **Resample** | 4 methods: nearest, bilinear, cubic, average |
| ✂️ **Clip** | Crop rasters by bounding-box extent |
| 🔷 **Polygon Mask** | Mask rasters using polygon geometries |
| 🖼️ **Raster Mask** | Mask rasters using another raster file |
| 🔄 **CRS Convert** | Reproject coordinates between WGS 84 ↔ UTM |
| 🎚️ **Multi-Band** | RGB, multispectral, any number of bands |
| 🧩 **Array Export** | Turn any numpy array into a georeferenced TIFF |

---

## Installation

```bash
pip install terratiff
```

### Dependencies only

```bash
pip install numpy tifffile pyproj
```

> **Note:** None of these dependencies require GDAL. `pyproj` uses the standalone PROJ library.

---

## Tutorial

### 1. Creating a GeoTIFF from a Numpy Array

The most common use case: you have a numpy array (e.g. from a computation, a model output, or a CSV) and you want to save it as a georeferenced TIFF file.

```python
import numpy as np
from terratiff import TerraTiff

# Create some data — for example, a 100×200 elevation grid
elevation = np.random.rand(100, 200).astype(np.float32) * 500  # 0–500m

# Wrap it in a TerraTiff with spatial metadata
raster = TerraTiff.from_array(
    elevation,
    origin_x=500000,       # easting of the top-left corner (meters)
    origin_y=4500000,      # northing of the top-left corner (meters)
    pixel_width=30,        # 30 m pixel size in X
    pixel_height=-30,      # -30 m in Y (negative = north-up)
    crs="UTM:33N",         # UTM zone 33 North
)

# Save to disk
raster.save("elevation.tif", dtype="float32")
print("Saved! ✅")
```

**Key points:**
- `pixel_height` should be **negative** for standard north-up rasters
- `origin_x` / `origin_y` are the coordinates of the **top-left corner** of the top-left pixel
- The array can be 2-D (single band) or 3-D (multi-band)
- By default, 3-D arrays are expected to be in `(bands, rows, cols)` format. If your array is `(rows, cols, bands)` (e.g. from an image library), pass `first_band=False` to automatically transpose it.

---

### 2. Reading an Existing GeoTIFF

```python
from terratiff import TerraTiff

# Open and read a GeoTIFF file
# Use first_band=False if the TIFF is pixel-interleaved (rows, cols, bands)
raster = TerraTiff.open("elevation.tif", first_band=True)

# Inspect the metadata
print(f"Shape:        {raster.shape}")         # (bands, rows, cols)
print(f"Data type:    {raster.dtype}")          # e.g. float32
print(f"CRS:          {raster.crs}")            # e.g. EPSG:32633
print(f"Origin:       ({raster.origin_x}, {raster.origin_y})")
print(f"Pixel size:   ({raster.pixel_width}, {raster.pixel_height})")
print(f"Bands:        {raster.bands}")
print(f"Rows × Cols:  {raster.rows} × {raster.cols}")

# Access the raw numpy array
data = raster.data         # shape: (bands, rows, cols)
band1 = raster.data[0]    # first band as 2-D array
```

**Output example:**
```
Shape:        (1, 100, 200)
Data type:    float32
CRS:          EPSG:32633
Origin:       (500000.0, 4500000.0)
Pixel size:   (30.0, -30.0)
Bands:        1
Rows × Cols:  100 × 200
```

---

### 3. Choosing a Coordinate Reference System (CRS)

TerraTiff supports multiple ways to specify a CRS:

```python
# WGS 84 (latitude / longitude) — for global geographic data
raster1 = TerraTiff.from_array(data,
    origin_x=-93.5, origin_y=42.0,         # lon, lat
    pixel_width=0.001, pixel_height=-0.001, # degrees
    crs="WGS84",
)

# UTM zone — for local projected data in meters
raster2 = TerraTiff.from_array(data,
    origin_x=500000, origin_y=4500000,     # easting, northing (meters)
    pixel_width=30, pixel_height=-30,       # meters
    crs="UTM:33N",                          # UTM zone 33, Northern hemisphere
)

# Southern hemisphere UTM
raster3 = TerraTiff.from_array(data,
    origin_x=300000, origin_y=6200000,
    pixel_width=10, pixel_height=-10,
    crs="UTM:55S",                          # UTM zone 55, Southern hemisphere
)

# Any EPSG code — as a string
raster4 = TerraTiff.from_array(data,
    origin_x=0, origin_y=0,
    pixel_width=1, pixel_height=-1,
    crs="EPSG:32635",
)

# Any EPSG code — as an integer
raster5 = TerraTiff.from_array(data,
    origin_x=0, origin_y=0,
    pixel_width=1, pixel_height=-1,
    crs=32635,
)
```

**When to use which CRS:**

| CRS | Unit | Best for |
|-----|------|----------|
| `"WGS84"` | degrees | Global datasets, GPS coordinates |
| `"UTM:ZoneN/S"` | meters | Local/regional data, distance calculations |
| `"EPSG:NNNNN"` | varies | Any specific projection you need |

---

### 4. Exporting with Different Data Types

Control the output data type when saving. This is useful for reducing file size or matching a required format.

```python
import numpy as np
from terratiff import TerraTiff

# Create some floating-point data
data = np.random.rand(50, 50).astype(np.float64) * 1000

raster = TerraTiff.from_array(data,
    origin_x=0, origin_y=0,
    pixel_width=10, pixel_height=-10,
    crs="UTM:33N",
)

# === Integer types (truncates decimals) ===
raster.save("as_uint8.tif",  dtype="uint8")    # 0 – 255
raster.save("as_uint16.tif", dtype="uint16")   # 0 – 65,535
raster.save("as_uint32.tif", dtype="uint32")   # 0 – 4,294,967,295
raster.save("as_int8.tif",   dtype="int8")     # -128 – 127
raster.save("as_int16.tif",  dtype="int16")    # -32,768 – 32,767
raster.save("as_int32.tif",  dtype="int32")    # -2B – 2B

# === Floating-point types ===
raster.save("as_float16.tif", dtype="float16") # half precision
raster.save("as_float32.tif", dtype="float32") # single precision (recommended)
raster.save("as_float64.tif", dtype="float64") # double precision

# === Binary mask ===
raster.save("as_binary.tif",  dtype="binary")  # 0 or 1 only

# === Default — keeps original dtype ===
raster.save("as_default.tif")                  # float64 in this case
```

**Choosing the right data type:**

| Type | Size/pixel | Use case |
|------|-----------|----------|
| `uint8` | 1 byte | RGB images, classification maps (≤ 255 classes) |
| `int16` | 2 bytes | Elevation (DEM), temperature, signed integer data |
| `float32` | 4 bytes | General-purpose scientific data (recommended) |
| `float64` | 8 bytes | High-precision data (coordinates, large values) |
| `binary` | 1 byte | Masks (land/water, cloud/clear, building/no-building) |

---

### 5. Working with Multi-Band Rasters

Multi-band rasters store multiple layers in a single file — common for satellite imagery (RGB, multispectral).

#### Creating a multi-band raster

```python
import numpy as np
from terratiff import TerraTiff

# 3-band RGB image (bands, rows, cols)
red   = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
green = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
blue  = np.random.randint(0, 255, (512, 512), dtype=np.uint8)

# Stack into (3, 512, 512)
rgb = np.stack([red, green, blue], axis=0)

raster = TerraTiff.from_array(
    rgb,
    origin_x=-93.5, origin_y=42.0,
    pixel_width=0.0001, pixel_height=-0.0001,
    crs="WGS84",
)

raster.save("rgb_image.tif", dtype="uint8")
print(f"Bands: {raster.bands}")  # 3
```

#### Creating a multispectral raster (7 bands)

```python
# Simulate 7-band Landsat-like data
bands_data = np.random.rand(7, 256, 256).astype(np.float32)

raster = TerraTiff.from_array(
    bands_data,
    origin_x=500000, origin_y=4500000,
    pixel_width=30, pixel_height=-30,
    crs="UTM:33N",
)

raster.save("multispectral.tif", dtype="float32")
print(f"Shape: {raster.shape}")  # (7, 256, 256)
```

#### Reading a multi-band raster

```python
raster = TerraTiff.open("multispectral.tif")
print(f"Number of bands: {raster.bands}")

# Access individual bands (0-indexed)
band1 = raster.get_band(0)  # first band, shape (256, 256)
band4 = raster.get_band(3)  # fourth band

# Or via the data array directly
all_bands = raster.data  # shape (7, 256, 256)
```

---

### 6. Band Operations

Add, remove, and manipulate individual bands.

```python
import numpy as np
from terratiff import TerraTiff

# Start with a single-band raster
dem = np.random.rand(100, 100).astype(np.float32) * 500
raster = TerraTiff.from_array(dem,
    origin_x=0, origin_y=0,
    pixel_width=30, pixel_height=-30,
    crs="UTM:33N",
)
print(f"Bands: {raster.bands}")  # 1

# === Add a slope band ===
slope = np.gradient(dem)[0].astype(np.float32)
raster.add_band(slope)
print(f"Bands: {raster.bands}")  # 2

# === Add an aspect band ===
aspect = np.gradient(dem)[1].astype(np.float32)
raster.add_band(aspect)
print(f"Bands: {raster.bands}")  # 3

# === Retrieve specific bands ===
dem_band   = raster.get_band(0)  # shape (100, 100)
slope_band = raster.get_band(1)
aspect_band = raster.get_band(2)

# === Save the multi-band result ===
raster.save("dem_slope_aspect.tif", dtype="float32")
```

---

### 7. Binary Mask Export

Create and export binary (0/1) masks — useful for land-use classification, cloud masks, etc.

```python
import numpy as np
from terratiff import TerraTiff

# Simulate an elevation array
elevation = np.random.rand(200, 300).astype(np.float32) * 2000  # 0–2000 m

# Create a binary mask: 1 where elevation > 1000m, 0 elsewhere
high_ground = (elevation > 1000).astype(np.uint8)

raster = TerraTiff.from_array(
    high_ground,
    origin_x=500000, origin_y=4500000,
    pixel_width=30, pixel_height=-30,
    crs="UTM:33N",
)

# dtype="binary" ensures output contains only 0 and 1
raster.save("high_ground_mask.tif", dtype="binary")

# Verify
loaded = TerraTiff.open("high_ground_mask.tif")
unique_values = np.unique(loaded.data)
print(f"Unique values: {unique_values}")  # [0 1]
```

**How `"binary"` works:** any non-zero value in the array is mapped to `1`, and zeros stay `0`. The output dtype is `uint8`.

---

### 8. Changing Spatial Resolution (Resampling)

Resample a raster to a coarser or finer pixel size. Choose from **4 interpolation methods** — the geographic extent is preserved; only the grid dimensions change.

#### Resampling methods

| Method | Description | Best for |
|--------|-------------|----------|
| `"nearest"` | Nearest-neighbour — fast, no blending | Categorical data, masks, classification maps |
| `"bilinear"` | 2×2 weighted average — smooth transitions | Continuous surfaces (elevation, temperature) |
| `"cubic"` | 4×4 Catmull-Rom — sharp, high quality | Photographic imagery, high-detail surfaces |
| `"average"` | Block-mean aggregation — anti-aliased | Downsampling any data type |

#### Basic usage

```python
from terratiff import TerraTiff

raster = TerraTiff.open("elevation.tif")
print(f"Original: {raster.shape}, pixel: {raster.pixel_width}m")

# === Downsample to 90m with nearest (default) ===
coarse = raster.resample(pixel_width=90, pixel_height=-90)
coarse.save("elevation_90m.tif", dtype="float32")

# === Upsample to 10m with bilinear (smooth) ===
fine = raster.resample(pixel_width=10, pixel_height=-10, method="bilinear")
fine.save("elevation_10m.tif", dtype="float32")
```

#### Comparing methods

```python
import numpy as np
from terratiff import TerraTiff

# Create a gradient surface
arr = np.linspace(0, 100, 100 * 100).reshape(100, 100).astype(np.float32)
raster = TerraTiff.from_array(
    arr, origin_x=0, origin_y=0,
    pixel_width=10, pixel_height=-10, crs="UTM:33N",
)

# Downsample with each method
for method in ["nearest", "bilinear", "cubic", "average"]:
    resampled = raster.resample(pixel_width=50, pixel_height=-50, method=method)
    resampled.save(f"gradient_{method}_50m.tif", dtype="float32")
    print(f"{method:10s}  shape={resampled.shape}  "
          f"min={resampled.data.min():.1f}  max={resampled.data.max():.1f}")
```

#### When to use each method

- **`nearest`** — Land-cover maps, classification rasters, binary masks. Zero blending means class values are never mixed.
- **`bilinear`** — DEMs, temperature grids, NDVI. Smooth interpolation avoids staircase artifacts.
- **`cubic`** — Satellite imagery, aerial photos. Sharper edges than bilinear with minimal ringing.
- **`average`** — Downsampling anything. Each output pixel is the mean of all source pixels it covers, avoiding aliasing.

---

### 9. Clipping by Extent

Crop a raster to a bounding box. The output is a new raster trimmed to the intersection of the requested extent and the original raster.

```python
from terratiff import TerraTiff

raster = TerraTiff.open("elevation.tif")
print(f"Original bounds: {raster.get_bounds()}")
# (500000.0, 4497000.0, 506000.0, 4500000.0)

# Clip to a smaller area (xmin, ymin, xmax, ymax)
clipped = raster.clip(501000, 4498000, 504000, 4500000)
print(f"Clipped shape:  {clipped.shape}")
print(f"Clipped bounds: {clipped.get_bounds()}")

clipped.save("elevation_clipped.tif", dtype="float32")
```

**Features:**
- Automatically clamps to the raster extent if the box extends beyond
- Raises `ValueError` if there is no overlap
- Preserves pixel size and CRS

---

### 10. Masking with a Polygon

Mask a raster using a polygon geometry — pixels outside the polygon are set to NoData.

```python
import numpy as np
from terratiff import TerraTiff

# Create a raster
arr = np.ones((100, 100), dtype=np.float32) * 500
raster = TerraTiff.from_array(
    arr, origin_x=500000, origin_y=4500000,
    pixel_width=30, pixel_height=-30, crs="UTM:33N",
)

# Define a polygon (list of (x, y) vertices in map coordinates)
# This rectangle covers the centre of the raster
polygon = [
    (500900, 4499100),   # bottom-left
    (502100, 4499100),   # bottom-right
    (502100, 4499700),   # top-right
    (500900, 4499700),   # top-left
]

# Mask: pixels outside the polygon → NoData
masked = raster.mask_with_polygon(polygon, nodata=-9999)
masked.save("polygon_masked.tif", dtype="float32")

# Invert: mask the INSIDE of the polygon instead
inverted = raster.mask_with_polygon(polygon, invert=True, nodata=0)
inverted.save("polygon_inverted.tif", dtype="float32")
```

**Features:**
- Polygon is automatically closed (last vertex connects to first)
- Coordinates must be in the same CRS as the raster
- Works with any number of bands (all bands are masked)
- `invert=True` flips the mask — useful for cutting holes

---

### 11. Masking with a Raster

Use another GeoTIFF file as a mask layer — for example, a land/water mask or a classification raster.

```python
import numpy as np
from terratiff import TerraTiff

# Load your data raster
data_raster = TerraTiff.open("elevation.tif")

# Create (or load) a mask raster with the same grid dimensions
# 1 = valid, 0 = masked
mask_arr = np.ones((data_raster.rows, data_raster.cols), dtype=np.uint8)
mask_arr[0:20, 0:20] = 0    # mask the top-left corner
mask_arr[80:, 80:] = 0       # mask the bottom-right corner

mask_raster = TerraTiff.from_array(
    mask_arr,
    origin_x=data_raster.origin_x,
    origin_y=data_raster.origin_y,
    pixel_width=data_raster.pixel_width,
    pixel_height=data_raster.pixel_height,
    crs=data_raster.crs,
)

# Apply the mask
result = data_raster.mask_with_raster(mask_raster, nodata=-9999)
result.save("raster_masked.tif", dtype="float32")

print(f"Masked pixels:  {(result.data[0] == -9999).sum()}")
print(f"Valid pixels:   {(result.data[0] != -9999).sum()}")
```

**Rules:**
- Mask raster must have the **same grid dimensions** (rows × cols) as the data raster
- Pixels where mask = `0` → set to NoData
- Pixels where mask = mask's own NoData value → also set to NoData
- All bands in the data raster are masked

---

### 12. Converting Between Coordinate Systems

Transform the raster's spatial metadata from one CRS to another — for example, WGS 84 ↔ UTM.

```python
from terratiff import TerraTiff

# A raster in UTM coordinates
raster_utm = TerraTiff.from_array(
    data,
    origin_x=500000, origin_y=4500000,
    pixel_width=30, pixel_height=-30,
    crs="UTM:33N",
)
print(f"UTM origin: ({raster_utm.origin_x}, {raster_utm.origin_y})")

# Convert metadata to WGS 84 (lat/lon)
raster_wgs = raster_utm.to_crs("WGS84")
print(f"WGS84 origin: ({raster_wgs.origin_x:.6f}, {raster_wgs.origin_y:.6f})")
print(f"WGS84 CRS:    {raster_wgs.crs}")  # EPSG:4326

# Save in WGS 84
raster_wgs.save("elevation_wgs84.tif", dtype="float32")

# Convert from WGS 84 to a specific UTM zone
raster_utm15 = raster_wgs.to_crs("UTM:15N")
print(f"UTM15N CRS: {raster_utm15.crs}")  # EPSG:32615
```

> **Important:** `to_crs()` performs a **metadata-only** transformation — it reprojects the origin coordinates and pixel scale but does not warp (re-grid) the pixel data. Use this when your data is already aligned to the target grid, or when you need to update the CRS tag.

---

### 13. Querying Spatial Metadata

```python
from terratiff import TerraTiff

raster = TerraTiff.open("elevation.tif")

# === Bounding box ===
xmin, ymin, xmax, ymax = raster.get_bounds()
print(f"Bounds: W={xmin}, S={ymin}, E={xmax}, N={ymax}")

# === Transform (origin + pixel size) ===
origin_x, origin_y, pixel_w, pixel_h = raster.get_transform()
print(f"Origin:     ({origin_x}, {origin_y})")
print(f"Pixel size: ({pixel_w}, {pixel_h})")

# === CRS info ===
print(f"CRS string:     {raster.crs}")            # "EPSG:32633"
print(f"CRS name:       {raster.crs_info.name}")   # "WGS 84 / UTM zone 33N"
print(f"Is projected:   {raster.crs_info.is_projected}")  # True
print(f"EPSG code:      {raster.crs_info.epsg}")   # 32633

# === Data info ===
print(f"Shape:      {raster.shape}")     # (bands, rows, cols)
print(f"Dtype:      {raster.dtype}")      # float32
print(f"Bands:      {raster.bands}")
print(f"Rows:       {raster.rows}")
print(f"Cols:       {raster.cols}")
print(f"NoData:     {raster.nodata}")
```

---

### 14. Working with NoData Values

Mark missing or invalid pixels with a NoData sentinel value.

```python
import numpy as np
from terratiff import TerraTiff

# Create data with holes
data = np.random.rand(100, 100).astype(np.float32) * 100
data[20:40, 30:60] = -9999  # mark a region as missing

raster = TerraTiff.from_array(
    data,
    origin_x=0, origin_y=0,
    pixel_width=10, pixel_height=-10,
    crs="UTM:33N",
    nodata=-9999.0,          # ← set the NoData value
)

raster.save("with_nodata.tif", dtype="float32")

# Read it back — NoData is preserved
loaded = TerraTiff.open("with_nodata.tif")
print(f"NoData value: {loaded.nodata}")  # -9999.0

# Create a validity mask
valid_mask = loaded.data[0] != loaded.nodata
print(f"Valid pixels: {valid_mask.sum()}")
```

---

### 15. UTM Zone Utilities

Automatically determine the correct UTM zone for any location.

```python
from terratiff import utm_zone_from_latlon, utm_epsg, parse_crs

# Find the UTM zone for a given lat/lon
zone, hemisphere = utm_zone_from_latlon(lat=42.0, lon=-93.5)
print(f"Zone: {zone}{hemisphere}")        # 15N

# Get the EPSG code
epsg = utm_epsg(zone, hemisphere)
print(f"EPSG code: {epsg}")               # 32615

# Parse it into a CRS object
crs = parse_crs(f"UTM:{zone}{hemisphere}")
print(f"CRS: {crs}")                      # CRSInfo(EPSG:32615, WGS 84 / UTM zone 15N)

# === More examples ===
print(utm_zone_from_latlon(51.5, -0.1))   # (30, 'N') — London
print(utm_zone_from_latlon(-33.9, 18.4))  # (34, 'S') — Cape Town
print(utm_zone_from_latlon(35.7, 139.7))  # (54, 'N') — Tokyo
print(utm_zone_from_latlon(-22.9, -43.2)) # (23, 'S') — Rio de Janeiro
```



## API Reference

### Class: `TerraTiff`

#### Constructors

| Method | Description |
|--------|-------------|
| `TerraTiff.open(filepath)` | Read a GeoTIFF file from disk. Returns a `TerraTiff` instance. |
| `TerraTiff.from_array(array, origin_x, origin_y, pixel_width, pixel_height, crs, nodata=None)` | Create a `TerraTiff` from a numpy array with user-supplied spatial metadata. |

#### I/O

| Method | Description |
|--------|-------------|
| `save(filepath, dtype=None)` | Write to a GeoTIFF file. Optionally cast to a specific dtype. |

#### Spatial Operations

| Method | Description |
|--------|-------------|
| `resample(pixel_width, pixel_height, method="nearest")` → `TerraTiff` | Resample to a new pixel size. Methods: `"nearest"`, `"bilinear"`, `"cubic"`, `"average"`. |
| `to_crs(target_crs)` → `TerraTiff` | Reproject origin coordinates to a new CRS. Returns a new instance. |
| `clip(xmin, ymin, xmax, ymax)` → `TerraTiff` | Crop the raster to a bounding-box extent. |
| `mask_with_polygon(polygon, invert=False, nodata=None)` → `TerraTiff` | Mask using polygon vertices. Outside → NoData. |
| `mask_with_raster(mask, nodata=None)` → `TerraTiff` | Mask using another raster (0 = masked). |

#### Band Access

| Method | Description |
|--------|-------------|
| `get_band(index)` → `np.ndarray` | Return band at `index` as a 2-D array (0-indexed). |
| `add_band(array)` | Append a 2-D array as a new band. Modifies in place. |

#### Queries

| Method | Description |
|--------|-------------|
| `get_bounds()` → `(xmin, ymin, xmax, ymax)` | Spatial extent in the raster's CRS units. |
| `get_transform()` → `(origin_x, origin_y, pixel_width, pixel_height)` | Origin and pixel scale. |
| `copy()` → `TerraTiff` | Deep copy of the raster. |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `data` | `np.ndarray` | Raw array, shape `(bands, rows, cols)`. |
| `crs` | `str` | CRS as `"EPSG:NNNNN"`. |
| `crs_info` | `CRSInfo` | Detailed CRS object with `.epsg`, `.name`, `.is_projected`. |
| `shape` | `tuple` | `(bands, rows, cols)`. |
| `bands` | `int` | Number of bands. |
| `rows` | `int` | Number of rows. |
| `cols` | `int` | Number of columns. |
| `dtype` | `np.dtype` | Data type of the array. |
| `origin_x` | `float` | X coordinate of the top-left corner. |
| `origin_y` | `float` | Y coordinate of the top-left corner. |
| `pixel_width` | `float` | Pixel size in X direction. |
| `pixel_height` | `float` | Pixel size in Y direction (negative = north-up). |
| `nodata` | `float/int/None` | NoData sentinel value. |

---

### Utility Functions

| Function | Description |
|----------|-------------|
| `parse_crs(crs_input)` → `CRSInfo` | Parse `"WGS84"`, `"UTM:33N"`, `"EPSG:4326"`, or `int` → `CRSInfo`. |
| `utm_zone_from_latlon(lat, lon)` → `(zone, hemisphere)` | Get UTM zone number and `"N"`/`"S"` from coordinates. |
| `utm_epsg(zone, hemisphere)` → `int` | Get EPSG code for a UTM zone (e.g. 33, "N" → 32633). |
| `supported_dtypes()` → `list[str]` | List all supported dtype strings. |

---

## Supported CRS Formats

| Input Format | Example | Description |
|--------------|---------|-------------|
| `"WGS84"` | `"WGS84"` | WGS 84 geographic (EPSG:4326) |
| `"EPSG:NNNNN"` | `"EPSG:32633"` | Any EPSG code as string |
| `int` | `4326` | Any EPSG code as integer |
| `"UTM:ZoneN"` | `"UTM:33N"` | WGS 84 / UTM zone, Northern hemisphere |
| `"UTM:ZoneS"` | `"UTM:55S"` | WGS 84 / UTM zone, Southern hemisphere |

All 120 UTM zones (1–60, N and S) are supported.

---

## Supported Data Types

| Type | numpy dtype | Size | Range | Best for |
|------|-------------|------|-------|----------|
| `"uint8"` | `uint8` | 1 byte | 0 – 255 | RGB images, class maps |
| `"uint16"` | `uint16` | 2 bytes | 0 – 65,535 | Satellite imagery (raw DN) |
| `"uint32"` | `uint32` | 4 bytes | 0 – 4.3B | Large ID rasters |
| `"int8"` | `int8` | 1 byte | -128 – 127 | Small signed values |
| `"int16"` | `int16` | 2 bytes | -32,768 – 32,767 | DEM, temperature |
| `"int32"` | `int32` | 4 bytes | -2.1B – 2.1B | Large signed values |
| `"float16"` | `float16` | 2 bytes | ±65,504 | Compact float storage |
| `"float32"` | `float32` | 4 bytes | ±3.4×10³⁸ | General scientific data |
| `"float64"` | `float64` | 8 bytes | ±1.8×10³⁰⁸ | High-precision data |
| `"binary"` | `uint8` | 1 byte | 0 or 1 | Masks |

---

## License

This software is licensed under a Custom License that allows free use for **non-commercial** applications only. See the [LICENSE](file:///f:/projects/TerraTiff/LICENSE) file for more details.
