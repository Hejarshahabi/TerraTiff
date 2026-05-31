"""
Data-type mapping and validation for the TerraTiff package.

Maps user-facing dtype strings (e.g. "float32", "binary") to numpy dtypes
and provides helpers for TIFF sample-format metadata.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Public dtype registry
# ---------------------------------------------------------------------------

# Maps user-facing name  →  numpy dtype
DTYPE_MAP: dict[str, np.dtype] = {
    "uint8":   np.dtype(np.uint8),
    "uint16":  np.dtype(np.uint16),
    "uint32":  np.dtype(np.uint32),
    "int8":    np.dtype(np.int8),
    "int16":   np.dtype(np.int16),
    "int32":   np.dtype(np.int32),
    "float16": np.dtype(np.float16),
    "float32": np.dtype(np.float32),
    "float64": np.dtype(np.float64),
    "binary":  np.dtype(np.uint8),   # stored as uint8, values clamped to 0/1
}

# Reverse lookup: numpy dtype → preferred user-facing name
_NUMPY_TO_NAME: dict[np.dtype, str] = {
    np.dtype(np.uint8):   "uint8",
    np.dtype(np.uint16):  "uint16",
    np.dtype(np.uint32):  "uint32",
    np.dtype(np.int8):    "int8",
    np.dtype(np.int16):   "int16",
    np.dtype(np.int32):   "int32",
    np.dtype(np.float16): "float16",
    np.dtype(np.float32): "float32",
    np.dtype(np.float64): "float64",
}


def supported_dtypes() -> list[str]:
    """Return the list of supported dtype strings."""
    return list(DTYPE_MAP.keys())


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def resolve_dtype(dtype_str: str | None, current_dtype: np.dtype | None = None) -> np.dtype:
    """
    Resolve a user-supplied dtype string to a numpy dtype.

    Parameters
    ----------
    dtype_str : str or None
        One of the supported dtype names (e.g. ``"float32"``, ``"binary"``).
        If *None*, falls back to *current_dtype*.
    current_dtype : numpy.dtype or None
        The array's current dtype used as a fallback.

    Returns
    -------
    numpy.dtype

    Raises
    ------
    ValueError
        If *dtype_str* is not recognised.
    """
    if dtype_str is None:
        if current_dtype is not None:
            return current_dtype
        return np.dtype(np.float32)  # sensible default

    key = dtype_str.strip().lower()
    if key not in DTYPE_MAP:
        raise ValueError(
            f"Unsupported dtype '{dtype_str}'. "
            f"Choose from: {', '.join(DTYPE_MAP)}"
        )
    return DTYPE_MAP[key]


def is_binary(dtype_str: str | None) -> bool:
    """Return *True* when the user explicitly requested binary output."""
    if dtype_str is None:
        return False
    return dtype_str.strip().lower() == "binary"


def cast_array(data: np.ndarray, dtype_str: str | None) -> np.ndarray:
    """
    Cast *data* to the target dtype, applying binary clamping when needed.

    Parameters
    ----------
    data : numpy.ndarray
    dtype_str : str or None

    Returns
    -------
    numpy.ndarray
        A new array (or view) with the target dtype.
    """
    target = resolve_dtype(dtype_str, data.dtype)

    if is_binary(dtype_str):
        # Binarise: anything != 0 becomes 1
        return (data != 0).astype(np.uint8)

    if data.dtype == target:
        return data

    return data.astype(target)


def dtype_name(dt: np.dtype) -> str:
    """Return the canonical user-facing name for a numpy dtype."""
    return _NUMPY_TO_NAME.get(np.dtype(dt), str(dt))
