"""Parse MODTRAN tape7 spectral output files into pandas DataFrames."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..exceptions import ModtranParseError


# ---------------------------------------------------------------------------
# Known column-name sets by IEMSCT mode
# ---------------------------------------------------------------------------
# MODTRAN writes different columns depending on the calculation mode.
# The column names below match the tape7 header labels (normalized to lowercase).

# IEMSCT = 0 — transmittance only
_COLS_IEMSCT0 = [
    "freq",
    "total_trans",
    "pth_thrml",
    "thrml_sct",
    "solar_scat",
    "grnd_rflt",
    "drct_rflt",
    "total_rad",
    "ref_sol",
    "sol/scat",
    "obs_trans",
]

# IEMSCT = 2 — solar + thermal radiance
_COLS_IEMSCT2 = [
    "freq",
    "total_trans",
    "pth_thrml",
    "thrml_sct",
    "solar_scat",
    "grnd_rflt",
    "drct_rflt",
    "total_rad",
    "ref_sol",
    "sol/scat",
    "obs_trans",
]

# IEMSCT = 3 — solar irradiance (downwelling)
_COLS_IEMSCT3 = [
    "freq",
    "total_trans",
    "pth_thrml",
    "thrml_sct",
    "solar_scat",
    "grnd_rflt",
    "drct_rflt",
    "total_rad",
    "ref_sol",
    "sol/scat",
    "obs_trans",
]

# Fallback — use when column count doesn't match any known set
_COL_PREFIX = "col"


def read_tape7(path: str | Path, iemsct: int | None = None) -> pd.DataFrame:
    """Parse a MODTRAN tape7 spectral output file.

    Parameters
    ----------
    path : str or Path
        Path to the tape7 file.
    iemsct : int or None
        IEMSCT mode used to produce the file.  When ``None`` the reader
        attempts to auto-detect the mode from the file header.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``wavenumber`` (cm⁻¹) as the index and one column
        per spectral quantity.  Column names are normalised to lowercase
        with spaces replaced by underscores.
    """
    path = Path(path)
    if not path.exists():
        raise ModtranParseError(f"tape7 file not found: {path}")

    raw = path.read_text(errors="replace")
    lines = raw.splitlines()

    if not lines:
        raise ModtranParseError(f"tape7 file is empty: {path}")

    # ------------------------------------------------------------------
    # Locate the header line and the first data line
    # ------------------------------------------------------------------
    header_line_idx, data_start_idx = _find_header(lines)

    if header_line_idx is None:
        raise ModtranParseError(
            f"Could not find column header in tape7 file: {path}"
        )

    # ------------------------------------------------------------------
    # Parse column names from header
    # ------------------------------------------------------------------
    col_names = _parse_header(lines[header_line_idx], iemsct)

    # ------------------------------------------------------------------
    # Parse data lines
    # ------------------------------------------------------------------
    data_lines = [
        line for line in lines[data_start_idx:]
        if line.strip() and not line.strip().startswith("!")
    ]

    if not data_lines:
        raise ModtranParseError(f"No data rows found in tape7: {path}")

    records = []
    for lineno, line in enumerate(data_lines, start=data_start_idx + 1):
        try:
            values = [float(v) for v in line.split()]
        except ValueError:
            continue  # skip non-numeric lines (footers, etc.)
        if len(values) >= 2:
            records.append(values)

    if not records:
        raise ModtranParseError(f"Could not parse any numeric rows from tape7: {path}")

    n_cols_data = len(records[0])

    # Adjust column name list to match actual data width
    if len(col_names) < n_cols_data:
        col_names += [f"{_COL_PREFIX}{i}" for i in range(len(col_names), n_cols_data)]
    elif len(col_names) > n_cols_data:
        col_names = col_names[:n_cols_data]

    arr = np.array(records)
    df = pd.DataFrame(arr, columns=col_names)

    # Use wavenumber (first column) as index
    wavenum_col = col_names[0]
    df = df.set_index(wavenum_col)
    df.index.name = "wavenumber_cm1"

    return df


def _find_header(lines: list[str]) -> tuple[int | None, int]:
    """Return (header_line_index, first_data_line_index).

    The header is the last non-numeric line before the data block.
    """
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        # Check if this line starts with a number (data line)
        parts = stripped.split()
        try:
            float(parts[0])
            # First numeric line found — header is the line before
            header_idx = None
            for j in range(i - 1, -1, -1):
                if lines[j].strip():
                    header_idx = j
                    break
            return header_idx, i
        except (ValueError, IndexError):
            continue
    return None, len(lines)


def _parse_header(header_line: str, iemsct: int | None) -> list[str]:
    """Extract column names from a tape7 header line.

    MODTRAN 5 sometimes runs column names together without spaces.
    This function handles both space-separated and run-together formats.
    """
    # Try splitting on whitespace first
    raw_names = header_line.split()

    if len(raw_names) >= 5:
        # Looks like normal space-separated header
        names = [_normalise_colname(n) for n in raw_names]
    else:
        # Possibly a run-together MODTRAN 5 header — split on known tokens
        names = _split_runtogether_header(header_line)

    return names if names else [_COL_PREFIX + str(i) for i in range(20)]


def _normalise_colname(name: str) -> str:
    """Lowercase, replace spaces and slashes with underscores."""
    name = name.lower().strip()
    name = re.sub(r"[/\s]+", "_", name)
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name or "col"


_KNOWN_TOKENS = [
    "FREQ", "TOTAL_TRANS", "PTH_THRML", "THRML_SCT", "SOLAR_SCT",
    "GRND_RFLT", "DRCT_RFLT", "TOTAL_RAD", "REF_SOL", "SOL/SCT",
    "OBS_TRANS", "SOL_TRANS", "TOTAL_EMIS", "SOLSCAT", "BBODY_T[K]",
]


def _split_runtogether_header(line: str) -> list[str]:
    """Split a run-together tape7 header by known token boundaries."""
    # Build a regex that matches any known token (longest first to avoid ambiguity)
    tokens_sorted = sorted(_KNOWN_TOKENS, key=len, reverse=True)
    pattern = "|".join(re.escape(t) for t in tokens_sorted)
    found = re.findall(pattern, line, flags=re.IGNORECASE)
    if found:
        return [_normalise_colname(t) for t in found]
    # Last resort: treat the whole line as one token per word
    return [_normalise_colname(t) for t in line.split()]
