"""Parse MODTRAN .flx flux output files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..exceptions import ModtranParseError


def read_flux(path: str | Path) -> pd.DataFrame:
    """Parse a MODTRAN .flx spectral flux file.

    The .flx file contains spectral irradiance components at each
    atmospheric level.  This reader returns a DataFrame indexed by
    wavenumber (cm⁻¹) with flux components as columns.

    The surface-level (ground) rows are tagged with level index 0.
    The returned DataFrame contains only the ground level by default.
    Pass ``level=None`` to get all levels (MultiIndex on wavenumber + level).

    Parameters
    ----------
    path : str or Path
        Path to the .flx file.

    Returns
    -------
    pd.DataFrame
        Columns: ``wavenumber_cm1``, ``direct_solar``, ``diffuse_down``,
        ``diffuse_up``, ``direct_solar_trans`` (units: W/cm²/cm⁻¹).
        Index: wavenumber_cm1.
    """
    path = Path(path)
    if not path.exists():
        raise ModtranParseError(f".flx file not found: {path}")

    raw = path.read_text(errors="replace")
    lines = raw.splitlines()

    if not lines:
        raise ModtranParseError(f".flx file is empty: {path}")

    # ------------------------------------------------------------------
    # The .flx file structure:
    #   Header lines (start with spaces or letters)
    #   Repeated blocks — one block per spectral point:
    #       Line 1: wavenumber, n_levels, ...
    #       Lines 2..n_levels+1: level_index, direct, diffuse_dn, diffuse_up, ...
    # ------------------------------------------------------------------
    # Alternatively, MODTRAN 6 .flx uses a simpler tabular format.
    # Try tabular first, then fall back to block format.
    # ------------------------------------------------------------------

    try:
        return _read_tabular(lines, path)
    except ModtranParseError:
        pass

    return _read_block(lines, path)


def _read_tabular(lines: list[str], path: Path) -> pd.DataFrame:
    """Read a tabular (MODTRAN 6 style) .flx file."""
    header_idx = None
    data_start = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        try:
            float(parts[0])
            if header_idx is None:
                raise ModtranParseError("No header found before data")
            data_start = i
            break
        except ValueError:
            header_idx = i

    if data_start is None:
        raise ModtranParseError(f"No numeric data in .flx file: {path}")

    col_names = lines[header_idx].split() if header_idx is not None else []
    col_names = [c.lower().replace("/", "_") for c in col_names]

    records = []
    for line in lines[data_start:]:
        parts = line.split()
        if not parts:
            continue
        try:
            records.append([float(v) for v in parts])
        except ValueError:
            continue

    if not records:
        raise ModtranParseError(f"No parseable rows in .flx file: {path}")

    arr = np.array(records)
    if len(col_names) != arr.shape[1]:
        col_names = [f"col{i}" for i in range(arr.shape[1])]

    df = pd.DataFrame(arr, columns=col_names)

    # Identify the wavenumber column
    wavenum_col = _find_wavenum_col(col_names)
    if wavenum_col:
        df = df.set_index(wavenum_col)
        df.index.name = "wavenumber_cm1"

    return df


def _read_block(lines: list[str], path: Path) -> pd.DataFrame:
    """Read a block-format (MODTRAN 5 style) .flx file.

    Each spectral point has a block:
        wavenumber  n_levels
        level  direct  diffuse_down  diffuse_up  [direct_trans]
        ...
    """
    records = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        parts = line.split()
        try:
            wavenum = float(parts[0])
            n_levels = int(parts[1])
        except (ValueError, IndexError):
            continue

        # Read level rows
        for _ in range(n_levels):
            if i >= n:
                break
            lparts = lines[i].strip().split()
            i += 1
            try:
                level_idx = int(lparts[0])
                direct = float(lparts[1])
                diff_dn = float(lparts[2])
                diff_up = float(lparts[3])
                d_trans = float(lparts[4]) if len(lparts) > 4 else float("nan")
            except (ValueError, IndexError):
                continue
            records.append((wavenum, level_idx, direct, diff_dn, diff_up, d_trans))

    if not records:
        raise ModtranParseError(f"Could not parse block-format .flx: {path}")

    df = pd.DataFrame(
        records,
        columns=["wavenumber_cm1", "level", "direct_solar",
                 "diffuse_down", "diffuse_up", "direct_trans"],
    )
    # Return surface level (level 0) only
    df_surface = df[df["level"] == 0].drop(columns="level").copy()
    df_surface = df_surface.set_index("wavenumber_cm1")
    return df_surface


def _find_wavenum_col(col_names: list[str]) -> str | None:
    for name in col_names:
        if "freq" in name or "wave" in name or name in ("cm-1", "cm_1", "wn"):
            return name
    return col_names[0] if col_names else None
