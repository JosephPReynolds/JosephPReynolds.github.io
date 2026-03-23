"""Parse MODTRAN 6 CSV spectral output files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..exceptions import ModtranParseError


def read_csv(path: str | Path) -> pd.DataFrame:
    """Parse a MODTRAN 6 CSV spectral output file.

    MODTRAN 6 can produce CSV output when the JSON input format is used.
    The CSV has a header row followed by spectral data rows.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file (typically ``<case_name>.csv``).

    Returns
    -------
    pd.DataFrame
        DataFrame with ``wavenumber_cm1`` as the index.
    """
    path = Path(path)
    if not path.exists():
        raise ModtranParseError(f"CSV file not found: {path}")

    try:
        df = pd.read_csv(path, comment="!")
    except Exception as exc:
        raise ModtranParseError(f"Failed to read CSV file {path}: {exc}") from exc

    if df.empty:
        raise ModtranParseError(f"CSV file is empty: {path}")

    # Normalise column names
    df.columns = [
        c.strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_")
        for c in df.columns
    ]

    # Set wavenumber as index
    wavenum_col = _find_wavenum_col(list(df.columns))
    if wavenum_col:
        df = df.set_index(wavenum_col)
        df.index.name = "wavenumber_cm1"

    return df


def _find_wavenum_col(col_names: list[str]) -> str | None:
    candidates = ("freq", "wavenumber", "cm_1", "cm1", "wn", "wavelength")
    for name in col_names:
        if any(c in name for c in candidates):
            return name
    return col_names[0] if col_names else None
