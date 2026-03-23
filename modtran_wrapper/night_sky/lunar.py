"""
Lunar top-of-atmosphere irradiance model.

Loads a two-column (wavelength_nm, irradiance_W_m2_nm) text file representing
the average TOA lunar irradiance and provides interpolation utilities.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

_DEFAULT_DATA_FILE = Path(__file__).parent / "data" / "lunar_toa_avg.txt"


class LunarModel:
    """
    Top-of-atmosphere average lunar irradiance model.

    Loads spectral irradiance data from a two-column ASCII text file
    (wavelength in nm, irradiance in W/m²/nm) and provides interpolation
    to arbitrary wavelength grids.

    Parameters
    ----------
    data_file : str or Path, optional
        Path to the two-column data file.  Defaults to
        ``night_sky/data/lunar_toa_avg.txt`` relative to this file.
    comments : str, optional
        Character that marks comment lines.  Defaults to ``'#'``.
    delimiter : str or None, optional
        Column delimiter.  ``None`` triggers auto-detection (whitespace
        or comma).

    Raises
    ------
    FileNotFoundError
        If *data_file* does not exist.
    ValueError
        If the file has the wrong shape or contains non-positive wavelengths.
    """

    def __init__(
        self,
        data_file: Optional[str | os.PathLike] = None,
        comments: str = "#",
        delimiter: Optional[str] = None,
    ) -> None:
        if data_file is None:
            data_file = _DEFAULT_DATA_FILE
            if not Path(data_file).is_file():
                raise FileNotFoundError(
                    f"Default lunar TOA data file not found at:\n"
                    f"  {data_file}\n"
                    "Please place a two-column (wavelength_nm, irradiance_W_m2_nm) "
                    "ASCII text file at that location, or supply a custom path via "
                    "the `data_file` argument."
                )

        data_file = Path(data_file)
        if not data_file.is_file():
            raise FileNotFoundError(
                f"Lunar TOA data file not found: {data_file}"
            )

        # Auto-detect delimiter when not supplied
        resolved_delimiter = delimiter
        if resolved_delimiter is None:
            resolved_delimiter = self._detect_delimiter(data_file, comments)

        raw = np.loadtxt(
            data_file,
            comments=comments,
            delimiter=resolved_delimiter,
        )

        if raw.ndim != 2 or raw.shape[1] < 2:
            raise ValueError(
                f"Lunar data file must have at least two columns "
                f"(wavelength_nm, irradiance_W_m2_nm); got shape {raw.shape}."
            )

        wl = raw[:, 0].copy()
        irr = raw[:, 1].copy()

        if np.any(wl <= 0.0):
            raise ValueError(
                "Wavelength column contains non-positive values.  "
                "All wavelengths must be > 0 nm."
            )

        # Ensure monotonically increasing wavelength
        sort_idx = np.argsort(wl)
        self._wavelength_nm: np.ndarray = wl[sort_idx]
        self._irradiance_W_m2_nm: np.ndarray = irr[sort_idx]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def wavelength_nm(self) -> np.ndarray:
        """Raw wavelength grid from the data file (nm)."""
        return self._wavelength_nm.copy()

    @property
    def irradiance_W_m2_nm(self) -> np.ndarray:
        """Raw spectral irradiance from the data file (W/m²/nm)."""
        return self._irradiance_W_m2_nm.copy()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def toa_irradiance(self, wavelength_nm: np.ndarray) -> np.ndarray:
        """
        Interpolate TOA lunar irradiance onto *wavelength_nm*.

        Values outside the data range are clamped to zero (no extrapolation).

        Parameters
        ----------
        wavelength_nm : np.ndarray
            Target wavelength grid (nm).

        Returns
        -------
        np.ndarray
            Spectral irradiance (W/m²/nm), same shape as *wavelength_nm*.
        """
        wavelength_nm = np.asarray(wavelength_nm, dtype=float)
        result = np.interp(
            wavelength_nm,
            self._wavelength_nm,
            self._irradiance_W_m2_nm,
            left=0.0,
            right=0.0,
        )
        return result

    def as_modtran_source(self, wavelength_nm: np.ndarray) -> np.ndarray:
        """
        Return the TOA lunar irradiance on *wavelength_nm*.

        Alias for :meth:`toa_irradiance`, provided for clarity when using
        the result as a MODTRAN external source spectrum.

        Parameters
        ----------
        wavelength_nm : np.ndarray
            Target wavelength grid (nm).

        Returns
        -------
        np.ndarray
            Spectral irradiance (W/m²/nm).
        """
        return self.toa_irradiance(wavelength_nm)

    def integrated_irradiance(
        self,
        wl_start_nm: float,
        wl_end_nm: float,
    ) -> float:
        """
        Integrate the TOA lunar irradiance over a wavelength band.

        Uses :func:`numpy.trapz` on the native data grid clipped to
        [*wl_start_nm*, *wl_end_nm*].  For sub-grid-resolution bands the
        method interpolates endpoint values and inserts them.

        Parameters
        ----------
        wl_start_nm : float
            Lower wavelength bound (nm).
        wl_end_nm : float
            Upper wavelength bound (nm).

        Returns
        -------
        float
            Band-integrated irradiance (W/m²).
        """
        if wl_start_nm >= wl_end_nm:
            raise ValueError(
                f"wl_start_nm ({wl_start_nm}) must be less than "
                f"wl_end_nm ({wl_end_nm})."
            )

        # Build integration grid: interior data points + interpolated endpoints
        mask = (self._wavelength_nm >= wl_start_nm) & (
            self._wavelength_nm <= wl_end_nm
        )
        wl_interior = self._wavelength_nm[mask]
        irr_interior = self._irradiance_W_m2_nm[mask]

        # Prepend start point if not already at a data node
        if len(wl_interior) == 0 or wl_interior[0] > wl_start_nm:
            irr_start = float(
                np.interp(
                    wl_start_nm,
                    self._wavelength_nm,
                    self._irradiance_W_m2_nm,
                    left=0.0,
                    right=0.0,
                )
            )
            wl_interior = np.concatenate([[wl_start_nm], wl_interior])
            irr_interior = np.concatenate([[irr_start], irr_interior])

        # Append end point if not already at a data node
        if len(wl_interior) == 0 or wl_interior[-1] < wl_end_nm:
            irr_end = float(
                np.interp(
                    wl_end_nm,
                    self._wavelength_nm,
                    self._irradiance_W_m2_nm,
                    left=0.0,
                    right=0.0,
                )
            )
            wl_interior = np.concatenate([wl_interior, [wl_end_nm]])
            irr_interior = np.concatenate([irr_interior, [irr_end]])

        return float(np.trapz(irr_interior, wl_interior))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_delimiter(path: Path, comments: str) -> Optional[str]:
        """
        Peek at the first non-comment, non-empty line of *path* and infer
        whether the delimiter is a comma or whitespace.

        Returns ``','`` if a comma is detected, otherwise ``None``
        (which makes :func:`numpy.loadtxt` split on whitespace).
        """
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped or stripped.startswith(comments):
                    continue
                if "," in stripped:
                    return ","
                return None  # whitespace-separated
        return None
