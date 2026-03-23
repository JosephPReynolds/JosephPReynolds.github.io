"""ModtranResult: container for parsed MODTRAN output products."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd

from .units import (
    irrad_wavenum_to_nm,
    irrad_wavenum_to_um,
    convert_spectrum_to_nm,
    wavenum_to_nm,
)

if TYPE_CHECKING:
    from .inputs.case import ModtranCase


@dataclass
class ModtranResult:
    """Container for all output products from a single MODTRAN run.

    Attributes
    ----------
    tape7 : pd.DataFrame or None
        Full spectral tape7 output (wavenumber index, spectral columns).
    flux : pd.DataFrame or None
        Spectral flux at each atmospheric level from the .flx file.
    csv : pd.DataFrame or None
        MODTRAN 6 CSV spectral output (only populated for version 6 runs).
    case : ModtranCase
        The input case that produced this result.
    run_dir : Path
        Working directory where MODTRAN was executed.
    """

    tape7: Optional[pd.DataFrame]
    flux: Optional[pd.DataFrame]
    csv: Optional[pd.DataFrame]
    case: "ModtranCase"
    run_dir: Path

    # ------------------------------------------------------------------
    # Downwelling irradiance access
    # ------------------------------------------------------------------

    def downwelling_irradiance(self, units: str = "W/m2/nm") -> pd.DataFrame:
        """Return total downwelling spectral irradiance (direct + diffuse).

        Tries the .flx file first (most direct source), then falls back to
        deriving from tape7.

        Parameters
        ----------
        units : str
            Output units.  One of:
            - ``'W/cm2/cm1'`` — native MODTRAN units
            - ``'W/m2/nm'``   — W per m² per nm  (default)
            - ``'W/m2/um'``   — W per m² per µm
            - ``'mW/m2/nm'``  — mW per m² per nm

        Returns
        -------
        pd.DataFrame
            Single-column DataFrame with downwelling irradiance.
            Index is wavenumber (cm⁻¹) for native units, wavelength (nm or µm)
            for converted units.
        """
        df = self._get_downwelling_native()
        return self._convert_units(df, "downwelling_irradiance", units)

    def direct_irradiance(self, units: str = "W/m2/nm") -> pd.DataFrame:
        """Return direct (beam) solar irradiance at ground level."""
        df = self._get_direct_native()
        return self._convert_units(df, "direct_irradiance", units)

    def diffuse_irradiance(self, units: str = "W/m2/nm") -> pd.DataFrame:
        """Return diffuse (scattered) downwelling irradiance at ground level."""
        df = self._get_diffuse_native()
        return self._convert_units(df, "diffuse_irradiance", units)

    def transmittance(self) -> pd.DataFrame:
        """Return total atmospheric transmittance (dimensionless).

        Returns a DataFrame with wavenumber index.
        """
        if self.tape7 is None:
            raise ValueError("No tape7 output available.")
        t7 = self.tape7
        # Find transmittance column
        col = _find_col(t7, ("total_trans", "trans", "totaltrans", "transmittance"))
        if col is None:
            raise ValueError(
                f"No transmittance column found in tape7. "
                f"Available columns: {list(t7.columns)}"
            )
        return t7[[col]].copy()

    # ------------------------------------------------------------------
    # Wavelength convenience
    # ------------------------------------------------------------------

    def to_wavelength_nm(self) -> pd.DataFrame:
        """Return tape7 data with wavenumber index converted to wavelength (nm).

        Spectral density columns are converted using the chain rule; purely
        dimensionless columns (e.g. transmittance) are interpolated onto the
        new wavelength grid unchanged.
        """
        if self.tape7 is None:
            raise ValueError("No tape7 output available.")
        nu = self.tape7.index.values
        lam_nm = wavenum_to_nm(nu)
        df_out = self.tape7.copy()
        df_out.index = pd.Index(lam_nm, name="wavelength_nm")
        return df_out.sort_index()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_downwelling_native(self) -> pd.DataFrame:
        """Return total downwelling irradiance in W/cm²/cm⁻¹."""
        # Try flux file first (ground level direct + diffuse)
        if self.flux is not None:
            direct = _find_col(self.flux, ("direct_solar", "direct", "dir"))
            diffuse = _find_col(self.flux, ("diffuse_down", "diffuse_dn", "diff_dn", "dif_dn"))
            if direct and diffuse:
                total = self.flux[direct] + self.flux[diffuse]
                return total.rename("downwelling_irradiance").to_frame()
            # Single column flux — assume it's total downwelling
            if len(self.flux.columns) == 1:
                return self.flux.rename(
                    columns={self.flux.columns[0]: "downwelling_irradiance"}
                )

        # Fall back to tape7
        if self.tape7 is not None:
            col = _find_col(
                self.tape7,
                ("total_rad", "sol_scat", "solar_scat", "grnd_rflt", "ref_sol"),
            )
            if col:
                return self.tape7[[col]].rename(
                    columns={col: "downwelling_irradiance"}
                )

        raise ValueError(
            "Cannot derive downwelling irradiance: no suitable flux or tape7 data available."
        )

    def _get_direct_native(self) -> pd.DataFrame:
        if self.flux is not None:
            col = _find_col(self.flux, ("direct_solar", "direct", "dir"))
            if col:
                return self.flux[[col]].rename(columns={col: "direct_irradiance"})
        if self.tape7 is not None:
            col = _find_col(self.tape7, ("drct_rflt", "direct", "ref_sol"))
            if col:
                return self.tape7[[col]].rename(columns={col: "direct_irradiance"})
        raise ValueError("Cannot derive direct irradiance from available output.")

    def _get_diffuse_native(self) -> pd.DataFrame:
        if self.flux is not None:
            col = _find_col(
                self.flux, ("diffuse_down", "diffuse_dn", "diff_dn", "dif_dn")
            )
            if col:
                return self.flux[[col]].rename(columns={col: "diffuse_irradiance"})
        if self.tape7 is not None:
            col = _find_col(self.tape7, ("solar_scat", "thrml_sct", "sol_scat"))
            if col:
                return self.tape7[[col]].rename(columns={col: "diffuse_irradiance"})
        raise ValueError("Cannot derive diffuse irradiance from available output.")

    def _convert_units(
        self, df: pd.DataFrame, value_col: str, units: str
    ) -> pd.DataFrame:
        """Convert a single-column native (W/cm²/cm⁻¹) DataFrame."""
        if units == "W/cm2/cm1":
            return df

        nu = df.index.values.astype(float)
        F_nu = df.iloc[:, 0].values.astype(float)

        if units in ("W/m2/nm", "mW/m2/nm"):
            lam, F_lam = irrad_wavenum_to_nm(F_nu, nu)
            scale = 1000.0 if units == "mW/m2/nm" else 1.0
            idx_name = "wavelength_nm"
            return pd.DataFrame(
                {value_col: F_lam * scale},
                index=pd.Index(lam, name=idx_name),
            )
        elif units == "W/m2/um":
            lam, F_lam = irrad_wavenum_to_um(F_nu, nu)
            return pd.DataFrame(
                {value_col: F_lam},
                index=pd.Index(lam, name="wavelength_um"),
            )
        else:
            raise ValueError(
                f"Unknown units '{units}'. "
                "Choose from: 'W/cm2/cm1', 'W/m2/nm', 'mW/m2/nm', 'W/m2/um'."
            )


def _find_col(df: pd.DataFrame, candidates: tuple) -> str | None:
    """Return the first column name that matches any candidate (case-insensitive)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    # Partial match
    for cand in candidates:
        for col_lower, col_orig in cols_lower.items():
            if cand.lower() in col_lower:
                return col_orig
    return None
