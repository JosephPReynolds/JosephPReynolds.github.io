"""Unit conversion utilities for MODTRAN spectral output.

MODTRAN native units
--------------------
Wavenumber    : cm⁻¹
Radiance      : W / (cm² sr cm⁻¹)
Irradiance    : W / (cm² cm⁻¹)
Transmittance : dimensionless

Common output units
-------------------
Wavelength    : nm  or  µm
Radiance      : W / (m² sr nm)
Irradiance    : W / (m² nm)  or  mW / (m² nm)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Spectral coordinate conversions
# ---------------------------------------------------------------------------

def wavenum_to_nm(wavenum_cm1: np.ndarray) -> np.ndarray:
    """Convert wavenumber (cm⁻¹) to wavelength (nm).

    λ(nm) = 1e7 / ν(cm⁻¹)
    """
    wavenum_cm1 = np.asarray(wavenum_cm1, dtype=float)
    return 1.0e7 / wavenum_cm1


def nm_to_wavenum(wavelength_nm: np.ndarray) -> np.ndarray:
    """Convert wavelength (nm) to wavenumber (cm⁻¹).

    ν(cm⁻¹) = 1e7 / λ(nm)
    """
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    return 1.0e7 / wavelength_nm


def wavenum_to_um(wavenum_cm1: np.ndarray) -> np.ndarray:
    """Convert wavenumber (cm⁻¹) to wavelength (µm)."""
    wavenum_cm1 = np.asarray(wavenum_cm1, dtype=float)
    return 1.0e4 / wavenum_cm1


def um_to_wavenum(wavelength_um: np.ndarray) -> np.ndarray:
    """Convert wavelength (µm) to wavenumber (cm⁻¹)."""
    wavelength_um = np.asarray(wavelength_um, dtype=float)
    return 1.0e4 / wavelength_um


# ---------------------------------------------------------------------------
# Spectral density conversions (irradiance)
# ---------------------------------------------------------------------------

def irrad_wavenum_to_nm(
    irrad_per_wavenum: np.ndarray,
    wavenum_cm1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert spectral irradiance per wavenumber to per wavelength (nm).

    Applies the chain rule:
        F_λ(W/m²/nm) = F_ν(W/cm²/cm⁻¹) × |dν/dλ| × unit_factors

    where |dν/dλ| = ν²/c = (1/λ²) → after unit conversion:
        F_λ [W/m²/nm] = F_ν [W/cm²/cm⁻¹] × ν²(cm⁻¹) × 1e7(nm/cm) × 1e−4(m²/cm²)
                       = F_ν × ν² × 1e3

    Parameters
    ----------
    irrad_per_wavenum : array-like
        Spectral irradiance in W / (cm² cm⁻¹).
    wavenum_cm1 : array-like
        Corresponding wavenumber axis in cm⁻¹.

    Returns
    -------
    wavelength_nm : np.ndarray
        Wavelength axis in nm (note: order reversed relative to wavenumber).
    irrad_per_nm : np.ndarray
        Spectral irradiance in W / (m² nm).
    """
    F_nu = np.asarray(irrad_per_wavenum, dtype=float)
    nu = np.asarray(wavenum_cm1, dtype=float)

    lam_nm = wavenum_to_nm(nu)
    # |dν/dλ|  in cm⁻¹/nm  =  ν² / 1e7
    # F_λ [W/cm²/nm] = F_ν × ν²/1e7
    # F_λ [W/m²/nm]  = F_ν × ν²/1e7 × 1e4  (cm² → m²)
    F_lam = F_nu * nu**2 / 1.0e3

    # Sort by increasing wavelength
    sort_idx = np.argsort(lam_nm)
    return lam_nm[sort_idx], F_lam[sort_idx]


def irrad_wavenum_to_um(
    irrad_per_wavenum: np.ndarray,
    wavenum_cm1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert spectral irradiance W/(cm² cm⁻¹) → W/(m² µm).

    Returns
    -------
    wavelength_um : np.ndarray
    irrad_per_um : np.ndarray
        In W / (m² µm).
    """
    F_nu = np.asarray(irrad_per_wavenum, dtype=float)
    nu = np.asarray(wavenum_cm1, dtype=float)

    lam_um = wavenum_to_um(nu)
    # |dν/dλ| in cm⁻¹/µm = ν² / 1e4
    # F_λ [W/m²/µm] = F_ν [W/cm²/cm⁻¹] × (ν²/1e4) × 1e4
    F_lam = F_nu * nu**2  # W / (m² µm) — the 1e4 factors cancel

    sort_idx = np.argsort(lam_um)
    return lam_um[sort_idx], F_lam[sort_idx]


# ---------------------------------------------------------------------------
# Radiance conversions (per steradian)
# ---------------------------------------------------------------------------

def radiance_wavenum_to_nm(
    radiance_per_wavenum: np.ndarray,
    wavenum_cm1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert spectral radiance W/(cm² sr cm⁻¹) → W/(m² sr nm)."""
    # Same conversion factor as irradiance (sr cancels)
    return irrad_wavenum_to_nm(radiance_per_wavenum, wavenum_cm1)


# ---------------------------------------------------------------------------
# Convenience: convert a DataFrame column
# ---------------------------------------------------------------------------

def convert_spectrum_to_nm(
    df,
    irrad_col: str,
    wavenum_index_name: str = "wavenumber_cm1",
    output_col: str | None = None,
):
    """Convert a spectral irradiance column in a tape7/flux DataFrame to nm basis.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with wavenumber index.
    irrad_col : str
        Column name of the spectral irradiance (W/cm²/cm⁻¹).
    wavenum_index_name : str
        Name of the wavenumber index.
    output_col : str or None
        Column name for the result.  Defaults to ``irrad_col + '_per_nm'``.

    Returns
    -------
    pd.DataFrame
        New DataFrame indexed by wavelength_nm with the converted column.
    """
    import pandas as pd

    nu = df.index.values.astype(float)
    F_nu = df[irrad_col].values.astype(float)

    lam_nm, F_lam = irrad_wavenum_to_nm(F_nu, nu)

    out_col = output_col or (irrad_col + "_per_nm")
    return pd.DataFrame({out_col: F_lam}, index=pd.Index(lam_nm, name="wavelength_nm"))
