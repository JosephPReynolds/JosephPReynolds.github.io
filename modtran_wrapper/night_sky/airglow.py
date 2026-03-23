"""
Nightglow / airglow spectral radiance model.

Emission line parameters are drawn from:
  - Roach & Gordon (1973), "The Light of the Night Sky", Reidel.
  - Khomich, Semenov & Shefov (2008), "Airglow as an Indicator of Upper
    Atmospheric Structure and Dynamics", Springer.

All radiance values are representative of quiet (solar-minimum) conditions
at a mid-latitude site with no moonlight and no aurora.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Layer altitude constants (km)
# ---------------------------------------------------------------------------

OH_ALTITUDE_KM: float = 87.0    # OH Meinel band emission layer
OI_ALTITUDE_KM: float = 97.0    # Atomic oxygen green/red-line layer
NA_ALTITUDE_KM: float = 92.0    # Sodium D-line layer


# ---------------------------------------------------------------------------
# Emission-line database
# ---------------------------------------------------------------------------
#
# Each row:
#   species        -- string label
#   wavelength_nm  -- central wavelength (nm)
#   peak_radiance  -- peak spectral radiance at zenith, solar minimum
#                     (W/m²/sr/nm)
#   fwhm_nm        -- full-width at half-maximum of the line (nm)
#   altitude_km    -- approximate emission layer altitude
#
# Unit conversion note:
#   1 Rayleigh (R) = 10^10 photons/m²/sr/s
#   At 589 nm: 1 R ≈ 3.37e-12 W/m²/sr  (line-integrated)
#   Typical NaD: ~50 R → ~1.7e-10 W/m²/sr integrated over ~0.3 nm FWHM
#   → peak ≈ 5.6e-10 W/m²/sr/nm.
#
# We store representative peak spectral radiance values (W/m²/sr/nm).
#
_EMISSION_LINES: list[dict] = [
    # --- OH Meinel bands (strong NIR, very complex; representative features)
    # P-branch heads of (6-2), (8-3), (9-4), (7-2), (5-1) etc.
    {"species": "OH",  "wavelength_nm": 590.0, "peak_radiance": 8.0e-10, "fwhm_nm": 1.5,  "altitude_km": OH_ALTITUDE_KM},
    {"species": "OH",  "wavelength_nm": 628.0, "peak_radiance": 6.5e-10, "fwhm_nm": 2.0,  "altitude_km": OH_ALTITUDE_KM},
    {"species": "OH",  "wavelength_nm": 695.0, "peak_radiance": 1.4e-9,  "fwhm_nm": 3.0,  "altitude_km": OH_ALTITUDE_KM},
    {"species": "OH",  "wavelength_nm": 715.0, "peak_radiance": 9.0e-10, "fwhm_nm": 3.5,  "altitude_km": OH_ALTITUDE_KM},
    {"species": "OH",  "wavelength_nm": 732.0, "peak_radiance": 1.1e-9,  "fwhm_nm": 3.0,  "altitude_km": OH_ALTITUDE_KM},
    {"species": "OH",  "wavelength_nm": 748.0, "peak_radiance": 7.5e-10, "fwhm_nm": 4.0,  "altitude_km": OH_ALTITUDE_KM},
    {"species": "OH",  "wavelength_nm": 770.0, "peak_radiance": 2.0e-9,  "fwhm_nm": 5.0,  "altitude_km": OH_ALTITUDE_KM},
    {"species": "OH",  "wavelength_nm": 789.0, "peak_radiance": 1.6e-9,  "fwhm_nm": 4.5,  "altitude_km": OH_ALTITUDE_KM},
    {"species": "OH",  "wavelength_nm": 824.0, "peak_radiance": 2.5e-9,  "fwhm_nm": 5.0,  "altitude_km": OH_ALTITUDE_KM},
    {"species": "OH",  "wavelength_nm": 840.0, "peak_radiance": 2.8e-9,  "fwhm_nm": 5.5,  "altitude_km": OH_ALTITUDE_KM},
    {"species": "OH",  "wavelength_nm": 865.0, "peak_radiance": 3.2e-9,  "fwhm_nm": 6.0,  "altitude_km": OH_ALTITUDE_KM},
    # --- O2 Atmospheric band (b¹Σ → X³Σ)
    {"species": "O2",  "wavelength_nm": 762.0, "peak_radiance": 3.0e-9,  "fwhm_nm": 0.5,  "altitude_km": OH_ALTITUDE_KM},
    {"species": "O2",  "wavelength_nm": 689.0, "peak_radiance": 4.0e-10, "fwhm_nm": 0.8,  "altitude_km": OH_ALTITUDE_KM},
    # --- Sodium D doublet (²P → ²S)
    {"species": "NaD", "wavelength_nm": 589.0, "peak_radiance": 5.6e-10, "fwhm_nm": 0.06, "altitude_km": NA_ALTITUDE_KM},
    {"species": "NaD", "wavelength_nm": 589.6, "peak_radiance": 2.8e-10, "fwhm_nm": 0.06, "altitude_km": NA_ALTITUDE_KM},
    # --- O I green line (¹S → ¹D)
    {"species": "OI",  "wavelength_nm": 557.7, "peak_radiance": 2.2e-9,  "fwhm_nm": 0.03, "altitude_km": OI_ALTITUDE_KM},
    # --- O I red doublet (¹D → ³P)
    {"species": "OI",  "wavelength_nm": 630.0, "peak_radiance": 5.0e-10, "fwhm_nm": 0.04, "altitude_km": OI_ALTITUDE_KM},
    {"species": "OI",  "wavelength_nm": 636.4, "peak_radiance": 1.7e-10, "fwhm_nm": 0.04, "altitude_km": OI_ALTITUDE_KM},
    # --- N2+ first negative (B²Σ → X²Σ) — faint
    {"species": "N2+", "wavelength_nm": 391.4, "peak_radiance": 4.0e-11, "fwhm_nm": 0.5,  "altitude_km": OI_ALTITUDE_KM},
]

# Continuum parameters (W/m²/sr/nm at reference wavelengths)
_CONTINUUM_WL_NM = np.array([300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0])
_CONTINUUM_BASE  = np.array([2.0e-8, 3.5e-8, 5.0e-8, 6.0e-8, 7.0e-8, 8.5e-8, 1.0e-7, 1.2e-7])


class AirglowModel:
    """
    Nightglow / airglow spectral radiance model.

    Models the night-sky emission spectrum as a sum of Gaussian emission
    lines plus a smooth pseudo-continuum.  Values are representative of
    quiet (solar-minimum) nightglow at a mid-latitude site.

    Class-level altitude constants
    --------------------------------
    OH_ALTITUDE_KM : float = 87.0
    OI_ALTITUDE_KM : float = 97.0
    NA_ALTITUDE_KM : float = 92.0
    """

    OH_ALTITUDE_KM: float = OH_ALTITUDE_KM
    OI_ALTITUDE_KM: float = OI_ALTITUDE_KM
    NA_ALTITUDE_KM: float = NA_ALTITUDE_KM

    def __init__(self) -> None:
        # Build structured arrays from the line database for fast vectorised
        # evaluation.
        self._lines = _EMISSION_LINES

        wl   = np.array([l["wavelength_nm"]  for l in self._lines], dtype=float)
        pk   = np.array([l["peak_radiance"]  for l in self._lines], dtype=float)
        fwhm = np.array([l["fwhm_nm"]        for l in self._lines], dtype=float)

        # Convert FWHM to Gaussian sigma: FWHM = 2*sqrt(2*ln2)*sigma
        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

        self._line_wl_nm: np.ndarray    = wl
        self._line_peak: np.ndarray     = pk
        self._line_sigma_nm: np.ndarray = sigma

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def zenith_radiance(
        self,
        wavelength_nm: np.ndarray,
        solar_activity: float = 1.0,
    ) -> np.ndarray:
        """
        Compute zenith spectral radiance of the night sky (airglow only).

        Parameters
        ----------
        wavelength_nm : array_like
            Wavelength grid (nm).
        solar_activity : float, optional
            Dimensionless solar activity scale factor.  ``1.0`` corresponds
            to solar minimum; ``1.5`` is representative of solar maximum.
            Scales the smooth continuum component.  Emission-line strengths
            also scale linearly (they are weakly correlated with solar flux
            in the mesosphere, but for simplicity the same factor is applied).

        Returns
        -------
        np.ndarray
            Spectral radiance (W/m²/sr/nm), same shape as *wavelength_nm*.
        """
        wavelength_nm = np.asarray(wavelength_nm, dtype=float)
        if wavelength_nm.size == 0:
            return np.empty_like(wavelength_nm)

        # --- Continuum (scales with solar activity) ---
        continuum = np.interp(
            wavelength_nm,
            _CONTINUUM_WL_NM,
            _CONTINUUM_BASE * solar_activity,
            left=_CONTINUUM_BASE[0] * solar_activity,
            right=_CONTINUUM_BASE[-1] * solar_activity,
        )

        # --- Emission lines ---
        # Shape: (n_lines, n_wavelengths)
        delta = wavelength_nm[np.newaxis, :] - self._line_wl_nm[:, np.newaxis]
        sigma = self._line_sigma_nm[:, np.newaxis]
        peak  = self._line_peak[:, np.newaxis] * solar_activity

        # Gaussian profile: G(λ) = peak * exp(-0.5*(Δλ/σ)²)
        gaussians = peak * np.exp(-0.5 * (delta / sigma) ** 2)
        line_contribution = gaussians.sum(axis=0)

        return continuum + line_contribution

    def line_strengths(self) -> pd.DataFrame:
        """
        Return a DataFrame summarising all emission lines.

        Returns
        -------
        pd.DataFrame
            Columns: ``species``, ``wavelength_nm``, ``peak_radiance_W_m2_sr_nm``,
            ``fwhm_nm``, ``altitude_km``.
        """
        rows = [
            {
                "species":                  l["species"],
                "wavelength_nm":            l["wavelength_nm"],
                "peak_radiance_W_m2_sr_nm": l["peak_radiance"],
                "fwhm_nm":                  l["fwhm_nm"],
                "altitude_km":              l["altitude_km"],
            }
            for l in self._lines
        ]
        df = pd.DataFrame(rows)
        df.sort_values("wavelength_nm", inplace=True, ignore_index=True)
        return df
