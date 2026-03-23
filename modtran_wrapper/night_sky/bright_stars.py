"""Bright star catalog and stellar spectral irradiance model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Physical constants
_H = 6.62607015e-34   # Planck constant  J·s
_C = 2.99792458e8     # Speed of light    m/s
_K = 1.380649e-23     # Boltzmann constant  J/K

# Default bundled catalog path
_DEFAULT_CATALOG = Path(__file__).parent / "data" / "yale_bsc_bright.csv"


@dataclass
class StarEntry:
    """Single entry from the Yale Bright Star Catalog."""

    hr: int           # Harvard Revised catalogue number
    name: str
    ra_deg: float
    dec_deg: float
    vmag: float
    bv: float         # B−V colour index
    sptype: str       # simplified spectral type


class BrightStarCatalog:
    """Yale Bright Star Catalog wrapper with stellar irradiance modelling.

    Parameters
    ----------
    catalog_file:
        Path to the CSV file.  Defaults to the bundled
        ``night_sky/data/yale_bsc_bright.csv``.
    """

    # Vega flux at V=0 in W m⁻² nm⁻¹ at ~555.6 nm
    VEGA_FLUX_V_BAND: float = 3.53e-12  # W/m²/nm

    def __init__(self, catalog_file: Optional[Path] = None) -> None:
        path = Path(catalog_file) if catalog_file is not None else _DEFAULT_CATALOG
        if not path.exists():
            raise FileNotFoundError(f"Bright star catalog not found: {path}")
        self._df = pd.read_csv(path, comment="#")
        self._stars: list[StarEntry] = [
            StarEntry(
                hr=int(row["hr"]),
                name=str(row["name"]),
                ra_deg=float(row["ra_deg"]),
                dec_deg=float(row["dec_deg"]),
                vmag=float(row["vmag"]),
                bv=float(row["bv"]),
                sptype=str(row["sptype"]),
            )
            for _, row in self._df.iterrows()
        ]

    # ------------------------------------------------------------------
    # Catalog queries
    # ------------------------------------------------------------------

    @property
    def all_stars(self) -> list[StarEntry]:
        """All stars in the loaded catalog."""
        return list(self._stars)

    def __len__(self) -> int:
        return len(self._stars)

    def stars_in_fov(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_deg: float,
        vmag_limit: float = 6.5,
    ) -> list[StarEntry]:
        """Stars within *radius_deg* of (*ra_deg*, *dec_deg*) and ≤ *vmag_limit*.

        Great-circle separation:
        ``cos(sep) = sin(d1)*sin(d2) + cos(d1)*cos(d2)*cos(ra1−ra2)``
        """
        ra0 = np.radians(ra_deg)
        dec0 = np.radians(dec_deg)
        cos_radius = np.cos(np.radians(radius_deg))

        result: list[StarEntry] = []
        for s in self._stars:
            if s.vmag > vmag_limit:
                continue
            ra_s = np.radians(s.ra_deg)
            dec_s = np.radians(s.dec_deg)
            cos_sep = (
                np.sin(dec0) * np.sin(dec_s)
                + np.cos(dec0) * np.cos(dec_s) * np.cos(ra0 - ra_s)
            )
            # Clamp for numerical safety
            cos_sep = float(np.clip(cos_sep, -1.0, 1.0))
            if cos_sep >= cos_radius:
                result.append(s)
        return result

    # ------------------------------------------------------------------
    # Spectral irradiance
    # ------------------------------------------------------------------

    def spectral_irradiance(
        self,
        star: StarEntry,
        wavelength_nm: np.ndarray,
    ) -> np.ndarray:
        """Stellar spectral irradiance W m⁻² nm⁻¹ (outside atmosphere).

        Steps
        -----
        1. Ballesteros (2012) B−V → Teff conversion.
        2. Planck function B(λ, T) in W m⁻² sr⁻¹ nm⁻¹.
        3. Synthetic V-band integration (Gaussian BP, centre 550 nm, FWHM 85 nm)
           → scale to match observed magnitude.
        4. M-star TiO absorption correction (bv > 1.4).
        """
        wavelength_nm = np.asarray(wavelength_nm, dtype=float)

        # 1. Effective temperature via Ballesteros (2012)
        bv = star.bv
        # Protect against extreme B-V
        bv_safe = float(np.clip(bv, -0.4, 2.5))
        denom1 = 0.92 * bv_safe + 1.7
        denom2 = 0.92 * bv_safe + 0.62
        # Guard against zero denominators
        denom1 = denom1 if abs(denom1) > 1e-6 else 1e-6
        denom2 = denom2 if abs(denom2) > 1e-6 else 1e-6
        teff = 4600.0 * (1.0 / denom1 + 1.0 / denom2)
        teff = float(np.clip(teff, 2500.0, 50000.0))

        # 2. Planck function (per nm)
        planck = _planck_per_nm(wavelength_nm, teff)

        # 3. Scale to V magnitude
        #    Synthetic V-band: Gaussian centred at 550 nm, FWHM 85 nm
        vband_wl = np.linspace(400.0, 700.0, 3000)
        sigma_v = 85.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        vband_T = np.exp(-0.5 * ((vband_wl - 550.0) / sigma_v) ** 2)
        planck_vband = _planck_per_nm(vband_wl, teff)
        synthetic_V = float(np.trapezoid(vband_T * planck_vband, vband_wl))

        target_flux = self.VEGA_FLUX_V_BAND * 10.0 ** (-star.vmag / 2.5)

        if synthetic_V <= 0.0:
            scale = 0.0
        else:
            scale = target_flux / synthetic_V

        irradiance = planck * scale

        # 4. M-star TiO absorption correction
        if bv > 1.4:
            depth = np.clip((bv - 1.4) / 0.6, 0.0, 1.0)  # 0→1 over bv 1.4–2.0
            # TiO band at 715 nm
            tio715 = depth * 0.4 * np.exp(-0.5 * ((wavelength_nm - 715.0) / 40.0) ** 2)
            # TiO band at 800 nm
            tio800 = depth * 0.25 * np.exp(-0.5 * ((wavelength_nm - 800.0) / 35.0) ** 2)
            absorption = 1.0 - tio715 - tio800
            absorption = np.clip(absorption, 0.0, 1.0)
            irradiance = irradiance * absorption

        return irradiance

    def total_fov_irradiance(
        self,
        ra_deg: float,
        dec_deg: float,
        fov_radius_deg: float,
        wavelength_nm: np.ndarray,
        vmag_limit: float = 6.5,
    ) -> np.ndarray:
        """Sum of spectral irradiance of all stars in the field of view.

        Returns W m⁻² nm⁻¹.
        """
        wavelength_nm = np.asarray(wavelength_nm, dtype=float)
        stars = self.stars_in_fov(ra_deg, dec_deg, fov_radius_deg, vmag_limit)
        total = np.zeros_like(wavelength_nm)
        for s in stars:
            total += self.spectral_irradiance(s, wavelength_nm)
        return total


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _planck_per_nm(wavelength_nm: np.ndarray, teff: float) -> np.ndarray:
    """Planck function B(λ, T) in W m⁻² sr⁻¹ nm⁻¹.

    Parameters
    ----------
    wavelength_nm:
        Wavelength array in nm.
    teff:
        Temperature in K.

    Notes
    -----
    Standard formula in SI (λ in metres), then convert per-metre to per-nm
    by dividing by 1e9.
    """
    wl_m = wavelength_nm * 1e-9   # nm → m
    # Avoid division by zero
    safe_wl = np.where(wl_m > 0, wl_m, 1e-15)

    # Exponent  hc / (λ k T)
    exponent = (_H * _C) / (safe_wl * _K * teff)
    # Clip to avoid overflow in exp
    exponent = np.clip(exponent, 0.0, 700.0)

    numerator = 2.0 * _H * _C ** 2 / safe_wl ** 5  # W/m²/sr/m
    denominator = np.exp(exponent) - 1.0
    # Guard against denominator underflow
    denominator = np.where(denominator > 0, denominator, np.finfo(float).tiny)

    B_per_m = numerator / denominator  # W/m²/sr/m
    B_per_nm = B_per_m / 1e9           # W/m²/sr/nm
    return B_per_nm
