"""Bandpass filter definitions for spectral integration."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.interpolate import interp1d

# NumPy 2.0 renamed trapz -> trapezoid; keep compatibility with both.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))  # NumPy 1.x / 2.x compat


@dataclass
class BandpassFilter:
    """Spectral bandpass filter defined on an arbitrary wavelength grid.

    Parameters
    ----------
    wavelength_nm:
        Wavelength grid in nm.  Must be positive and strictly ascending.
    transmission:
        Transmission values in [0, 1] on the same grid.
    label:
        Human-readable name for the filter (used in plots and DataFrames).
    """

    wavelength_nm: np.ndarray
    transmission: np.ndarray
    label: str = ""

    def __post_init__(self) -> None:
        self.wavelength_nm = np.asarray(self.wavelength_nm, dtype=float)
        self.transmission = np.asarray(self.transmission, dtype=float)

        if self.wavelength_nm.shape != self.transmission.shape:
            raise ValueError(
                f"wavelength_nm and transmission must have the same shape; "
                f"got {self.wavelength_nm.shape} vs {self.transmission.shape}."
            )
        if self.wavelength_nm.ndim != 1:
            raise ValueError("wavelength_nm must be a 1-D array.")
        if len(self.wavelength_nm) < 2:
            raise ValueError("wavelength_nm must contain at least 2 points.")
        if np.any(self.wavelength_nm <= 0):
            raise ValueError("All wavelength values must be positive.")
        if not np.all(np.diff(self.wavelength_nm) > 0):
            raise ValueError("wavelength_nm must be strictly ascending.")
        if np.any(self.transmission < 0) or np.any(self.transmission > 1):
            raise ValueError("Transmission values must lie in [0, 1].")

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def gaussian(
        cls,
        center_nm: float,
        fwhm_nm: float,
        wavelength_nm: np.ndarray,
        peak: float = 1.0,
        label: str = "",
    ) -> BandpassFilter:
        """Gaussian bandpass: T(λ) = peak · exp(−4 ln2 · (λ − center)² / fwhm²).

        Parameters
        ----------
        center_nm:
            Central wavelength in nm.
        fwhm_nm:
            Full-width at half-maximum in nm.  Must be > 0.
        wavelength_nm:
            Wavelength grid on which to evaluate the filter.
        peak:
            Peak transmission, in (0, 1].
        label:
            Optional label string.
        """
        if fwhm_nm <= 0:
            raise ValueError("fwhm_nm must be positive.")
        wavelength_nm = np.asarray(wavelength_nm, dtype=float)
        sigma2 = fwhm_nm ** 2 / (8.0 * math.log(2.0))
        trans = peak * np.exp(-((wavelength_nm - center_nm) ** 2) / (2.0 * sigma2))
        trans = np.clip(trans, 0.0, 1.0)
        return cls(wavelength_nm=wavelength_nm, transmission=trans, label=label)

    @classmethod
    def rectangular(
        cls,
        wl_start_nm: float,
        wl_end_nm: float,
        wavelength_nm: np.ndarray,
        peak: float = 1.0,
        label: str = "",
    ) -> BandpassFilter:
        """Rectangular (top-hat) bandpass.

        Parameters
        ----------
        wl_start_nm:
            Short-wavelength edge in nm.
        wl_end_nm:
            Long-wavelength edge in nm.
        wavelength_nm:
            Wavelength grid on which to evaluate the filter.
        peak:
            In-band transmission, in (0, 1].
        label:
            Optional label string.
        """
        if wl_start_nm >= wl_end_nm:
            raise ValueError("wl_start_nm must be less than wl_end_nm.")
        wavelength_nm = np.asarray(wavelength_nm, dtype=float)
        trans = np.where(
            (wavelength_nm >= wl_start_nm) & (wavelength_nm <= wl_end_nm),
            float(peak),
            0.0,
        )
        return cls(wavelength_nm=wavelength_nm, transmission=trans, label=label)

    @classmethod
    def from_file(
        cls,
        path: Path,
        label: str = "",
        delimiter: Optional[str] = None,
        comments: str = "#",
    ) -> BandpassFilter:
        """Load a bandpass from a 2-column text file (wavelength_nm, transmission).

        Parameters
        ----------
        path:
            Path to the text/CSV file.
        label:
            Optional label (defaults to the file stem).
        delimiter:
            Column delimiter.  ``None`` means any whitespace.
        comments:
            Character used to mark comment lines.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Bandpass file not found: {path}")
        data = np.loadtxt(str(path), delimiter=delimiter, comments=comments)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(
                f"Expected a file with at least 2 columns; got shape {data.shape}."
            )
        wl = data[:, 0]
        tr = data[:, 1]
        # Sort by wavelength in case file is not ordered
        order = np.argsort(wl)
        wl, tr = wl[order], tr[order]
        lbl = label if label else path.stem
        return cls(wavelength_nm=wl, transmission=tr, label=lbl)

    # ------------------------------------------------------------------
    # Common detector / image-intensifier responses
    # ------------------------------------------------------------------

    @classmethod
    def gen3_intensifier(cls, wavelength_nm: np.ndarray) -> BandpassFilter:
        """Approximate spectral response of a Gen-3 image intensifier (GaAs photocathode).

        The GaAs photocathode response spans ~450 – 900 nm with a broad peak
        near 700–750 nm.  This tabulation is based on published GaAs quantum-
        efficiency curves (Photonis, ITT datasheets).  Values are normalised to
        a peak of 1.0.
        """
        # Reference nodes (nm, QE_normalised)
        _wl_ref = np.array([
            350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000,
        ], dtype=float)
        _tr_ref = np.array([
            0.00, 0.02, 0.10, 0.28, 0.50, 0.68, 0.82, 0.92, 1.00, 0.88, 0.60,
            0.25, 0.05, 0.00,
        ], dtype=float)

        wavelength_nm = np.asarray(wavelength_nm, dtype=float)
        interp = interp1d(
            _wl_ref, _tr_ref, kind="linear", bounds_error=False, fill_value=0.0
        )
        trans = np.clip(interp(wavelength_nm), 0.0, 1.0)
        return cls(wavelength_nm=wavelength_nm, transmission=trans, label="Gen3-GaAs")

    @classmethod
    def swir_ingaas(cls, wavelength_nm: np.ndarray) -> BandpassFilter:
        """Approximate spectral response of an InGaAs SWIR focal-plane array.

        Typical cut-on ~950 nm, flat response from ~1000 – 1600 nm, cut-off
        ~1700 nm.  Based on published Sensors Unlimited / FLIR InGaAs curves.
        """
        _wl_ref = np.array([
            800, 900, 950, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1650, 1700, 1750,
        ], dtype=float)
        _tr_ref = np.array([
            0.00, 0.00, 0.15, 0.80, 0.90, 0.92, 0.91, 0.90, 0.90, 0.88, 0.70, 0.10, 0.00,
        ], dtype=float)

        wavelength_nm = np.asarray(wavelength_nm, dtype=float)
        interp = interp1d(
            _wl_ref, _tr_ref, kind="linear", bounds_error=False, fill_value=0.0
        )
        trans = np.clip(interp(wavelength_nm), 0.0, 1.0)
        return cls(wavelength_nm=wavelength_nm, transmission=trans, label="SWIR-InGaAs")

    @classmethod
    def mwir_hgcdte(cls, wavelength_nm: np.ndarray) -> BandpassFilter:
        """Approximate spectral response of an HgCdTe MWIR detector (3–5 µm band).

        Cut-on near 3000 nm, peak QE from 3500 – 4800 nm, cut-off ~5000 nm.
        Based on published Teledyne/DRS HgCdTe MWIR data.
        """
        _wl_ref = np.array([
            2500, 2800, 3000, 3200, 3500, 4000, 4500, 4700, 4900, 5000, 5200,
        ], dtype=float)
        _tr_ref = np.array([
            0.00, 0.00, 0.10, 0.55, 0.85, 0.90, 0.88, 0.80, 0.40, 0.05, 0.00,
        ], dtype=float)

        wavelength_nm = np.asarray(wavelength_nm, dtype=float)
        interp = interp1d(
            _wl_ref, _tr_ref, kind="linear", bounds_error=False, fill_value=0.0
        )
        trans = np.clip(interp(wavelength_nm), 0.0, 1.0)
        return cls(wavelength_nm=wavelength_nm, transmission=trans, label="MWIR-HgCdTe")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def center_nm(self) -> float:
        """Centroid wavelength, weighted by transmission.

        Defined as ∫ λ T(λ) dλ / ∫ T(λ) dλ.
        Returns the midpoint of the wavelength range if the integral is zero.
        """
        denom = float(_trapz(self.transmission, self.wavelength_nm))
        if denom == 0.0:
            return float(0.5 * (self.wavelength_nm[0] + self.wavelength_nm[-1]))
        numer = float(_trapz(self.wavelength_nm * self.transmission, self.wavelength_nm))
        return numer / denom

    @property
    def bandwidth_nm(self) -> float:
        """Equivalent noise bandwidth: ∫ T(λ) dλ / max(T).

        Returns 0 if the filter is identically zero.
        """
        peak = float(np.max(self.transmission))
        if peak == 0.0:
            return 0.0
        return float(_trapz(self.transmission, self.wavelength_nm)) / peak

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------

    def integrated_response(
        self,
        wavelength_nm: np.ndarray,
        spectrum_W_m2_sr_nm: np.ndarray,
    ) -> float:
        """Integrate *spectrum* through this bandpass.

        Computes ∫ T(λ) · spectrum(λ) dλ in W m⁻² sr⁻¹.

        The filter transmission is interpolated onto the spectrum's wavelength
        grid before integration (zero outside the filter's wavelength range).

        Parameters
        ----------
        wavelength_nm:
            Wavelength grid of the input spectrum in nm.
        spectrum_W_m2_sr_nm:
            Spectral radiance in W m⁻² sr⁻¹ nm⁻¹.

        Returns
        -------
        float
            Integrated radiance in W m⁻² sr⁻¹.
        """
        wavelength_nm = np.asarray(wavelength_nm, dtype=float)
        spectrum = np.asarray(spectrum_W_m2_sr_nm, dtype=float)
        if wavelength_nm.shape != spectrum.shape:
            raise ValueError(
                "wavelength_nm and spectrum_W_m2_sr_nm must have the same shape."
            )
        # Interpolate filter onto spectrum grid
        interp = interp1d(
            self.wavelength_nm,
            self.transmission,
            kind="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        t_interp = np.clip(interp(wavelength_nm), 0.0, 1.0)
        return float(_trapz(t_interp * spectrum, wavelength_nm))

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_transmission(self) -> None:
        """Quick matplotlib plot of the filter transmission curve."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install it with: pip install matplotlib"
            )
        fig, ax = plt.subplots()
        ax.plot(self.wavelength_nm, self.transmission)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Transmission")
        title = self.label if self.label else "Bandpass Filter"
        ax.set_title(title)
        ax.set_ylim(0, max(1.05, float(np.max(self.transmission)) * 1.05))
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
