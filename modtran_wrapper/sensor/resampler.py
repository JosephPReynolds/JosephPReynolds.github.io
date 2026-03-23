"""Spectral resampling and unit-conversion utilities for NV-PM / IPM preparation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

if TYPE_CHECKING:
    from .bandpass import BandpassFilter

# Physical constants
_H = 6.62607015e-34    # Planck constant  J·s
_C_NM = 2.99792458e17  # Speed of light   nm/s  (= 2.998e8 m/s × 1e9 nm/m)


class SpectralResampler:
    """Resamples and converts spectral data for NV-PM / IPM input preparation.

    All methods are stateless; a single instance can be reused for any number
    of spectra.

    Typical workflow
    ----------------
    1. Obtain a spectral radiance array ``L(λ)`` from MODTRAN or a night-sky
       model (W m⁻² sr⁻¹ nm⁻¹).
    2. Call :meth:`resample` to project onto the sensor wavelength grid.
    3. Call :meth:`apply_bandpass` or :meth:`multi_band_resampling` to
       integrate through one or more :class:`~modtran_wrapper.sensor.BandpassFilter`
       instances.
    4. Optionally call :meth:`to_ipm_format` to prepare the final array for
       writing into IPM / NV-PM input files.
    """

    # ------------------------------------------------------------------
    # Basic resampling
    # ------------------------------------------------------------------

    def resample(
        self,
        wavelength_nm: np.ndarray,
        spectrum: np.ndarray,
        target_wavelengths_nm: np.ndarray,
        method: str = "linear",
    ) -> np.ndarray:
        """Resample *spectrum* onto *target_wavelengths_nm*.

        Values outside the input wavelength range are set to zero.

        Parameters
        ----------
        wavelength_nm:
            Input wavelength grid in nm (must be ascending).
        spectrum:
            Input spectral values.  Any physical units are preserved.
        target_wavelengths_nm:
            Output wavelength grid in nm.
        method:
            Interpolation method: ``"linear"``, ``"cubic"``, or ``"nearest"``.
            ``"cubic"`` uses a not-a-knot cubic spline.

        Returns
        -------
        np.ndarray
            Resampled spectrum on *target_wavelengths_nm*.
        """
        wavelength_nm = np.asarray(wavelength_nm, dtype=float)
        spectrum = np.asarray(spectrum, dtype=float)
        target_wavelengths_nm = np.asarray(target_wavelengths_nm, dtype=float)

        _validate_1d_matching(wavelength_nm, spectrum, "wavelength_nm", "spectrum")

        kind_map = {"linear": "linear", "cubic": "cubic", "nearest": "nearest"}
        if method not in kind_map:
            raise ValueError(
                f"method must be one of {list(kind_map)}; got {method!r}."
            )

        interp = interp1d(
            wavelength_nm,
            spectrum,
            kind=kind_map[method],
            bounds_error=False,
            fill_value=0.0,
        )
        return interp(target_wavelengths_nm)

    # ------------------------------------------------------------------
    # SRF application
    # ------------------------------------------------------------------

    def apply_srf(
        self,
        wavelength_nm: np.ndarray,
        spectrum: np.ndarray,
        srf: np.ndarray,
        target_wavelengths_nm: np.ndarray,
    ) -> np.ndarray:
        """Apply a spectral response function (SRF) matrix to *spectrum*.

        Each row of *srf* is the normalised response of one output band,
        evaluated at *wavelength_nm*.  The method computes:

        .. math::

            \\text{result}[i] = \\int \\text{srf}[i, \\lambda]
                                       \\cdot \\text{spectrum}[\\lambda]\\, d\\lambda

        using ``numpy.trapz`` along the *wavelength_nm* axis.

        Parameters
        ----------
        wavelength_nm:
            Input wavelength grid in nm.
        spectrum:
            Input spectrum (any units).
        srf:
            2-D array of shape ``(n_bands, len(wavelength_nm))``.  Each row is
            the (not necessarily normalised) response for one output band.
        target_wavelengths_nm:
            Output wavelength grid (used only to verify *srf* column count and
            as a convenience — *srf* must already be evaluated on
            *wavelength_nm*, not on this grid).

        Returns
        -------
        np.ndarray
            Shape ``(n_bands,)`` – integrated radiance per band in the same
            units as *spectrum* × nm (e.g., W m⁻² sr⁻¹ if input is
            W m⁻² sr⁻¹ nm⁻¹).

        Notes
        -----
        *srf* rows should already be evaluated on *wavelength_nm*.  The
        parameter *target_wavelengths_nm* is accepted for API completeness but
        is not used for interpolation here; the caller is responsible for
        ensuring *srf* columns correspond to *wavelength_nm*.
        """
        wavelength_nm = np.asarray(wavelength_nm, dtype=float)
        spectrum = np.asarray(spectrum, dtype=float)
        srf = np.asarray(srf, dtype=float)

        _validate_1d_matching(wavelength_nm, spectrum, "wavelength_nm", "spectrum")
        if srf.ndim != 2:
            raise ValueError(f"srf must be 2-D; got shape {srf.shape}.")
        if srf.shape[1] != len(wavelength_nm):
            raise ValueError(
                f"srf.shape[1] ({srf.shape[1]}) must equal len(wavelength_nm) "
                f"({len(wavelength_nm)})."
            )

        n_bands = srf.shape[0]
        result = np.empty(n_bands, dtype=float)
        for i in range(n_bands):
            result[i] = float(np.trapz(srf[i] * spectrum, wavelength_nm))
        return result

    # ------------------------------------------------------------------
    # Gaussian convolution
    # ------------------------------------------------------------------

    def convolve_gaussian(
        self,
        wavelength_nm: np.ndarray,
        spectrum: np.ndarray,
        fwhm_nm: float,
    ) -> np.ndarray:
        """Convolve *spectrum* with a Gaussian kernel of given FWHM.

        The convolution is performed in pixel space using
        :func:`scipy.ndimage.gaussian_filter1d`.  If the wavelength grid is
        not uniform, the spectrum is first resampled to a uniform grid with the
        same number of points, convolved, and then resampled back.

        Parameters
        ----------
        wavelength_nm:
            Wavelength grid in nm (need not be uniform).
        spectrum:
            Input spectral values.
        fwhm_nm:
            FWHM of the Gaussian kernel in nm.

        Returns
        -------
        np.ndarray
            Convolved spectrum on the original *wavelength_nm* grid.
        """
        wavelength_nm = np.asarray(wavelength_nm, dtype=float)
        spectrum = np.asarray(spectrum, dtype=float)
        _validate_1d_matching(wavelength_nm, spectrum, "wavelength_nm", "spectrum")

        if fwhm_nm <= 0:
            raise ValueError("fwhm_nm must be positive.")

        diffs = np.diff(wavelength_nm)
        # Consider grid uniform if all spacings agree to 1 part in 1e4
        is_uniform = bool(
            np.max(np.abs(diffs - diffs[0])) / (diffs[0] + 1e-30) < 1e-4
        )

        if is_uniform:
            d_lam = float(diffs[0])
            sigma_pixels = fwhm_nm / (2.0 * math.sqrt(2.0 * math.log(2.0)) * d_lam)
            return gaussian_filter1d(spectrum, sigma=sigma_pixels, mode="constant", cval=0.0)
        else:
            # Resample to uniform grid, convolve, resample back
            n = len(wavelength_nm)
            wl_uniform = np.linspace(wavelength_nm[0], wavelength_nm[-1], n)
            d_lam = float(wl_uniform[1] - wl_uniform[0])
            spec_uniform = self.resample(wavelength_nm, spectrum, wl_uniform, method="linear")
            sigma_pixels = fwhm_nm / (2.0 * math.sqrt(2.0 * math.log(2.0)) * d_lam)
            spec_conv = gaussian_filter1d(
                spec_uniform, sigma=sigma_pixels, mode="constant", cval=0.0
            )
            # Resample back onto original grid
            return self.resample(wl_uniform, spec_conv, wavelength_nm, method="linear")

    # ------------------------------------------------------------------
    # Bandpass integration
    # ------------------------------------------------------------------

    def apply_bandpass(
        self,
        wavelength_nm: np.ndarray,
        spectrum: np.ndarray,
        bandpass: BandpassFilter,
    ) -> float:
        """Integrate *spectrum* through *bandpass*.

        Computes ∫ T(λ) · spectrum(λ) dλ using the spectrum's wavelength grid
        as the integration variable.

        Parameters
        ----------
        wavelength_nm:
            Wavelength grid of the input spectrum in nm.
        spectrum:
            Input spectral radiance (W m⁻² sr⁻¹ nm⁻¹ or similar).
        bandpass:
            :class:`~modtran_wrapper.sensor.BandpassFilter` instance.

        Returns
        -------
        float
            Integrated value in the same units as *spectrum* × nm
            (e.g., W m⁻² sr⁻¹).
        """
        return bandpass.integrated_response(wavelength_nm, spectrum)

    # ------------------------------------------------------------------
    # Unit conversion
    # ------------------------------------------------------------------

    def to_photon_rate(
        self,
        wavelength_nm: np.ndarray,
        irradiance_W_m2_nm: np.ndarray,
    ) -> np.ndarray:
        """Convert spectral irradiance W m⁻² nm⁻¹ to photons s⁻¹ m⁻² nm⁻¹.

        Uses the relation N = E λ / (h c):

        .. math::

            N(\\lambda) = \\frac{E(\\lambda)\\,\\lambda}{h c}

        with *λ* in nm and *c* in nm s⁻¹ so that the result is in the same
        per-nm spectral density units.

        Parameters
        ----------
        wavelength_nm:
            Wavelength grid in nm.
        irradiance_W_m2_nm:
            Spectral irradiance in W m⁻² nm⁻¹.

        Returns
        -------
        np.ndarray
            Photon flux in photons s⁻¹ m⁻² nm⁻¹.
        """
        wavelength_nm = np.asarray(wavelength_nm, dtype=float)
        irradiance = np.asarray(irradiance_W_m2_nm, dtype=float)
        _validate_1d_matching(wavelength_nm, irradiance, "wavelength_nm", "irradiance_W_m2_nm")

        # E_photon = h*c/λ with c in nm/s gives J/photon when λ is in nm.
        # N = E / E_photon = E * λ / (h*c_nm)
        safe_wl = np.where(wavelength_nm > 0, wavelength_nm, 1.0)
        return irradiance * safe_wl / (_H * _C_NM)

    # ------------------------------------------------------------------
    # IPM / NV-PM output formatting
    # ------------------------------------------------------------------

    def to_ipm_format(
        self,
        wavelength_nm: np.ndarray,
        radiance_W_m2_sr_nm: np.ndarray,
        bandpass: Optional[BandpassFilter] = None,
        output_unit: str = "W/m2/sr/nm",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare spectral data for NV-PM / IPM input files.

        Parameters
        ----------
        wavelength_nm:
            Input wavelength grid in nm.
        radiance_W_m2_sr_nm:
            Spectral radiance in W m⁻² sr⁻¹ nm⁻¹.
        bandpass:
            Optional :class:`~modtran_wrapper.sensor.BandpassFilter`.  When
            provided the radiance is integrated over the band and the returned
            wavelength is the bandpass centroid.
        output_unit:
            One of:

            * ``"W/m2/sr/nm"``   – spectral radiance (default, no conversion)
            * ``"ph/s/m2/sr/nm"``– photon spectral radiance
            * ``"W/m2/sr"``      – spectrally integrated radiance (requires
              *bandpass*)

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(wavelength_nm, spectrum)`` in the requested format.  If
            *bandpass* is provided and *output_unit* is ``"W/m2/sr"``, both
            returned arrays are length-1.
        """
        _VALID_UNITS = {"W/m2/sr/nm", "ph/s/m2/sr/nm", "W/m2/sr"}
        if output_unit not in _VALID_UNITS:
            raise ValueError(
                f"output_unit must be one of {_VALID_UNITS}; got {output_unit!r}."
            )

        wavelength_nm = np.asarray(wavelength_nm, dtype=float)
        radiance = np.asarray(radiance_W_m2_sr_nm, dtype=float)

        if bandpass is not None:
            # Integrate through the bandpass
            integrated = self.apply_bandpass(wavelength_nm, radiance, bandpass)
            center = bandpass.center_nm
            if output_unit == "W/m2/sr":
                return np.array([center]), np.array([integrated])
            elif output_unit == "ph/s/m2/sr/nm":
                # Convert the integrated value back to approximate photon rate
                # by treating the band as a single point at center_nm.
                # For a full photon-rate spectrum integrate before calling this.
                phot = float(integrated * center / (_H * _C_NM))
                return np.array([center]), np.array([phot])
            else:  # W/m2/sr/nm – just return the integrated scalar
                return np.array([center]), np.array([integrated])

        # No bandpass – return spectral array in requested unit
        if output_unit == "W/m2/sr/nm":
            return wavelength_nm.copy(), radiance.copy()
        elif output_unit == "ph/s/m2/sr/nm":
            photon_rad = self.to_photon_rate(wavelength_nm, radiance)
            return wavelength_nm.copy(), photon_rad
        else:
            raise ValueError(
                "output_unit='W/m2/sr' requires a bandpass to be provided."
            )

    # ------------------------------------------------------------------
    # Multi-band
    # ------------------------------------------------------------------

    def multi_band_resampling(
        self,
        wavelength_nm: np.ndarray,
        spectrum: np.ndarray,
        bandpasses: list[BandpassFilter],
    ) -> pd.DataFrame:
        """Apply multiple bandpass filters and return a summary DataFrame.

        Parameters
        ----------
        wavelength_nm:
            Wavelength grid of the input spectrum in nm.
        spectrum:
            Input spectral radiance in W m⁻² sr⁻¹ nm⁻¹.
        bandpasses:
            List of :class:`~modtran_wrapper.sensor.BandpassFilter` instances.

        Returns
        -------
        pd.DataFrame
            Columns:

            * ``band_label``          – filter label (str)
            * ``center_nm``           – centroid wavelength of the filter (nm)
            * ``bandwidth_nm``        – equivalent noise bandwidth (nm)
            * ``integrated_radiance`` – ∫ T(λ) L(λ) dλ in W m⁻² sr⁻¹
        """
        if not bandpasses:
            return pd.DataFrame(
                columns=["band_label", "center_nm", "bandwidth_nm", "integrated_radiance"]
            )

        rows = []
        for bp in bandpasses:
            integrated = self.apply_bandpass(wavelength_nm, spectrum, bp)
            rows.append(
                {
                    "band_label": bp.label,
                    "center_nm": bp.center_nm,
                    "bandwidth_nm": bp.bandwidth_nm,
                    "integrated_radiance": integrated,
                }
            )
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Optional import guard (avoid circular import at module-level)
# ---------------------------------------------------------------------------
try:
    from .bandpass import BandpassFilter as _BandpassFilter  # noqa: F401
except ImportError:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_1d_matching(
    a: np.ndarray,
    b: np.ndarray,
    name_a: str,
    name_b: str,
) -> None:
    """Raise ValueError if *a* and *b* are not matching 1-D arrays."""
    if a.ndim != 1:
        raise ValueError(f"{name_a} must be 1-D.")
    if b.ndim != 1:
        raise ValueError(f"{name_b} must be 1-D.")
    if len(a) != len(b):
        raise ValueError(
            f"{name_a} and {name_b} must have the same length; "
            f"got {len(a)} vs {len(b)}."
        )


# ---------------------------------------------------------------------------
# Optional TYPE_CHECKING shim
# ---------------------------------------------------------------------------
from typing import Optional  # noqa: E402 – kept at bottom to avoid shadowing
