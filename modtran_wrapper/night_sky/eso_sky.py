"""ESO SkyCalc REST API wrapper for night sky radiance modelling."""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
import urllib.error
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


# Physical constants
_H = 6.62607015e-34   # Planck constant  J·s
_C = 2.99792458e8     # Speed of light    m/s
_ARCSEC2_PER_SR = 206265.0 ** 2   # arcsec² per steradian


@dataclass
class SkyCalcParams:
    """Parameters for the ESO SkyCalc sky background model.

    All angular quantities in degrees unless noted.
    """

    airmass: float = 1.0
    pwv_mode: str = "pwv"          # "pwv" or "season"
    pwv: float = 3.5               # precipitable water vapour  mm
    msolflux: float = 130.0        # solar radio flux  sfu
    incl_moon: bool = True
    moon_sun_sep_deg: float = 90.0   # moon–sun separation
    moon_target_sep_deg: float = 45.0  # moon–target separation
    moon_alt_deg: float = 45.0     # moon altitude above horizon
    moon_earth_dist: float = 1.0   # relative to mean distance  0.91–1.08
    incl_starlight: bool = True
    incl_zodiacal: bool = True
    ecl_lon_deg: float = 135.0     # target ecliptic longitude
    ecl_lat_deg: float = 90.0      # target ecliptic latitude (90 = pole = no zodi)
    incl_loweratm: bool = True
    incl_upperatm: bool = True
    incl_airglow: bool = True
    incl_thermal: bool = False
    wmin_nm: float = 300.0
    wmax_nm: float = 2000.0
    wdelta_nm: float = 0.5         # wavelength step  nm
    vacair: str = "vac"            # "vac" or "air"
    observatory: str = "paranal"


class EsoSkyModel:
    """Calls the ESO SkyCalc REST API and returns calibrated sky radiance.

    Parameters
    ----------
    timeout_s:
        HTTP request timeout in seconds.
    cache_dir:
        Directory for caching API responses.  Responses are stored as
        ``<md5_of_params>.pkl`` files.  Pass ``None`` to disable caching.
    """

    API_URL = "https://www.eso.org/observing/etc/bin/simu/skycalc"

    def __init__(
        self,
        timeout_s: float = 60.0,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self._timeout = timeout_s
        self._cache_dir: Optional[Path] = Path(cache_dir) if cache_dir is not None else None
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def radiance(
        self,
        wavelength_nm: np.ndarray,
        params: Optional[SkyCalcParams] = None,
        **kwargs,
    ) -> np.ndarray:
        """Return total sky spectral radiance interpolated onto *wavelength_nm*.

        Units: W m⁻² sr⁻¹ nm⁻¹.

        Parameters
        ----------
        wavelength_nm:
            Target wavelength grid in nm.
        params:
            :class:`SkyCalcParams` instance.  Defaults to a default-constructed
            instance if not provided.
        **kwargs:
            Override individual ``SkyCalcParams`` fields, e.g.
            ``airmass=1.5``.
        """
        if params is None:
            params = SkyCalcParams()
        if kwargs:
            params = _replace_params(params, kwargs)

        df = self._fetch(params)

        wl = df["wavelength_nm"].values
        flux_ph = df["flux_ph_s_m2_um_arcsec2"].values

        # Convert to W/m²/sr/nm
        flux_W = self._photon_to_radiance(flux_ph, wl)

        # Interpolate onto requested grid (zero outside range)
        wl_min, wl_max = wl[0], wl[-1]
        interp = interp1d(wl, flux_W, kind="linear", bounds_error=False, fill_value=0.0)
        result = interp(np.asarray(wavelength_nm, dtype=float))

        # Ensure non-negative
        return np.maximum(result, 0.0)

    def fetch_raw(self, params: SkyCalcParams) -> pd.DataFrame:
        """Return raw SkyCalc data as a DataFrame.

        Columns: ``wavelength_nm``, ``flux_ph_s_m2_um_arcsec2``, ``trans``.
        """
        return self._fetch(params)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_request_body(self, params: SkyCalcParams) -> dict:
        """Map :class:`SkyCalcParams` onto the ESO SkyCalc JSON field names."""

        def _yn(value: bool) -> str:
            return "Y" if value else "N"

        body: dict = {
            "airmass": params.airmass,
            "pwv_mode": params.pwv_mode,
            "pwv": params.pwv,
            "msolflux": params.msolflux,
            "incl_moon": _yn(params.incl_moon),
            "moon_sun_sep": params.moon_sun_sep_deg,
            "moon_target_sep": params.moon_target_sep_deg,
            "moon_alt": params.moon_alt_deg,
            "moon_earth_dist": params.moon_earth_dist,
            "incl_starlight": _yn(params.incl_starlight),
            "incl_zodiacal": _yn(params.incl_zodiacal),
            "ecl_lon": params.ecl_lon_deg,
            "ecl_lat": params.ecl_lat_deg,
            "incl_loweratm": _yn(params.incl_loweratm),
            "incl_upperatm": _yn(params.incl_upperatm),
            "incl_airglow": _yn(params.incl_airglow),
            "incl_thermal": _yn(params.incl_thermal),
            "wmin": params.wmin_nm,
            "wmax": params.wmax_nm,
            "wdelta": params.wdelta_nm,
            "vacair": params.vacair,
            "observatory": params.observatory,
        }
        return body

    def _cache_path(self, params: SkyCalcParams) -> Optional[Path]:
        if self._cache_dir is None:
            return None
        body = self._build_request_body(params)
        key = json.dumps(body, sort_keys=True).encode()
        digest = hashlib.md5(key).hexdigest()
        return self._cache_dir / f"{digest}.pkl"

    def _fetch(self, params: SkyCalcParams) -> pd.DataFrame:
        """POST to the SkyCalc API, cache the result, and return a DataFrame."""
        import pickle  # stdlib – safe for local caching

        cache_path = self._cache_path(params)

        # Try to load from cache
        if cache_path is not None and cache_path.exists():
            try:
                with cache_path.open("rb") as fh:
                    return pickle.load(fh)
            except Exception:
                pass  # stale / corrupt cache – re-fetch

        body = self._build_request_body(params)
        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.API_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                content = resp.read()
                content_type = resp.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode(errors="replace")
            raise RuntimeError(
                f"ESO SkyCalc HTTP {exc.code}: {body_text[:500]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"ESO SkyCalc network error: {exc.reason}") from exc

        # Parse response
        df = self._parse_response(content, content_type)

        # Write to cache
        if cache_path is not None:
            try:
                with cache_path.open("wb") as fh:
                    pickle.dump(df, fh)
            except Exception:
                pass  # non-fatal

        return df

    def _parse_response(self, content: bytes, content_type: str) -> pd.DataFrame:
        """Try FITS first, then JSON."""
        # Heuristic: FITS files start with "SIMPLE  ="
        is_fits = content[:6] == b"SIMPLE" or b"XTENSION" in content[:40]
        if is_fits or "fits" in content_type.lower():
            try:
                return self._parse_fits_response(content)
            except Exception as fits_exc:
                pass  # fall through to JSON

        # Try JSON fallback
        try:
            data = json.loads(content.decode("utf-8", errors="replace"))
            return self._parse_json_response(data)
        except Exception as json_exc:
            raise RuntimeError(
                "Could not parse ESO SkyCalc response as FITS or JSON. "
                f"First 200 bytes: {content[:200]!r}"
            )

    def _parse_fits_response(self, content: bytes) -> pd.DataFrame:
        """Parse FITS binary table response bytes using astropy."""
        try:
            from astropy.io import fits as astropy_fits
        except ImportError as exc:
            raise RuntimeError(
                "astropy is required to parse FITS responses from ESO SkyCalc. "
                "Install it with: pip install astropy"
            ) from exc

        with astropy_fits.open(io.BytesIO(content), memmap=False) as hdul:
            # The sky spectrum is in extension 1 (BinTableHDU)
            tbl = None
            for ext in hdul[1:]:
                if hasattr(ext, "columns") and ext.columns is not None:
                    tbl = ext
                    break
            if tbl is None:
                raise RuntimeError("No binary table found in FITS response.")

            col_names = [c.name.lower() for c in tbl.columns]

            # Wavelength – may be 'lam' or 'wave' or 'wavelength'
            wl_key = _find_column(col_names, ["lam", "wave", "wavelength"])
            if wl_key is None:
                raise RuntimeError(f"Cannot find wavelength column in: {col_names}")

            # Flux – 'flux' or 'flux_sml' etc.
            flux_key = _find_column(col_names, ["flux", "flux_sml", "radiance"])
            if flux_key is None:
                raise RuntimeError(f"Cannot find flux column in: {col_names}")

            # Transmission – 'trans' or 'trans_ma'
            trans_key = _find_column(col_names, ["trans", "trans_ma", "transmission"])

            wl_nm = np.array(tbl.data[wl_key], dtype=float)
            flux = np.array(tbl.data[flux_key], dtype=float)
            trans = (
                np.array(tbl.data[trans_key], dtype=float)
                if trans_key is not None
                else np.ones_like(wl_nm)
            )

            # ESO SkyCalc returns wavelength in nm natively for wmin/wmax in nm
            # but historically it was µm – detect by magnitude
            if wl_nm.max() < 100:
                # Probably µm – convert to nm
                wl_nm = wl_nm * 1000.0

        df = pd.DataFrame(
            {
                "wavelength_nm": wl_nm,
                "flux_ph_s_m2_um_arcsec2": flux,
                "trans": trans,
            }
        )
        return df

    def _parse_json_response(self, data: dict) -> pd.DataFrame:
        """Parse JSON fallback response (non-standard but handle gracefully)."""
        if "error" in data:
            raise RuntimeError(f"ESO SkyCalc error: {data['error']}")

        # Try common field names
        wl = np.array(data.get("lam", data.get("wavelength", [])), dtype=float)
        flux = np.array(data.get("flux", data.get("radiance", [])), dtype=float)
        trans = np.array(
            data.get("trans", data.get("transmission", np.ones_like(wl))),
            dtype=float,
        )

        if wl.size == 0:
            raise RuntimeError("ESO SkyCalc JSON response missing wavelength data.")

        if wl.max() < 100:
            wl = wl * 1000.0

        return pd.DataFrame(
            {
                "wavelength_nm": wl,
                "flux_ph_s_m2_um_arcsec2": flux,
                "trans": trans,
            }
        )

    def _photon_to_radiance(
        self,
        flux_ph_s_m2_um_arcsec2: np.ndarray,
        wavelength_nm: np.ndarray,
    ) -> np.ndarray:
        """Convert ph s⁻¹ m⁻² µm⁻¹ arcsec⁻² to W m⁻² sr⁻¹ nm⁻¹.

        The conversion is:

        * Multiply by E_photon = hc/λ  [J/photon]  (λ in metres)
        * Multiply by (206265)²  to go from arcsec⁻² to sr⁻¹
        * Divide by 1000  to go from µm⁻¹ to nm⁻¹  (1 µm = 1000 nm)
        """
        wl_m = wavelength_nm * 1e-9  # nm → m
        # Guard against zero wavelength
        safe_wl = np.where(wl_m > 0, wl_m, 1e-9)
        e_photon = (_H * _C) / safe_wl  # J

        result = (
            flux_ph_s_m2_um_arcsec2
            * e_photon
            * _ARCSEC2_PER_SR
            / 1000.0
        )
        return result


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _find_column(col_names: list[str], candidates: list[str]) -> Optional[str]:
    """Return the first candidate that exists in *col_names*, or None."""
    for c in candidates:
        if c in col_names:
            return c
    # Partial match fallback
    for c in candidates:
        for name in col_names:
            if c in name:
                return name
    return None


def _replace_params(params: SkyCalcParams, overrides: dict) -> SkyCalcParams:
    """Return a new :class:`SkyCalcParams` with *overrides* applied."""
    d = asdict(params)
    for k, v in overrides.items():
        if k not in d:
            raise ValueError(
                f"Unknown SkyCalcParams field: {k!r}. "
                f"Valid fields: {list(d.keys())}"
            )
        d[k] = v
    return SkyCalcParams(**d)
