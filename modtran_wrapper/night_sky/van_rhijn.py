"""
Van Rhijn enhancement factor and FOV-integrated sky background irradiance.

The Van Rhijn function accounts for the increased path length through a thin
emitting atmospheric layer (e.g. OH at ~87 km, NaD at ~92 km, OI at ~97 km)
as a function of zenith angle.  The resulting radiance enhancement relative
to zenith is:

    W(θ) = 1 / sqrt(1 - (R/(R+h))^2 * sin^2(θ))

References
----------
van Rhijn, P. J. (1921), Publ. Astron. Lab. Groningen, 31.
Roach & Gordon (1973), "The Light of the Night Sky", Reidel.
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np
from numpy.polynomial.legendre import leggauss

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

_FloatOrArray = Union[float, np.ndarray]


class VanRhijn:
    """
    Van Rhijn path-length enhancement and sky-background integration.

    The class is stateless (all parameters are passed per-call), but is
    provided as a class for API consistency with the rest of the package and
    to hold the :attr:`EARTH_RADIUS_KM` class constant.

    Class Attributes
    ----------------
    EARTH_RADIUS_KM : float
        Mean Earth radius used as default in all calculations.
    """

    EARTH_RADIUS_KM: float = 6371.0

    # ------------------------------------------------------------------
    # Core geometry
    # ------------------------------------------------------------------

    @staticmethod
    def enhancement_factor(
        zenith_angle_deg: _FloatOrArray,
        layer_altitude_km: float = 90.0,
        earth_radius_km: float = 6371.0,
    ) -> _FloatOrArray:
        """
        Van Rhijn path-length enhancement factor W(θ).

        W(θ) = 1 / sqrt(1 - (R / (R + h))^2 * sin^2(θ))

        Returns the ratio of the slant path length through the emitting
        layer at zenith angle *θ* to the vertical path length (zenith = 0).
        At θ = 0, W = 1 exactly.

        Parameters
        ----------
        zenith_angle_deg : float or np.ndarray
            Zenith angle(s) in degrees.  Must be < critical angle (see below).
        layer_altitude_km : float, optional
            Altitude of the emitting layer above the ground (km).
        earth_radius_km : float, optional
            Mean Earth radius (km).  Defaults to :attr:`EARTH_RADIUS_KM`.

        Returns
        -------
        float or np.ndarray
            Enhancement factor W ≥ 1, same type/shape as *zenith_angle_deg*.

        Raises
        ------
        ValueError
            If any *zenith_angle_deg* equals or exceeds the critical angle
            at which the ray becomes tangent to the emitting layer.
        """
        scalar_input = np.ndim(zenith_angle_deg) == 0
        theta_deg = np.asarray(zenith_angle_deg, dtype=float)
        theta_rad = np.deg2rad(theta_deg)

        R = earth_radius_km
        h = layer_altitude_km
        ratio = R / (R + h)               # dimensionless, < 1

        # Critical zenith angle (ray tangent to layer)
        critical_rad = np.arcsin(1.0 / ratio) if ratio <= 1.0 else np.inf
        critical_deg = np.degrees(critical_rad)

        if np.any(theta_deg >= critical_deg):
            raise ValueError(
                f"zenith_angle_deg must be < critical angle "
                f"({critical_deg:.4f}°) for layer_altitude_km={layer_altitude_km} km. "
                "The ray would be tangent to or pass below the emitting layer."
            )

        sin2 = np.sin(theta_rad) ** 2
        discriminant = 1.0 - ratio ** 2 * sin2
        # Clamp tiny negative values from floating-point noise
        discriminant = np.maximum(discriminant, 0.0)

        W = 1.0 / np.sqrt(discriminant)
        # At θ=0, sin=0, discriminant=1, W=1 exactly — no clamping needed.

        return float(W) if scalar_input else W

    # ------------------------------------------------------------------
    # FOV-integrated irradiance
    # ------------------------------------------------------------------

    def fov_integral(
        self,
        zenith_radiance_W_m2_sr_nm: np.ndarray,
        wavelength_nm: np.ndarray,
        layer_altitude_km: float = 90.0,
        fov_half_angle_deg: float = 2.0,
        boresight_zenith_deg: float = 0.0,
        n_rings: int = 64,
    ) -> np.ndarray:
        """
        Spectrally-resolved sky background irradiance (W/m²/nm) at the aperture
        from a Van Rhijn-enhanced emitting layer within the camera FOV cone.

        For a zenith-pointing camera (*boresight_zenith_deg* = 0):

            I(λ) = 2π ∫₀^α L(λ) · W(θ) · cos(θ) · sin(θ) dθ

        where α is *fov_half_angle_deg* in radians.

        For an off-zenith boresight the integral is tiled with *n_rings*
        concentric Gauss-Legendre rings in polar angle θ' (measured from the
        boresight), averaged over azimuth φ'.  The true zenith angle z for
        each (θ', φ') sample is computed by the spherical law of cosines:

            cos(z) = cos(boresight) · cos(θ') + sin(boresight) · sin(θ') · cos(φ')

        and the result integrated:

            I(λ) = ∫₀^α ∫₀^{2π} L(λ) · W(z(θ',φ')) · cos(z) · sin(θ') dφ' dθ'

        Samples where z >= critical angle are excluded (set to zero weight).

        Parameters
        ----------
        zenith_radiance_W_m2_sr_nm : np.ndarray
            Zenith spectral radiance L(λ) (W/m²/sr/nm), shape (n_wl,).
        wavelength_nm : np.ndarray
            Wavelength grid (nm), shape (n_wl,).
        layer_altitude_km : float, optional
            Emitting layer altitude (km).
        fov_half_angle_deg : float, optional
            Half-angle of the camera FOV cone (degrees).
        boresight_zenith_deg : float, optional
            Zenith angle of the camera boresight (degrees).  0 = zenith.
        n_rings : int, optional
            Number of Gauss-Legendre quadrature points over θ' ∈ [0, α].

        Returns
        -------
        np.ndarray
            Irradiance (W/m²/nm), shape (n_wl,).
        """
        L = np.asarray(zenith_radiance_W_m2_sr_nm, dtype=float)
        wl = np.asarray(wavelength_nm, dtype=float)
        if L.size == 0 or wl.size == 0:
            return np.zeros_like(L)

        alpha_rad = math.radians(fov_half_angle_deg)
        bs_rad = math.radians(boresight_zenith_deg)

        R = self.EARTH_RADIUS_KM
        h = layer_altitude_km
        ratio = R / (R + h)
        # Critical zenith angle in radians
        critical_rad = math.asin(min(1.0 / ratio, 1.0)) if ratio < 1.0 else math.pi / 2.0

        if abs(boresight_zenith_deg) < 1e-10:
            # ----------------------------------------------------------
            # Fast path: zenith-pointing camera
            # Gauss-Legendre over θ ∈ [0, α]
            # I(λ) = 2π * L(λ) * ∫₀^α W(θ) cos(θ) sin(θ) dθ
            # ----------------------------------------------------------
            nodes_m1p1, weights = leggauss(n_rings)
            # Map from [-1,1] to [0, alpha]
            theta_vals = 0.5 * (nodes_m1p1 + 1.0) * alpha_rad   # shape (n_rings,)
            jac = 0.5 * alpha_rad                                  # Jacobian

            sin_t = np.sin(theta_vals)
            cos_t = np.cos(theta_vals)
            sin2 = sin_t ** 2
            disc = 1.0 - ratio ** 2 * sin2
            disc = np.maximum(disc, 0.0)
            W = np.where(theta_vals < critical_rad, 1.0 / np.sqrt(disc), 0.0)

            integrand = W * cos_t * sin_t          # shape (n_rings,)
            integral_scalar = float(np.dot(weights * jac, integrand))

            return 2.0 * math.pi * L * integral_scalar

        else:
            # ----------------------------------------------------------
            # General off-zenith boresight
            # Double integral over (θ', φ') using:
            #   θ' ∈ [0, α]  — Gauss-Legendre quadrature (n_rings points)
            #   φ' ∈ [0, 2π] — trapezoidal rule (n_phi points)
            # ----------------------------------------------------------
            n_phi = max(64, 2 * n_rings)
            phi_vals = np.linspace(0.0, 2.0 * math.pi, n_phi, endpoint=False)
            dphi = 2.0 * math.pi / n_phi

            nodes_m1p1, gl_weights = leggauss(n_rings)
            theta_p_vals = 0.5 * (nodes_m1p1 + 1.0) * alpha_rad   # (n_rings,)
            gl_jac = 0.5 * alpha_rad

            cos_bs = math.cos(bs_rad)
            sin_bs = math.sin(bs_rad)

            integral_scalar = 0.0
            for i, theta_p in enumerate(theta_p_vals):
                cos_tp = math.cos(theta_p)
                sin_tp = math.sin(theta_p)
                # True zenith angle z for each azimuth sample
                # cos(z) = cos(bs)*cos(θ') + sin(bs)*sin(θ')*cos(φ')
                cos_z = cos_bs * cos_tp + sin_bs * sin_tp * np.cos(phi_vals)  # (n_phi,)
                # Clamp numerical noise
                cos_z = np.clip(cos_z, -1.0, 1.0)
                z_vals = np.arccos(cos_z)   # (n_phi,)

                sin_z = np.sin(z_vals)
                sin2_z = sin_z ** 2
                disc = 1.0 - ratio ** 2 * sin2_z
                disc = np.maximum(disc, 0.0)
                valid = z_vals < critical_rad
                W_phi = np.where(valid, 1.0 / np.sqrt(np.where(disc > 0, disc, 1.0)), 0.0)

                # Integrand: W(z) * cos(z) * sin(θ') dφ'
                # (using cos(z) as the projection onto the aperture normal)
                ring_integrand = W_phi * cos_z * sin_tp   # (n_phi,)
                phi_sum = float(np.sum(ring_integrand) * dphi)

                integral_scalar += gl_weights[i] * gl_jac * phi_sum

            return L * integral_scalar

    # ------------------------------------------------------------------
    # Hemispherical integration
    # ------------------------------------------------------------------

    def hemispherical_irradiance(
        self,
        zenith_radiance_W_m2_sr_nm: np.ndarray,
        wavelength_nm: np.ndarray,
        layer_altitude_km: float = 90.0,
        n_points: int = 256,
    ) -> np.ndarray:
        """
        Full hemisphere (2π sr) integrated irradiance W/m²/nm.

        Evaluates:

            I(λ) = 2π ∫₀^{π/2} L(λ) · W(θ) · cos(θ) · sin(θ) dθ

        The upper limit is clamped to just below the critical zenith angle
        when that angle is less than 90°.  Integration uses the trapezoidal
        rule with *n_points* uniformly-spaced samples in θ.

        Parameters
        ----------
        zenith_radiance_W_m2_sr_nm : np.ndarray
            Zenith spectral radiance L(λ) (W/m²/sr/nm), shape (n_wl,).
        wavelength_nm : np.ndarray
            Wavelength grid (nm), shape (n_wl,).
        layer_altitude_km : float, optional
            Emitting layer altitude (km).
        n_points : int, optional
            Number of quadrature points over θ ∈ [0, θ_max].

        Returns
        -------
        np.ndarray
            Hemispherical irradiance (W/m²/nm), shape (n_wl,).
        """
        L = np.asarray(zenith_radiance_W_m2_sr_nm, dtype=float)
        wl = np.asarray(wavelength_nm, dtype=float)
        if L.size == 0 or wl.size == 0:
            return np.zeros_like(L)

        R = self.EARTH_RADIUS_KM
        h = layer_altitude_km
        ratio = R / (R + h)

        # Critical zenith angle (ray tangent to layer)
        if ratio < 1.0:
            critical_rad = math.asin(1.0 / ratio)
        else:
            critical_rad = math.pi / 2.0

        theta_max = min(math.pi / 2.0, critical_rad - 1e-6)

        theta = np.linspace(0.0, theta_max, n_points)
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        sin2 = sin_t ** 2
        disc = np.maximum(1.0 - ratio ** 2 * sin2, 0.0)
        W = 1.0 / np.sqrt(disc)

        integrand_theta = W * cos_t * sin_t   # shape (n_points,)
        integral_scalar = float(np.trapz(integrand_theta, theta))

        return 2.0 * math.pi * L * integral_scalar

    # ------------------------------------------------------------------
    # Radiance map
    # ------------------------------------------------------------------

    def sky_radiance_map(
        self,
        zenith_radiance_W_m2_sr_nm: np.ndarray,
        wavelength_nm: np.ndarray,
        zenith_angles_deg: np.ndarray,
        layer_altitude_km: float = 90.0,
    ) -> np.ndarray:
        """
        Compute spectrally-resolved sky radiance as a function of zenith angle.

        L(θ, λ) = L_zenith(λ) · W(θ, h)

        Parameters
        ----------
        zenith_radiance_W_m2_sr_nm : np.ndarray
            Zenith spectral radiance L(λ) (W/m²/sr/nm), shape (n_wl,).
        wavelength_nm : np.ndarray
            Wavelength grid (nm), shape (n_wl,).
        zenith_angles_deg : np.ndarray
            Array of zenith angles (degrees) at which to evaluate the map,
            shape (n_ang,).  Angles at or beyond the critical angle are
            set to NaN rather than raising an error, to allow smooth maps.
        layer_altitude_km : float, optional
            Emitting layer altitude (km).

        Returns
        -------
        np.ndarray
            Radiance map (W/m²/sr/nm), shape (n_ang, n_wl).
        """
        L = np.asarray(zenith_radiance_W_m2_sr_nm, dtype=float)
        wl = np.asarray(wavelength_nm, dtype=float)
        theta_deg = np.asarray(zenith_angles_deg, dtype=float)

        if L.size == 0 or wl.size == 0 or theta_deg.size == 0:
            return np.zeros((theta_deg.size, wl.size), dtype=float)

        R = self.EARTH_RADIUS_KM
        h = layer_altitude_km
        ratio = R / (R + h)

        # Critical zenith angle
        if ratio < 1.0:
            critical_deg = math.degrees(math.asin(1.0 / ratio))
        else:
            critical_deg = 90.0

        theta_rad = np.deg2rad(theta_deg)
        sin2 = np.sin(theta_rad) ** 2
        disc = 1.0 - ratio ** 2 * sin2            # shape (n_ang,)

        # Angles beyond critical get NaN
        beyond = theta_deg >= critical_deg
        disc = np.where(beyond, np.nan, np.maximum(disc, 0.0))
        W = 1.0 / np.sqrt(disc)                   # shape (n_ang,)

        # Outer product: (n_ang, n_wl)
        radiance_map = W[:, np.newaxis] * L[np.newaxis, :]
        return radiance_map
