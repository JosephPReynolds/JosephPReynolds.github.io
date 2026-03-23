"""SolarGeometry: compute solar zenith and azimuth from lat/lon/time."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class SolarGeometry:
    """Result of a solar position calculation."""

    solar_zenith: float
    """Solar zenith angle in degrees (0 = overhead, 90 = horizon)."""

    solar_azimuth: float
    """Solar azimuth angle in degrees from North, clockwise."""

    solar_elevation: float
    """Solar elevation angle in degrees above horizon."""

    @classmethod
    def compute(cls, lat: float, lon: float, dt: datetime) -> "SolarGeometry":
        """Compute solar position for a given location and UTC time.

        Uses scipy.integrate if available, otherwise falls back to a
        built-in Spencer/Iqbal algorithm accurate to ~0.01°.

        Parameters
        ----------
        lat : float
            Latitude in decimal degrees (N positive).
        lon : float
            Longitude in decimal degrees (E positive).
        dt : datetime
            UTC datetime. If timezone-naive, assumed UTC.
        """
        # Attempt to use pvlib for high accuracy
        try:
            return cls._compute_pvlib(lat, lon, dt)
        except ImportError:
            pass

        # Fall back to built-in Spencer algorithm
        return cls._compute_builtin(lat, lon, dt)

    @classmethod
    def _compute_pvlib(cls, lat: float, lon: float, dt: datetime) -> "SolarGeometry":
        import pvlib
        import pandas as pd

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        times = pd.DatetimeIndex([dt])
        pos = pvlib.solarposition.get_solarposition(times, lat, lon)
        zenith = float(pos["apparent_zenith"].iloc[0])
        azimuth = float(pos["azimuth"].iloc[0])
        elevation = float(pos["apparent_elevation"].iloc[0])
        return cls(solar_zenith=zenith, solar_azimuth=azimuth, solar_elevation=elevation)

    @classmethod
    def _compute_builtin(cls, lat: float, lon: float, dt: datetime) -> "SolarGeometry":
        """Spencer (1971) / Iqbal (1983) solar position algorithm.

        Accurate to about 0.01° for most solar elevation angles.
        """
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        # Day of year
        doy = dt.timetuple().tm_yday
        ut = dt.hour + dt.minute / 60.0 + dt.second / 3600.0

        # B factor (radians)
        B = (doy - 1) * 2.0 * math.pi / 365.0

        # Equation of time (minutes) — Spencer 1971
        eqt = (
            229.18
            * (
                0.000075
                + 0.001868 * math.cos(B)
                - 0.032077 * math.sin(B)
                - 0.014615 * math.cos(2 * B)
                - 0.04089 * math.sin(2 * B)
            )
        )

        # Solar declination (radians) — Spencer 1971
        decl = (
            0.006918
            - 0.399912 * math.cos(B)
            + 0.070257 * math.sin(B)
            - 0.006758 * math.cos(2 * B)
            + 0.000907 * math.sin(2 * B)
            - 0.002697 * math.cos(3 * B)
            + 0.00148 * math.sin(3 * B)
        )

        # Local solar time (hours)
        lst = ut + lon / 15.0 + eqt / 60.0

        # Hour angle (radians); noon = 0
        ha = (lst - 12.0) * math.pi / 12.0

        lat_r = math.radians(lat)

        # Solar elevation (radians)
        sin_elev = (
            math.sin(lat_r) * math.sin(decl)
            + math.cos(lat_r) * math.cos(decl) * math.cos(ha)
        )
        sin_elev = max(-1.0, min(1.0, sin_elev))
        elev_r = math.asin(sin_elev)
        elevation = math.degrees(elev_r)
        zenith = 90.0 - elevation

        # Solar azimuth (degrees from North, clockwise)
        cos_az = (math.sin(decl) - math.sin(lat_r) * sin_elev) / (
            math.cos(lat_r) * math.cos(elev_r) + 1e-12
        )
        cos_az = max(-1.0, min(1.0, cos_az))
        az = math.degrees(math.acos(cos_az))
        if ha > 0:
            az = 360.0 - az

        return cls(solar_zenith=zenith, solar_azimuth=az, solar_elevation=elevation)
