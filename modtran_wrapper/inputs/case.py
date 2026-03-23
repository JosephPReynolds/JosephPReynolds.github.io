"""ModtranCase: central dataclass holding all MODTRAN input parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .atmosphere import AtmosphericProfile


# ---------------------------------------------------------------------------
# Atmospheric model constants
# ---------------------------------------------------------------------------
TROPICAL = 1
MID_LATITUDE_SUMMER = 2
MID_LATITUDE_WINTER = 3
SUB_ARCTIC_SUMMER = 4
SUB_ARCTIC_WINTER = 5
US_STANDARD_1976 = 6

# IEMSCT modes
TRANSMITTANCE_ONLY = 0
THERMAL_RADIANCE = 1
SOLAR_RADIANCE = 2
SOLAR_IRRADIANCE = 3

# IHAZE aerosol models
AEROSOL_NONE = 0
AEROSOL_RURAL = 1
AEROSOL_MARITIME = 4
AEROSOL_URBAN = 5
AEROSOL_TROPOSPHERIC = 6


@dataclass
class ModtranCase:
    """All MODTRAN input parameters in a single Python dataclass.

    Parameters are named after their MODTRAN 5/6 counterparts.
    Sensible defaults are set for a ground-level downwelling solar irradiance
    calculation using the US Standard 1976 atmosphere.

    For parameters not explicitly listed, use the ``extra`` dict which is
    passed through verbatim to the writer (tape5 or JSON).
    """

    # ------------------------------------------------------------------
    # Atmosphere
    # ------------------------------------------------------------------
    model: int = US_STANDARD_1976
    """Standard atmospheric model (1=Tropical … 6=US Std 1976, 0/7/8=user-supplied)."""

    h2ostr: float = 0.0
    """Water vapour column. 0 = use model default.
    Positive value: scale factor on model; negative: absolute (atm-cm or g/cm²)."""

    o3str: float = 0.0
    """Ozone column. 0 = use model default. Same sign convention as h2ostr."""

    co2mx: float = 380.0
    """CO2 mixing ratio (ppmv). MODTRAN default is 370 ppmv."""

    atmosphere: "AtmosphericProfile | None" = field(default=None, repr=False)
    """User-supplied atmospheric profile. If set, overrides *model*."""

    # ------------------------------------------------------------------
    # Solar / observer geometry
    # ------------------------------------------------------------------
    itype: int = 3
    """Path type. 3 = vertical/slant path to space or ground (most common for
    downwelling). 1 = horizontal path; 2 = slant path between two altitudes."""

    iemsct: int = SOLAR_IRRADIANCE
    """Radiation transport mode.
    0=transmittance, 1=thermal, 2=solar radiance, 3=solar irradiance."""

    imult: int = 1
    """Multiple scattering. 0=disabled, 1=enabled (recommended for diffuse irradiance)."""

    iday: int = 172
    """Day of year (1–366) for solar distance and irradiance."""

    gmtime: float = 12.0
    """GMT time in decimal hours (e.g. 14.5 = 14:30 UTC)."""

    solzen: float = 30.0
    """Solar zenith angle at target (degrees)."""

    azimuth: float = 0.0
    """Relative azimuth between observer line-of-sight and sun (degrees)."""

    iparm: int = 12
    """Geometry specification mode.
    12 = explicit solzen + azimuth; 1 = lat/lon + date/time (requires parm1/parm2)."""

    parm1: float = 0.0
    """First geometry parameter — meaning depends on IPARM."""

    parm2: float = 0.0
    """Second geometry parameter — meaning depends on IPARM."""

    # ------------------------------------------------------------------
    # Observer / target altitudes
    # ------------------------------------------------------------------
    h1: float = 100.0
    """Observer (sensor) altitude km. For downwelling: top of atmosphere (100 km)."""

    h2: float = 0.0
    """Target altitude km ASL. For downwelling: ground level (0 km)."""

    angle: float = 180.0
    """Initial zenith angle of the line of sight (degrees).
    180° = straight down (nadir), used for downwelling."""

    gndalt: float = 0.0
    """Ground altitude km ASL."""

    # ------------------------------------------------------------------
    # Spectral range (wavenumber, cm⁻¹)
    # ------------------------------------------------------------------
    v1: float = 4000.0
    """Start wavenumber (cm⁻¹). 4000 cm⁻¹ ≈ 2.5 µm."""

    v2: float = 25000.0
    """End wavenumber (cm⁻¹). 25000 cm⁻¹ ≈ 0.4 µm (full solar range)."""

    dv: float = 1.0
    """Spectral increment (cm⁻¹). Use 1.0 for MODTRAN 5, 0.1 for MODTRAN 6."""

    fwhm: float = 2.0
    """Slit function full-width at half-maximum (cm⁻¹).
    Use 2.0 for MODTRAN 5 (2 cm⁻¹ resolution), 0.2 for MODTRAN 6."""

    # ------------------------------------------------------------------
    # Aerosols
    # ------------------------------------------------------------------
    ihaze: int = AEROSOL_RURAL
    """Boundary layer aerosol extinction model.
    0=none, 1=rural, 4=maritime, 5=urban, 6=tropospheric."""

    vis: float = 23.0
    """Meteorological range / visibility (km). Overrides IHAZE default."""

    iseasn: int = 0
    """Seasonal aerosol model. 0=default, 1=spring/summer, 2=fall/winter."""

    icld: int = 0
    """Cloud/rain model. 0=none."""

    ivulc: int = 0
    """Stratospheric aerosol (volcanic). 0=background."""

    # ------------------------------------------------------------------
    # Surface
    # ------------------------------------------------------------------
    surref: float = 0.0
    """Lambertian surface reflectance (0–1). 0 = black body / no reflection."""

    # ------------------------------------------------------------------
    # Output control
    # ------------------------------------------------------------------
    iform: int = 1
    """Output format. 0=tape7 only, 1=tape7 + 7sc convolved output."""

    iout: int = 5
    """Output spectral quantity selection. 5 = all quantities."""

    iprt: int = 0
    """Print flag for diagnostic output (0=minimal)."""

    # ------------------------------------------------------------------
    # Pass-through parameters
    # ------------------------------------------------------------------
    extra: dict = field(default_factory=dict)
    """Any additional MODTRAN parameter not explicitly listed above.
    Keys should be MODTRAN parameter names (case-insensitive for tape5)."""

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_solar_position(
        cls,
        lat: float,
        lon: float,
        dt: datetime,
        **kwargs,
    ) -> "ModtranCase":
        """Create a ModtranCase with solar geometry auto-computed from coordinates.

        Uses scipy to compute solar zenith and azimuth angles for the given
        location and UTC datetime.

        Parameters
        ----------
        lat : float
            Observer latitude (degrees N, negative for S).
        lon : float
            Observer longitude (degrees E, negative for W).
        dt : datetime
            UTC datetime for the observation.
        **kwargs
            Any additional ModtranCase field overrides.
        """
        from .geometry import SolarGeometry

        geo = SolarGeometry.compute(lat, lon, dt)
        iday = dt.timetuple().tm_yday
        gmtime = dt.hour + dt.minute / 60.0 + dt.second / 3600.0

        return cls(
            iday=iday,
            gmtime=gmtime,
            solzen=geo.solar_zenith,
            azimuth=geo.solar_azimuth,
            iparm=12,
            **kwargs,
        )

    @classmethod
    def for_downwelling_irradiance(cls, **kwargs) -> "ModtranCase":
        """Return a case pre-configured for spectral downwelling irradiance.

        Sets ``iemsct=3`` (solar irradiance), ``imult=1`` (multiple scattering),
        ``itype=3`` with nadir geometry from TOA to ground. All other parameters
        can be overridden via kwargs.
        """
        defaults = dict(
            itype=3,
            iemsct=SOLAR_IRRADIANCE,
            imult=1,
            h1=100.0,
            h2=0.0,
            angle=180.0,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    def replace(self, **changes) -> "ModtranCase":
        """Return a new ModtranCase with the specified fields changed."""
        import dataclasses
        return dataclasses.replace(self, **changes)
