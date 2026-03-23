"""modtran_wrapper — Python interface for MODTRAN 5 and 6.

Quick start
-----------
::

    from modtran_wrapper import ModtranRunner, ModtranCase

    runner = ModtranRunner(executable="/opt/modtran6/bin/mod6con")

    case = ModtranCase.for_downwelling_irradiance(
        model=6,          # US Standard 1976
        solzen=30.0,      # 30° solar zenith
        iday=172,         # summer solstice
        ihaze=1,          # rural aerosol
        v1=4000.0,        # 2.5 µm
        v2=25000.0,       # 0.4 µm
    )

    result = runner.run(case)
    irrad = result.downwelling_irradiance(units="W/m2/nm")
    print(irrad.head())

Batch / parameter sweep
-----------------------
::

    from modtran_wrapper import ModtranRunner, ModtranCase, BatchRunner

    runner = ModtranRunner(executable="/opt/modtran6/bin/mod6con")
    batch  = BatchRunner(runner)

    df = batch.sweep(
        base_case=ModtranCase.for_downwelling_irradiance(),
        param_grid={"solzen": [0, 20, 40, 60], "ihaze": [1, 4]},
        n_workers=4,
    )

Solar geometry from coordinates
--------------------------------
::

    from datetime import datetime
    from modtran_wrapper import ModtranCase

    case = ModtranCase.from_solar_position(
        lat=34.0, lon=-118.0,
        dt=datetime(2024, 6, 21, 18, 0, 0),   # UTC
    )
"""

from .runner import ModtranRunner
from .batch import BatchRunner
from .output import ModtranResult
from .inputs.case import (
    ModtranCase,
    TROPICAL,
    MID_LATITUDE_SUMMER,
    MID_LATITUDE_WINTER,
    SUB_ARCTIC_SUMMER,
    SUB_ARCTIC_WINTER,
    US_STANDARD_1976,
    AEROSOL_NONE,
    AEROSOL_RURAL,
    AEROSOL_MARITIME,
    AEROSOL_URBAN,
    AEROSOL_TROPOSPHERIC,
    TRANSMITTANCE_ONLY,
    THERMAL_RADIANCE,
    SOLAR_RADIANCE,
    SOLAR_IRRADIANCE,
)
from .inputs.atmosphere import AtmosphericProfile
from .inputs.geometry import SolarGeometry
from .exceptions import (
    ModtranError,
    ModtranRunError,
    ModtranParseError,
    ModtranConfigError,
    ModtranVersionError,
)
from . import units

__all__ = [
    # Core
    "ModtranRunner",
    "BatchRunner",
    "ModtranResult",
    "ModtranCase",
    # Inputs
    "AtmosphericProfile",
    "SolarGeometry",
    # Constants
    "TROPICAL",
    "MID_LATITUDE_SUMMER",
    "MID_LATITUDE_WINTER",
    "SUB_ARCTIC_SUMMER",
    "SUB_ARCTIC_WINTER",
    "US_STANDARD_1976",
    "AEROSOL_NONE",
    "AEROSOL_RURAL",
    "AEROSOL_MARITIME",
    "AEROSOL_URBAN",
    "AEROSOL_TROPOSPHERIC",
    "TRANSMITTANCE_ONLY",
    "THERMAL_RADIANCE",
    "SOLAR_RADIANCE",
    "SOLAR_IRRADIANCE",
    # Utilities
    "units",
    # Exceptions
    "ModtranError",
    "ModtranRunError",
    "ModtranParseError",
    "ModtranConfigError",
    "ModtranVersionError",
]
