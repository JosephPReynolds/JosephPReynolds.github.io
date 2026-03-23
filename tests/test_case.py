"""Tests for ModtranCase construction and serialization."""

import dataclasses
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modtran_wrapper.inputs.case import (
    ModtranCase,
    US_STANDARD_1976,
    SOLAR_IRRADIANCE,
    AEROSOL_RURAL,
)


def test_default_construction():
    case = ModtranCase()
    assert case.model == US_STANDARD_1976
    assert case.iemsct == SOLAR_IRRADIANCE
    assert case.ihaze == AEROSOL_RURAL
    assert case.imult == 1
    assert case.h1 == 100.0
    assert case.h2 == 0.0


def test_for_downwelling_irradiance():
    case = ModtranCase.for_downwelling_irradiance(model=2, solzen=45.0)
    assert case.iemsct == SOLAR_IRRADIANCE
    assert case.imult == 1
    assert case.itype == 3
    assert case.angle == 180.0
    assert case.model == 2
    assert case.solzen == 45.0


def test_replace():
    base = ModtranCase()
    modified = base.replace(solzen=60.0, ihaze=4)
    assert modified.solzen == 60.0
    assert modified.ihaze == 4
    # Original unchanged
    assert base.solzen == 30.0
    assert base.ihaze == AEROSOL_RURAL


def test_extra_passthrough():
    case = ModtranCase(extra={"CUSTOM_PARAM": 42})
    assert case.extra["CUSTOM_PARAM"] == 42


def test_from_solar_position():
    # Summer solstice, Los Angeles, noon UTC
    dt = datetime(2024, 6, 21, 20, 0, 0)  # 20:00 UTC ≈ noon local in LA
    case = ModtranCase.from_solar_position(lat=34.0, lon=-118.0, dt=dt)
    assert case.iday == 173  # day of year for June 21, 2024
    assert abs(case.gmtime - 20.0) < 0.01
    # Solar zenith should be non-negative
    assert 0.0 <= case.solzen <= 90.0
    assert case.iparm == 12


def test_spectral_range_defaults():
    case = ModtranCase()
    assert case.v1 < case.v2
    assert case.dv > 0
    assert case.fwhm > 0


def test_all_fields_are_dataclass_fields():
    """Ensure ModtranCase is a proper dataclass with expected fields."""
    field_names = {f.name for f in dataclasses.fields(ModtranCase)}
    for expected in (
        "model", "h2ostr", "o3str", "iday", "gmtime", "solzen", "azimuth",
        "v1", "v2", "dv", "fwhm", "itype", "iemsct", "imult",
        "ihaze", "vis", "gndalt", "h1", "h2",
    ):
        assert expected in field_names, f"Missing field: {expected}"
