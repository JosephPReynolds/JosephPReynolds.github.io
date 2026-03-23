"""Tests for tape5 and JSON input writers."""

import json
import tempfile
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modtran_wrapper.inputs.case import ModtranCase
from modtran_wrapper.writers.tape5_writer import write_tape5
from modtran_wrapper.writers.json_writer import write_json


# ---------------------------------------------------------------------------
# tape5 writer tests
# ---------------------------------------------------------------------------

def test_tape5_creates_file():
    case = ModtranCase()
    with tempfile.TemporaryDirectory() as tmp:
        path = write_tape5(case, Path(tmp) / "tape5")
        assert path.exists()
        assert path.stat().st_size > 0


def test_tape5_ends_with_minus1():
    case = ModtranCase()
    with tempfile.TemporaryDirectory() as tmp:
        path = write_tape5(case, Path(tmp) / "tape5")
        lines = path.read_text().strip().splitlines()
        assert lines[-1].strip() == "-1"


def test_tape5_contains_spectral_values():
    case = ModtranCase(v1=5000.0, v2=20000.0, dv=2.0, fwhm=4.0)
    with tempfile.TemporaryDirectory() as tmp:
        path = write_tape5(case, Path(tmp) / "tape5")
        text = path.read_text()
        assert "5000.00" in text
        assert "20000.00" in text


def test_tape5_contains_solar_zenith():
    case = ModtranCase(solzen=45.0, iparm=12)
    with tempfile.TemporaryDirectory() as tmp:
        path = write_tape5(case, Path(tmp) / "tape5")
        text = path.read_text()
        assert "45.0000" in text


def test_tape5_with_custom_atmosphere():
    import numpy as np
    from modtran_wrapper.inputs.atmosphere import AtmosphericProfile

    n = 5
    atm = AtmosphericProfile(
        altitude_km=np.linspace(0, 40, n),
        pressure_mb=np.array([1013, 800, 600, 400, 100], dtype=float),
        temperature_k=np.array([288, 275, 260, 240, 220], dtype=float),
        h2o_ppmv=np.array([10000, 5000, 1000, 100, 10], dtype=float),
        o3_ppmv=np.array([0.06, 0.08, 0.5, 4.0, 1.0], dtype=float),
    )
    case = ModtranCase(atmosphere=atm)
    with tempfile.TemporaryDirectory() as tmp:
        path = write_tape5(case, Path(tmp) / "tape5")
        text = path.read_text()
        assert str(n) in text  # number of levels present


# ---------------------------------------------------------------------------
# JSON writer tests
# ---------------------------------------------------------------------------

def test_json_creates_file():
    case = ModtranCase()
    with tempfile.TemporaryDirectory() as tmp:
        path = write_json(case, Path(tmp) / "input.json")
        assert path.exists()
        assert path.stat().st_size > 0


def test_json_is_valid_json():
    case = ModtranCase()
    with tempfile.TemporaryDirectory() as tmp:
        path = write_json(case, Path(tmp) / "input.json")
        doc = json.loads(path.read_text())
        assert "MODTRANINPUT" in doc


def test_json_spectral_values():
    case = ModtranCase(v1=5000.0, v2=20000.0, dv=2.0)
    with tempfile.TemporaryDirectory() as tmp:
        path = write_json(case, Path(tmp) / "input.json")
        doc = json.loads(path.read_text())
        spectral = doc["MODTRANINPUT"]["SPECTRAL"]
        assert spectral["V1"] == 5000.0
        assert spectral["V2"] == 20000.0
        assert spectral["DV"] == 2.0


def test_json_geometry():
    case = ModtranCase(solzen=60.0, iday=180, gmtime=14.5)
    with tempfile.TemporaryDirectory() as tmp:
        path = write_json(case, Path(tmp) / "input.json")
        doc = json.loads(path.read_text())
        geo = doc["MODTRANINPUT"]["GEOMETRY"]
        assert geo["SOLDEG"] == 60.0
        assert geo["IDAY"] == 180
        assert abs(geo["GMTIME"] - 14.5) < 0.001


def test_json_atmosphere_model():
    case = ModtranCase(model=2)  # Mid-Latitude Summer
    with tempfile.TemporaryDirectory() as tmp:
        path = write_json(case, Path(tmp) / "input.json")
        doc = json.loads(path.read_text())
        atm = doc["MODTRANINPUT"]["ATMOSPHERE"]
        assert "MIDLAT_SUMMER" in atm["MODEL"]


def test_json_custom_atmosphere():
    import numpy as np
    from modtran_wrapper.inputs.atmosphere import AtmosphericProfile

    n = 4
    atm = AtmosphericProfile(
        altitude_km=np.array([0.0, 5.0, 20.0, 50.0]),
        pressure_mb=np.array([1013.0, 540.0, 55.0, 0.8]),
        temperature_k=np.array([288.0, 256.0, 217.0, 271.0]),
        h2o_ppmv=np.array([12000.0, 700.0, 5.0, 2.0]),
        o3_ppmv=np.array([0.06, 0.12, 4.5, 0.5]),
    )
    case = ModtranCase(atmosphere=atm)
    with tempfile.TemporaryDirectory() as tmp:
        path = write_json(case, Path(tmp) / "input.json")
        doc = json.loads(path.read_text())
        profiles = doc["MODTRANINPUT"]["ATMOSPHERE"]["PROFILES"]
        assert len(profiles) == n
        assert profiles[0]["ALT"] == 0.0
        assert profiles[0]["PRES"] == 1013.0
