"""Tests for tape7, .flx, and CSV output parsers."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _fixture(name):
    return os.path.join(FIXTURES, name)


# ---------------------------------------------------------------------------
# tape7 reader tests
# ---------------------------------------------------------------------------

def test_tape7_loads():
    from modtran_wrapper.readers.tape7_reader import read_tape7
    df = read_tape7(_fixture("sample.tape7"))
    assert not df.empty
    assert df.index.name == "wavenumber_cm1"


def test_tape7_row_count():
    from modtran_wrapper.readers.tape7_reader import read_tape7
    df = read_tape7(_fixture("sample.tape7"))
    assert len(df) == 5


def test_tape7_wavenumber_values():
    from modtran_wrapper.readers.tape7_reader import read_tape7
    df = read_tape7(_fixture("sample.tape7"))
    assert abs(df.index[0] - 4000.0) < 0.01
    assert abs(df.index[-1] - 4004.0) < 0.01


def test_tape7_has_transmittance_column():
    from modtran_wrapper.readers.tape7_reader import read_tape7
    df = read_tape7(_fixture("sample.tape7"))
    # Column normalisation — should find something with 'trans'
    trans_cols = [c for c in df.columns if "trans" in c.lower()]
    assert trans_cols, f"No transmittance column found; columns: {list(df.columns)}"


def test_tape7_transmittance_range():
    from modtran_wrapper.readers.tape7_reader import read_tape7
    df = read_tape7(_fixture("sample.tape7"))
    trans_cols = [c for c in df.columns if "trans" in c.lower()]
    col = trans_cols[0]
    assert (df[col] >= 0.0).all()
    assert (df[col] <= 1.0).all()


# ---------------------------------------------------------------------------
# .flx reader tests
# ---------------------------------------------------------------------------

def test_flux_loads():
    from modtran_wrapper.readers.flux_reader import read_flux
    df = read_flux(_fixture("sample.flx"))
    assert not df.empty
    assert df.index.name == "wavenumber_cm1"


def test_flux_row_count():
    from modtran_wrapper.readers.flux_reader import read_flux
    df = read_flux(_fixture("sample.flx"))
    assert len(df) == 5


def test_flux_has_direct_and_diffuse():
    from modtran_wrapper.readers.flux_reader import read_flux
    df = read_flux(_fixture("sample.flx"))
    cols_lower = [c.lower() for c in df.columns]
    assert any("direct" in c for c in cols_lower), f"No direct column; cols: {list(df.columns)}"
    assert any("diffuse" in c for c in cols_lower), f"No diffuse column; cols: {list(df.columns)}"


def test_flux_positive_values():
    from modtran_wrapper.readers.flux_reader import read_flux
    df = read_flux(_fixture("sample.flx"))
    # Direct and diffuse irradiance should be non-negative
    for col in df.columns:
        if "direct" in col.lower() or "diffuse_down" in col.lower():
            assert (df[col] >= 0.0).all(), f"Negative values in {col}"


# ---------------------------------------------------------------------------
# CSV reader tests
# ---------------------------------------------------------------------------

def test_csv_loads():
    from modtran_wrapper.readers.csv_reader import read_csv
    df = read_csv(_fixture("sample.csv"))
    assert not df.empty
    assert df.index.name == "wavenumber_cm1"


def test_csv_row_count():
    from modtran_wrapper.readers.csv_reader import read_csv
    df = read_csv(_fixture("sample.csv"))
    assert len(df) == 5


def test_csv_columns_normalised():
    from modtran_wrapper.readers.csv_reader import read_csv
    df = read_csv(_fixture("sample.csv"))
    # All column names should be lowercase
    for col in df.columns:
        assert col == col.lower(), f"Column not normalised: {col}"


# ---------------------------------------------------------------------------
# AtmosphericProfile CSV loading
# ---------------------------------------------------------------------------

def test_atmosphere_from_csv():
    from modtran_wrapper.inputs.atmosphere import AtmosphericProfile
    atm = AtmosphericProfile.from_csv(_fixture("sample_atmosphere.csv"))
    assert atm.n_levels == 8
    assert atm.altitude_km[0] == 0.0
    assert atm.pressure_mb[0] == 1013.25
    assert atm.temperature_k[0] == 288.15
    assert atm.h2o_ppmv[0] > 0
    assert atm.o3_ppmv[0] > 0
