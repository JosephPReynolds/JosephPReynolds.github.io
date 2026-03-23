"""Tests for unit conversion utilities."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from modtran_wrapper import units


# ---------------------------------------------------------------------------
# Spectral coordinate round-trips
# ---------------------------------------------------------------------------

def test_wavenum_to_nm_roundtrip():
    nu = np.array([4000.0, 10000.0, 25000.0])
    lam = units.wavenum_to_nm(nu)
    nu_back = units.nm_to_wavenum(lam)
    np.testing.assert_allclose(nu_back, nu, rtol=1e-10)


def test_wavenum_to_um_roundtrip():
    nu = np.array([4000.0, 10000.0, 25000.0])
    lam = units.wavenum_to_um(nu)
    nu_back = units.um_to_wavenum(lam)
    np.testing.assert_allclose(nu_back, nu, rtol=1e-10)


def test_known_wavelength_values():
    # 10000 cm⁻¹ = 1000 nm = 1 µm
    lam_nm = units.wavenum_to_nm(np.array([10000.0]))
    assert abs(lam_nm[0] - 1000.0) < 0.001

    lam_um = units.wavenum_to_um(np.array([10000.0]))
    assert abs(lam_um[0] - 1.0) < 0.0001


def test_nm_to_wavenum_known():
    # 500 nm = 20000 cm⁻¹
    nu = units.nm_to_wavenum(np.array([500.0]))
    assert abs(nu[0] - 20000.0) < 0.001


# ---------------------------------------------------------------------------
# Irradiance conversion
# ---------------------------------------------------------------------------

def test_irrad_conversion_preserves_energy():
    """Integrated irradiance should be approximately preserved across conversion.

    ∫ F_ν dν  ≈  ∫ F_λ dλ  (in consistent SI units)
    """
    nu = np.linspace(5000.0, 10000.0, 1000)  # cm⁻¹
    F_nu = np.ones_like(nu) * 1e-6  # W/cm²/cm⁻¹ — flat spectrum

    lam_nm, F_lam = units.irrad_wavenum_to_nm(F_nu, nu)

    # Integrate in both domains (trapezoidal)
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    E_nu = trapz(F_nu, nu) * 1e4  # → W/m² (cm² → m²)
    E_lam = trapz(F_lam, lam_nm)  # W/m²/nm × nm → W/m²

    # Should agree to within a few percent
    assert abs(E_nu - E_lam) / E_nu < 0.02, (
        f"Energy not conserved: E_nu={E_nu:.4f} W/m², E_lam={E_lam:.4f} W/m²"
    )


def test_irrad_wavenum_to_nm_output_sorted():
    nu = np.array([4000.0, 6000.0, 8000.0, 10000.0])
    F_nu = np.ones_like(nu) * 1e-6
    lam_nm, F_lam = units.irrad_wavenum_to_nm(F_nu, nu)
    assert (np.diff(lam_nm) > 0).all(), "Wavelength output should be sorted ascending"


def test_irrad_wavenum_to_um_output_sorted():
    nu = np.array([4000.0, 6000.0, 8000.0])
    F_nu = np.ones_like(nu) * 1e-6
    lam_um, F_lam = units.irrad_wavenum_to_um(F_nu, nu)
    assert (np.diff(lam_um) > 0).all()


def test_irrad_conversion_positive():
    nu = np.linspace(4000.0, 25000.0, 500)
    F_nu = np.abs(np.random.default_rng(42).normal(1e-6, 1e-7, len(nu)))
    lam_nm, F_lam = units.irrad_wavenum_to_nm(F_nu, nu)
    assert (F_lam >= 0).all()


# ---------------------------------------------------------------------------
# DataFrame conversion helper
# ---------------------------------------------------------------------------

def test_convert_spectrum_to_nm():
    import pandas as pd
    nu = np.array([4000.0, 5000.0, 8000.0, 10000.0])
    F_nu = np.array([1e-6, 2e-6, 3e-6, 4e-6])
    df = pd.DataFrame({"solar_scat": F_nu}, index=pd.Index(nu, name="wavenumber_cm1"))
    result = units.convert_spectrum_to_nm(df, "solar_scat")
    assert result.index.name == "wavelength_nm"
    assert len(result) == len(nu)
    assert (result.iloc[:, 0] >= 0).all()
