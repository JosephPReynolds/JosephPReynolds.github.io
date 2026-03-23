"""Write MODTRAN 5 tape5 input files (fixed-column FORTRAN card format)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..inputs.case import ModtranCase


def write_tape5(case: "ModtranCase", path: str | Path) -> Path:
    """Serialise a ModtranCase to a MODTRAN 5 tape5 file.

    The tape5 format uses fixed FORTRAN column positions.  This writer
    generates the standard card deck for a solar irradiance run.

    Parameters
    ----------
    case : ModtranCase
        Input parameters to write.
    path : str or Path
        Destination file path (typically ``<run_dir>/tape5``).

    Returns
    -------
    Path
        Absolute path to the written file.
    """
    path = Path(path)
    lines = []

    # ------------------------------------------------------------------
    # CARD 1  — main control parameters
    # Columns  1-  1  MODTRN  (T/M/C/K/F/B/S)
    # Columns  2-  2  SPEED   (S/M)
    # Columns  3-  3  BINARY  (F/T)
    # Columns  4-  4  LYMOLC  ( / +)
    # Columns  5-  5  MODEL   (0-8)
    # Columns  6-  6  ITYPE   (1-3)
    # Columns  7-  7  IEMSCT  (0-3)
    # Columns  8-  8  IMULT   (0/1/-1)
    # Columns  9- 10  M1      (0-6, each 1 char)
    # Columns 11- 12  M2
    # Columns 13- 14  M3
    # Columns 15- 16  M4
    # Columns 17- 18  M5
    # Columns 19- 20  M6
    # Columns 21- 21  MDEF    (0/1/2)
    # Columns 22- 22  I_RD2C  (0/1)
    # Columns 23- 23  NOPRNT  (0/1/-1/-2)
    # Columns 24- 24  TPTEMP  (0.0 for no surface)
    # Columns 25- 26  SURREF  (reflectance or 'BRDF')
    # ------------------------------------------------------------------
    modtrn = "T"   # correlated-k, 15 cm⁻¹ spectral bins
    speed = "S"    # slow (accurate)
    binary = "F"
    lymolc = " "
    model = str(case.model)
    itype = str(case.itype)
    iemsct = str(case.iemsct)
    imult = str(case.imult)
    # Molecular profiles (0 = use model defaults)
    m1 = " 0"
    m2 = " 0"
    m3 = " 0"
    m4 = " 0"
    m5 = " 0"
    m6 = " 0"
    mdef = "0"
    i_rd2c = "1" if case.atmosphere is not None else "0"
    noprnt = str(case.iprt)
    tptemp = "0"
    surref = f"{case.surref:4.2f}"

    card1 = (
        f"{modtrn}{speed}{binary}{lymolc}"
        f"{model}{itype}{iemsct}{imult}"
        f"{m1}{m2}{m3}{m4}{m5}{m6}"
        f"{mdef}{i_rd2c}{noprnt}{tptemp}"
        f"{surref:>6}"
    )
    lines.append(card1)

    # ------------------------------------------------------------------
    # CARD 1A — spectral range and resolution
    # Columns  1-  8  V1       (cm⁻¹)
    # Columns  9- 16  V2       (cm⁻¹)
    # Columns 17- 24  DV       (cm⁻¹)
    # Columns 25- 32  FWHM     (cm⁻¹)
    # Columns 33- 33  IFORM    (0/1)
    # Columns 34- 37  IOUT     (integer)
    # Columns 38- 42  IRPT     (0=end, 1=repeat card1)
    # ------------------------------------------------------------------
    card1a = (
        f"{case.v1:8.2f}"
        f"{case.v2:8.2f}"
        f"{case.dv:8.4f}"
        f"{case.fwhm:8.4f}"
        f"{case.iform:1d}"
        f"{case.iout:4d}"
        f"{'0':>5}"
    )
    lines.append(card1a)

    # ------------------------------------------------------------------
    # CARD 2 — main geometry
    # Columns  1- 10  H1       (km)
    # Columns 11- 20  H2       (km)
    # Columns 21- 30  ANGLE    (degrees)
    # Columns 31- 40  RANGE    (km, 0=auto)
    # Columns 41- 50  BETA     (degrees, 0)
    # Columns 51- 55  RO       (earth radius, 0=default)
    # Columns 65- 65  LENN     (0/1)
    # Columns 68- 68  PHI      (0)
    # ------------------------------------------------------------------
    card2 = (
        f"{case.h1:10.3f}"
        f"{case.h2:10.3f}"
        f"{case.angle:10.3f}"
        f"{'0.0':>10}"
        f"{'0.0':>10}"
        f"{'0':>5}"
        f"{'':>10}"
        f"{'0':>1}"
        f"{'':>2}"
        f"{'0':>1}"
    )
    lines.append(card2)

    # ------------------------------------------------------------------
    # CARD 3 — solar/observer geometry
    # Columns  1-  5  IDAY   (1-366)
    # Columns  6- 12  GMTIME (decimal hours)
    # Columns 13- 20  GNDALT (km)
    # Columns 21- 28  IPH    (0/1)
    # Columns 29- 36  IDAY2  (repeat for IPARM)
    # Columns 37- 38  ISOURC (0=sun, 1=moon)
    # Columns 39- 48  PARM1
    # Columns 49- 58  PARM2
    # Columns 59- 65  IPARM  (geometry mode)
    # Columns 66- 69  IPH    (phase function)
    # Columns 70- 72  IDAY   again (3 digit)
    # ------------------------------------------------------------------
    # Simplified card3 — only fields needed for downwelling irradiance
    card3 = (
        f"{case.iday:5d}"
        f"{case.gmtime:7.4f}"
        f"{case.gndalt:8.3f}"
        f"{'0':>8}"   # IPH
        f"{'0':>8}"   # spare
        f"{'0':>2}"   # ISOURC (0=sun)
        f"{case.parm1:10.4f}"
        f"{case.parm2:10.4f}"
        f"{case.iparm:7d}"
    )
    lines.append(card3)

    # ------------------------------------------------------------------
    # CARD 3A1 — solar zenith / azimuth (when IPARM=12)
    # Only written when IPARM is 12 (explicit angles)
    # ------------------------------------------------------------------
    if case.iparm == 12:
        card3a1 = (
            f"{case.solzen:10.4f}"
            f"{case.azimuth:10.4f}"
        )
        lines.append(card3a1)

    # ------------------------------------------------------------------
    # CARD 4 — boundary layer aerosols
    # Columns  1-  5  IHAZE   (0-10)
    # Columns  6- 10  ISEASN  (0-2)
    # Columns 11- 15  ARUSS   (0)
    # Columns 16- 20  IVULC   (0-8)
    # Columns 21- 25  ICSTL   (1-10)
    # Columns 26- 30  ICLD    (0-19)
    # Columns 31- 35  IVSA    (0)
    # Columns 36- 45  VIS     (km)
    # Columns 46- 55  WSS     (m/s, 0=default)
    # Columns 56- 65  WHH     (m/s, 0=default)
    # Columns 66- 75  RAINRT  (mm/hr, 0=none)
    # Columns 76- 85  GNDALT  (km)
    # ------------------------------------------------------------------
    card4 = (
        f"{case.ihaze:5d}"
        f"{case.iseasn:5d}"
        f"{'0':>5}"   # ARUSS
        f"{case.ivulc:5d}"
        f"{'5':>5}"   # ICSTL (continental aerosol strength, 1-10)
        f"{case.icld:5d}"
        f"{'0':>5}"   # IVSA
        f"{case.vis:10.3f}"
        f"{'0.0':>10}"  # WSS
        f"{'0.0':>10}"  # WHH
        f"{'0.0':>10}"  # RAINRT
        f"{case.gndalt:10.3f}"
    )
    lines.append(card4)

    # ------------------------------------------------------------------
    # CARD 4L — user atmosphere (if provided)
    # Written when I_RD2C = 1.  Contains the level count, then one
    # line per level: ALT(km) PRES(mb) TEMP(K) WMOL(n) ...
    # ------------------------------------------------------------------
    if case.atmosphere is not None:
        atm = case.atmosphere
        # Header: number of levels, units flags
        # JCHAR = A (altitude), B (pressure), C (temperature), B (H2O ppmv), B (O3 ppmv)
        card4l_hdr = f"{atm.n_levels:5d}    2    0    0    0    0    0    0    0    0    0    0    0"
        lines.append(card4l_hdr)
        for i in range(atm.n_levels):
            h2o = atm.h2o_ppmv[i]
            o3 = atm.o3_ppmv[i]
            # Optional species
            co2 = atm.co2_ppmv[i] if atm.co2_ppmv is not None else 0.0
            ch4 = atm.ch4_ppmv[i] if atm.ch4_ppmv is not None else 0.0
            n2o = atm.n2o_ppmv[i] if atm.n2o_ppmv is not None else 0.0
            co = atm.co_ppmv[i] if atm.co_ppmv is not None else 0.0
            line = (
                f"{atm.altitude_km[i]:10.3f}"
                f"{atm.pressure_mb[i]:10.4f}"
                f"{atm.temperature_k[i]:10.4f}"
                f"{h2o:10.4E}"
                f"{o3:10.4E}"
                f"{co2:10.4E}"
                f"{ch4:10.4E}"
                f"{n2o:10.4E}"
                f"{co:10.4E}"
                f"ABCBBBBBB"
            )
            lines.append(line)

    # End of case marker
    lines.append("-1")

    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return path.resolve()
