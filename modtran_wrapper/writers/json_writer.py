"""Write MODTRAN 6 JSON input files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..inputs.case import ModtranCase


def write_json(case: "ModtranCase", path: str | Path, case_name: str = "modtran_run") -> Path:
    """Serialise a ModtranCase to a MODTRAN 6 JSON input file.

    Parameters
    ----------
    case : ModtranCase
        Input parameters to write.
    path : str or Path
        Destination file path (typically ``<run_dir>/<case_name>.json``).
    case_name : str
        Name embedded in the JSON ``CASE`` block.

    Returns
    -------
    Path
        Absolute path to the written file.
    """
    path = Path(path)
    doc = _build_json(case, case_name)
    path.write_text(json.dumps(doc, indent=2))
    return path.resolve()


def _build_json(case: "ModtranCase", case_name: str) -> dict:
    """Build the MODTRAN 6 JSON input dictionary."""

    # ------------------------------------------------------------------
    # MODTRAN 6 top-level structure:
    #   MODTRANINPUT -> CASES -> [CASE ...]
    # ------------------------------------------------------------------

    atmosphere_block = _build_atmosphere(case)
    geometry_block = _build_geometry(case)
    spectral_block = _build_spectral(case)
    surface_block = _build_surface(case)
    aerosol_block = _build_aerosol(case)

    case_block = {
        "MODTRANINPUT": {
            "NAME": case_name,
            "ATMOSPHERE": atmosphere_block,
            "SURFACE": surface_block,
            "GEOMETRY": geometry_block,
            "SPECTRAL": spectral_block,
            "AEROSOL": aerosol_block,
            "RTOPTIONS": {
                "MODTRN": "RT_CORRK_FAST",
                "LYMOLC": False,
                "T_BEST": False,
                "IEMSCT": _iemsct_name(case.iemsct),
                "IMULT": _imult_name(case.imult),
                "DISALB": False,
                "NSTR": 4,
                "SOLCON": 0.0,    # 0 = use internal solar constant
            },
        }
    }

    # Merge any extra pass-through parameters at the top MODTRANINPUT level
    if case.extra:
        case_block["MODTRANINPUT"].update(case.extra)

    return case_block


def _iemsct_name(iemsct: int) -> str:
    mapping = {
        0: "RT_TRANSMITTANCE",
        1: "RT_THERMAL_RADIANCE",
        2: "RT_SOLAR_AND_THERMAL",
        3: "RT_SOLAR_IRRADIANCE",
    }
    return mapping.get(iemsct, "RT_SOLAR_IRRADIANCE")


def _imult_name(imult: int) -> str:
    if imult == 0:
        return "RT_NO_MULTIPLE_SCATTER"
    elif imult < 0:
        return "RT_SCALED_SS"
    return "RT_DISORT2"


def _build_atmosphere(case: "ModtranCase") -> dict:
    atm: dict = {
        "MODEL": _model_name(case.model),
        "M1": 0,
        "M2": 0,
        "M3": 0,
        "M4": 0,
        "M5": 0,
        "M6": 0,
        "MDEF": 0,
        "CO2MX": case.co2mx,
        "H2OSTR": case.h2ostr,
        "O3STR": case.o3str,
    }

    if case.atmosphere is not None:
        atm["MDEF"] = 1
        atm["NLAYERS"] = case.atmosphere.n_levels
        atm["PROFILES"] = _serialize_profile(case.atmosphere)

    return atm


def _model_name(model: int) -> str:
    mapping = {
        0: "ATM_USER_ALT_PRES",
        1: "ATM_TROPICAL",
        2: "ATM_MIDLAT_SUMMER",
        3: "ATM_MIDLAT_WINTER",
        4: "ATM_SUBARCTIC_SUMMER",
        5: "ATM_SUBARCTIC_WINTER",
        6: "ATM_US_STANDARD_1976",
        7: "ATM_USER_ALT_PRES",
        8: "ATM_USER_ALT_PRES",
    }
    return mapping.get(model, "ATM_US_STANDARD_1976")


def _serialize_profile(atm) -> list[dict]:
    """Convert AtmosphericProfile to MODTRAN 6 PROFILES list."""
    levels = []
    for i in range(atm.n_levels):
        level = {
            "ALT": float(atm.altitude_km[i]),
            "PRES": float(atm.pressure_mb[i]),
            "TEMP": float(atm.temperature_k[i]),
            "H2O": float(atm.h2o_ppmv[i]),
            "O3": float(atm.o3_ppmv[i]),
        }
        if atm.co2_ppmv is not None:
            level["CO2"] = float(atm.co2_ppmv[i])
        if atm.ch4_ppmv is not None:
            level["CH4"] = float(atm.ch4_ppmv[i])
        if atm.n2o_ppmv is not None:
            level["N2O"] = float(atm.n2o_ppmv[i])
        if atm.co_ppmv is not None:
            level["CO"] = float(atm.co_ppmv[i])
        levels.append(level)
    return levels


def _build_geometry(case: "ModtranCase") -> dict:
    geo: dict = {
        "ITYPE": _itype_name(case.itype),
        "H1ALT": case.h1,
        "H2ALT": case.h2,
        "OBSZEN": case.angle,
        "GNDALT": case.gndalt,
        "IPARM": case.iparm,
        "PARM1": case.parm1,
        "PARM2": case.parm2,
        "IDAY": case.iday,
        "GMTIME": case.gmtime,
    }
    if case.iparm == 12:
        geo["SOLDEG"] = case.solzen
        geo["SOLAZ"] = case.azimuth
    return geo


def _itype_name(itype: int) -> str:
    mapping = {
        1: "slant_path_at_altitude",
        2: "slant_path_between_altitudes",
        3: "vertical_or_slant_to_space_or_ground",
    }
    return mapping.get(itype, "vertical_or_slant_to_space_or_ground")


def _build_spectral(case: "ModtranCase") -> dict:
    return {
        "V1": case.v1,
        "V2": case.v2,
        "DV": case.dv,
        "FWHM": case.fwhm,
        "IFORM": case.iform,
        "IOUT": case.iout,
    }


def _build_surface(case: "ModtranCase") -> dict:
    return {
        "SURREF": str(case.surref),
        "TPTEMP": 0.0,
    }


def _build_aerosol(case: "ModtranCase") -> dict:
    return {
        "IHAZE": _ihaze_name(case.ihaze),
        "ISEASN": _iseasn_name(case.iseasn),
        "IVULC": case.ivulc,
        "ICLD": case.icld,
        "VIS": case.vis,
        "GNDALT": case.gndalt,
    }


def _ihaze_name(ihaze: int) -> str:
    mapping = {
        0: "AER_NONE",
        1: "AER_RURAL",
        2: "AER_RURAL",
        3: "AER_NAVY_MARITIME",
        4: "AER_MARITIME",
        5: "AER_URBAN",
        6: "AER_TROPOSPHERIC",
        10: "AER_FOG1",
    }
    return mapping.get(ihaze, "AER_RURAL")


def _iseasn_name(iseasn: int) -> str:
    mapping = {
        0: "SEASN_DEFAULT",
        1: "SEASN_SPRING_SUMMER",
        2: "SEASN_FALL_WINTER",
    }
    return mapping.get(iseasn, "SEASN_DEFAULT")
