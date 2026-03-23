"""AtmosphericProfile: user-supplied or standard atmospheric profile."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class AtmosphericProfile:
    """Altitude-resolved atmospheric state vector for use with MODTRAN.

    All arrays must have the same length (one entry per level).
    Levels should be ordered from surface (lowest altitude) to TOA.

    Units
    -----
    altitude_km      : km above sea level
    pressure_mb      : pressure in millibars (hPa)
    temperature_k    : temperature in Kelvin
    h2o_ppmv         : water vapour volume mixing ratio (ppmv)
    o3_ppmv          : ozone volume mixing ratio (ppmv)
    co2_ppmv         : CO2 mixing ratio (ppmv), optional
    ch4_ppmv         : CH4 mixing ratio (ppmv), optional
    n2o_ppmv         : N2O mixing ratio (ppmv), optional
    co_ppmv          : CO mixing ratio (ppmv), optional
    """

    altitude_km: np.ndarray
    pressure_mb: np.ndarray
    temperature_k: np.ndarray
    h2o_ppmv: np.ndarray
    o3_ppmv: np.ndarray
    co2_ppmv: Optional[np.ndarray] = field(default=None)
    ch4_ppmv: Optional[np.ndarray] = field(default=None)
    n2o_ppmv: Optional[np.ndarray] = field(default=None)
    co_ppmv: Optional[np.ndarray] = field(default=None)

    def __post_init__(self):
        n = len(self.altitude_km)
        for name in ("pressure_mb", "temperature_k", "h2o_ppmv", "o3_ppmv"):
            arr = getattr(self, name)
            if len(arr) != n:
                raise ValueError(
                    f"AtmosphericProfile: '{name}' has length {len(arr)}, "
                    f"expected {n} (same as altitude_km)."
                )
        # Ensure numpy arrays
        for name in (
            "altitude_km", "pressure_mb", "temperature_k",
            "h2o_ppmv", "o3_ppmv",
        ):
            setattr(self, name, np.asarray(getattr(self, name), dtype=float))
        for name in ("co2_ppmv", "ch4_ppmv", "n2o_ppmv", "co_ppmv"):
            val = getattr(self, name)
            if val is not None:
                setattr(self, name, np.asarray(val, dtype=float))

    @property
    def n_levels(self) -> int:
        return len(self.altitude_km)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(cls, path: str | Path) -> "AtmosphericProfile":
        """Load from a CSV file.

        Expected columns (header row required, case-insensitive):
            altitude_km, pressure_mb, temperature_k, h2o_ppmv, o3_ppmv
        Optional extra columns: co2_ppmv, ch4_ppmv, n2o_ppmv, co_ppmv
        """
        path = Path(path)
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Normalize column names to lowercase
        rows = [{k.strip().lower(): v for k, v in row.items()} for row in rows]

        required = ("altitude_km", "pressure_mb", "temperature_k", "h2o_ppmv", "o3_ppmv")
        for col in required:
            if col not in rows[0]:
                raise ValueError(f"AtmosphericProfile CSV missing required column: '{col}'")

        def col(name):
            return np.array([float(r[name]) for r in rows])

        optional = {}
        for name in ("co2_ppmv", "ch4_ppmv", "n2o_ppmv", "co_ppmv"):
            if name in rows[0]:
                optional[name] = col(name)

        return cls(
            altitude_km=col("altitude_km"),
            pressure_mb=col("pressure_mb"),
            temperature_k=col("temperature_k"),
            h2o_ppmv=col("h2o_ppmv"),
            o3_ppmv=col("o3_ppmv"),
            **optional,
        )

    @classmethod
    def from_arrays(
        cls,
        altitude_km: np.ndarray,
        pressure_mb: np.ndarray,
        temperature_k: np.ndarray,
        h2o_ppmv: np.ndarray,
        o3_ppmv: np.ndarray,
        **optional_species,
    ) -> "AtmosphericProfile":
        """Construct directly from numpy arrays."""
        return cls(
            altitude_km=np.asarray(altitude_km),
            pressure_mb=np.asarray(pressure_mb),
            temperature_k=np.asarray(temperature_k),
            h2o_ppmv=np.asarray(h2o_ppmv),
            o3_ppmv=np.asarray(o3_ppmv),
            **optional_species,
        )

    @classmethod
    def from_radiosonde(cls, path: str | Path) -> "AtmosphericProfile":
        """Load from a standard radiosonde text file.

        Expects a whitespace-delimited file with columns:
            pressure(mb)  altitude(m)  temperature(C)  dewpoint(C)

        Water vapour is derived from dewpoint using the Magnus formula.
        Ozone is not available from radiosondes and is set to zero
        (MODTRAN will use its built-in climatological ozone profile).
        """
        path = Path(path)
        data = np.loadtxt(path, comments="#")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] < 4:
            raise ValueError(
                "Radiosonde file must have at least 4 columns: "
                "pressure(mb), altitude(m), temperature(C), dewpoint(C)"
            )
        pressure_mb = data[:, 0]
        altitude_km = data[:, 1] / 1000.0
        temperature_k = data[:, 2] + 273.15
        dewpoint_c = data[:, 3]

        # Magnus formula for saturation vapour pressure at dewpoint (hPa)
        e_sat = 6.112 * np.exp(17.67 * dewpoint_c / (dewpoint_c + 243.5))
        # Water vapour mixing ratio: e / (p - e) * 1e6 * (Mair/Mwater) ≈ ppmv
        # Approximate conversion: wmr(g/kg) ≈ 622 * e / (p - e), then to ppmv
        h2o_ppmv = 622.0 * e_sat / (pressure_mb - e_sat) * 1000.0  # ppmv approx

        return cls(
            altitude_km=altitude_km,
            pressure_mb=pressure_mb,
            temperature_k=temperature_k,
            h2o_ppmv=np.clip(h2o_ppmv, 0.0, None),
            o3_ppmv=np.zeros_like(pressure_mb),
        )

    def to_csv(self, path: str | Path) -> None:
        """Write the profile to a CSV file."""
        path = Path(path)
        columns = ["altitude_km", "pressure_mb", "temperature_k", "h2o_ppmv", "o3_ppmv"]
        arrays = [self.altitude_km, self.pressure_mb, self.temperature_k,
                  self.h2o_ppmv, self.o3_ppmv]
        for name in ("co2_ppmv", "ch4_ppmv", "n2o_ppmv", "co_ppmv"):
            val = getattr(self, name)
            if val is not None:
                columns.append(name)
                arrays.append(val)

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for row in zip(*arrays):
                writer.writerow([f"{v:.6g}" for v in row])
