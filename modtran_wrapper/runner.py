"""ModtranRunner: execute MODTRAN and return parsed results."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from .exceptions import ModtranConfigError, ModtranRunError
from .inputs.case import ModtranCase
from .output import ModtranResult
from .readers.tape7_reader import read_tape7
from .readers.flux_reader import read_flux
from .readers.csv_reader import read_csv


class ModtranRunner:
    """Execute MODTRAN 5 or 6 and return structured output.

    Parameters
    ----------
    executable : str or None
        Path to the MODTRAN executable.  Falls back to the ``MODTRAN_EXE``
        environment variable if not provided.
    data_dir : str or None
        Path to the MODTRAN DATA directory.  Falls back to ``MODTRAN_DATA``
        environment variable.
    version : int or None
        MODTRAN version (5 or 6).  Auto-detected from the executable name
        if not specified.
    work_dir : str or None
        Persistent working directory for all runs.  A unique subdirectory is
        created inside it for each run.  If ``None``, a temporary directory
        is used per run and deleted after parsing (unless ``keep_files=True``).
    keep_files : bool
        When ``True``, do not delete per-run working directories after parsing.
    timeout : int
        Subprocess timeout in seconds.  Default 300 (5 minutes).
    """

    def __init__(
        self,
        executable: Optional[str] = None,
        data_dir: Optional[str] = None,
        version: Optional[int] = None,
        work_dir: Optional[str] = None,
        keep_files: bool = False,
        timeout: int = 300,
    ):
        self.executable = self._resolve_executable(executable)
        self.data_dir = self._resolve_data_dir(data_dir)
        self.version = version or self._detect_version()
        self.work_dir = Path(work_dir) if work_dir else None
        self.keep_files = keep_files
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, case: ModtranCase) -> ModtranResult:
        """Execute a single MODTRAN case and return the parsed result.

        Parameters
        ----------
        case : ModtranCase
            Input parameters for the run.

        Returns
        -------
        ModtranResult
        """
        run_dir = self._make_run_dir()
        try:
            self._write_input(case, run_dir)
            self._invoke(run_dir)
            result = self._parse_outputs(run_dir, case)
        except Exception:
            if not self.keep_files:
                _safe_rmtree(run_dir)
            raise

        if not self.keep_files:
            _safe_rmtree(run_dir)

        return result

    def run_batch(
        self,
        cases: list[ModtranCase],
        n_workers: int = 4,
    ) -> list[ModtranResult]:
        """Execute multiple MODTRAN cases in parallel.

        Parameters
        ----------
        cases : list of ModtranCase
            Input cases to run.
        n_workers : int
            Maximum number of concurrent MODTRAN processes.

        Returns
        -------
        list of ModtranResult
            Results in the same order as ``cases``.
        """
        results: list[Optional[ModtranResult]] = [None] * len(cases)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            future_to_idx = {pool.submit(self.run, case): i for i, case in enumerate(cases)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()  # propagates exceptions
        return results  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Internal: setup
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_executable(exe: Optional[str]) -> Path:
        if exe:
            p = Path(exe)
            if not p.exists():
                raise ModtranConfigError(f"MODTRAN executable not found: {p}")
            return p.resolve()
        env_exe = os.environ.get("MODTRAN_EXE")
        if env_exe:
            p = Path(env_exe)
            if not p.exists():
                raise ModtranConfigError(
                    f"MODTRAN_EXE env var points to non-existent file: {p}"
                )
            return p.resolve()
        raise ModtranConfigError(
            "MODTRAN executable not found.  Provide 'executable' argument or "
            "set the MODTRAN_EXE environment variable."
        )

    @staticmethod
    def _resolve_data_dir(data_dir: Optional[str]) -> Optional[Path]:
        if data_dir:
            p = Path(data_dir)
            if not p.is_dir():
                raise ModtranConfigError(f"MODTRAN data directory not found: {p}")
            return p.resolve()
        env_data = os.environ.get("MODTRAN_DATA")
        if env_data:
            p = Path(env_data)
            if p.is_dir():
                return p.resolve()
        return None  # optional — MODTRAN may find it relative to executable

    def _detect_version(self) -> int:
        """Guess MODTRAN version from the executable name."""
        name = self.executable.name.lower()
        if "6" in name or "mod6" in name:
            return 6
        if "5" in name or "mod5" in name:
            return 5
        # Default to 6 for modern installations
        return 6

    # ------------------------------------------------------------------
    # Internal: per-run lifecycle
    # ------------------------------------------------------------------

    def _make_run_dir(self) -> Path:
        """Create and return a unique directory for this run."""
        run_id = str(uuid.uuid4())[:8]
        if self.work_dir is not None:
            self.work_dir.mkdir(parents=True, exist_ok=True)
            run_dir = self.work_dir / run_id
            run_dir.mkdir()
        else:
            run_dir = Path(tempfile.mkdtemp(prefix="modtran_"))
        return run_dir

    def _write_input(self, case: ModtranCase, run_dir: Path) -> Path:
        """Write the appropriate input file for the detected MODTRAN version."""
        if self.version == 6:
            from .writers.json_writer import write_json
            return write_json(case, run_dir / "modtran_input.json")
        else:
            from .writers.tape5_writer import write_tape5
            return write_tape5(case, run_dir / "tape5")

    def _invoke(self, run_dir: Path) -> None:
        """Invoke the MODTRAN executable in *run_dir*."""
        env = os.environ.copy()
        if self.data_dir is not None:
            env["MODTRAN_DATA"] = str(self.data_dir)

        # MODTRAN 6: pass the input filename; MODTRAN 5: no argument needed
        if self.version == 6:
            cmd = [str(self.executable), "modtran_input.json"]
        else:
            cmd = [str(self.executable)]

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(run_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise ModtranRunError(
                f"MODTRAN timed out after {self.timeout}s in {run_dir}"
            ) from exc
        except OSError as exc:
            raise ModtranRunError(
                f"Failed to launch MODTRAN ({self.executable}): {exc}"
            ) from exc

        if proc.returncode != 0:
            raise ModtranRunError(
                f"MODTRAN exited with code {proc.returncode}.\n"
                f"stderr: {proc.stderr[-2000:] if proc.stderr else '(empty)'}",
                returncode=proc.returncode,
                stderr=proc.stderr,
            )

    def _parse_outputs(self, run_dir: Path, case: ModtranCase) -> ModtranResult:
        """Locate and parse all output files in *run_dir*."""
        tape7_df = None
        flux_df = None
        csv_df = None

        # tape7 (MODTRAN 5 and 6)
        for candidate in ("tape7", "modtran_input.tp7", "tape7.7"):
            p = run_dir / candidate
            if p.exists():
                try:
                    tape7_df = read_tape7(p, iemsct=case.iemsct)
                except Exception:
                    pass
                break

        # .flx flux file
        for p in sorted(run_dir.glob("*.flx")) + [run_dir / "tape7.flx"]:
            if p.exists():
                try:
                    flux_df = read_flux(p)
                except Exception:
                    pass
                break

        # .csv (MODTRAN 6 only)
        if self.version == 6:
            for p in sorted(run_dir.glob("*.csv")):
                if p.exists():
                    try:
                        csv_df = read_csv(p)
                    except Exception:
                        pass
                    break

        return ModtranResult(
            tape7=tape7_df,
            flux=flux_df,
            csv=csv_df,
            case=case,
            run_dir=run_dir,
        )


def _safe_rmtree(path: Path) -> None:
    """Remove a directory tree, ignoring errors."""
    try:
        shutil.rmtree(path)
    except Exception:
        pass
