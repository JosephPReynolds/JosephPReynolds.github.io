"""BatchRunner: parameter sweep over ModtranCase fields."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .runner import ModtranRunner
    from .inputs.case import ModtranCase


class BatchRunner:
    """Run parameter sweeps over a base ModtranCase.

    Parameters
    ----------
    runner : ModtranRunner
        Configured runner to use for each case.
    """

    def __init__(self, runner: "ModtranRunner"):
        self.runner = runner

    def sweep(
        self,
        base_case: "ModtranCase",
        param_grid: dict[str, list],
        n_workers: int = 4,
        output_method: str = "downwelling_irradiance",
        output_units: str = "W/m2/nm",
    ) -> pd.DataFrame:
        """Run all combinations in the Cartesian product of *param_grid*.

        Parameters
        ----------
        base_case : ModtranCase
            Template case.  Each combination overrides fields in this case.
        param_grid : dict
            Mapping of ``ModtranCase`` field name → list of values.
            Example: ``{"solzen": [0, 20, 40, 60], "ihaze": [1, 4]}``.
        n_workers : int
            Number of parallel MODTRAN processes.
        output_method : str
            ``ModtranResult`` method name to call to get spectral output.
            Defaults to ``'downwelling_irradiance'``.
        output_units : str
            Units argument forwarded to *output_method*.

        Returns
        -------
        pd.DataFrame
            Multi-indexed DataFrame.  The outermost index levels correspond to
            each swept parameter; the innermost is the spectral axis
            (wavelength or wavenumber).  Columns are the spectral values.

        Example
        -------
        ::

            runner = ModtranRunner(executable="/path/to/mod6con")
            batch = BatchRunner(runner)
            df = batch.sweep(
                base_case=ModtranCase.for_downwelling_irradiance(),
                param_grid={"solzen": [20, 40, 60]},
            )
            # df has a 2-level MultiIndex: (solzen, wavelength_nm)
        """
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))

        # Build case list
        cases = []
        for combo in combinations:
            overrides = dict(zip(param_names, combo))
            cases.append(base_case.replace(**overrides))

        # Execute in parallel
        results = self.runner.run_batch(cases, n_workers=n_workers)

        # Collect spectral DataFrames and label with parameter values
        frames = []
        for combo, result in zip(combinations, results):
            method = getattr(result, output_method)
            # Call with or without units argument
            try:
                spec_df = method(units=output_units)
            except TypeError:
                spec_df = method()

            # Add parameter values as additional index levels
            keys = dict(zip(param_names, combo))
            for k, v in reversed(list(keys.items())):
                spec_df = pd.concat({v: spec_df}, names=[k])

            frames.append(spec_df)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames)

    def run_cases(
        self,
        cases: list["ModtranCase"],
        n_workers: int = 4,
    ) -> list:
        """Run an arbitrary list of cases and return their results.

        Parameters
        ----------
        cases : list of ModtranCase
            Cases to execute.
        n_workers : int
            Parallelism.

        Returns
        -------
        list of ModtranResult
        """
        return self.runner.run_batch(cases, n_workers=n_workers)
