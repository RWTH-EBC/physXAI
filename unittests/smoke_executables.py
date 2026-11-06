"""
Smoke test of the executables: runs scripts with one training epoch and without opening plots.

The executables train with their full number of epochs when users run them. In the CI only their
correctness matters, so every keras `fit` (single step, recursive training, recurrent models) is
limited to one epoch here. The scripts themselves are not changed.

Usage (from the repository root, the scripts use relative data paths):

    python unittests/smoke_executables.py                                   # all executables
    python unittests/smoke_executables.py executables/bestest_hydronic_heat_pump/TAir.py ...
"""

from __future__ import annotations

import runpy
import sys
import time
import traceback
from pathlib import Path
from unittest.mock import patch

#: Experimental scripts, not part of the smoke test
EXCLUDED = {"regression_prototype.py"}


def _scripts(arguments: list[str]) -> list[Path]:
    if arguments:
        return [Path(a) for a in arguments]
    return sorted(p for p in Path("executables").rglob("*.py") if p.name not in EXCLUDED)


def main(arguments: list[str]) -> int:
    import keras

    fit = keras.Model.fit

    def one_epoch(self, *args, **kwargs):
        kwargs["epochs"] = 1
        return fit(self, *args, **kwargs)

    failed = list()
    for script in _scripts(arguments):
        start = time.perf_counter()
        with patch("plotly.graph_objects.Figure.show"), patch.object(keras.Model, "fit", one_epoch):
            try:
                runpy.run_path(str(script), run_name="__main__")
                status = "OK"
            except Exception:  # noqa: BLE001 - report every failing script
                traceback.print_exc()
                failed.append(str(script))
                status = "FAILED"
        print(f"{status}: {script} ({time.perf_counter() - start:.0f} s)", flush=True)
    if failed:
        print("Failed scripts:", *failed, sep="\n  ")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
