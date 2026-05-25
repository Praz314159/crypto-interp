"""Canonical, experiment-agnostic analyses for grokked modular-arithmetic models.

Each module is runnable as ``python -m crypto_interp.analysis.<name> --run-dir ...``
(single-run analyses) or ``--runs-dir ...`` (sweep scanners). They depend only on
``crypto_interp.interp`` and the model/dataset interface, so the same analysis runs
against any experiment's runs without copying code into the experiment folder.
"""
