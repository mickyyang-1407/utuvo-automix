"""UTUVO automix MVP.

Drop a folder of stems in, pick a style, get a mixed WAV plus an editable
MixPlan and a human-readable receipt.

Public entry points
-------------------
* :func:`automix.cli.main` -- the command line interface
* :func:`automix.analyze.load_stems` -- read and describe the input
* :func:`automix.plan.build_plan` -- preset + analysis -> MixPlan
* :func:`automix.pipeline.mix` -- the whole job: plan, render, and (in
  reference mode) re-plan against what the first render actually measured
* :func:`automix.reference.analyse` -- turn a reference track into targets
* :func:`automix.render.render` -- MixPlan + stems -> audio, plus measurements
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"