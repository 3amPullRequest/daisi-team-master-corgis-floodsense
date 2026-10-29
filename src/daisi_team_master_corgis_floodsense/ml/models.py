"""Load a saved FloodSense model of either kind: the ordinal model (ordinal.py, from 23 Oct) or the gated pair
(gated.py, the registered versions up to v5). Both have the same face: `predict` (ml.predictions rows), `chances`,
`features`, `area_priors` and `logged_cut_offs`, which is all live scoring, verify.py and the app use.

A model logged to MLflow keeps its files under artifacts/<name> of the downloaded model: `model` since 23 Oct, `pair`
before (train.log_model). Loading needs lightgbm, numpy and pandas only.
"""

from __future__ import annotations

from pathlib import Path

from .gated import GatedPair
from .ordinal import SAVED, OrdinalModel

# the artifact a logged model's files are in, newest first
ARTIFACTS = ("model", "pair")


def load(directory) -> OrdinalModel | GatedPair:
    """The model saved in `directory` (OrdinalModel.save or GatedPair.save)."""
    d = Path(directory)
    if (d / SAVED).exists():
        return OrdinalModel.load(d)
    return GatedPair.load(d)


def load_logged(root) -> OrdinalModel | GatedPair:
    """The model in a downloaded MLflow model directory (models:/<name>/<version>, downloaded to `root`)."""
    for name in ARTIFACTS:
        directory = Path(root) / "artifacts" / name
        if directory.exists():
            return load(directory)
    raise FileNotFoundError(f"no {' or '.join(ARTIFACTS)} artifact under {root}")
