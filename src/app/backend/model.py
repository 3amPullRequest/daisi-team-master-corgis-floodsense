"""The registered model, for replays: the @champion version is downloaded from Unity Catalog once and loaded with its
own logged code (ml.models.load_logged, which reads the ordinal model and the gated pair alike; a version logged before
23 Oct has no ml.models, and loads as the gated pair it was saved as). Loading needs only lightgbm, numpy and pandas.

Live tiers are not computed here: flood-live's score task writes them to ml.predictions every tick, and the app reads
them, so what the app shows is exactly what was scored. A replay scores a past day's gold.area_features rows with the
same model."""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_lock = threading.Lock()
_loaded: dict[str, Champion] = {}


@dataclass
class Champion:
    version: str
    name: str
    pair: object  # the OrdinalModel or GatedPair, imported from the model's own code
    features: tuple[str, ...]

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        frame = rows.astype({c: "float64" for c in self.features})
        return self.pair.predict(frame)


def champion(full_name: str, short_name: str) -> Champion:
    with _lock:
        if full_name not in _loaded:
            _loaded[full_name] = _load(full_name, short_name)
        return _loaded[full_name]


def _load(full_name: str, short_name: str) -> Champion:
    import mlflow

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    champ = mlflow.MlflowClient().get_model_version_by_alias(full_name, "champion")
    version = champ.version
    root = (
        Path(os.environ.get("FLOODSENSE_MODEL_CACHE", tempfile.gettempdir()))
        / f"floodsense-v{version}"
    )
    if not any((root / "artifacts" / name).exists() for name in ("model", "pair")):
        mlflow.artifacts.download_artifacts(
            f"models:/{full_name}@champion", dst_path=str(root)
        )
    code = root / "code"
    # the floodsense package, as logged with the model (code_paths)
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    contract = importlib.import_module(
        "daisi_team_master_corgis_floodsense.ml.contract"
    )
    try:
        loader = importlib.import_module("daisi_team_master_corgis_floodsense.ml.models")
        pair = loader.load_logged(root)
    except ModuleNotFoundError:  # logged before 23 Oct: a gated pair, in artifacts/pair
        gated = importlib.import_module("daisi_team_master_corgis_floodsense.ml.gated")
        pair = gated.GatedPair.load(root / "artifacts" / "pair")
    return Champion(
        version=str(version),
        name=version_name(short_name, version, champ.aliases),
        pair=pair,
        features=tuple(contract.MODEL_FEATURES),
    )


def version_name(short_name: str, version, aliases=()) -> str:
    """The alias the user named a version by (final_1), when it has one besides champion and challenger, else
    '<model> v<version>': what flood-live calls it in ml.predictions (contract.version_name)."""
    named = sorted(set(aliases or ()) - {"champion", "challenger"})
    return named[0] if named else f"{short_name} v{version}"
