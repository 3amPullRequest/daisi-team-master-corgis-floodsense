"""The registered model, for replays: the @champion version is downloaded from Unity Catalog once and loaded as the
gated pair it was saved as (GatedPair.load needs only lightgbm and numpy; its code travels with the model).

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
    pair: object  # GatedPair, imported from the model's own code
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
    version = (
        mlflow.MlflowClient().get_model_version_by_alias(full_name, "champion").version
    )
    root = (
        Path(os.environ.get("FLOODSENSE_MODEL_CACHE", tempfile.gettempdir()))
        / f"floodsense-v{version}"
    )
    if not (root / "artifacts" / "pair").exists():
        mlflow.artifacts.download_artifacts(
            f"models:/{full_name}@champion", dst_path=str(root)
        )
    code = root / "code"
    # the floodsense package, as logged with the model (code_paths)
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    gated = importlib.import_module("daisi_team_master_corgis_floodsense.ml.gated")
    contract = importlib.import_module(
        "daisi_team_master_corgis_floodsense.ml.contract"
    )
    pair = gated.GatedPair.load(root / "artifacts" / "pair")
    return Champion(
        version=str(version),
        name=f"{short_name} v{version}",
        pair=pair,
        features=tuple(contract.MODEL_FEATURES),
    )
