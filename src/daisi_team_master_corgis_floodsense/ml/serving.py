"""The gated pair as an MLflow pyfunc model, registered in Unity Catalog (train.py) and loaded for scoring with
mlflow.pyfunc.load_model("models:/<catalog>.<ml schema>.floodsense@champion").

predict takes gold.area_features rows (area, ts and contract.MODEL_FEATURES) and returns ml.predictions rows.

This is only the wrapper MLflow needs. train.log_model saves the fitted pair (gate.txt, rater.txt and pair.json, see
GatedPair.save) as the model's `pair` artifact, and logs this class with the floodsense package's source, so loading
the model from Unity Catalog needs no installed copy of the package, only lightgbm, numpy and pandas (gated.fit is the
only part that imports scikit-learn, and it imports it lazily for that reason). `load_context` rebuilds the GatedPair
from the artifact; `predict` hands the rows to GatedPair.predict, which gates, rates, calibrates, assigns the tier
and writes the plain-English reason. The registered model carries two aliases (contract.CHAMPION, CHALLENGER): the
live model, and the newest one trained.
"""

import mlflow

from .gated import GatedPair


class GatedPairModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.pair = GatedPair.load(context.artifacts["pair"])

    def predict(self, context, model_input, params=None):
        return self.pair.predict(model_input)
