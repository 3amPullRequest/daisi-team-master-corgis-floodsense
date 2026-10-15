"""The gated pair as an MLflow pyfunc model, registered in Unity Catalog (train.py) and loaded for scoring with
mlflow.pyfunc.load_model("models:/<catalog>.<ml schema>.floodsense@champion").

predict takes gold.area_features rows (area, ts and contract.MODEL_FEATURES) and returns ml.predictions rows.
"""

import mlflow

from .gated import GatedPair


class GatedPairModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.pair = GatedPair.load(context.artifacts["pair"])

    def predict(self, context, model_input, params=None):
        return self.pair.predict(model_input)
