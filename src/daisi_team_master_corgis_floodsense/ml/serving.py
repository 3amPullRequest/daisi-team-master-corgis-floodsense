"""The registered models as MLflow pyfunc models, registered in Unity Catalog (train.py) and loaded for scoring with
mlflow.pyfunc.load_model("models:/<catalog>.<ml schema>.floodsense@champion").

predict takes gold.area_features rows (area, ts and contract.MODEL_FEATURES) and returns ml.predictions rows.

These are only the wrappers MLflow needs. train.log_model saves the fitted model (OrdinalModel.save or GatedPair.save)
as the model's `model` artifact (`pair` for a gated pair), and logs the matching class with the floodsense package's
source, so loading the model from Unity Catalog needs no installed copy of the package, only lightgbm, numpy and pandas
(fitting is the only part that imports scikit-learn and scipy, and it imports them lazily for that reason).
`load_context` rebuilds the model from the artifact; `predict` hands the rows to its predict, which scores, calibrates,
assigns the tier and writes the plain-English reason. The registered model carries two aliases (contract.CHAMPION,
CHALLENGER): the live model, and the newest one trained.

- FloodSenseModel: any model models.load reads; the ordinal model since 23 Oct.
- GatedPairModel: the gated pair, as versions 1 to 5 were logged (their own logged code unpickles it).
"""

import mlflow

from .gated import GatedPair
from .models import load


class FloodSenseModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = load(context.artifacts["model"])

    def predict(self, context, model_input, params=None):
        return self.model.predict(model_input)


class GatedPairModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.pair = GatedPair.load(context.artifacts["pair"])

    def predict(self, context, model_input, params=None):
        return self.pair.predict(model_input)
