"""gold.model_rows: the area-ticks a model learns from (split 'train') and is tested on (split 'test'), as
contract.MODEL_ROWS_COLUMNS defines them.

A row is there only if its label window is recorded (gold.label_ticks), its feature window is recorded
(gold.feature_ticks) and none of its required features is null. So a missing interval never reaches a model: not its
quiet ticks, which may hide lost alerts, and not the alerts that did come back from it, which kept alone would inflate
the alert rate around every lost post. The expectations fail the update rather than let such a row through. The
optional features (contract.OPTIONAL_FEATURES: the weather, the rain upwind) may be null. Lightning isn't a
model feature yet (contract.HELD_BACK_FEATURES), so it isn't here.

This one table is where the rule "a missing interval is not a quiet one" is enforced for the models, because it is the
only table they read. It is built by inner joins, so a row that fails any condition simply is not there: area_features
rows with every required feature present, kept for the ticks of gold.feature_ticks whose window is recorded, and
joined to the area_labels rows with no missing reason, in the train or test split (the embargo and after_split ticks
fall out here). The three expectations (labels known, features known, split is train or test) are tripwires, not
filters: they fail the update if a join ever lets such a row through.
"""

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from utilities.contract_tables import MODEL_ROWS_SCHEMA

from daisi_team_master_corgis_floodsense.ml import contract as C

FEATURES_KNOWN = " AND ".join(f"{c} IS NOT NULL" for c in C.REQUIRED_FEATURES)


@dp.materialized_view(
    name=C.MODEL_ROWS,
    comment="The area-ticks models learn from (split = 'train') and are tested on once (split = 'test'): "
    "gold.area_features joined to gold.area_labels wherever both the label window and the feature window are "
    "recorded and no required feature is null (the optional ones may be). No missing interval reaches a model, not "
    "even one that saw an alert.",
    schema=MODEL_ROWS_SCHEMA,
    cluster_by=["ts", "area"],
)
@dp.expect_all_or_fail(
    {
        "labels_known": "alert_label IS NOT NULL AND flood_label IS NOT NULL",
        "features_known": FEATURES_KNOWN,
        "train_or_test": "split IN ('train', 'test')",
    }
)
def model_rows():
    labels = spark.read.table(C.AREA_LABELS).where(
        F.col("missing_reason").isNull() & F.col("split").isin("train", "test")
    )
    recorded = (
        spark.read.table(C.FEATURE_TICKS)
        .where(F.col("feature_window") == C.RECORDED)
        .select("ts")
    )
    features = spark.read.table(C.AREA_FEATURES).where(FEATURES_KNOWN)
    rows = features.join(recorded, "ts").join(
        labels.select("area", "ts", "alert_label", "flood_label", "split"),
        ["area", "ts"],
    )
    return rows.select(*C.MODEL_ROWS_COLUMNS)
