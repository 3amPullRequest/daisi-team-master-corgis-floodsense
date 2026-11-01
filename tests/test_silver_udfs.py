"""Silver's UDFs (utilities/silver.py) reach the executors by value: pickled as Spark pickles them, each carries its own
code, and neither a module whole nor anything of this package by reference.

A module pickled whole carries its __loader__ and __spec__, and on Databricks those come from dbruntime, which the
executors don't have: on 1 Nov 2026 read_jev failed silver.telegram_posts with "No module named 'dbruntime'", because
jev.py held alert_text as a module. utilities/silver.py needs a Spark session to import, so its registrations are
repeated here.
"""

import io
import pickle

import pytest
from pyspark import cloudpickle

from daisi_team_master_corgis_floodsense.labels import alert_text, jev, locations

BY_VALUE = (alert_text, jev, locations)  # as utilities/silver.py registers them
# how cloudpickle rebuilds a module pickled whole
WHOLE_MODULE = ("pyspark.cloudpickle.cloudpickle", "dynamic_subimport")
UDFS = (
    alert_text.parse_row,
    jev.read_row,
    locations.location_key,
    locations.name_match,
    locations.svy21_to_wgs84,
)


class Executor(pickle.Unpickler):
    """Unpickles as an executor would, noting each global it looks up as (module, name)."""

    def __init__(self, data: bytes):
        super().__init__(io.BytesIO(data))
        self.looked_up: set[tuple[str, str]] = set()

    def find_class(self, module: str, name: str):
        self.looked_up.add((module, name))
        return super().find_class(module, name)


@pytest.fixture(autouse=True)
def by_value():
    for module in BY_VALUE:
        cloudpickle.register_pickle_by_value(module)
    yield
    for module in BY_VALUE:
        cloudpickle.unregister_pickle_by_value(module)


@pytest.mark.parametrize("udf", UDFS, ids=lambda f: f.__name__)
def test_udfs_travel_by_value(udf):
    executor = Executor(cloudpickle.dumps(udf))
    executor.load()
    assert WHOLE_MODULE not in executor.looked_up
    assert not {
        module
        for module, _ in executor.looked_up
        if module.startswith("daisi_team_master_corgis_floodsense")
    }
