import argparse
import logging
import re
import sys
import types
from pathlib import Path

import databricks.sdk
import pytest

from daisi_team_master_corgis_floodsense.ingest import cli

KEY = "test-key-not-real"


class FakeSecrets:
    def __init__(self, value: str | Exception):
        self.value = value
        self.asked: list[tuple[str, str]] = []

    def get(self, scope: str, key: str) -> str:
        self.asked.append((scope, key))
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


def fake_workspace(monkeypatch, value) -> tuple[FakeSecrets, list]:
    """databricks.sdk.WorkspaceClient, answering secrets.get with `value`; records the profiles it was made with."""
    secrets, profiles = FakeSecrets(value), []

    class Client:
        def __init__(self, profile=None):
            profiles.append(profile)
            self.dbutils = types.SimpleNamespace(secrets=secrets)

    monkeypatch.setattr(databricks.sdk, "WorkspaceClient", Client)
    return secrets, profiles


@pytest.fixture(autouse=True)
def laptop(monkeypatch):
    monkeypatch.delenv(cli.API_KEY_ENV, raising=False)
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)


def test_the_environment_variable_wins(monkeypatch):
    monkeypatch.setenv(cli.API_KEY_ENV, f" {KEY}\n")
    _, profiles = fake_workspace(monkeypatch, "from-scope")
    assert cli.resolve_api_key("floodsense", "flood_dev") == KEY
    assert profiles == []  # no workspace call


def test_laptop_runs_read_the_secret_scope_as_their_profile(monkeypatch, caplog):
    secrets, profiles = fake_workspace(monkeypatch, f"{KEY}\n")
    with caplog.at_level(logging.INFO):
        assert cli.resolve_api_key("floodsense", "flood_dev") == KEY
    assert profiles == ["flood_dev"]
    assert secrets.asked == [("floodsense", cli.API_KEY_SECRET)]
    assert "secret scope 'floodsense'" in caplog.text and KEY not in caplog.text


def test_without_a_profile_a_laptop_run_goes_unauthenticated(monkeypatch, caplog):
    _, profiles = fake_workspace(monkeypatch, KEY)
    assert cli.resolve_api_key("floodsense", None) is None
    assert profiles == [] and "--profile" in caplog.text


def test_a_failed_read_goes_unauthenticated(monkeypatch, caplog):
    fake_workspace(monkeypatch, RuntimeError("RESOURCE_DOES_NOT_EXIST"))
    assert cli.resolve_api_key("floodsense", "flood_dev") is None
    assert "RuntimeError" in caplog.text
    fake_workspace(monkeypatch, "  \n")
    assert cli.resolve_api_key("floodsense", "flood_dev") is None  # an empty secret
    assert cli.resolve_api_key("", "flood_dev") is None  # --secret-scope ''


def test_jobs_read_it_with_their_dbutils(monkeypatch):
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "client.2.0")
    secrets = FakeSecrets(KEY)
    runtime = types.ModuleType("databricks.sdk.runtime")
    runtime.dbutils = types.SimpleNamespace(secrets=secrets)
    monkeypatch.setitem(sys.modules, "databricks.sdk.runtime", runtime)
    fake_workspace(monkeypatch, RuntimeError("a job mustn't need a profile"))
    assert cli.resolve_api_key("floodsense", None) == KEY
    assert secrets.asked == [("floodsense", cli.API_KEY_SECRET)]


def test_the_default_scope_is_the_bundle_variable(monkeypatch):
    monkeypatch.delenv("FLOODSENSE_SECRET_SCOPE", raising=False)
    parser = argparse.ArgumentParser()
    cli.add_common_args(parser)
    args = parser.parse_args(["--landing", "/tmp/x"])
    bundle = (Path(__file__).resolve().parents[1] / "databricks.yml").read_text()
    default = re.search(r"secret_scope:\n(?:\s+.*\n)*?\s+default: (\S+)", bundle)
    assert args.secret_scope == default.group(1) == cli.SECRET_SCOPE
