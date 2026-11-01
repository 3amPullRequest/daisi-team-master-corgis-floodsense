"""Where the app's data lives. On Databricks the bundle sets these (resources/app.yml); locally, export them or use
the dev defaults with DATABRICKS_CONFIG_PROFILE=flood_dev."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    warehouse_id: str
    catalog: str
    bronze: str
    silver: str
    gold: str
    ml: str
    model: str  # registered model name in <catalog>.<ml>
    dashboard: str  # the agencies' AI/BI scoreboard (resources/dashboard.yml); empty: none
    genie_space: str  # the agencies' Genie space (resources/genie.yml); empty: none
    app: str  # the app's own schema: warning subscriptions, which hold personal data (resources/uc.yml)
    telegram_bot: str  # the warning bot's username, for warnings wherever you are (ml/follow.py); empty: none

    def table(self, schema: str, name: str) -> str:
        return f"{self.catalog}.{getattr(self, schema)}.{name}"

    @property
    def model_full_name(self) -> str:
        return f"{self.catalog}.{self.ml}.{self.model}"


def load() -> Settings:
    env = os.environ.get
    prefix = env("FLOODSENSE_SCHEMA_PREFIX", "")
    return Settings(
        warehouse_id=env("DATABRICKS_WAREHOUSE_ID", ""),
        catalog=env("FLOODSENSE_CATALOG", "flood_dev"),
        bronze=env("FLOODSENSE_BRONZE", f"{prefix}bronze"),
        silver=env("FLOODSENSE_SILVER", f"{prefix}silver"),
        gold=env("FLOODSENSE_GOLD", f"{prefix}gold"),
        ml=env("FLOODSENSE_ML", f"{prefix}ml"),
        model=env("FLOODSENSE_MODEL", f"{prefix}floodsense"),
        dashboard=env("FLOODSENSE_DASHBOARD", ""),
        genie_space=env("FLOODSENSE_GENIE_SPACE", ""),
        app=env("FLOODSENSE_APP", f"{prefix}app"),
        telegram_bot=env("FLOODSENSE_TELEGRAM_BOT", "").lstrip("@"),
    )
