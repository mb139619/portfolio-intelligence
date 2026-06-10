"""
Configuration — single source of truth.
Edit values here or override via a .env file (PI_ prefix).
"""

from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_prefix="PI_",
        case_sensitive=False,
    )

    # --- Paths ---
    data_dir: Path = ROOT / "data"

    @property
    def prices_dir(self) -> Path:
        return self.data_dir / "raw" / "prices"

    @property
    def macro_dir(self) -> Path:
        return self.data_dir / "raw" / "macro"

    @property
    def factors_dir(self) -> Path:
        return self.data_dir / "raw" / "factors"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def features_dir(self) -> Path:
        return self.data_dir / "features"

    # --- Analytics defaults ---
    trading_days: int = 252
    risk_free_rate: float = 0.04
    ewma_lambda: float = 0.94
    var_confidence: float = 0.95

    # --- Ingestion defaults ---
    default_start: str = "2015-01-01"

    def ensure_dirs(self) -> None:
        for p in [
            self.prices_dir, self.macro_dir, self.factors_dir,
            self.processed_dir, self.features_dir,
        ]:
            p.mkdir(parents=True, exist_ok=True)


settings = Settings()
