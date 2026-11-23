from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

base_dir: Path = Path(__file__).parents[2]
load_dotenv(base_dir / ".env", override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=base_dir / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    data_dir: Path = base_dir / "data"
    raw_data_dir: Path = data_dir / "raw"
    interim_data_dir: Path = data_dir / "interim"
    processed_data_dir: Path = data_dir / "processed"
    cloud_provider: Literal["aws", "gcp"] | None = None
    raw_data_prefix: str = ""
    aws_region: str | None = None
    aws_data_bucket: str | None = None
    gcp_project_id: str | None = None
    gcp_region: str | None = None
    gcp_data_bucket: str | None = None
    train_transaction_file: str = "train_transaction.csv"
    train_identity_file: str = "train_identity.csv"
    test_transaction_file: str = "test_transaction.csv"
    test_identity_file: str = "test_identity.csv"
    sample_submission_file: str = "sample_submission.csv"


settings = Settings()
