import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

from botocore.exceptions import ClientError
from google.api_core.exceptions import NotFound
from pandas import DataFrame, read_csv

from fraud_detection.settings import settings

logger = logging.getLogger(__name__)


class BaseLoader:
    def __init__(self):
        self.raw_data_dir = settings.raw_data_dir
        self.interim_data_dir = settings.interim_data_dir
        self.cloud_provider = settings.cloud_provider
        self.raw_data_prefix = settings.raw_data_prefix
        self.aws_region = settings.aws_region
        self.aws_data_bucket = settings.aws_data_bucket
        self.gcp_project_id = settings.gcp_project_id
        self.gcp_data_bucket = settings.gcp_data_bucket

    def get_raw_data_path(self, filename: str | Path) -> Path:
        relative_path = Path(filename)
        raw_data_dir = self.raw_data_dir.resolve()
        file_path = (
            relative_path.resolve()
            if relative_path.is_absolute()
            else (raw_data_dir / relative_path).resolve()
        )
        try:
            object_path = file_path.relative_to(raw_data_dir)
        except ValueError as exc:
            raise ValueError(
                f"Raw data filename must be inside '{raw_data_dir}': {filename}"
            ) from exc

        if not file_path.is_file():
            self._download_from_cloud(object_path, file_path)

        return file_path

    def load_raw_data(self, filename: str | Path) -> DataFrame:
        return read_csv(self.get_raw_data_path(filename))

    def _download_from_cloud(self, object_path: Path, file_path: Path) -> None:
        provider = self._get_cloud_provider()
        bucket_name = (
            self.aws_data_bucket if provider == "aws" else self.gcp_data_bucket
        )
        if not bucket_name:
            raise FileNotFoundError(
                f"Raw data file '{file_path}' does not exist. "
                f"Set the {provider.upper()} data bucket to download it."
            )

        relative_key = object_path.as_posix().lstrip("/")
        prefixed_key = "/".join(
            part for part in (self.raw_data_prefix.strip("/"), relative_key) if part
        )
        object_keys = [prefixed_key]
        if prefixed_key != relative_key:
            object_keys.append(relative_key)

        file_path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            dir=file_path.parent,
            prefix=f".{file_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        try:
            for index, object_key in enumerate(object_keys):
                try:
                    self._download_object(
                        provider, bucket_name, object_key, temporary_path
                    )
                except ClientError as exc:
                    error_code = exc.response.get("Error", {}).get("Code")
                    if error_code not in {"404", "NoSuchKey", "NotFound"}:
                        raise
                    if index == len(object_keys) - 1:
                        raise
                    logger.warning(
                        "Cloud object not found at configured key %s; "
                        "trying the bucket-root key.",
                        object_key,
                    )
                except NotFound:
                    if index == len(object_keys) - 1:
                        raise
                    logger.warning(
                        "Cloud object not found at configured key %s; "
                        "trying the bucket-root key.",
                        object_key,
                    )
                else:
                    temporary_path.replace(file_path)
                    return
        finally:
            temporary_path.unlink(missing_ok=True)

    def _download_object(
        self,
        provider: Literal["aws", "gcp"],
        bucket_name: str,
        object_key: str,
        destination: Path,
    ) -> None:
        if provider == "aws":
            import boto3

            boto3.client("s3", region_name=self.aws_region).download_file(
                bucket_name, object_key, str(destination)
            )
        else:
            from google.cloud import storage

            client = storage.Client(project=self.gcp_project_id)
            client.bucket(bucket_name).blob(object_key).download_to_filename(
                str(destination)
            )

    def _get_cloud_provider(self) -> Literal["aws", "gcp"]:
        if self.cloud_provider:
            return self.cloud_provider

        configured_providers: list[Literal["aws", "gcp"]] = []
        if self.aws_data_bucket:
            configured_providers.append("aws")
        if self.gcp_data_bucket:
            configured_providers.append("gcp")

        if len(configured_providers) == 1:
            return configured_providers[0]
        if not configured_providers:
            raise FileNotFoundError(
                "Raw data file is missing and neither AWS_DATA_BUCKET nor "
                "GCP_DATA_BUCKET is configured."
            )
        raise ValueError(
            "Both cloud data buckets are configured. Set CLOUD_PROVIDER to "
            "'aws' or 'gcp' to choose the download source."
        )
