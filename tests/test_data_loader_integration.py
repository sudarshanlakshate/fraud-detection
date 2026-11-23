import os
from pathlib import Path

import boto3
import pytest
from botocore.exceptions import ClientError
from google.cloud import storage

from fraud_detection.data.data_loader import DataLoader
from fraud_detection.settings import settings

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_CLOUD_INTEGRATION_TESTS") != "1",
    reason="Set RUN_CLOUD_INTEGRATION_TESTS=1 to connect to cloud storage.",
)


def _dataset_files() -> tuple[str, ...]:
    return (
        settings.train_transaction_file,
        settings.train_identity_file,
        settings.test_transaction_file,
        settings.test_identity_file,
        settings.sample_submission_file,
    )


def _object_keys(filename: str) -> list[str]:
    relative_key = filename.replace("\\", "/").lstrip("/")
    prefixed_key = "/".join(
        part for part in (settings.raw_data_prefix.strip("/"), relative_key) if part
    )
    keys = [prefixed_key]
    if prefixed_key != relative_key:
        keys.append(relative_key)
    return keys


def test_aws_can_access_all_configured_dataset_objects() -> None:
    if not settings.aws_data_bucket:
        pytest.skip("AWS_DATA_BUCKET is not configured.")

    client = boto3.client("s3", region_name=settings.aws_region)
    for filename in _dataset_files():
        for key in _object_keys(filename):
            try:
                response = client.head_object(
                    Bucket=settings.aws_data_bucket,
                    Key=key,
                )
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code") not in {
                    "404",
                    "NoSuchKey",
                    "NotFound",
                }:
                    raise
            else:
                assert response["ContentLength"] > 0
                break
        else:
            pytest.fail(f"Configured dataset object was not found in S3: {filename}")


def test_gcp_can_access_all_configured_dataset_objects() -> None:
    if not settings.gcp_data_bucket:
        pytest.skip("GCP_DATA_BUCKET is not configured.")

    client = storage.Client(project=settings.gcp_project_id)
    for filename in _dataset_files():
        assert any(
            client.bucket(settings.gcp_data_bucket).blob(key).exists(client=client)
            for key in _object_keys(filename)
        ), f"Configured dataset object was not found in GCS: {filename}"


@pytest.mark.parametrize("provider", ["aws", "gcp"])
def test_cloud_download_and_local_cache(provider: str, tmp_path: Path) -> None:
    loader = DataLoader()
    loader.cloud_provider = provider
    loader.raw_data_dir = tmp_path

    data = loader.load_raw_data(settings.sample_submission_file)
    cached = loader.load_raw_data(settings.sample_submission_file)

    assert data.shape == (506_691, 2)
    assert cached.equals(data)
    assert (tmp_path / settings.sample_submission_file).is_file()
