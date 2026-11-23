from pathlib import Path
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError
from google.api_core.exceptions import NotFound

from fraud_detection.data.data_loader import DataLoader


def test_loads_existing_local_file_without_cloud_access(tmp_path: Path) -> None:
    source = tmp_path / "transactions.csv"
    source.write_text("TransactionID,isFraud\n1,0\n", encoding="utf-8")
    loader = DataLoader()
    loader.raw_data_dir = tmp_path

    data = loader.load_raw_data("transactions.csv")

    assert data.to_dict(orient="records") == [{"TransactionID": 1, "isFraud": 0}]


@pytest.mark.parametrize("provider", ["aws", "gcp"])
@pytest.mark.parametrize("fallback_to_root", [False, True])
def test_downloads_missing_file_and_reuses_cached_copy(
    provider: str,
    fallback_to_root: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = DataLoader()
    loader.raw_data_dir = tmp_path
    loader.cloud_provider = provider
    loader.raw_data_prefix = "ieee-cis/raw"
    downloads: list[tuple[str, str, str]] = []

    def write_csv(destination: str) -> None:
        Path(destination).write_text("TransactionID,isFraud\n2,1\n", encoding="utf-8")

    if provider == "aws":
        loader.aws_data_bucket = "fraud-datasets"

        def download_file(bucket: str, key: str, destination: str) -> None:
            downloads.append((bucket, key, "aws"))
            if fallback_to_root and key.startswith("ieee-cis/raw/"):
                raise ClientError({"Error": {"Code": "404"}}, "GetObject")
            write_csv(destination)

        def create_aws_client(service: str, region_name: str | None) -> SimpleNamespace:
            assert service == "s3"
            assert region_name == loader.aws_region
            return SimpleNamespace(download_file=download_file)

        monkeypatch.setattr("boto3.client", create_aws_client)
    else:
        loader.gcp_data_bucket = "fraud-datasets"

        class FakeBlob:
            def __init__(self, key: str):
                self.key = key

            def download_to_filename(self, destination: str) -> None:
                downloads.append(("fraud-datasets", self.key, "gcp"))
                if fallback_to_root and self.key.startswith("ieee-cis/raw/"):
                    raise NotFound("object not found")
                write_csv(destination)

        class FakeBucket:
            def blob(self, key: str) -> FakeBlob:
                assert key in {
                    "ieee-cis/raw/transactions.csv",
                    "transactions.csv",
                }
                return FakeBlob(key)

        class FakeStorageClient:
            def __init__(self, project: str | None):
                assert project == loader.gcp_project_id

            def bucket(self, bucket_name: str) -> FakeBucket:
                assert bucket_name == "fraud-datasets"
                return FakeBucket()

        monkeypatch.setattr("google.cloud.storage.Client", FakeStorageClient)

    first = loader.load_raw_data("transactions.csv")
    second = loader.load_raw_data("transactions.csv")

    assert first.equals(second)
    assert first.to_dict(orient="records") == [{"TransactionID": 2, "isFraud": 1}]
    expected_keys = ["ieee-cis/raw/transactions.csv"]
    if fallback_to_root:
        expected_keys.append("transactions.csv")
    assert downloads == [("fraud-datasets", key, provider) for key in expected_keys]
    assert (tmp_path / "transactions.csv").is_file()


def test_requires_provider_if_multiple_cloud_buckets_are_configured(
    tmp_path: Path,
) -> None:
    loader = DataLoader()
    loader.raw_data_dir = tmp_path
    loader.cloud_provider = None
    loader.aws_data_bucket = "aws-bucket"
    loader.gcp_data_bucket = "gcp-bucket"

    with pytest.raises(ValueError, match="Set CLOUD_PROVIDER"):
        loader.load_raw_data("missing.csv")


def test_missing_cloud_buckets_raise_clear_error(tmp_path: Path) -> None:
    loader = DataLoader()
    loader.raw_data_dir = tmp_path
    loader.cloud_provider = None
    loader.aws_data_bucket = None
    loader.gcp_data_bucket = None

    with pytest.raises(FileNotFoundError, match="neither AWS_DATA_BUCKET nor"):
        loader.load_raw_data("missing.csv")
