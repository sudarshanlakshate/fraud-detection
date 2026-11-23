import os
from dotenv import load_dotenv
from google.api_core.exceptions import Conflict, Forbidden, NotFound
from google.cloud import storage

# Automatically searches upward to find and load .env in the project root
load_dotenv()

PROJECT_NAME = os.getenv("PROJECT_NAME", "fraud-detection-mlops")
ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "fraud-detection-mlops-2026")
GCP_REGION = os.getenv("GCP_REGION", "asia-south1")  # Mumbai
BUCKET_TYPES = [
    b.strip()
    for b in os.getenv("BUCKET_TYPES", "data,mlflow-artifacts,pipelines").split(",")
    if b.strip()
]


def ensure_gcs_bucket(client: storage.Client, bucket_name: str, region: str) -> None:
    """Create a GCS bucket with Uniform Access, Public Access Prevention, versioning, and labels if missing."""
    try:
        client.get_bucket(bucket_name)
        print(f"[SKIP] Already exists: gs://{bucket_name}")
        return
    except NotFound:
        pass
    except Forbidden as exc:
        raise RuntimeError(
            f"Access denied checking gs://{bucket_name} (name may be taken globally or IAM missing): {exc}"
        ) from exc

    print(f"[CREATE] Creating gs://{bucket_name} in {region}...")
    bucket = client.bucket(bucket_name)
    bucket.storage_class = "STANDARD"

    # Security hardening (equivalent to AWS S3 BlockPublicAccess + ACL block)
    bucket.iam_configuration.uniform_bucket_level_access_enabled = True
    bucket.iam_configuration.public_access_prevention = "enforced"

    # Enable object versioning for reproducibility & rollback
    bucket.versioning_enabled = True

    # Apply FinOps labels (GCP equivalent of AWS Tags — must be lowercase)
    bucket.labels = {
        "project": PROJECT_NAME.lower(),
        "environment": ENVIRONMENT.lower(),
    }

    try:
        client.create_bucket(bucket, location=region)
    except Conflict:
        print(f"[SKIP] Bucket created concurrently: gs://{bucket_name}")
        return

    print(f"[DONE] Ready: gs://{bucket_name}")


def main() -> None:
    # Automatically picks up GOOGLE_APPLICATION_CREDENTIALS from .env or local ADC
    client = storage.Client(project=GCP_PROJECT_ID)

    for b_type in BUCKET_TYPES:
        # Appending GCP_PROJECT_ID guarantees global uniqueness across Google Cloud
        bucket_name = f"{GCP_PROJECT_ID}-{b_type}"
        ensure_gcs_bucket(client, bucket_name, GCP_REGION)


if __name__ == "__main__":
    main()
