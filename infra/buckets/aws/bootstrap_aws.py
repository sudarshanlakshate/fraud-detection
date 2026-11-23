import os
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Automatically searches upward from this file to find and load .env in the project root
load_dotenv()

PROJECT_NAME = os.getenv("PROJECT_NAME", "fraud-detection-mlops")
ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")  # Mumbai
BUCKET_TYPES = [
    b.strip()
    for b in os.getenv("BUCKET_TYPES", "data,mlflow-artifacts,pipelines").split(",")
    if b.strip()
]


def ensure_s3_bucket(s3_client, bucket_name: str, region: str) -> None:
    """Create S3 bucket with versioning, tags, and public-access block if it doesn't exist."""
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"[SKIP] Already exists: s3://{bucket_name}")
        return
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code")
        if error_code not in ("404", "NoSuchBucket"):
            raise RuntimeError(f"Error checking s3://{bucket_name}: {exc}") from exc

    print(f"[CREATE] Creating s3://{bucket_name} in {region}...")
    create_kwargs = {"Bucket": bucket_name}
    if region != "us-east-1":
        create_kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}

    s3_client.create_bucket(**create_kwargs)

    s3_client.put_public_access_block(
        Bucket=bucket_name,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )

    s3_client.put_bucket_versioning(
        Bucket=bucket_name,
        VersioningConfiguration={"Status": "Enabled"},
    )

    s3_client.put_bucket_tagging(
        Bucket=bucket_name,
        Tagging={
            "TagSet": [
                {"Key": "Project", "Value": PROJECT_NAME},
                {"Key": "Environment", "Value": ENVIRONMENT},
            ]
        },
    )
    print(f"[DONE] Ready: s3://{bucket_name}")


def main() -> None:
    # boto3 automatically picks up AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY from os.environ if loaded
    session = boto3.Session(region_name=AWS_REGION)
    account_id = session.client("sts").get_caller_identity()["Account"]
    s3 = session.client("s3")

    for b_type in BUCKET_TYPES:
        bucket_name = f"{PROJECT_NAME}-{b_type}-{account_id}"
        ensure_s3_bucket(s3, bucket_name, AWS_REGION)


if __name__ == "__main__":
    main()
