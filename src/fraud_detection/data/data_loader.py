from fraud_detection.data.base_loader import BaseLoader


class DataLoader(BaseLoader):
    """Load CSV data from the local raw cache or its configured S3 source."""
