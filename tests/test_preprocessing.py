import json
from pathlib import Path

import pandas as pd

from fraud_detection.data.data_loader import DataLoader
from fraud_detection.data.preprocessing import (
    TRANSACTION_CATEGORICAL_COLUMNS,
    TRANSACTION_NUMERIC_COLUMNS,
    run_preprocessing,
)
from fraud_detection.settings import settings


def _transaction_rows(
    transaction_ids: list[int],
    amounts: list[float],
    card1_values: list[str],
    targets: list[int] | None = None,
) -> pd.DataFrame:
    data: dict[str, list[object]] = {
        "TransactionID": transaction_ids,
        "TransactionDT": [
            86_400 * (index + 1) for index in range(len(transaction_ids))
        ],
        "TransactionAmt": amounts,
        "dist1": [0.0] * len(transaction_ids),
        "dist2": [None] * len(transaction_ids),
    }
    for column in TRANSACTION_NUMERIC_COLUMNS:
        if column not in data:
            data[column] = [1.0] * len(transaction_ids)
    for column in TRANSACTION_CATEGORICAL_COLUMNS:
        data[column] = ["value"] * len(transaction_ids)
    data["card1"] = card1_values
    data["card4"] = ["visa"] * len(transaction_ids)
    if targets is not None:
        data["isFraud"] = targets
    return pd.DataFrame(data)


def _identity_rows(
    transaction_ids: list[int], *, test_header: bool = False
) -> pd.DataFrame:
    data: dict[str, list[object]] = {"TransactionID": transaction_ids}
    for index in range(1, 39):
        column = f"id_{index:02}"
        if index in range(1, 12):
            data[column] = [0.5] * len(transaction_ids)
        else:
            data[column] = ["match"] * len(transaction_ids)
    data["DeviceType"] = ["mobile"] * len(transaction_ids)
    data["DeviceInfo"] = ["device"] * len(transaction_ids)
    identity = pd.DataFrame(data)
    if test_header:
        identity = identity.rename(
            columns={f"id_{index:02}": f"id-{index:02}" for index in range(1, 39)}
        )
    return identity


def test_pipeline_validates_joins_transforms_and_writes_processed_data(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir()
    _transaction_rows([1, 2], [100.5, 50.0], ["a", "a"], [0, 1]).to_csv(
        raw_dir / settings.train_transaction_file, index=False
    )
    _transaction_rows([3], [25.25], ["b"]).to_csv(
        raw_dir / settings.test_transaction_file, index=False
    )
    _identity_rows([1]).to_csv(raw_dir / settings.train_identity_file, index=False)
    _identity_rows([3], test_header=True).to_csv(
        raw_dir / settings.test_identity_file, index=False
    )
    pd.DataFrame({"TransactionID": [3], "isFraud": [0]}).to_csv(
        raw_dir / settings.sample_submission_file, index=False
    )

    loader = DataLoader()
    loader.raw_data_dir = raw_dir
    output_path = run_preprocessing(
        loader=loader,
        processed_data_dir=output_dir,
        chunk_size=1,
        expected_shapes=None,
    )

    train = pd.read_parquet(output_path / "train.parquet")
    test = pd.read_parquet(output_path / "test.parquet")
    manifest = json.loads((output_path / "manifest.json").read_text(encoding="utf-8"))

    assert len(train) == 2
    assert len(test) == 1
    assert "isFraud" in train.columns
    assert "isFraud" not in test.columns
    assert train["has_identity"].tolist() == [True, False]
    assert test["has_identity"].tolist() == [True]
    assert train.loc[0, "id_01"] == 0.5
    assert train.loc[0, "TransactionAmt_fractional_part"] == 0.5
    assert train.loc[0, "transaction_day_index"] == 1
    assert train.loc[0, "TransactionAmt_log1p"] > 4.0
    assert train.loc[0, "card1_count_full"] == 2
    assert manifest["identity_join"]["unmatched_train"] == 1
    assert manifest["identity_join"]["unmatched_test"] == 0
    assert (output_path / "missingness.csv").is_file()
    assert (output_path / "sample_submission.parquet").is_file()
