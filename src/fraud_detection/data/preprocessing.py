from __future__ import annotations

import hashlib
import json
import logging
import math
import tempfile
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fraud_detection.data.data_loader import DataLoader
from fraud_detection.settings import settings

logger = logging.getLogger(__name__)

FEATURE_SCHEMA_VERSION = "1.0"
EXPECTED_SOURCE_SHAPES = {
    "train_transaction": (590_540, 394),
    "train_identity": (144_233, 41),
    "test_transaction": (506_691, 393),
    "test_identity": (141_907, 41),
    "sample_submission": (506_691, 2),
}

TRANSACTION_NUMERIC_COLUMNS = (
    "TransactionDT",
    "TransactionAmt",
    "dist1",
    "dist2",
    *(f"C{i}" for i in range(1, 15)),
    *(f"D{i}" for i in range(1, 16)),
    *(f"V{i}" for i in range(1, 340)),
)
TRANSACTION_CATEGORICAL_COLUMNS = (
    "ProductCD",
    *(f"card{i}" for i in range(1, 7)),
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    *(f"M{i}" for i in range(1, 10)),
)
IDENTITY_NUMERIC_COLUMNS = {f"id_{i:02}" for i in range(1, 12)}
IDENTITY_CATEGORICAL_COLUMNS = {
    "DeviceType",
    "DeviceInfo",
    "id_12",
    "id_15",
    "id_16",
    "id_23",
    "id_27",
    "id_28",
    "id_29",
    "id_30",
    "id_31",
    "id_33",
    *(f"id_{i:02}" for i in range(34, 39)),
}
IDENTITY_ALIAS_MAP = {f"id-{i:02}": f"id_{i:02}" for i in range(1, 39)}
FREQUENCY_COLUMNS = (
    "card1",
    "card2",
    "card4",
    "addr1",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceInfo",
)
AGGREGATE_GROUP_COLUMNS = ("card1", "card4")
CHUNK_SIZE = 5_000


def normalize_identity_columns(identity: pd.DataFrame) -> pd.DataFrame:
    renamed_columns = [
        IDENTITY_ALIAS_MAP.get(column, column) for column in identity.columns
    ]
    if len(renamed_columns) != len(set(renamed_columns)):
        raise ValueError("Identity column normalization would create duplicate names.")
    return identity.rename(
        columns=dict(zip(identity.columns, renamed_columns, strict=True))
    )


def _transaction_columns(has_target: bool) -> set[str]:
    columns = {
        "TransactionID",
        *TRANSACTION_NUMERIC_COLUMNS,
        *TRANSACTION_CATEGORICAL_COLUMNS,
    }
    if has_target:
        columns.add("isFraud")
    return columns


def _coerce_transaction_types(
    transactions: pd.DataFrame, has_target: bool
) -> pd.DataFrame:
    expected_columns = _transaction_columns(has_target)
    unexpected_columns = set(transactions.columns) - expected_columns
    missing_columns = expected_columns - set(transactions.columns)
    if unexpected_columns or missing_columns:
        raise ValueError(
            "Transaction schema mismatch. "
            f"Missing columns: {sorted(missing_columns)}; "
            f"unexpected columns: {sorted(unexpected_columns)}."
        )

    dtype_map: dict[str, Any] = {
        "TransactionID": "int32",
        "TransactionDT": "int32",
        "TransactionAmt": "float64",
        **{
            column: "float32"
            for column in TRANSACTION_NUMERIC_COLUMNS
            if column not in {"TransactionDT", "TransactionAmt"}
        },
        **{column: "string" for column in TRANSACTION_CATEGORICAL_COLUMNS},
    }
    if has_target:
        target = pd.to_numeric(transactions["isFraud"], errors="raise")
        if target.isna().any() or not target.isin([0, 1]).all():
            raise ValueError("Training target isFraud must contain only 0 and 1.")
        dtype_map["isFraud"] = "int8"

    return transactions.astype(dtype_map).copy()


def _coerce_identity_types(identity: pd.DataFrame) -> pd.DataFrame:
    identity = normalize_identity_columns(identity)
    expected_columns = {
        "TransactionID",
        "DeviceType",
        "DeviceInfo",
        *(f"id_{i:02}" for i in range(1, 39)),
    }
    if set(identity.columns) != expected_columns:
        raise ValueError(
            "Identity schema mismatch after alias normalization. "
            f"Missing columns: {sorted(expected_columns - set(identity.columns))}; "
            f"unexpected columns: {sorted(set(identity.columns) - expected_columns)}."
        )

    dtype_map: dict[str, Any] = {
        "TransactionID": "int32",
        **{column: "float32" for column in IDENTITY_NUMERIC_COLUMNS},
        **{
            column: "string"
            for column in expected_columns
            - IDENTITY_NUMERIC_COLUMNS
            - {"TransactionID"}
        },
    }
    return identity.astype(dtype_map).copy()


def _merge_identity(transactions: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    merged = transactions.merge(
        identity,
        how="left",
        on="TransactionID",
        sort=False,
        validate="many_to_one",
        indicator="_identity_merge",
    )
    has_identity = merged.pop("_identity_merge").eq("both").astype(bool)
    return pd.concat([merged, has_identity.rename("has_identity")], axis=1)


def _record_frequency_counts(
    counts: dict[str, Counter[str | None]], frame: pd.DataFrame
) -> None:
    for column in FREQUENCY_COLUMNS:
        values = frame[column].astype("string")
        for value, count in values.value_counts(dropna=False).items():
            key = None if pd.isna(value) else str(value)
            counts[column][key] += int(count)


def _record_amount_aggregates(
    aggregates: dict[str, dict[str, list[float]]], frame: pd.DataFrame
) -> None:
    for group_column in AGGREGATE_GROUP_COLUMNS:
        valid = frame.loc[
            frame[group_column].notna() & frame["TransactionAmt"].notna(),
            [group_column, "TransactionAmt"],
        ].copy()
        if valid.empty:
            continue
        valid["_amount_squared"] = valid["TransactionAmt"] ** 2
        grouped = valid.groupby(group_column, sort=False, observed=True).agg(
            count=("TransactionAmt", "count"),
            total=("TransactionAmt", "sum"),
            sum_squared=("_amount_squared", "sum"),
        )
        for key, values in grouped.iterrows():
            state = aggregates[group_column].setdefault(str(key), [0.0, 0.0, 0.0])
            state[0] += int(values["count"])
            state[1] += float(values["total"])
            state[2] += float(values["sum_squared"])


def _load_identity(path: Path) -> pd.DataFrame:
    identity = _coerce_identity_types(pd.read_csv(path, dtype="string"))
    if identity["TransactionID"].isna().any():
        raise ValueError(f"Identity table has null TransactionID values: {path.name}")
    if identity["TransactionID"].duplicated().any():
        duplicate_ids = identity.loc[
            identity["TransactionID"].duplicated(keep=False), "TransactionID"
        ].head(5)
        raise ValueError(
            f"Identity table has duplicate TransactionID values in {path.name}: "
            f"{duplicate_ids.tolist()}"
        )
    return identity


def _precompute_aggregates(
    transaction_paths: Mapping[str, Path],
    identity_paths: Mapping[str, Path],
    chunk_size: int,
) -> tuple[dict[str, Counter[str | None]], dict[str, dict[str, list[float]]]]:
    frequency_counts = {column: Counter() for column in FREQUENCY_COLUMNS}
    amount_aggregates = {column: {} for column in AGGREGATE_GROUP_COLUMNS}

    for split in ("train", "test"):
        identity = _load_identity(identity_paths[split])
        transaction_path = transaction_paths[split]
        for transactions in pd.read_csv(
            transaction_path,
            chunksize=chunk_size,
            dtype={column: "string" for column in TRANSACTION_CATEGORICAL_COLUMNS},
        ):
            transactions = _coerce_transaction_types(
                transactions, has_target=split == "train"
            )
            if split == "train":
                transactions = transactions.drop(columns="isFraud")
            frame = _merge_identity(transactions, identity)
            _record_frequency_counts(frequency_counts, frame)
            _record_amount_aggregates(amount_aggregates, frame)
        del identity

    return frequency_counts, amount_aggregates


def _aggregate_moments(
    state: list[float],
) -> tuple[float, float | None]:
    count, total, sum_squared = state
    mean = total / count
    if count < 2:
        return mean, None
    variance = max((sum_squared - total * total / count) / (count - 1), 0.0)
    return mean, math.sqrt(variance)


def _frequency_feature(
    source: pd.Series, source_column: str, counts: Counter[str | None]
) -> pd.Series:
    source = source.astype("string")
    known_counts = {key: value for key, value in counts.items() if key is not None}
    encoded = source.map(known_counts)
    missing_values = source.isna()
    if missing_values.any():
        encoded.loc[missing_values] = counts.get(None, 0)
    if encoded.isna().any():
        raise ValueError(f"Unable to apply frequency map for {source_column}.")
    return encoded.astype("int32").rename(f"{source_column}_count_full")


def _apply_amount_ratio_features(
    frame: pd.DataFrame,
    aggregates: dict[str, dict[str, list[float]]],
) -> dict[str, pd.Series]:
    amount = frame["TransactionAmt"]
    features: dict[str, pd.Series] = {}
    for group_column, group_values in aggregates.items():
        means: dict[str, float] = {}
        standard_deviations: dict[str, float] = {}
        for key, state in group_values.items():
            mean, standard_deviation = _aggregate_moments(state)
            means[key] = mean
            if standard_deviation is not None and standard_deviation > 0:
                standard_deviations[key] = standard_deviation

        group = frame[group_column].astype("string")
        mean = group.map(means).astype("float64")
        standard_deviation = group.map(standard_deviations).astype("float64")
        features[f"TransactionAmt_to_mean_{group_column}"] = (amount / mean).astype(
            "float32"
        )
        features[f"TransactionAmt_to_std_{group_column}"] = (
            amount / standard_deviation
        ).astype("float32")
    return features


def transform_transaction_chunk(
    transactions: pd.DataFrame,
    identity: pd.DataFrame,
    has_target: bool,
    frequency_counts: dict[str, Counter[str | None]],
    amount_aggregates: dict[str, dict[str, list[float]]],
) -> pd.DataFrame:
    frame = _coerce_transaction_types(transactions, has_target)
    frame = _merge_identity(frame, identity)

    feature_columns = [
        column for column in frame.columns if column not in {"TransactionID", "isFraud"}
    ]
    features: dict[str, pd.Series] = {
        "row_missing_count": frame[feature_columns].isna().sum(axis=1).astype("int16")
    }

    for column in ("TransactionAmt", "dist1", "dist2"):
        values = frame[column]
        if values.lt(0).fillna(False).any():
            raise ValueError(f"{column} contains negative values; log1p is invalid.")
        features[f"{column}_log1p"] = np.log1p(values).astype("float32")
    features["TransactionAmt_fractional_part"] = (
        frame["TransactionAmt"] - np.floor(frame["TransactionAmt"])
    ).astype("float32")
    features["transaction_day_index"] = (frame["TransactionDT"] // 86_400).astype(
        "int32"
    )
    features["transaction_hour"] = (frame["TransactionDT"].mod(86_400) // 3_600).astype(
        "int8"
    )

    for column in FREQUENCY_COLUMNS:
        features[f"{column}_count_full"] = _frequency_feature(
            frame[column], column, frequency_counts[column]
        )
    features.update(_apply_amount_ratio_features(frame, amount_aggregates))
    return pd.concat([frame, pd.DataFrame(features, index=frame.index)], axis=1)


def _validate_key_column(
    path: Path, table_name: str, include_target: bool = False
) -> pd.Series:
    columns = ["TransactionID"]
    if include_target:
        columns.append("isFraud")
    data = pd.read_csv(
        path,
        usecols=columns,
        dtype={"TransactionID": "Int32"},
    )
    keys = data["TransactionID"]
    if keys.isna().any():
        rows = (np.flatnonzero(keys.isna().to_numpy())[:5] + 2).tolist()
        raise ValueError(f"{table_name} has null TransactionID at CSV rows {rows}.")
    duplicate_mask = keys.duplicated(keep=False)
    if duplicate_mask.any():
        rows = (np.flatnonzero(duplicate_mask.to_numpy())[:5] + 2).tolist()
        raise ValueError(
            f"{table_name} has duplicate TransactionID at CSV rows {rows}."
        )
    if include_target:
        target = pd.to_numeric(data["isFraud"], errors="raise")
        if target.isna().any() or not target.isin([0, 1]).all():
            raise ValueError(f"{table_name} isFraud must contain only 0 and 1.")
    return keys


def _validate_inputs(
    raw_paths: dict[str, Path],
    expected_shapes: Mapping[str, tuple[int, int]] | None,
) -> dict[str, dict[str, Any]]:
    headers = {
        table: list(pd.read_csv(path, nrows=0).columns)
        for table, path in raw_paths.items()
    }
    for table in (
        "train_transaction",
        "train_identity",
        "test_transaction",
        "test_identity",
        "sample_submission",
    ):
        if not raw_paths[table].is_file():
            raise FileNotFoundError(f"Required source file is missing: {table}")

    if "isFraud" not in headers["train_transaction"]:
        raise ValueError("Training transaction table must include isFraud.")
    if "isFraud" in headers["test_transaction"]:
        raise ValueError("Test transaction table must not include isFraud.")
    if [
        column for column in headers["train_transaction"] if column != "isFraud"
    ] != headers["test_transaction"]:
        raise ValueError("Train and test transaction feature schemas do not match.")

    train_identity_columns = headers["train_identity"]
    test_identity_columns = [
        IDENTITY_ALIAS_MAP.get(column, column) for column in headers["test_identity"]
    ]
    if len(test_identity_columns) != len(set(test_identity_columns)):
        raise ValueError("Test identity alias normalization creates duplicate columns.")
    if train_identity_columns != test_identity_columns:
        raise ValueError(
            "Train and test identity schemas do not match after normalization."
        )
    if headers["sample_submission"] != ["TransactionID", "isFraud"]:
        raise ValueError("Sample submission schema must be TransactionID,isFraud.")

    include_target = {
        "train_transaction": True,
        "train_identity": False,
        "test_transaction": False,
        "test_identity": False,
        "sample_submission": False,
    }
    metadata: dict[str, dict[str, Any]] = {}
    keys_by_table: dict[str, pd.Series] = {}
    for table, path in raw_paths.items():
        keys = _validate_key_column(path, table, include_target=include_target[table])
        shape = (len(keys), len(headers[table]))
        if expected_shapes is not None and shape != expected_shapes[table]:
            raise ValueError(
                f"{table} source profile drift: expected {expected_shapes[table]}, "
                f"found {shape}."
            )
        metadata[table] = {"rows": shape[0], "columns": shape[1]}
        if table in {"train_identity", "test_identity"}:
            # Ensure all source identity fields and the transaction key are present.
            _coerce_identity_types(pd.read_csv(path, nrows=1))
        keys_by_table[table] = keys

    if not keys_by_table["test_transaction"].equals(keys_by_table["sample_submission"]):
        raise ValueError(
            "Sample-submission TransactionID values do not match test transactions."
        )
    return metadata


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_parquet_chunks(
    source_path: Path,
    identity: pd.DataFrame,
    destination: Path,
    has_target: bool,
    frequency_counts: dict[str, Counter[str | None]],
    amount_aggregates: dict[str, dict[str, list[float]]],
    chunk_size: int,
) -> tuple[int, Counter[str], int]:
    writer: pq.ParquetWriter | None = None
    rows = 0
    unmatched_identity_rows = 0
    missing_counts: Counter[str] = Counter()
    try:
        for transactions in pd.read_csv(
            source_path,
            chunksize=chunk_size,
            dtype={column: "string" for column in TRANSACTION_CATEGORICAL_COLUMNS},
        ):
            frame = transform_transaction_chunk(
                transactions,
                identity,
                has_target,
                frequency_counts,
                amount_aggregates,
            )
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
            elif not writer.schema.equals(table.schema):
                raise ValueError(
                    f"Parquet schema changed while processing {source_path.name}."
                )
            writer.write_table(table)
            rows += len(frame)
            unmatched_identity_rows += int((~frame["has_identity"]).sum())
            missing_counts.update(frame.isna().sum().to_dict())
    finally:
        if writer is not None:
            writer.close()
    return rows, missing_counts, unmatched_identity_rows


def _write_missingness_report(
    destination: Path,
    table_rows: dict[str, int],
    missing_counts: dict[str, Counter[str]],
) -> None:
    records = []
    for table, counts in missing_counts.items():
        total_rows = table_rows[table]
        for column, missing in sorted(counts.items()):
            records.append(
                {
                    "table": table,
                    "column": column,
                    "missing_count": missing,
                    "missing_rate": missing / total_rows if total_rows else 0.0,
                    "row_count": total_rows,
                }
            )
    pd.DataFrame(records).to_csv(destination, index=False)


def run_preprocessing(
    loader: DataLoader | None = None,
    processed_data_dir: Path | None = None,
    chunk_size: int = CHUNK_SIZE,
    expected_shapes: Mapping[str, tuple[int, int]] | None = EXPECTED_SOURCE_SHAPES,
) -> Path:
    if chunk_size < 1:
        raise ValueError("chunk_size must be greater than zero.")
    loader = loader or DataLoader()
    processed_data_dir = processed_data_dir or settings.processed_data_dir

    configured_files = {
        "train_transaction": settings.train_transaction_file,
        "train_identity": settings.train_identity_file,
        "test_transaction": settings.test_transaction_file,
        "test_identity": settings.test_identity_file,
        "sample_submission": settings.sample_submission_file,
    }
    raw_paths = {
        table: loader.get_raw_data_path(filename)
        for table, filename in configured_files.items()
    }
    source_metadata = _validate_inputs(raw_paths, expected_shapes)

    transaction_paths = {
        split: raw_paths[f"{split}_transaction"] for split in ("train", "test")
    }
    identity_paths = {
        split: raw_paths[f"{split}_identity"] for split in ("train", "test")
    }
    frequency_counts, amount_aggregates = _precompute_aggregates(
        transaction_paths, identity_paths, chunk_size
    )

    processed_data_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    output_dir = processed_data_dir / f"ieee_cis_{run_id}"
    missing_counts: dict[str, Counter[str]] = {}
    output_rows: dict[str, int] = {}
    unmatched_identity_rows: dict[str, int] = {}

    with tempfile.TemporaryDirectory(
        prefix=".ieee_cis_staging_", dir=processed_data_dir
    ) as staging_name:
        staging_dir = Path(staging_name)
        for split in ("train", "test"):
            identity = _load_identity(identity_paths[split])
            destination = staging_dir / f"{split}.parquet"
            rows, counts, unmatched_rows = _write_parquet_chunks(
                transaction_paths[split],
                identity,
                destination,
                has_target=split == "train",
                frequency_counts=frequency_counts,
                amount_aggregates=amount_aggregates,
                chunk_size=chunk_size,
            )
            if rows != source_metadata[f"{split}_transaction"]["rows"]:
                raise ValueError(
                    f"{split} output row count changed: expected "
                    f"{source_metadata[f'{split}_transaction']['rows']}, found {rows}."
                )
            output_rows[split] = rows
            missing_counts[split] = counts
            unmatched_identity_rows[split] = unmatched_rows
            del identity

        sample_submission = pd.read_csv(raw_paths["sample_submission"])
        sample_submission.to_parquet(
            staging_dir / "sample_submission.parquet",
            index=False,
            compression="zstd",
        )
        output_rows["sample_submission"] = len(sample_submission)

        _write_missingness_report(
            staging_dir / "missingness.csv", output_rows, missing_counts
        )

        source_checksums = {
            table: _sha256_file(path) for table, path in raw_paths.items()
        }
        output_files = [
            "train.parquet",
            "test.parquet",
            "sample_submission.parquet",
            "missingness.csv",
        ]
        manifest = {
            "run_id": run_id,
            "generated_at": datetime.now(UTC).isoformat(),
            "dataset": "IEEE-CIS Fraud Detection",
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "source_revision": None,
            "source_revision_note": "Exact Kaggle download revision was not recorded.",
            "source_tables": source_metadata,
            "source_sha256": source_checksums,
            "output_rows": output_rows,
            "output_sha256": {
                name: _sha256_file(staging_dir / name) for name in output_files
            },
            "identity_join": {
                "type": "left",
                "key": "TransactionID",
                "unmatched_train": unmatched_identity_rows["train"],
                "unmatched_test": unmatched_identity_rows["test"],
                "identity_column_aliases": IDENTITY_ALIAS_MAP,
            },
            "features_added": [
                "has_identity",
                "row_missing_count",
                "TransactionAmt_log1p",
                "dist1_log1p",
                "dist2_log1p",
                "TransactionAmt_fractional_part",
                "transaction_day_index",
                "transaction_hour",
                *(f"{column}_count_full" for column in FREQUENCY_COLUMNS),
                *(
                    f"TransactionAmt_to_{stat}_{column}"
                    for column in AGGREGATE_GROUP_COLUMNS
                    for stat in ("mean", "std")
                ),
            ],
            "aggregate_scope": "Unsupervised counts and amount aggregates use train+test covariates; transductive.",
            "missing_value_policy": "Preserve nulls; no row/column dropping or imputation.",
            "target_policy": "isFraud is retained for train only; sample_submission is kept separate.",
            "temporal_policy": "TransactionDT remains a relative offset; only day index and hour-of-offset are derived.",
            "validation_caveat": "Refit transductive aggregates inside temporal validation folds; do not use labels to encode features.",
        }
        (staging_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        staging_dir.rename(output_dir)

    logger.info("Preprocessing complete: %s", output_dir)
    return output_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    output_dir = run_preprocessing()
    print(output_dir)


if __name__ == "__main__":
    main()
