import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl
import pyarrow.parquet as pq

COUNT_COLUMNS = ["arr_flights", "arr_del15", "arr_cancelled", "arr_diverted"]
CAUSE_CT_COLUMNS = ["carrier_ct", "weather_ct", "nas_ct", "security_ct", "late_aircraft_ct"]
CAUSE_DELAY_COLUMNS = ["carrier_delay", "weather_delay", "nas_delay", "security_delay", "late_aircraft_delay"]
NUMERIC_COLUMNS = COUNT_COLUMNS + CAUSE_CT_COLUMNS + ["arr_delay"] + CAUSE_DELAY_COLUMNS

DEL15_THRESHOLD_MINUTES = 15
MAX_AVG_DELAY_MINUTES = 24 * 60
SUM_TOLERANCE = 1

PARTITION_COLUMN = "carrier"
SORTED_ROW_GROUP_SIZE = 16_384

_cause_ct_sum = pl.sum_horizontal(CAUSE_CT_COLUMNS)
_cause_delay_sum = pl.sum_horizontal(CAUSE_DELAY_COLUMNS)
_operated_flights = pl.col("arr_flights") - pl.col("arr_cancelled") - pl.col("arr_diverted")
_avg_delay_per_delayed_flight = pl.col("arr_delay") / pl.col("arr_del15")

BUSINESS_RULES: Dict[str, pl.Expr] = {
    "missing_measurements": pl.any_horizontal(pl.col(NUMERIC_COLUMNS).is_null()),
    "negative_values": pl.any_horizontal(pl.col(NUMERIC_COLUMNS) < 0),
    "delayed_exceeds_flights": pl.col("arr_del15") > pl.col("arr_flights"),
    "outcomes_exceed_flights": (pl.col("arr_del15") + pl.col("arr_cancelled") + pl.col("arr_diverted"))
    > pl.col("arr_flights"),
    "delay_minutes_sum_mismatch": (pl.col("arr_delay") - _cause_delay_sum).abs() > SUM_TOLERANCE,
    "cause_counts_sum_mismatch": (pl.col("arr_del15") - _cause_ct_sum).abs() > SUM_TOLERANCE,
}

TELEMETRY_RULES: Dict[str, pl.Expr] = {
    "no_operated_flights": _operated_flights <= 0,
    "delay_minutes_without_delayed_flights": (pl.col("arr_del15") == 0) & (pl.col("arr_delay") > 0),
    "avg_delay_below_del15_threshold": (pl.col("arr_del15") > 0)
    & (_avg_delay_per_delayed_flight < DEL15_THRESHOLD_MINUTES),
    "avg_delay_above_24h": (pl.col("arr_del15") > 0) & (_avg_delay_per_delayed_flight > MAX_AVG_DELAY_MINUTES),
}


def load_raw(csv_path: str | Path) -> pl.LazyFrame:
    return pl.scan_csv(csv_path)


def anomaly_report(lf: pl.LazyFrame) -> pl.DataFrame:
    rules = {**BUSINESS_RULES, **TELEMETRY_RULES}
    counts = (
        lf.select([expr.fill_null(False).sum().alias(name) for name, expr in rules.items()])
        .collect()
        .unpivot(variable_name="rule", value_name="flagged_rows")
    )
    categories = pl.DataFrame(
        {
            "rule": list(rules),
            "category": ["business"] * len(BUSINESS_RULES) + ["telemetry"] * len(TELEMETRY_RULES),
        }
    )
    return categories.join(counts, on="rule")


def _drop_rules(lf: pl.LazyFrame, rules: Dict[str, pl.Expr]) -> pl.LazyFrame:
    any_violation = pl.any_horizontal([expr.fill_null(False) for expr in rules.values()])
    return lf.filter(~any_violation)


def clean_business_anomalies(lf: pl.LazyFrame) -> pl.LazyFrame:
    return _drop_rules(lf, BUSINESS_RULES)


def clean_telemetry_anomalies(lf: pl.LazyFrame) -> pl.LazyFrame:
    return _drop_rules(lf, TELEMETRY_RULES)


def _season(month: pl.Expr) -> pl.Expr:
    return (
        pl.when(month.is_in([12, 1, 2]))
        .then(pl.lit("winter"))
        .when(month.is_in([3, 4, 5]))
        .then(pl.lit("spring"))
        .when(month.is_in([6, 7, 8]))
        .then(pl.lit("summer"))
        .otherwise(pl.lit("autumn"))
    )


def add_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    month = pl.col("month")
    lf = lf.with_columns(
        period_start=pl.date(pl.col("year"), month, 1),
        quarter=((month - 1) // 3 + 1).cast(pl.Int8),
        season=_season(month),
        is_peak_travel_month=month.is_in([6, 7, 8, 12]),
        is_covid_period=pl.date(pl.col("year"), month, 1).is_between(
            pl.date(2020, 3, 1), pl.date(2021, 12, 1)
        ),
        operated_flights=_operated_flights,
        arr_delay_hours=pl.col("arr_delay") / 60,
    )

    lf = lf.with_columns(
        delay_rate=pl.col("arr_del15") / pl.col("arr_flights"),
        cancellation_rate=pl.col("arr_cancelled") / pl.col("arr_flights"),
        diversion_rate=pl.col("arr_diverted") / pl.col("arr_flights"),
        avg_delay_minutes_per_delayed_flight=pl.when(pl.col("arr_del15") > 0)
        .then(_avg_delay_per_delayed_flight)
        .otherwise(0.0),
        avg_delay_minutes_per_operated_flight=pl.col("arr_delay") / pl.col("operated_flights"),
        **{
            f"{c}_share": pl.when(pl.col("arr_delay") > 0).then(pl.col(c) / pl.col("arr_delay")).otherwise(0.0)
            for c in CAUSE_DELAY_COLUMNS
        },
    )

    return lf.with_columns(
        dominant_delay_cause=pl.when(pl.col("arr_delay") > 0)
        .then(
            pl.concat_list(CAUSE_DELAY_COLUMNS)
            .list.arg_max()
            .replace_strict(dict(enumerate(c.removesuffix("_delay") for c in CAUSE_DELAY_COLUMNS)))
        )
        .otherwise(pl.lit("none"))
    )


def build_clean_dataset(csv_path: str | Path) -> Tuple[pl.DataFrame, pl.DataFrame]:
    raw = load_raw(csv_path)
    after_business = clean_business_anomalies(raw)
    after_telemetry = clean_telemetry_anomalies(after_business)
    clean = add_features(after_telemetry).collect()

    funnel = pl.DataFrame(
        {
            "stage": ["raw", "after_business_rules", "after_telemetry_rules"],
            "rows": [
                raw.select(pl.len()).collect().item(),
                after_business.select(pl.len()).collect().item(),
                clean.height,
            ],
        }
    ).with_columns(dropped=pl.col("rows").shift(1) - pl.col("rows"))
    return clean, funnel


def write_partitioned_parquet(
    df: pl.DataFrame,
    root_path: str | Path,
    partition_cols: Optional[List[str]] = None,
    compression: str = "snappy",
) -> Path:
    root_path = Path(root_path)
    if root_path.exists():
        shutil.rmtree(root_path)
    partition_cols = partition_cols or [PARTITION_COLUMN]
    # pyarrow emits one row group per input batch per partition, so unsorted input fragments every partition file
    pq.write_to_dataset(
        df.sort(*partition_cols, "year", "month").to_arrow(),
        root_path=str(root_path),
        partition_cols=partition_cols,
        compression=compression,
    )
    return root_path


def write_sorted_parquet(
    df: pl.DataFrame,
    path: str | Path,
    row_group_size: int = SORTED_ROW_GROUP_SIZE,
    compression: str = "snappy",
) -> Path:
    path = Path(path)
    df.sort(PARTITION_COLUMN, "year", "month").write_parquet(
        path, compression=compression, row_group_size=row_group_size
    )
    return path


def describe_partitions(root_path: str | Path) -> pl.DataFrame:
    root_path = Path(root_path)
    rows = []
    for part_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
        files = sorted(part_dir.glob("*.parquet"))
        rows.append(
            {
                "partition": part_dir.name,
                "files": len(files),
                "rows": sum(pq.ParquetFile(f).metadata.num_rows for f in files),
                "size_kb": round(sum(f.stat().st_size for f in files) / 1024, 1),
                "compression": pq.ParquetFile(files[0]).metadata.row_group(0).column(0).compression,
            }
        )
    return pl.DataFrame(rows)


def main() -> None:
    from src.config import BTS_CSV_PATH, PARTITIONED_DATA_DIR, PROCESSED_DATA_DIR

    clean, funnel = build_clean_dataset(BTS_CSV_PATH)
    print(funnel)
    clean.write_parquet(PROCESSED_DATA_DIR / "bts_clean.parquet", compression="snappy")
    root = write_partitioned_parquet(clean, PARTITIONED_DATA_DIR / "bts_by_carrier")
    print(describe_partitions(root))


if __name__ == "__main__":
    main()
