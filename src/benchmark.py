import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import duckdb
import polars as pl
import pyarrow.parquet as pq
from polars.testing import assert_frame_equal

PARTITION_COLUMN = "carrier"
ATHENA_USD_PER_TB = 5.0

LAYOUT_FLAT = "A: flat file (full scan)"
LAYOUT_PARTITIONED = "B: partitioned (pruning)"


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    columns: Sequence[str]
    carriers: Optional[Sequence[str]]
    sql: str
    polars: Callable[[pl.LazyFrame], pl.LazyFrame]


def _carrier_filter(carriers: Sequence[str]) -> pl.Expr:
    return pl.col(PARTITION_COLUMN).is_in(list(carriers))


def _sql_in(carriers: Sequence[str]) -> str:
    return ", ".join(f"'{c}'" for c in carriers)


def monthly_kpis(carrier: str) -> Scenario:
    return Scenario(
        name=f"monthly_kpis[{carrier}]",
        description=f"Avg delay & delay rate by month, carrier = '{carrier}'",
        columns=["month", "avg_delay_minutes_per_delayed_flight", "delay_rate"],
        carriers=[carrier],
        sql=f"""
            SELECT month,
                   AVG(avg_delay_minutes_per_delayed_flight) AS avg_delay_min,
                   AVG(delay_rate) AS avg_delay_rate
            FROM {{source}}
            WHERE carrier = '{carrier}'
            GROUP BY month ORDER BY month
        """,
        polars=lambda lf: lf.filter(pl.col(PARTITION_COLUMN) == carrier)
        .group_by("month")
        .agg(
            avg_delay_min=pl.col("avg_delay_minutes_per_delayed_flight").mean(),
            avg_delay_rate=pl.col("delay_rate").mean(),
        )
        .sort("month"),
    )


def yearly_volume(carrier: str) -> Scenario:
    return Scenario(
        name=f"yearly_volume[{carrier}]",
        description=f"Flights, delayed flights and delay hours by year, carrier = '{carrier}'",
        columns=["year", "arr_flights", "arr_del15", "arr_delay"],
        carriers=[carrier],
        sql=f"""
            SELECT year,
                   SUM(arr_flights) AS flights,
                   SUM(arr_del15) / SUM(arr_flights) AS delay_rate,
                   SUM(arr_delay) / 60 AS delay_hours
            FROM {{source}}
            WHERE carrier = '{carrier}'
            GROUP BY year ORDER BY year
        """,
        polars=lambda lf: lf.filter(pl.col(PARTITION_COLUMN) == carrier)
        .group_by("year")
        .agg(
            flights=pl.col("arr_flights").sum(),
            delay_rate=pl.col("arr_del15").sum() / pl.col("arr_flights").sum(),
            delay_hours=pl.col("arr_delay").sum() / 60,
        )
        .sort("year"),
    )


def top_airports(carrier: str) -> Scenario:
    return Scenario(
        name=f"top_airports[{carrier}]",
        description=f"Top 10 airports by delay minutes, carrier = '{carrier}'",
        columns=["airport", "airport_name", "arr_delay", "arr_flights"],
        carriers=[carrier],
        sql=f"""
            SELECT airport, MAX(airport_name) AS airport_name,
                   SUM(arr_delay) AS delay_min, SUM(arr_flights) AS flights
            FROM {{source}}
            WHERE carrier = '{carrier}'
            GROUP BY airport ORDER BY delay_min DESC LIMIT 10
        """,
        polars=lambda lf: lf.filter(pl.col(PARTITION_COLUMN) == carrier)
        .group_by("airport")
        .agg(
            airport_name=pl.col("airport_name").max(),
            delay_min=pl.col("arr_delay").sum(),
            flights=pl.col("arr_flights").sum(),
        )
        .sort("delay_min", descending=True)
        .head(10),
    )


def multi_carrier_kpis(carriers: Sequence[str]) -> Scenario:
    label = "+".join(carriers)
    return Scenario(
        name=f"monthly_kpis[{label}]",
        description=f"Avg delay & delay rate by month for {len(carriers)} carriers",
        columns=["month", "avg_delay_minutes_per_delayed_flight", "delay_rate"],
        carriers=list(carriers),
        sql=f"""
            SELECT carrier, month,
                   AVG(avg_delay_minutes_per_delayed_flight) AS avg_delay_min,
                   AVG(delay_rate) AS avg_delay_rate
            FROM {{source}}
            WHERE carrier IN ({_sql_in(carriers)})
            GROUP BY carrier, month ORDER BY carrier, month
        """,
        polars=lambda lf: lf.filter(_carrier_filter(carriers))
        .group_by(PARTITION_COLUMN, "month")
        .agg(
            avg_delay_min=pl.col("avg_delay_minutes_per_delayed_flight").mean(),
            avg_delay_rate=pl.col("delay_rate").mean(),
        )
        .sort(PARTITION_COLUMN, "month"),
    )


def all_carriers_kpis() -> Scenario:
    return Scenario(
        name="monthly_kpis[ALL]",
        description="Control: avg delay & delay rate by month, no carrier filter",
        columns=["month", "avg_delay_minutes_per_delayed_flight", "delay_rate"],
        carriers=None,
        sql="""
            SELECT month,
                   AVG(avg_delay_minutes_per_delayed_flight) AS avg_delay_min,
                   AVG(delay_rate) AS avg_delay_rate
            FROM {source}
            GROUP BY month ORDER BY month
        """,
        polars=lambda lf: lf.group_by("month")
        .agg(
            avg_delay_min=pl.col("avg_delay_minutes_per_delayed_flight").mean(),
            avg_delay_rate=pl.col("delay_rate").mean(),
        )
        .sort("month"),
    )


def parquet_files(path: Path) -> List[Path]:
    return [path] if path.is_file() else sorted(path.rglob("*.parquet"))


def size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in parquet_files(path)) / 1024**2


def evict_page_cache(path: Path) -> None:
    os.sync()
    for f in parquet_files(path):
        fd = os.open(f, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def _files_for(path: Path, carriers: Optional[Sequence[str]]) -> List[Path]:
    files = parquet_files(path)
    if path.is_file() or carriers is None:
        return files
    wanted = {f"{PARTITION_COLUMN}={c}" for c in carriers}
    return [f for f in files if f.parent.name in wanted]


def scanned_bytes(path: Path, scenario: Scenario) -> Dict[str, float]:
    """Compressed bytes of the column chunks an engine must read (row groups pruned via min/max stats)."""
    needed = set(scenario.columns) | ({PARTITION_COLUMN} if path.is_file() and scenario.carriers else set())
    total_files = len(parquet_files(path))
    files = _files_for(path, scenario.carriers)
    data_bytes = 0
    row_groups_read = row_groups_total = 0
    for f in files:
        meta = pq.ParquetFile(f).metadata
        for i in range(meta.num_row_groups):
            rg = meta.row_group(i)
            row_groups_total += 1
            cols = {rg.column(j).path_in_schema: rg.column(j) for j in range(rg.num_columns)}
            if scenario.carriers and PARTITION_COLUMN in cols:
                stats = cols[PARTITION_COLUMN].statistics
                if stats is not None and stats.has_min_max and not any(
                    stats.min <= c <= stats.max for c in scenario.carriers
                ):
                    continue
            row_groups_read += 1
            data_bytes += sum(cols[c].total_compressed_size for c in needed if c in cols)
    return {
        "files_read": len(files),
        "files_total": total_files,
        "row_groups_read": row_groups_read,
        "row_groups_total": row_groups_total,
        "scanned_mb": data_bytes / 1024**2,
    }


def _duckdb_source(path: Path) -> str:
    if path.is_file():
        return f"read_parquet('{path}')"
    return f"read_parquet('{path}/**/*.parquet', hive_partitioning = true)"


def _polars_source(path: Path) -> pl.LazyFrame:
    if path.is_file():
        return pl.scan_parquet(path)
    return pl.scan_parquet(path, hive_partitioning=True)


def make_runner(engine: str, path: Path, scenario: Scenario) -> Callable[[], pl.DataFrame]:
    if engine == "duckdb":
        sql = scenario.sql.format(source=_duckdb_source(path))
        con = duckdb.connect()
        # without this DuckDB keeps file bytes in its own RAM cache and cold runs would never hit the OS
        con.execute("SET enable_external_file_cache = false")
        con.execute("SET parquet_metadata_cache = false")
        return lambda: con.execute(sql).pl()
    if engine == "polars":
        return lambda: scenario.polars(_polars_source(path)).collect()
    raise ValueError(f"Unknown engine: {engine}")


def run_ab_benchmark(
    layouts: Dict[str, Path],
    scenarios: Sequence[Scenario],
    engines: Sequence[str] = ("duckdb", "polars"),
    cache_modes: Sequence[str] = ("warm", "cold"),
    repeats: int = 15,
    warmup: int = 2,
    seed: int = 42,
    extra: Optional[Dict[str, object]] = None,
) -> pl.DataFrame:
    """Runs every (scenario, engine, cache mode) on all layouts, interleaving layouts in random order per repeat."""
    rng = random.Random(seed)
    rows = []
    for scenario in scenarios:
        for engine in engines:
            runners = {name: make_runner(engine, path, scenario) for name, path in layouts.items()}
            reference = None
            for name, runner in runners.items():
                result = runner()
                for _ in range(warmup - 1):
                    runner()
                if reference is None:
                    reference = result
                else:
                    assert_frame_equal(result, reference, check_exact=False, check_dtypes=False)
            for mode in cache_modes:
                for rep in range(repeats):
                    order = list(layouts)
                    rng.shuffle(order)
                    for name in order:
                        if mode == "cold":
                            evict_page_cache(layouts[name])
                        start = time.perf_counter()
                        runners[name]()
                        elapsed_ms = (time.perf_counter() - start) * 1000
                        rows.append(
                            {
                                **(extra or {}),
                                "scenario": scenario.name,
                                "engine": engine,
                                "cache": mode,
                                "layout": name,
                                "repeat": rep,
                                "ms": elapsed_ms,
                            }
                        )
    return pl.DataFrame(rows)


def summarize(runs: pl.DataFrame, by: Sequence[str] = ("scenario", "engine", "cache")) -> pl.DataFrame:
    """Per-layout stats plus the A vs B comparison (gain % = (A - B) / A on mean latency)."""
    by = list(by)
    stats = runs.group_by(*by, "layout").agg(
        n=pl.len(),
        mean_ms=pl.col("ms").mean(),
        std_ms=pl.col("ms").std(),
        median_ms=pl.col("ms").median(),
        min_ms=pl.col("ms").min(),
        p95_ms=pl.col("ms").quantile(0.95),
    )
    flat = stats.filter(pl.col("layout") == LAYOUT_FLAT).drop("layout").rename(
        lambda c: c if c in by else f"A_{c}"
    )
    part = stats.filter(pl.col("layout") == LAYOUT_PARTITIONED).drop("layout").rename(
        lambda c: c if c in by else f"B_{c}"
    )
    return (
        flat.join(part, on=by)
        .with_columns(
            gain_pct=(pl.col("A_mean_ms") - pl.col("B_mean_ms")) / pl.col("A_mean_ms") * 100,
            speedup=pl.col("A_mean_ms") / pl.col("B_mean_ms"),
            gain_pct_median=(pl.col("A_median_ms") - pl.col("B_median_ms")) / pl.col("A_median_ms") * 100,
        )
        .select(
            *by,
            "A_n",
            "A_mean_ms",
            "A_std_ms",
            "A_median_ms",
            "B_mean_ms",
            "B_std_ms",
            "B_median_ms",
            "gain_pct",
            "gain_pct_median",
            "speedup",
        )
        .sort(by)
    )


def io_report(layouts: Dict[str, Path], scenarios: Sequence[Scenario]) -> pl.DataFrame:
    rows = []
    for scenario in scenarios:
        for name, path in layouts.items():
            rows.append({"scenario": scenario.name, "layout": name, **scanned_bytes(path, scenario)})
    df = pl.DataFrame(rows)
    flat = df.filter(pl.col("layout") == LAYOUT_FLAT).select("scenario", A_scanned_mb="scanned_mb")
    return df.join(flat, on="scenario").with_columns(
        io_reduction_pct=(1 - pl.col("scanned_mb") / pl.col("A_scanned_mb")) * 100,
        usd_per_1k_queries=pl.col("scanned_mb") / 1024**2 * ATHENA_USD_PER_TB * 1000,
    ).drop("A_scanned_mb")


def build_scaled_copies(
    source_parquet: Path, out_dir: Path, scale: int, row_group_size: int = 122_880
) -> Dict[str, Path]:
    """Replicates the clean dataset `scale` times with DuckDB (streaming, out-of-core) into both layouts.

    Copies are appended as whole blocks so every row group looks like a slice of the real data;
    interleaving duplicate rows would let RLE/dictionary encoding compress them away.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    flat = out_dir / f"flat_x{scale}.parquet"
    part = out_dir / f"partitioned_x{scale}"
    if part.exists():
        shutil.rmtree(part)
    flat.unlink(missing_ok=True)
    sources = ", ".join(f"'{source_parquet}'" for _ in range(scale))
    select = f"SELECT * FROM read_parquet([{sources}])"
    with duckdb.connect() as con:
        con.execute(
            f"COPY ({select}) TO '{flat}' (FORMAT parquet, COMPRESSION snappy, ROW_GROUP_SIZE {row_group_size})"
        )
        con.execute(
            f"COPY ({select}) TO '{part}' "
            f"(FORMAT parquet, COMPRESSION snappy, ROW_GROUP_SIZE {row_group_size}, PARTITION_BY ({PARTITION_COLUMN}))"
        )
    return {LAYOUT_FLAT: flat, LAYOUT_PARTITIONED: part}


def run_scale_experiment(
    source_parquet: Path,
    out_dir: Path,
    scales: Sequence[int],
    scenarios: Sequence[Scenario],
    repeats: int = 10,
    **kwargs,
) -> Dict[str, pl.DataFrame]:
    runs, io, storage = [], [], []
    for scale in scales:
        layouts = build_scaled_copies(source_parquet, out_dir, scale)
        rows = pq.ParquetFile(layouts[LAYOUT_FLAT]).metadata.num_rows
        storage.extend(
            {"scale": scale, "rows": rows, "layout": name, "size_mb": size_mb(path), "files": len(parquet_files(path))}
            for name, path in layouts.items()
        )
        io.append(io_report(layouts, scenarios).with_columns(scale=pl.lit(scale)))
        runs.append(run_ab_benchmark(layouts, scenarios, repeats=repeats, extra={"scale": scale}, **kwargs))
    return {"runs": pl.concat(runs), "io": pl.concat(io), "storage": pl.DataFrame(storage)}
