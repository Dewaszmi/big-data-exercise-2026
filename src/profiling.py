import io
from pathlib import Path
from typing import Any, Dict, Optional

import duckdb
import polars as pl


def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(database=":memory:")


def _wide_row_to_metric_value(df: pl.DataFrame) -> pl.DataFrame:
    return df.unpivot(on=df.columns, variable_name="metric", value_name="value")


def _frame_to_text(df: pl.DataFrame) -> str:
    buf = io.StringIO()
    df.write_csv(buf)
    return buf.getvalue().rstrip("\n")


def get_schema_description(file_path: str, con: Optional[duckdb.DuckDBPyConnection] = None) -> pl.DataFrame:
    if con is None:
        con = get_duckdb_connection()
    query = f"DESCRIBE SELECT * FROM '{file_path}'"
    return con.execute(query).pl()


def profile_bts_flight_delays(
    file_path: str, con: Optional[duckdb.DuckDBPyConnection] = None
) -> Dict[str, pl.DataFrame]:
    if con is None:
        con = get_duckdb_connection()

    con.execute(f"CREATE OR REPLACE TEMP VIEW bts_source AS SELECT * FROM '{file_path}'")

    null_sql = """
    SELECT
        COUNT(*) AS total_rows,
        COUNT(*) - COUNT(year) AS null_year,
        COUNT(*) - COUNT(month) AS null_month,
        COUNT(*) - COUNT(carrier) AS null_carrier,
        COUNT(*) - COUNT(carrier_name) AS null_carrier_name,
        COUNT(*) - COUNT(airport) AS null_airport,
        COUNT(*) - COUNT(airport_name) AS null_airport_name,
        COUNT(*) - COUNT(arr_flights) AS null_arr_flights,
        COUNT(*) - COUNT(arr_del15) AS null_arr_del15,
        COUNT(*) - COUNT(carrier_ct) AS null_carrier_ct,
        COUNT(*) - COUNT(arr_cancelled) AS null_arr_cancelled,
        COUNT(*) - COUNT(arr_diverted) AS null_arr_diverted,
        COUNT(*) - COUNT(arr_delay) AS null_arr_delay
    FROM bts_source
    """
    df_nulls = _wide_row_to_metric_value(con.execute(null_sql).pl())

    stats_sql = """
    SELECT
        MIN(year) AS min_year, MAX(year) AS max_year,
        MIN(month) AS min_month, MAX(month) AS max_month,
        MIN(arr_flights) AS min_arr_flights, MAX(arr_flights) AS max_arr_flights,
        ROUND(AVG(arr_flights), 2) AS avg_arr_flights,
        MIN(arr_del15) AS min_arr_del15, MAX(arr_del15) AS max_arr_del15,
        ROUND(AVG(arr_del15), 2) AS avg_arr_del15,
        MIN(arr_cancelled) AS min_arr_cancelled, MAX(arr_cancelled) AS max_arr_cancelled,
        MIN(arr_delay) AS min_arr_delay, MAX(arr_delay) AS max_arr_delay,
        ROUND(AVG(arr_delay), 2) AS avg_arr_delay
    FROM bts_source
    """
    df_stats = con.execute(stats_sql).pl()

    anomaly_sql = """
    SELECT
        COUNT(*) FILTER (WHERE arr_flights <= 0) AS zero_or_negative_flights,
        COUNT(*) FILTER (WHERE arr_del15 > arr_flights) AS delayed_exceeds_total_flights,
        COUNT(*) FILTER (WHERE arr_cancelled > arr_flights) AS cancelled_exceeds_total_flights,
        COUNT(*) FILTER (WHERE arr_delay < 0) AS negative_delay_minutes,
        COUNT(*) FILTER (WHERE ABS(arr_delay - (carrier_delay + weather_delay + nas_delay + security_delay + late_aircraft_delay)) > 1) AS delay_components_sum_mismatch,
        COUNT(*) FILTER (WHERE carrier_ct > 0 AND carrier_ct != ROUND(carrier_ct)) AS fractional_carrier_counts,
        COUNT(*) FILTER (WHERE arr_cancelled = arr_flights AND arr_flights > 0) AS all_flights_cancelled_records
    FROM bts_source
    """
    df_anomalies = _wide_row_to_metric_value(con.execute(anomaly_sql).pl())
    df_anomalies = df_anomalies.rename({"metric": "anomaly_type", "value": "detected_count"})

    return {
        "null_profile": df_nulls,
        "descriptive_stats": df_stats,
        "anomalies": df_anomalies,
    }


def save_profiling_results_to_txt(
    file_path: str,
    output_txt_path: Optional[str] = None,
    con: Optional[duckdb.DuckDBPyConnection] = None,
) -> str:
    if con is None:
        con = get_duckdb_connection()
    if output_txt_path is None:
        from src.config import PROFILING_TXT_PATH

        output_txt_path = str(PROFILING_TXT_PATH)

    profiles = profile_bts_flight_delays(file_path, con=con)

    top_airports_query = """
    SELECT 
        airport,
        airport_name,
        SUM(arr_flights) AS total_flights,
        SUM(arr_del15) AS total_delayed_flights,
        ROUND(SUM(arr_del15) * 100.0 / NULLIF(SUM(arr_flights), 0), 2) AS delay_percentage,
        ROUND(SUM(arr_delay) / 60.0 / 1000.0, 1) AS total_delay_thousands_of_hours
    FROM bts_source
    WHERE arr_flights > 0
    GROUP BY airport, airport_name
    HAVING SUM(arr_flights) > 1000000
    ORDER BY delay_percentage DESC
    LIMIT 10
    """
    df_top_airports = con.execute(top_airports_query).pl()

    causes_query = """
    SELECT
        ROUND(SUM(carrier_delay) / 60.0 / 1000.0, 1) AS carrier_hours_k,
        ROUND(SUM(weather_delay) / 60.0 / 1000.0, 1) AS weather_hours_k,
        ROUND(SUM(nas_delay) / 60.0 / 1000.0, 1) AS nas_hours_k,
        ROUND(SUM(security_delay) / 60.0 / 1000.0, 1) AS security_hours_k,
        ROUND(SUM(late_aircraft_delay) / 60.0 / 1000.0, 1) AS late_aircraft_hours_k
    FROM bts_source
    """
    df_causes = con.execute(causes_query).pl()

    stats_vertical = profiles["descriptive_stats"].transpose(
        include_header=True,
        header_name="metric",
        column_names=["value"],
    )

    lines = [
        _frame_to_text(profiles["null_profile"]),
        "",
        _frame_to_text(stats_vertical),
        "",
        _frame_to_text(profiles["anomalies"]),
        "",
        _frame_to_text(df_top_airports),
        "",
        _frame_to_text(df_causes),
    ]

    content = "\n".join(lines) + "\n"
    Path(output_txt_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(content)

    return output_txt_path


def main() -> None:
    from src.config import BTS_CSV_PATH, PROFILING_TXT_PATH

    save_profiling_results_to_txt(str(BTS_CSV_PATH), str(PROFILING_TXT_PATH))


if __name__ == "__main__":
    main()
