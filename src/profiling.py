"""Data profiling utilities using DuckDB in-process SQL OLAP engine."""
from typing import Dict, Any, Optional
from pathlib import Path
import duckdb
import pandas as pd


def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """Create and return an in-process DuckDB connection."""
    return duckdb.connect(database=":memory:")


def get_schema_description(file_path: str, con: Optional[duckdb.DuckDBPyConnection] = None) -> pd.DataFrame:
    """
    Inspect the schema of a file using DuckDB without loading full data into memory.
    
    Args:
        file_path: Path to Parquet or CSV file.
        con: Optional DuckDB connection.
        
    Returns:
        DataFrame containing column names, types, nullability, etc.
    """
    if con is None:
        con = get_duckdb_connection()
    query = f"DESCRIBE SELECT * FROM '{file_path}'"
    return con.execute(query).df()


def profile_bts_flight_delays(file_path: str, con: Optional[duckdb.DuckDBPyConnection] = None) -> Dict[str, pd.DataFrame]:
    """
    Execute comprehensive analytical profiling queries for the BTS Airline Delay dataset.
    
    Args:
        file_path: Path to the Airline Delay CSV/Parquet file.
        con: Optional DuckDB connection.
        
    Returns:
        Dictionary of DataFrames with profiling results.
    """
    if con is None:
        con = get_duckdb_connection()
        
    # 1. Volume and Null Profile
    null_sql = f"""
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
    FROM '{file_path}'
    """
    df_nulls = con.execute(null_sql).df().T.reset_index()
    df_nulls.columns = ["metric", "value"]
    
    # 2. Extreme Values and Descriptive Statistics
    stats_sql = f"""
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
    FROM '{file_path}'
    """
    df_stats = con.execute(stats_sql).df()

    # 3. Anomaly and Inconsistency Audit
    anomaly_sql = f"""
    SELECT
        COUNT(*) FILTER (WHERE arr_flights <= 0) AS zero_or_negative_flights,
        COUNT(*) FILTER (WHERE arr_del15 > arr_flights) AS delayed_exceeds_total_flights,
        COUNT(*) FILTER (WHERE arr_cancelled > arr_flights) AS cancelled_exceeds_total_flights,
        COUNT(*) FILTER (WHERE arr_delay < 0) AS negative_delay_minutes,
        COUNT(*) FILTER (WHERE ABS(arr_delay - (carrier_delay + weather_delay + nas_delay + security_delay + late_aircraft_delay)) > 1) AS delay_components_sum_mismatch,
        COUNT(*) FILTER (WHERE carrier_ct > 0 AND carrier_ct != ROUND(carrier_ct)) AS fractional_carrier_counts,
        COUNT(*) FILTER (WHERE arr_cancelled = arr_flights AND arr_flights > 0) AS all_flights_cancelled_records
    FROM '{file_path}'
    """
    df_anomalies = con.execute(anomaly_sql).df().T.reset_index()
    df_anomalies.columns = ["anomaly_type", "detected_count"]

    return {
        "null_profile": df_nulls,
        "descriptive_stats": df_stats,
        "anomalies": df_anomalies,
    }
