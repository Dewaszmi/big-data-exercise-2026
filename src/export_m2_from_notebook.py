"""Rebuild Milestone 2 result artifacts from the executed notebook outputs.

Use when ``data/benchmark/`` was wiped (gitignored) but
``notebooks/02_cleaning_partitioning.ipynb`` still holds the last run's tables.
Re-running the notebook is preferred; this only recovers presentation inputs.
"""

from __future__ import annotations

import json
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import List

import polars as pl

from src.config import (
    BENCHMARK_DIR,
    CLEANING_BENCHMARK_TXT_PATH,
    NOTEBOOKS_DIR,
    PRESENTATION_RESULTS_TXT_PATH,
)
from src.results_export import save_milestone2_results, write_csv


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: List[List[List[str]]] = []
        self._table: List[List[str]] | None = None
        self._row: List[str] | None = None
        self._cell: List[str] | None = None
        self._in_cell = False

    def handle_starttag(self, tag, attrs):  # noqa: ANN001
        if tag == "table":
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
            self._in_cell = True

    def handle_endtag(self, tag):  # noqa: ANN001
        if tag in ("td", "th") and self._in_cell and self._row is not None and self._cell is not None:
            self._row.append(unescape("".join(self._cell)).strip())
            self._in_cell = False
            self._cell = None
        elif tag == "tr" and self._row is not None and self._table is not None:
            if self._row:
                self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            self.tables.append(self._table)
            self._table = None

    def handle_data(self, data):  # noqa: ANN001
        if self._in_cell and self._cell is not None:
            self._cell.append(data)


def _cell_html_tables(cell: dict) -> List[List[List[str]]]:
    tables: List[List[List[str]]] = []
    for output in cell.get("outputs") or []:
        html = (output.get("data") or {}).get("text/html")
        if not html:
            continue
        html_str = "".join(html) if isinstance(html, list) else html
        parser = _TableParser()
        parser.feed(html_str)
        tables.extend(parser.tables)
    return tables


def _polars_html_table_to_df(rows: List[List[str]]) -> pl.DataFrame:
    """Convert a Polars HTML preview (header + dtype row + data) to a DataFrame of strings."""
    if len(rows) < 2:
        raise ValueError("Expected at least a header and a dtype row")
    header = [c.strip('"') for c in rows[0]]
    # drop dtype row
    data_rows = rows[2:]
    cleaned = []
    for row in data_rows:
        values = []
        for cell in row:
            cell = cell.strip()
            if cell == "null":
                values.append(None)
            elif len(cell) >= 2 and cell[0] == '"' and cell[-1] == '"':
                values.append(cell[1:-1])
            elif cell.startswith('"') and cell.endswith("…"):
                values.append(cell[1:])
            else:
                values.append(cell)
        # pad / trim if polars truncated with an ellipsis column
        if len(values) < len(header):
            values.extend([None] * (len(header) - len(values)))
        cleaned.append(values[: len(header)])
    return pl.DataFrame(cleaned, schema=header, orient="row")


def export_from_notebook(notebook_path: Path | None = None) -> dict[str, Path]:
    notebook_path = notebook_path or (NOTEBOOKS_DIR / "02_cleaning_partitioning.ipynb")
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = nb["cells"]

    def table(cell_idx: int, which: int = 0) -> pl.DataFrame:
        tables = _cell_html_tables(cells[cell_idx])
        if which >= len(tables):
            raise IndexError(f"Cell {cell_idx} has {len(tables)} HTML tables; asked for {which}")
        return _polars_html_table_to_df(tables[which])

    artifacts = {
        "anomaly_report": table(4, 0),
        "negative_examples": table(6, 0),
        "telemetry_avg_delay_above_24h_examples": table(8, 0),
        "cleaning_funnel": table(8, 1),
        "feature_verification": table(13, 0),
        "feature_worked_example_dl_atl": table(13, 1),
        "partition_key_candidates": table(16, 0),
        "partitions": table(19, 0),
        "io_report": table(23, 0),
        "summary_real": table(25, 0),
        "task_query_comparison_monthly_kpis_DL": table(27, 0),
        "storage_by_scale": table(33, 0),
        "summary_scale_task_query": table(34, 0),
        "finops_scale": table(37, 0),
        "csv_vs_parquet": table(39, 0),
        "layout_storage_real": table(41, 0),
        "io_sorted_vs_partitioned": table(41, 1),
        "replication_storage": table(42, 0),
        "replication_io": table(43, 0),
    }

    # Drop the ellipsis column Polars injects when the HTML preview is too wide.
    if "…" in artifacts["summary_real"].columns:
        artifacts["summary_real"] = artifacts["summary_real"].drop("…")

    # Polars HTML truncates long cell values with an ellipsis; restore known rule names.
    artifacts["anomaly_report"] = artifacts["anomaly_report"].with_columns(
        pl.col("rule")
        .str.replace(r"delay_minutes_without_delayed_…", "delay_minutes_without_delayed_flights")
        .str.replace(r"avg_delay_below_del15_threshol…", "avg_delay_below_del15_threshold")
    )

    csv_paths = {}
    for name, df in artifacts.items():
        csv_paths[name] = write_csv(df, BENCHMARK_DIR / f"{name}.csv")

    written = save_milestone2_results(
        artifacts,
        txt_path=CLEANING_BENCHMARK_TXT_PATH,
        also_copy_to=PRESENTATION_RESULTS_TXT_PATH,
    )
    written.update(csv_paths)
    return written


def main() -> None:
    written = export_from_notebook()
    for label, path in written.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
