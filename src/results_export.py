"""Export Milestone 2 tables as raw artifacts for the presentation (same idea as profiling_results.txt)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Mapping, Optional, Union

import polars as pl

FrameLike = Union[pl.DataFrame, Path, str]


def _frame_to_text(df: pl.DataFrame) -> str:
    buf = io.StringIO()
    df.write_csv(buf)
    return buf.getvalue().rstrip("\n")


def _as_frame(value: FrameLike) -> pl.DataFrame:
    if isinstance(value, pl.DataFrame):
        return value
    path = Path(value)
    if path.suffix.lower() == ".csv":
        return pl.read_csv(path)
    raise TypeError(f"Unsupported result artifact: {value!r}")


def save_tables_to_txt(
    tables: Mapping[str, FrameLike],
    output_txt_path: str | Path,
    *,
    title: str = "Milestone 2 raw results (no narrative)",
) -> Path:
    """Write named Polars frames (or CSV paths) to a single plain-text dump.

    Sections are separated by a header line. Content is CSV only — the presentation
    is meant to be generated from this file plus the Jupyter notebook, analogous to
    ``data/profiling_results.txt`` in Milestone 1.
    """
    output_txt_path = Path(output_txt_path)
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)

    blocks = [title, ""]
    for name, value in tables.items():
        df = _as_frame(value)
        blocks.append(f"=== {name} ===")
        blocks.append(_frame_to_text(df))
        blocks.append("")

    output_txt_path.write_text("\n".join(blocks).rstrip() + "\n", encoding="utf-8")
    return output_txt_path


def write_csv(df: pl.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)
    return path


def save_milestone2_results(
    tables: Mapping[str, FrameLike],
    *,
    txt_path: Optional[str | Path] = None,
    also_copy_to: Optional[str | Path] = None,
) -> Dict[str, Path]:
    """Persist the consolidated Milestone 2 results txt (and an optional presentation copy)."""
    from src.config import CLEANING_BENCHMARK_TXT_PATH, PRESENTATION_RESULTS_TXT_PATH

    txt_path = Path(txt_path or CLEANING_BENCHMARK_TXT_PATH)
    primary = save_tables_to_txt(tables, txt_path)
    written = {"txt": primary}

    copy_path = Path(also_copy_to) if also_copy_to is not None else PRESENTATION_RESULTS_TXT_PATH
    if copy_path is not None:
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        copy_path.write_text(primary.read_text(encoding="utf-8"), encoding="utf-8")
        written["presentation_copy"] = copy_path
    return written
