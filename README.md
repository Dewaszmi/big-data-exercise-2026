# End-to-End Data Engineering Lifecycle (Big Data 2026)

## Project Overview
This project implements a complete **Data Engineering Lifecycle** pipeline on real-world civil aviation telemetry data from the **US Bureau of Transportation Statistics (BTS)**, tracking flight delays across 425 airports and all major US carriers over 23 years (2003–2025).

The project spans three incremental milestones:
1. **Milestone 1:** Schema Discovery, DuckDB In-Process OLAP SQL Profiling, Anomaly Audit & Business Presentation.
2. **Milestone 2:** Polars Data Transformation, Apache Parquet Snappy Physical Layout, Hive Partitioning & Partition Pruning Benchmark.
3. **Milestone 3:** Feature Store Engineering & Predictive Machine Learning Model (Scikit-Learn).

---

## Repository Structure

```
big-data-exercise-2026/
├── .venv/                         # Virtual environment (dependencies installed)
├── data/
│   ├── raw/                       # Downloaded raw datasets (gitignored)
│   │   ├── Airline_Delay_Cause.csv
│   │   └── Download_Column_Definitions.xlsx
│   ├── processed/                 # Cleaned single-file Parquet (benchmark variant A)
│   ├── partitioned/               # Hive-partitioned Parquet store (benchmark variant B)
│   ├── benchmark/                 # Raw Milestone 2 tables (CSV) + scaled Parquet copies
│   ├── profiling_results.txt      # Milestone 1: narrative-free profiling dump (presentation input)
│   └── cleaning_benchmark_results.txt # Milestone 2: narrative-free cleaning/partition/A-B dump
├── notebooks/
│   ├── 01_data_profiling.ipynb    # Milestone 1: Executed DuckDB profiling notebook
│   └── 02_cleaning_partitioning.ipynb # Milestone 2: Polars cleaning, features, Parquet partitioning, A/B benchmark
├── presentation/
│   ├── bts_flight_delays_profilowanie_i_audyt_danych.html             # Milestone 1 slides
│   ├── bts_flight_delays_czyszczenie_partycjonowanie_benchmark.html   # Milestone 2 slides
│   ├── figures/                   # Charts exported by notebook 02 (used by the Milestone 2 slides)
│   └── results/                   # Committed copy of cleaning_benchmark_results.txt (attachable artifact)
├── src/
│   ├── __init__.py
│   ├── config.py                  # Project paths and configuration
│   ├── download_data.py           # Ingestion script (BTS Flight Delays from Kaggle)
│   ├── profiling.py               # Reusable DuckDB SQL analytical profiling functions
│   ├── cleaning.py                # Milestone 2: Polars cleaning rules, feature engineering, partitioned Parquet writer
│   ├── benchmark.py               # Milestone 2: A/B benchmark (scenarios, warm/cold cache, I/O accounting, scale-up)
│   ├── results_export.py          # Shared CSV/TXT dump helpers for presentation inputs
│   └── export_m2_from_notebook.py # Rebuild M2 result TXT/CSV from executed notebook outputs if data/ was wiped
├── pyproject.toml                 # Project metadata and dependencies (uv)
├── uv.lock                        # Locked dependency versions
└── README.md                      # Documentation
```

---

## Getting Started

### 1. Environment Setup
Install [uv](https://docs.astral.sh/uv/) if needed, then sync dependencies and activate the virtual environment:
```bash
uv sync
source .venv/bin/activate
```

On another machine, clone the repo and run `uv sync` (uses `pyproject.toml` and `uv.lock`).

### 2. Running Data Ingestion
To verify or redownload raw files:
```bash
uv run download-bts-data
# or: uv run python -m src.download_data
```

### 3. Cleaning & Partitioning (Milestone 2)
Writes `data/processed/bts_clean.parquet` and the Hive-partitioned `data/partitioned/bts_by_carrier/carrier=XX/` dataset (Snappy):
```bash
uv run clean-bts-data
# or: uv run python -m src.cleaning
```

### 4. Launching the Notebooks
Launch Jupyter or open VS Code / Jupyter Lab:
```bash
jupyter notebook notebooks/01_data_profiling.ipynb
jupyter notebook notebooks/02_cleaning_partitioning.ipynb
```
Select the **Python (Big Data .venv)** kernel. All cells have been pre-executed and contain hard numbers and visualizations.

Notebook 02 runs the full A/B benchmark (~4 minutes): 1,120 timed runs on the real data and a 1x-50x scale-up experiment that writes ~2.6 GB of temporary Parquet to `data/benchmark/`. It also re-exports the charts in `presentation/figures/` and dumps presentation inputs to `data/cleaning_benchmark_results.txt` (plus a copy under `presentation/results/`), same pattern as Milestone 1's `profiling_results.txt`. To re-execute it headlessly:
```bash
cd notebooks && uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace 02_cleaning_partitioning.ipynb
```

If `data/` was wiped but the notebook still contains executed outputs, rebuild the result artifacts with:
```bash
uv run export-m2-results-from-notebook
```

### 5. Viewing & Exporting the Presentations (PDF)
Open the slides in your web browser:
```bash
xdg-open presentation/bts_flight_delays_profilowanie_i_audyt_danych.html             # Milestone 1
xdg-open presentation/bts_flight_delays_czyszczenie_partycjonowanie_benchmark.html   # Milestone 2
```
Click **"🖨️ Save as PDF / Print"** at the top or press `Cmd + P` to export the slides as a PDF for the instructor.