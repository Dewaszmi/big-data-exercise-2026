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
│   ├── processed/                 # Cleaned datasets
│   └── partitioned/               # Partitioned Parquet store (Milestone 2)
├── notebooks/
│   └── 01_data_profiling.ipynb    # Milestone 1: Executed DuckDB profiling notebook
├── presentation/
│   ├── milestone_1_presentation.md   # Presentation deck in Markdown
│   └── milestone_1_presentation.html # Modern HTML slides with "Save as PDF" button
├── src/
│   ├── __init__.py
│   ├── config.py                  # Project paths and configuration
│   ├── download_data.py           # Ingestion script (BTS Flight Delays from Kaggle)
│   ├── profiling.py               # Reusable DuckDB SQL analytical profiling functions
│   └── create_notebook.py         # Notebook generator script
├── requirements.txt               # Frozen dependencies
└── README.md                      # Documentation
```

---

## Getting Started

### 1. Environment Setup
The virtual environment is already initialized. To activate it manually:
```bash
source .venv/bin/activate
```

To install dependencies on another machine:
```bash
pip install -r requirements.txt
```

### 2. Running Data Ingestion
To verify or redownload raw files:
```bash
python src/download_data.py
```

### 3. Launching the Notebook
Launch Jupyter or open VS Code / Jupyter Lab:
```bash
jupyter notebook notebooks/01_data_profiling.ipynb
```
Select the **Python (Big Data .venv)** kernel. All cells have been pre-executed and contain hard numbers and visualizations.

### 4. Viewing & Exporting the Presentation (PDF)
Open `presentation/milestone_1_presentation.html` in your web browser:
```bash
open presentation/milestone_1_presentation.html
```
Click **"🖨️ Save as PDF / Print"** at the top or press `Cmd + P` to export the slides as a PDF for the instructor.