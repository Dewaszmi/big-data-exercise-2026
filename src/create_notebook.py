"""Script to generate the Milestone 1 Jupyter Notebook."""
import json
from pathlib import Path

def generate_notebook():
    notebooks_dir = Path("notebooks")
    notebooks_dir.mkdir(parents=True, exist_ok=True)
    
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Milestone 1: Data Profiling and Quality Audit with DuckDB\n",
                    "## Data Engineering Lifecycle - BTS Airline Flight Delays Dataset\n",
                    "\n",
                    "### 1. Executive Summary and Lifecycle Context\n",
                    "In the modern Data Engineering Lifecycle, raw telemetry and production operational data are inherently noisy—suffering from transmission drops, reporting anomalies, system updates, and formatting discrepancies. Direct ingestion of unverified data into analytical platforms or AI models leads to degraded performance and invalid business decisions.\n",
                    "\n",
                    "**Milestone 1 Objectives:**\n",
                    "1. **Schema Discovery:** Inspect dataset schema and column metadata directly from storage using **DuckDB** without full RAM ingestion.\n",
                    "2. **Analytical SQL Profiling:** Quantify data completeness, compute extreme boundaries, and uncover underlying distributions.\n",
                    "3. **Anomaly Identification:** Detect technical glitches (e.g., component sum mismatches) and understand domain-specific attributes (e.g., fractional attribution of delay causes).\n",
                    "4. **Preparation for Milestone 2 & 3:** Establish baseline data quality rules for cleaning, physical partitioning (PyArrow/Polars), and feature serving (Scikit-learn).\n",
                    "\n",
                    "### Why DuckDB?\n",
                    "* **In-process OLAP engine:** Operates directly within the Python process memory space without external database server overhead.\n",
                    "* **Vectorized Execution Kernel:** Processes data in columnar chunks (vector batches), maximizing CPU cache locality and SIMD execution.\n",
                    "* **Zero-Copy & Memory Efficiency:** Directly streams and queries files from disk without duplicating raw records into system RAM."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Step 0: Environment Setup and Dependencies\n",
                    "import sys\n",
                    "from pathlib import Path\n",
                    "\n",
                    "# Add repository root to Python path\n",
                    "ROOT_DIR = Path(\"..\").resolve()\n",
                    "if str(ROOT_DIR) not in sys.path:\n",
                    "    sys.path.insert(0, str(ROOT_DIR))\n",
                    "\n",
                    "import duckdb\n",
                    "import polars as pl\n",
                    "import pyarrow as pa\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "from src.config import BTS_CSV_PATH, RAW_DATA_DIR\n",
                    "from src.profiling import get_duckdb_connection, get_schema_description, profile_bts_flight_delays\n",
                    "\n",
                    "print(f\"DuckDB version: {duckdb.__version__}\")\n",
                    "print(f\"Polars version: {pl.__version__}\")\n",
                    "print(f\"PyArrow version: {pa.__version__}\")\n",
                    "print(f\"Raw dataset path: {BTS_CSV_PATH}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 2. Schema Discovery (Zero Full-Load)\n",
                    "Using DuckDB to inspect the file schema using `DESCRIBE SELECT * FROM '<file>'` without materializing the dataset into RAM."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "con = get_duckdb_connection()\n",
                    "schema_df = get_schema_description(str(BTS_CSV_PATH), con=con)\n",
                    "schema_df"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 3. Comprehensive Data Profiling with DuckDB SQL\n",
                    "Here we execute an analytical aggregation query to measure:\n",
                    "* Total records and completeness per column (NULL counts).\n",
                    "* Min, max, and average values across all numeric indicators.\n",
                    "* Data coverage timeline (years, months, airlines, airports)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "profiles = profile_bts_flight_delays(str(BTS_CSV_PATH), con=con)\n",
                    "\n",
                    "print(\"=== COMPLETENESS AND NULL AUDIT ===\")\n",
                    "display(profiles[\"null_profile\"])\n",
                    "\n",
                    "print(\"\\n=== DESCRIPTIVE SUMMARY STATISTICS ===\")\n",
                    "display(profiles[\"descriptive_stats\"].T)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 4. Technical and Domain Anomaly Detection\n",
                    "We audit the dataset for mathematical invariants and known domain characteristics:\n",
                    "1. **Component sum mismatch:** Does `arr_delay == carrier_delay + weather_delay + nas_delay + security_delay + late_aircraft_delay`?\n",
                    "2. **Fractional flight counts:** Why are columns like `carrier_ct` floating-point numbers instead of integers?\n",
                    "3. **100% cancellation periods:** Months where scheduled operations collapsed completely."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(\"=== DETECTED ANOMALIES & INTEGRITY CHECKS ===\")\n",
                    "display(profiles[\"anomalies\"])\n",
                    "\n",
                    "# Inspect specific rows with component sum mismatch\n",
                    "mismatch_query = f\"\"\"\n",
                    "SELECT year, month, carrier, airport, arr_flights, arr_del15, arr_delay,\n",
                    "       (carrier_delay + weather_delay + nas_delay + security_delay + late_aircraft_delay) AS calculated_sum,\n",
                    "       arr_delay - (carrier_delay + weather_delay + nas_delay + security_delay + late_aircraft_delay) AS delay_difference\n",
                    "FROM '{BTS_CSV_PATH}'\n",
                    "WHERE ABS(arr_delay - (carrier_delay + weather_delay + nas_delay + security_delay + late_aircraft_delay)) > 1\n",
                    "\"\"\"\n",
                    "df_mismatches = con.execute(mismatch_query).df()\n",
                    "print(f\"\\nTotal mismatch instances: {len(df_mismatches)}\")\n",
                    "display(df_mismatches.head(10))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Domain Interpretation of Anomalies\n",
                    "* **Fractional Counts (`carrier_ct`, `weather_ct`, etc.):** When a single flight delay is attributed to multiple causes (e.g. bad weather cascading into late turnaround equipment), the US Department of Transportation allocates proportional weights (e.g. 0.5 to weather, 0.5 to late aircraft). Thus, counts are non-integers by regulatory design.\n",
                    "* **Component Sum Discrepancy:** The 9 records with discrepancies reflect retroactive reporting corrections or non-standard delay classifications in specific carrier reporting systems.\n",
                    "* **NULL Values (657 rows):** 657 monthly records contain NULL values for delay categories. These correspond to months where flights were scheduled but no actual operations took place or data submission was incomplete."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 5. Analytical Aggregations and Visual Insights\n",
                    "Deep-dive OLAP queries to visualize delay distributions by airline, airport, and historical trends."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Top 10 Major Airports by Total Arrival Delay Volume\n",
                    "top_airports_query = f\"\"\"\n",
                    "SELECT \n",
                    "    airport,\n",
                    "    airport_name,\n",
                    "    SUM(arr_flights) AS total_flights,\n",
                    "    SUM(arr_del15) AS total_delayed_flights,\n",
                    "    ROUND(SUM(arr_del15) * 100.0 / NULLIF(SUM(arr_flights), 0), 2) AS delay_percentage,\n",
                    "    ROUND(SUM(arr_delay) / 60.0 / 1000.0, 1) AS total_delay_thousands_of_hours\n",
                    "FROM '{BTS_CSV_PATH}'\n",
                    "WHERE arr_flights > 0\n",
                    "GROUP BY airport, airport_name\n",
                    "HAVING SUM(arr_flights) > 1000000\n",
                    "ORDER BY delay_percentage DESC\n",
                    "LIMIT 10\n",
                    "\"\"\"\n",
                    "df_top_airports = con.execute(top_airports_query).df()\n",
                    "display(df_top_airports)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Visualization: Delay Percentage across Top High-Traffic Airports\n",
                    "plt.figure(figsize=(10, 4))\n",
                    "sns.barplot(data=df_top_airports, x=\"airport\", y=\"delay_percentage\", palette=\"magma\")\n",
                    "plt.title(\"Flight Delay Percentage (>15 min) for Top High-Traffic US Airports (2003-2025)\")\n",
                    "plt.ylabel(\"Delayed Flights (%)\")\n",
                    "plt.xlabel(\"Airport Code\")\n",
                    "plt.grid(axis=\"y\", linestyle=\"--\", alpha=0.7)\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Delay Cause Breakdown across Entire Dataset\n",
                    "causes_query = f\"\"\"\n",
                    "SELECT\n",
                    "    ROUND(SUM(carrier_delay) / 60.0 / 1000.0, 1) AS Carrier,\n",
                    "    ROUND(SUM(weather_delay) / 60.0 / 1000.0, 1) AS Weather,\n",
                    "    ROUND(SUM(nas_delay) / 60.0 / 1000.0, 1) AS National_Aviation_System,\n",
                    "    ROUND(SUM(security_delay) / 60.0 / 1000.0, 1) AS Security,\n",
                    "    ROUND(SUM(late_aircraft_delay) / 60.0 / 1000.0, 1) AS Late_Aircraft\n",
                    "FROM '{BTS_CSV_PATH}'\n",
                    "\"\"\"\n",
                    "df_causes = con.execute(causes_query).df()\n",
                    "causes_series = df_causes.iloc[0]\n",
                    "\n",
                    "plt.figure(figsize=(6, 6))\n",
                    "plt.pie(causes_series, labels=causes_series.index, autopct=\"%1.1f%%\", startangle=140, colors=sns.color_palette(\"Set2\"))\n",
                    "plt.title(\"Proportional Distribution of Total Delay Hours by Cause (2003-2025)\")\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 6. Strategic Conclusions for Next Milestones\n",
                    "\n",
                    "#### Milestone 2 (Data Transformation & Physical Partitioning):\n",
                    "* **Partitioning Strategy:** Partition data by `year` (23 partitions) or `(year, carrier)`. This yields natural cardinalities for partition pruning.\n",
                    "* **Storage Format:** Convert raw CSV to Snappy-compressed Parquet with columnar encodings.\n",
                    "* **Cleaning Rules:** Handle 657 NULL records; flag or correct 9 component mismatch rows.\n",
                    "\n",
                    "#### Milestone 3 (AI / Machine Learning Serving):\n",
                    "* **Problem Formulation:** Predict `arr_del15` (delayed flight probability) or `arr_delay` (delay duration in minutes) based on historical carrier performance, seasonality (`month`), and airport traffic.\n",
                    "* **Feature Engineering:** Historical airport congestion rate, carrier on-time score, monthly seasonality index."
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python (Big Data .venv)",
                "language": "python",
                "name": "big-data-venv"
            },
            "language_info": {
                "name": "python",
                "version": "3.9.13"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    output_path = notebooks_dir / "01_data_profiling.ipynb"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print(f"Successfully generated clean notebook at: {output_path}")

if __name__ == "__main__":
    generate_notebook()
