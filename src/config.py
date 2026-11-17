from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PARTITIONED_DATA_DIR = DATA_DIR / "partitioned"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
PRESENTATION_DIR = BASE_DIR / "presentation"

BTS_CSV_PATH = RAW_DATA_DIR / "Airline_Delay_Cause.csv"
BTS_EXCEL_DEFS = RAW_DATA_DIR / "Download_Column_Definitions.xlsx"
PROFILING_TXT_PATH = DATA_DIR / "profiling_results.txt"
BENCHMARK_DIR = DATA_DIR / "benchmark"
CLEANING_BENCHMARK_TXT_PATH = DATA_DIR / "cleaning_benchmark_results.txt"
PRESENTATION_RESULTS_DIR = PRESENTATION_DIR / "results"
PRESENTATION_RESULTS_TXT_PATH = PRESENTATION_RESULTS_DIR / "cleaning_benchmark_results.txt"

for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    PARTITIONED_DATA_DIR,
    BENCHMARK_DIR,
    NOTEBOOKS_DIR,
    PRESENTATION_DIR,
    PRESENTATION_RESULTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
