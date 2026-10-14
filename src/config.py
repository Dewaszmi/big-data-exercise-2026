"""Project configuration and path definitions."""
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PARTITIONED_DATA_DIR = DATA_DIR / "partitioned"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
PRESENTATION_DIR = BASE_DIR / "presentation"

# Raw dataset files
BTS_CSV_PATH = RAW_DATA_DIR / "Airline_Delay_Cause.csv"
BTS_EXCEL_DEFS = RAW_DATA_DIR / "Download_Column_Definitions.xlsx"

# Ensure required directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, PARTITIONED_DATA_DIR, NOTEBOOKS_DIR, PRESENTATION_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
