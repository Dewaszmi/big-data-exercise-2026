import shutil
from pathlib import Path
import kagglehub
from src.config import RAW_DATA_DIR, BTS_CSV_PATH

def download_bts_dataset() -> Path:
    print("Fetching BTS Airline Delay dataset from Kaggle...")
    path = kagglehub.dataset_download("daryaheyko/airline-on-time-statistics-and-delay-causes-bts")
    print(f"Downloaded to cache: {path}")
    
    source_dir = Path(path)
    target_csv = RAW_DATA_DIR / "Airline_Delay_Cause.csv"
    source_csv = source_dir / "Airline_Delay_Cause.csv"
    
    if source_csv.exists() and not target_csv.exists():
        shutil.copy(source_csv, target_csv)
        print(f"Copied dataset CSV to: {target_csv}")
        
    source_xlsx = source_dir / "Download_Column_Definitions.xlsx"
    target_xlsx = RAW_DATA_DIR / "Download_Column_Definitions.xlsx"
    if source_xlsx.exists() and not target_xlsx.exists():
        shutil.copy(source_xlsx, target_xlsx)
        print(f"Copied column definitions to: {target_xlsx}")
        
    return target_csv

def main() -> None:
    download_bts_dataset()
    print("Dataset download completed successfully.")


if __name__ == "__main__":
    main()
