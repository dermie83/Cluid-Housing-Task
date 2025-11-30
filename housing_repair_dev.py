import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")

def read_excel_file(path: Path, sheet=None):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    # choose engine based on extension
    ext = path.suffix.lower()
    if ext == ".xlsx":
        engine = "openpyxl"
    elif ext == ".xls":
        engine = "xlrd"
    else:
        engine = None
    return pd.read_excel(path, sheet_name=sheet, engine=engine)

def add_time_until_repair(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 'time_until_repair' = repair_year - construction_year if both columns exist.
    Non-numeric values are coerced to NaN.
    """
    if not isinstance(df, pd.DataFrame):
        return df
    if {"construction_year", "repair_year"}.issubset(df.columns):
        df = df.copy()
        cons = pd.to_numeric(df["construction_year"], errors="coerce")
        repair = pd.to_numeric(df["repair_year"], errors="coerce")
        df["time_until_repair"] = repair - cons
    return df

def summarize_df(df: pd.DataFrame, head_rows: int = 5):
    print("\nPreview:")
    print(df.head(head_rows).to_string(index=False))
    print(f"\nShape: {df.shape}")
    print("\nColumns and dtypes:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isna().sum())
    print("\nNumeric summary:")
    print(df.describe(include=[np.number]).T)


def main():
    parser = argparse.ArgumentParser(description="Read and inspect an Excel file")
    parser.add_argument("-f", "--file", type=str, default="synthetic_property_data.xlsx",
                        help="Path to the Excel file (default: synthetic_property_data.xlsx)")
    parser.add_argument("-s", "--sheet", type=str, default=None, help="Sheet name or index to read")
    parser.add_argument("-r", "--rows", type=int, default=5, help="Number of rows to preview")
    parser.add_argument("--to-csv", type=str, default=None, help="Optional: save read sheet to CSV")
    parser.add_argument("--plot", action="store_true", help="Optional: show a histogram for numeric columns")
    args = parser.parse_args()

    path = Path(args.file)
    try:
        df = read_excel_file(path, sheet=args.sheet)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    if isinstance(df, dict):
        # multiple sheets returned
        for name, sheet_df in df.items():
            print(f"\n=== Sheet: {name} ===")
            sheet_df = add_time_until_repair(sheet_df)
            summarize_df(sheet_df, head_rows=args.rows)
            if args.to_csv:
                out = Path(args.to_csv)
                stem = out.stem
                out_sheet = out.with_name(f"{stem}_{name}.csv")
                sheet_df.to_csv(out_sheet, index=False)
                print(f"Saved sheet to {out_sheet}")
            if args.plot:
                sheet_df.select_dtypes(include=[np.number]).hist(figsize=(8,6))
                if "time_until_repair" in sheet_df.columns:
                    sheet_df["time_until_repair"].dropna().hist(figsize=(6,4))
                    plt.suptitle(f"Histograms: {name} (includes time_until_repair)")
                else:
                    plt.suptitle(f"Histograms: {name}")
                plt.show()
    else:
        df = add_time_until_repair(df)
        summarize_df(df, head_rows=args.rows)
        if args.to_csv:
            Path(args.to_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.to_csv, index=False)
            print(f"Saved to {args.to_csv}")
        if args.plot:
            df.select_dtypes(include=[np.number]).hist(figsize=(10,6))
            if "time_until_repair" in df.columns:
                df["time_until_repair"].dropna().hist(figsize=(6,4))
                plt.suptitle("Numeric column histograms (includes time_until_repair)")
            else:
                plt.suptitle("Numeric column histograms")
            plt.show()


if __name__ == "__main__":
    main()
# ...existing code...
