import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")


# ---------------------------------------------------------
# Utility: Safe Excel reader
# ---------------------------------------------------------
def read_excel_file(path: Path, sheet=None):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext == ".xlsx":
        engine = "openpyxl"
    elif ext == ".xls":
        engine = "xlrd"
    else:
        engine = None

    return pd.read_excel(path, sheet_name=sheet, engine=engine)


# ---------------------------------------------------------
# Add derived fields
# ---------------------------------------------------------
def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add time_until_repair, property_age, repairs_per_year safely."""
    df = df.copy()

    # Time until repair
    if {"construction_year", "repair_year"}.issubset(df.columns):
        cons = pd.to_numeric(df["construction_year"], errors="coerce")
        rep = pd.to_numeric(df["repair_year"], errors="coerce")
        df["time_until_repair"] = rep - cons

    # Property age
    if "construction_year" in df.columns:
        df["property_age"] = 2025 - pd.to_numeric(df["construction_year"], errors="coerce")

    # Repairs per year
    if "repair_count" in df.columns:
        df["repairs_per_year"] = df["repair_count"] / (df.get("property_age", 0) + 1)

    return df


# ---------------------------------------------------------
# Summary printing
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Plotting engine
# ---------------------------------------------------------
def generate_plots(df: pd.DataFrame, title_prefix=""):
    """Centralized function to generate ALL plots exactly once."""

    plt.close("all")  # ensure nothing duplicates

    # Histograms
    df.select_dtypes(include=[np.number]).hist(figsize=(10, 6))
    plt.suptitle(f"{title_prefix} Numeric Column Histograms")
    plt.show()

    # -------- Scatter 1: total_repair_cost vs repair_year --------
    if {"repair_year", "total_repair_cost"}.issubset(df.columns):
        plt.figure(figsize=(8, 5))
        plt.scatter(df["repair_year"], df["total_repair_cost"], alpha=0.5)
        plt.title(f"{title_prefix} Total Repair Cost vs Repair Year")
        plt.xlabel("Repair Year")
        plt.ylabel("Total Repair Cost")
        plt.tight_layout()
        plt.show()

     # -------- Scatter 2: total_repair_cost vs construction_year --------
    if {"construction_year", "total_repair_cost"}.issubset(df.columns):
        plt.figure(figsize=(8, 5))
        plt.scatter(df["construction_year"], df["total_repair_cost"], alpha=0.5)
        plt.title(f"{title_prefix} Total Repair Cost vs Construction Year")
        plt.xlabel("Construction Year")
        plt.ylabel("Total Repair Cost")
        plt.tight_layout()
        plt.show()

    # -------- Scatter 3: repair_count vs repair_year --------
    if {"repair_year", "repair_count"}.issubset(df.columns):
        plt.figure(figsize=(8, 5))
        plt.scatter(df["repair_year"], df["repair_count"], alpha=0.5)
        plt.title(f"{title_prefix} Repair Count vs Repair Year")
        plt.xlabel("Repair Year")
        plt.ylabel("Repair Count")
        plt.tight_layout()
        plt.show()
    
     # -------- Scatter 4: Repair Count vs Construction Year --------
    if {"construction_year", "repair_count"}.issubset(df.columns):
        plt.figure(figsize=(8, 5))
        plt.scatter(df["construction_year"], df["repair_count"], alpha=0.5)
        plt.title(f"{title_prefix} Repair Count vs Construction Year")
        plt.xlabel("Construction Year")
        plt.ylabel("Repair Count")
        plt.tight_layout()
        plt.show()

    # -------- Scatter 5: Repair Count vs Property Age --------
    if {"property_age", "repair_count"}.issubset(df.columns):
        plt.figure(figsize=(8, 5))
        plt.scatter(df["property_age"], df["repair_count"], alpha=0.5)
        plt.title(f"{title_prefix} Repair Count vs Property Age")
        plt.xlabel("Property Age")
        plt.ylabel("Repair Count")
        plt.tight_layout()
        plt.show()

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Read, inspect, and plot an Excel dataset")
    parser.add_argument("-f", "--file", type=str, default="synthetic_property_data.xlsx",
                        help="Path to the Excel file")
    parser.add_argument("-s", "--sheet", type=str, default=None, help="Sheet name/index")
    parser.add_argument("-r", "--rows", type=int, default=5, help="Number of preview rows")
    parser.add_argument("--to-csv", type=str, default=None, help="Save processed sheet(s) to CSV")
    parser.add_argument("--plot", action="store_true", help="Show plots")

    args = parser.parse_args()
    path = Path(args.file)

    try:
        df = read_excel_file(path, sheet=args.sheet)
    except Exception as e:
        print(f"Error reading Excel: {e}")
        return

    # -----------------------------------------------------
    # MULTI-SHEET MODE
    # -----------------------------------------------------
    if isinstance(df, dict):
        for name, sheet_df in df.items():
            print(f"\n=== SHEET: {name} ===")

            sheet_df = prepare_dataframe(sheet_df)
            summarize_df(sheet_df, args.rows)

            # Save per-sheet CSV
            if args.to_csv:
                out_path = Path(args.to_csv)
                out_sheet = out_path.with_name(f"{out_path.stem}_{name}.csv")
                sheet_df.to_csv(out_sheet, index=False)
                print(f"Saved sheet to {out_sheet}")

            if args.plot:
                generate_plots(sheet_df, title_prefix=f"[{name}] ")

    # -----------------------------------------------------
    # SINGLE-SHEET MODE
    # -----------------------------------------------------
    else:
        df = prepare_dataframe(df)
        summarize_df(df, args.rows)

        if args.to_csv:
            Path(args.to_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.to_csv, index=False)
            print(f"Saved to {args.to_csv}")

        if args.plot:
            generate_plots(df, title_prefix="")


# ---------------------------------------------------------
if __name__ == "__main__":
    main()

