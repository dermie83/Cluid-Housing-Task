import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# ---------- 1. Load Excel ----------
file = "synthetic_property_data.xlsx"
path = Path(file)

def read_excel_file(path: Path, sheet=None):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    ext = path.suffix.lower()
    engine = None
    if ext == ".xlsx":
        engine = "openpyxl"
    elif ext == ".xls":
        engine = "xlrd"

    df = pd.read_excel(path, sheet_name=sheet, engine=engine)

    if isinstance(df, dict):
        first_sheet_name = list(df.keys())[0]
        print(f"Multiple sheets found. Using the first sheet: {first_sheet_name}")
        df = df[first_sheet_name]

    return df

# ---------- 2. Feature engineering ----------
def add_time_until_repair(df: pd.DataFrame) -> pd.DataFrame:
    if {"construction_year", "repair_year"}.issubset(df.columns):
        df = df.copy()
        cons = pd.to_numeric(df["construction_year"], errors="coerce")
        repair = pd.to_numeric(df["repair_year"], errors="coerce")
        df["time_until_repair"] = repair - cons
    return df

df_ = read_excel_file(path)
df = add_time_until_repair(df_)
df["property_age"] = 2025 - df["construction_year"]
# Avoid division by zero by adding 1 to property_age
df["repairs_per_year"] = df["repair_count"] / (df["property_age"] + 1)
print("Columns in the dataframe:", df.columns.tolist())

# ---------- 3. Balance / correct target ----------
# Clip extreme high repair costs to the 99th percentile
upper_bound = df["total_repair_cost"].quantile(0.99)
df["total_repair_cost"] = df["total_repair_cost"].clip(upper=upper_bound)

# Optional: check distribution
print("Total repair cost after clipping (max):", df["total_repair_cost"].max())

# ---------- 4. Correlation check ----------
corr = df[["repair_count", "total_repair_cost", "time_until_repair", "occupants", "property_age", "repairs_per_year"]].corr()
print("Correlation between repair_count and total_repair_cost:", corr)

# ---------- 5. Feature selection ----------
numeric_features = ["occupants", "time_until_repair", "property_age", "repair_count", "repairs_per_year"]

# Drop rows with missing numeric features or target
df = df.dropna(subset=numeric_features + ["total_repair_cost"])

X = df[numeric_features].values
y = df["total_repair_cost"].values

# ---------- 6. Log-transform target ----------
log_transform = True
if log_transform:
    y = np.log1p(y)  # log(1 + y) reduces skew

# ---------- 7. Train/test split ----------
# Optional stratified split by cost bins for better balance
y_bins = pd.qcut(df["total_repair_cost"], q=5, labels=False)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_bins
)

# ---------- 8. Train Decision Tree ----------
tree = DecisionTreeRegressor(max_depth=5, random_state=42)
tree.fit(X_train, y_train)

# ---------- 9. Predict ----------
y_pred = tree.predict(X_test)

# ---------- 10. Convert back from log if needed ----------
if log_transform:
    y_pred_original = np.expm1(y_pred)
    y_test_original = np.expm1(y_test)
else:
    y_pred_original = y_pred
    y_test_original = y_test

# ---------- 11. Evaluate ----------
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
print(f"Decision Tree RMSE on original scale: {rmse:.2f}")

# ---------- 12. Feature importance ----------
importances = pd.Series(tree.feature_importances_, index=numeric_features)
print("\nTop features driving total_repair_cost:")
print(importances.sort_values(ascending=False))
