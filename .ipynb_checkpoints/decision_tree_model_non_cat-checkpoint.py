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
print("Columns in the dataframe:", df.columns.tolist())

# ---------- 3. Feature selection ----------
numeric_features = ["construction_year", "repair_year", "occupants", "time_until_repair", "total_repair_cost"]

# Drop rows with missing numeric features
df = df.dropna(subset=numeric_features + ["total_repair_cost"])

X = df[numeric_features].values
y = df["repair_count"].values

# ---------- 4. Optional log transformation ----------
log_transform = True
if log_transform:
    y = np.log1p(y)  # log(1 + y) to handle zeros

# ---------- 5. Train/test split ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- 6. Train Decision Tree ----------
tree = DecisionTreeRegressor(max_depth=5, random_state=42)
tree.fit(X_train, y_train)

# ---------- 7. Predict ----------
y_pred = tree.predict(X_test)

# ---------- 8. Convert back from log if needed ----------
if log_transform:
    y_pred_original = np.expm1(y_pred)
    y_test_original = np.expm1(y_test)
else:
    y_pred_original = y_pred
    y_test_original = y_test

# ---------- 9. Evaluate ----------
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
print(f"Decision Tree RMSE on original scale: {rmse:.2f}")

# ---------- 10. Feature importance ----------
importances = pd.Series(tree.feature_importances_, index=numeric_features)
print("\nTop features driving repair count:")
print(importances.sort_values(ascending=False))
