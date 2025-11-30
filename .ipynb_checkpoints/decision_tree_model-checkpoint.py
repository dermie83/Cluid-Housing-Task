import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from scipy import sparse

file = "synthetic_property_data.xlsx"
path = Path(file)
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

    # Read Excel file
    df = pd.read_excel(path, sheet_name=sheet, engine=engine)

    # If df is a dict (multiple sheets), pick the first sheet automatically
    if isinstance(df, dict):
        first_sheet_name = list(df.keys())[0]
        print(f"Multiple sheets found. Using the first sheet: {first_sheet_name}")
        df = df[first_sheet_name]

    return df



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


df_ = read_excel_file(path)
df = add_time_until_repair(df_)
print("Columns in the dataframe:", df.columns.tolist())

# ---------- 3. Feature selection ----------
# Avoid high-cardinality categorical features like property_id
categorical_features = ['region_name']  # small-cardinality categories
numeric_features = ["construction_year", "repair_year", "occupants", "repair_count", "time_until_repair"]

X_num = df[numeric_features].values
y = df['total_repair_cost'].values

# ---------- 4. Optional log transformation ----------
log_transform = True
if log_transform:
    y = np.log1p(y)  # log(1 + y) to handle zeros

# ---------- 5. Encode categorical features using sparse OneHotEncoder ----------
encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
X_cat_sparse = encoder.fit_transform(df[categorical_features])

# Combine numeric and categorical features into one sparse matrix
X_final = sparse.hstack([X_num, X_cat_sparse])

# ---------- 6. Train/test split ----------
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# ---------- 7. Train Decision Tree ----------
tree = DecisionTreeRegressor(max_depth=5, random_state=42)  # limit depth to avoid overfitting
tree.fit(X_train, y_train)

# ---------- 8. Predict ----------
y_pred = tree.predict(X_test)

# ---------- 9. Convert back from log if needed ----------
if log_transform:
    y_pred_original = np.expm1(y_pred)
    y_test_original = np.expm1(y_test)
else:
    y_pred_original = y_pred
    y_test_original = y_test

# ---------- 10. Evaluate ----------
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
print(f"Decision Tree RMSE on original scale: {rmse:.2f}")

# ---------- 11. Feature importance ----------
# Combine numeric + categorical feature names
cat_feature_names = encoder.get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(cat_feature_names)

importances = pd.Series(tree.feature_importances_, index=all_feature_names)
print("\nTop features driving repair costs:")
print(importances.sort_values(ascending=False).head(10))
