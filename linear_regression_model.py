import pandas as pd
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Load dataset
# df = pd.read_csv('housing_data.csv')

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


def main():
    try:
        df_read = read_excel_file(path, sheet=None)
        df = add_time_until_repair(df_read)
        print("Columns in the dataframe:", df.columns)
       
        print("\nDataframe preview:")
        # print(df.head())
        print(f"Successfully read Excel file: {path}")
        # Select features for simple model
        features = ["region_name", "construction_year",  "repair_year",
                    "occupants",  "repair_count",  "time_until_repair"]
        X = df[features]
        print(f"Selected features: {X}")
        y = df['total_repair_cost']
        print(f"Target variable 'total_repair_cost': {y}")

        # Handle categorical variables with one-hot encoding
        categorical_features = ["region_name"]
        print("\nApplying One-Hot Encoding to categorical features...")
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_encoded = encoder.fit_transform(X[categorical_features])  
        encoded_feature_names = encoder.get_feature_names_out(categorical_features)
        print(f"Encoded feature names: {encoded_feature_names}")
        X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_feature_names, index=X.index)   
        X_numeric = X.drop(columns=categorical_features).reset_index(drop=True)
        X_prepared = pd.concat([X_encoded_df.reset_index(drop=True), X_numeric], axis=1)
        print("Prepared feature set after encoding:")
        print(X_prepared.head())

        X_train, X_test, y_train, y_test = train_test_split(X_prepared, y, test_size=0.2, random_state=42)

        # model = LinearRegression()
        # model.fit(X_train, y_train)
        # y_pred = model.predict(X_test)
        # mse = mean_squared_error(y_test, y_pred)
        # print(f"\nMean Squared Error on test set: {mse}")

        # Apply log transformation to the target
        y_train_log = np.log1p(y_train)
        y_test_log = np.log1p(y_test)

        # Train model on log-transformed target
        model = LinearRegression()
        model.fit(X_train, y_train_log)

        # Predict in log scale
        y_pred_log = model.predict(X_test)

        # Convert predictions back to original scale
        y_pred = np.expm1(y_pred_log)  # inverse of log1p

        # Evaluate performance on original scale
        mse = mean_squared_error(y_test, y_pred)
        print(f"\nMean Squared Error on test set (original scale): {mse:.2f}")

        rmse = np.sqrt(mse)
        print(f"Root Mean Squared Error on test set (original scale): {rmse:.2f}")

    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return
    
if __name__ == "__main__":
    main()