import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_clean(path: str) -> pd.DataFrame:
    """Load heart disease dataset and perform cleaning steps."""
    df = pd.read_csv(path)
    
    # Replace '?' with NaN (UCI dataset uses '?' for missing values)
    df.replace("?", np.nan, inplace=True)
    
    # Convert all columns to numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    
    # Drop rows with missing values (only 6 rows affected in this dataset)
    original_len = len(df)
    df.dropna(inplace=True)
    print(f"Dropped {original_len - len(df)} rows with missing values.")
    
    # Binarize target: 0 = no disease, 1 = disease (values 1-4 → 1)
    df["target"] = (df["num"] > 0).astype(int)
    df.drop(columns=["num"], inplace=True)
    
    return df


def split_and_scale(df: pd.DataFrame):
    """Split into train/val/test and scale features."""
    X = df.drop(columns=["target"])
    y = df["target"]

    # 70% train, 15% val, 15% test — stratified to preserve class balance
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler