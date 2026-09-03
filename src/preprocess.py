"""
Preprocessing script for the credit default ML pipeline.

This script loads the raw UCI credit default dataset, cleans column names,
renames features into readable names, drops non-predictive columns, and
saves the processed dataset.
"""

from pathlib import Path
import pandas as pd
from src.utils import load_config, create_directories

def load_raw_data(config: dict) -> pd.DataFrame:
    """
    Load the raw credit default dataset.

    Parameters
    ----------
    config : dict
        Project configuration dictionary.

    Returns
    -------
    pd.DataFrame
        Raw dataset.
    """

    raw_path = Path(config['data']['raw_path'])
    csv_header_row = config['data']['csv_header_row']

    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw dataset not found at:{raw_path}")

    df = pd.read_csv(raw_path, header=csv_header_row)
    return df

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip leading and trailing spaces from column names.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Returns
    -------
    pd.DataFrame
        Dataset with cleaned column names.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df

def rename_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename features into readable names.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Returns
    -------
    pd.DataFrame
        Dataset with renamed features.
    """
    rename_map = {
        "LIMIT_BAL": "credit_limit",
        "SEX": "sex",
        "EDUCATION": "education",
        "MARRIAGE": "marriage",
        "AGE": "age",
        "PAY_0": "repayment_status_sep",
        "PAY_2": "repayment_status_aug",
        "PAY_3": "repayment_status_jul",
        "PAY_4": "repayment_status_jun",
        "PAY_5": "repayment_status_may",
        "PAY_6": "repayment_status_apr",
        "BILL_AMT1": "bill_amount_sep",
        "BILL_AMT2": "bill_amount_aug",
        "BILL_AMT3": "bill_amount_jul",
        "BILL_AMT4": "bill_amount_jun",
        "BILL_AMT5": "bill_amount_may",
        "BILL_AMT6": "bill_amount_apr",
        "PAY_AMT1": "payment_amount_sep",
        "PAY_AMT2": "payment_amount_aug",
        "PAY_AMT3": "payment_amount_jul",
        "PAY_AMT4": "payment_amount_jun",
        "PAY_AMT5": "payment_amount_may",
        "PAY_AMT6": "payment_amount_apr",
        "default payment next month": "default_next_month",
    }
    df = df.rename(columns=rename_map)
    return df


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop non-predictive columns from the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with renamed features.

    Returns
    -------
    pd.DataFrame
        Dataset with unnecessary columns dropped.
    """
    df = df.copy()
    columns_to_drop = ['ID']
    df = df.drop([col for col in columns_to_drop if col in df.columns], axis = 1)
    return df

def preprocess_data(config: dict) -> pd.DataFrame:
    """
    Preprocess the raw dataset by cleaning column names, renaming features,
    and dropping non-predictive columns.

    Parameters
    ----------
    config : dict
        Project configuration dictionary.

    Returns
    -------
    pd.DataFrame
        Processed dataset.
    """
    df = load_raw_data(config)
    df = clean_column_names(df)
    df = rename_features(df)
    df = drop_unnecessary_columns(df)

    target_column = config['data']['target_column']

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")

    return df

def save_processed_data(df: pd.DataFrame, config: dict) -> None:
    """
    Save the processed dataset to the specified path in the configuration.

    Parameters
    ----------
    df : pd.DataFrame
        Processed dataset.
    config : dict
        Project configuration dictionary.
    """
    processed_path = Path(config['data']['processed_path'])
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)

    print(f"Processed data saved to: {processed_path}")

def main():
    """
    Run preprocessing from command line.
    """
    print("Loading config...")
    config = load_config()
    create_directories(config)

    print("Preprocessing raw data...")
    df_processed = preprocess_data(config)

    print("Processed dataset shape:", df_processed.shape)
    print("Processed columns:")
    print(df_processed.columns.tolist())

    save_processed_data(df_processed, config)

    print("Preprocessing completed successfully.")

if __name__ == "__main__":
    main()