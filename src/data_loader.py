"""
Data loading functions for the credit default ML pipeline.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

def load_processed_data(config: dict)-> pd.DataFrame:
    """
    Load the processed data from the specified path in the configuration.
    parameters
    ----------
    config : dict
        The configuration dictionary containing the data path.
    returns
    -------
    pd.DataFrame
        The loaded processed data as a pandas DataFrame.
    """
    data_path = Path(config['data']['processed_path'])

    if not data_path.is_file():
        raise FileNotFoundError(f"Processed data file not found: {data_path}")

    df = pd.read_csv(data_path)

    return df

def split_features_target(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.series]:
    """
    Split the DataFrame into features and target based on the specified target column.
    parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing features and target.
    target_column : str
        The name of the target column in the DataFrame.
    returns
    -------
    tuple[pd.DataFrame, pd.Series]
        A tuple containing the features DataFrame and the target Series.
    """
    if target_column not in df.columns:
        raise ValueError(f'Target column not found: {target_column}')

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y

def create_train_test_split(X: pd.DataFrame, y: pd.Series, config:dict):
    """
    Create a stratified train-test split.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.

    y : pd.Series
        Target vector.

    config : dict
        Project configuration dictionary loaded from config.yaml.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = config['data']['test_size'],
        random_state = config['random_state'],
        stratify = y
    )

    return X_train, X_test, y_train, y_test