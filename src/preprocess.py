"""
Preprocessing script for the credit default ML pipeline.

This script loads the raw UCI credit default dataset, cleans column names,
renames features into readable names, drops non-predictive columns, and
saves the processed dataset.
"""

from pathlib import Path
import pandas as pd
from xgboost import config
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