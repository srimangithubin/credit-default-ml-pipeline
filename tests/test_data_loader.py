"""
Tests for data loading and splitting utilities.
"""

import pandas as pd

from src.data_loader import split_features_target, load_processed_data, create_train_test_split
from src.utils import load_config

def test_load_processed_data():
    """
    Test that config.yaml loads as python  dictionary
    """

    config = load_config()

    assert isinstance(config, dict)

def test_config_contains_required_sections():
    config = load_config()

    required_sections = [
        "project",
        "data",
        "random_state",
        "cross_validation",
        "training",
        "outputs",
        "models",
    ]

    for section in required_sections:
        assert section in config

def test_split_features_target():
    df = pd.DataFrame(
        {
            "credit_limit": [20000, 50000, 100000],
            "age": [24, 35, 41],
            "default_next_month": [1, 0, 0],
        }
    )
    X, y = split_features_target(df, "default_next_month")

    assert 'default_next_month' not in X.columns
    assert list(X.columns) == ['credit_limit', 'age']
    assert list(y) == [1, 0, 0]

def test_split_features_target_raises_error_for_missing_target():
    df = pd.DataFrame(
        {
            "credit_limit": [20000, 50000, 100000],
            "age": [24, 35, 41],
        }
    )

    try:
        split_features_target(df, "default_next_month")
        assert False
    except ValueError:
        assert True

def test_create_train_test_split():
    """
    Test that train_test_split returns expected number of rows
    """

    df = pd.DataFrame(
        {
            "credit_limit": list(range(20)),
            "age": list(range(20, 40)),
            "default_next_month": [0, 1] * 10,
        }
    )

    X, y = split_features_target(df, 'default_next_month')

    config = {
        "data": {
            "test_size": 0.25,
        },
        "random_state": 999,
    }

    X_train, X_test, y_train, y_test = create_train_test_split(X, y, config)

    assert X_train.shape[0] == 15
    assert X_test.shape[0] == 5
    assert y_train.shape[0] == 15
    assert y_test.shape[0] == 5
    
