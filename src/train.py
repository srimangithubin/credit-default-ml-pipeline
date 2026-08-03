"""
Training script for the credit default ML pipeline.

This script trains, tunes, evaluates, and saves the best machine learning model.
It uses the processed dataset created by src/preprocess.py.
"""

from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import (
    cross_validate,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

from src.data_loader import (
    load_processed_data,
    split_features_target,
    create_train_test_split,
)
from src.utils import load_config, create_directories

def build_models(random_state: int, n_jobs: int) -> dict:
    """
    Build machine learning pipelines.

    Parameters
    ----------
    random_state : int
        Random seed for reproducibility.

    n_jobs : int
        Number of CPU cores to use where supported.

    Returns
    -------
    dict
        Dictionary of model names and pipelines.
    """
    models = {
        "dummy_classifier": Pipeline(
            steps=[
                ("model", DummyClassifier(strategy="most_frequent"))
            ]
        ),

        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        random_state=random_state,
                        max_iter=1000,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),

        "random_forest": Pipeline(
            steps=[
                (
                    "model",
                    RandomForestClassifier(
                        random_state=random_state,
                        class_weight="balanced",
                        n_jobs=n_jobs,
                    ),
                )
            ]
        ),

        "gradient_boosting": Pipeline(
            steps=[
                (
                    "model",
                    GradientBoostingClassifier(
                        random_state=random_state,
                    ),
                )
            ]
        ),

        "xgboost": Pipeline(
            steps=[
                (
                    "model",
                    XGBClassifier(
                        random_state=random_state,
                        eval_metric="logloss",
                        n_jobs=n_jobs,
                    ),
                )
            ]
        ),
    }

    return models

def run_cross_validation(
    model_name: str,
    pipeline: Pipeline,
    X_train : pd.DataFrame,
    y_train : pd.Series,
    cv: StratifiedKFold,
    n_jobs: int,
):
    """
    Run cross-validation for one model.

    Returns
    -------
    dict
        Mean train and validation scores.
    """
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    scores = cross_validate(
        estimator=pipeline,
        X=X_train,
        y=y_train,
        cv = cv,
        scoring=scoring,
        n_jobs=n_jobs,
        return_train_score=True,
    )

    result = {
        'model': model_name,
        'train_accuracy': scores['train_accuracy'].mean(),
        'train_precision': scores['train_precision'].mean(),
        'train_recall': scores['train_recall'].mean(),
        'train_f1': scores['train_f1'].mean(),
        'train_roc_auc': scores['train_roc_auc'].mean(),
        'cv_accuracy': scores['test_accuracy'].mean(),
        'cv_precision': scores['test_precision'].mean(),
        'cv_recall': scores['test_recall'].mean(),
        'cv_f1': scores['test_f1'].mean(),
        'cv_roc_auc': scores['test_roc_auc'].mean(),
    }

    return result

def tune_model(
        model_name:str,
        pipeline: Pipeline,
        param_grid: dict,
        n_iter: int,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv: StratifiedKFold,
        scoring: dict,
        n_jobs: int,
        random_state: int,
) -> dict:
    """
    Tune one model using RandomizedSearchCV.

    Returns
    -------
    dict
        Best estimator, best parameters, and best CV score.
    """
    print(f"Tuning {model_name}...")

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=1,
        refit=True,
    )

    search.fit(X_train, y_train)

    print(f"Best CV {scoring}: {search.best_score_:.4f}")
    print("Best parameters:")
    print(search.best_params_)

    return {
        "model": model_name,
        "best_cv_score": search.best_score_,
        "best_params": search.best_params_,
        "best_estimator": search.best_estimator_,
    }

def evaluate_model(
    model_name: str,
    fitted_model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    confusion_matrix_path: Path,
    roc_curve_path: Path,
) -> dict:
    """
    Evaluate the best fitted model on the test set.
    """
    y_pred = fitted_model.predict(X_test)
    y_proba = fitted_model.predict_proba(X_test)[:, 1]

    results = {
        'model': model_name,
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_precision': precision_score(y_test, y_pred),
        'test_recall': recall_score(y_test, y_pred),
        'test_f1': f1_score(y_test, y_pred),
        'test_roc_auc': roc_auc_score(y_test, y_proba),
    }   

    print(f"Test set evaluation for {model_name}:")
    print(results)

    print("\n Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["No Default", "Default"])
    plt.title(f" Confusion Matrix for {model_name}")
    plt.tight_layout()
    plt.savefig(confusion_matrix_path, dpi=300)
    plt.close()

    RocCurveDisplay.from_estimator(
        fitted_model,
        X_test,
        y_test,
    )
    plt.title(f"ROC Curve - {model_name}")
    plt.tight_layout()
    plt.savefig(roc_curve_path, dpi=300)
    plt.close()

    return results

def save_feature_importance(
        model_name: str,
        fitted_model: Pipeline,
        X_train: pd.DataFrame,
        report_dir: Path,
        image_dir: Path,

) -> None:
    """
    Save feature importance for tree-based models.
    """

    model = fitted_model.named_steps['model']

    if not hasattr(model, 'feature_importances_'):
        print(f"Model {model_name} does not have feature importances.")
        return

    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)

    importance_Path = report_dir / 'feature_importance.csv'
    importance_df.to_csv(importance_Path, index=False)

    importance_path = report_dir / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)

    top_features = importance_df.head(15)

    plt.figure(figsize=(10, 6))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.gca().invert_yaxis()
    plt.title(f"Top 15 Feature Importances - {model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(image_dir / "feature_importance.png", dpi=300)
    plt.close()

    print("\nTop 15 important features:")
    print(top_features)
