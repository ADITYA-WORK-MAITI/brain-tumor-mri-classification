"""SVM classification and evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data import CLASS_NAMES

RANDOM_STATE = 42

# The scaler sits inside the model so that it is fitted on the training folds only.
PARAM_GRID = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": ["scale", 0.01, 0.001],
    "svc__class_weight": [None, "balanced"],
}


def build_model(random_state: int = RANDOM_STATE) -> GridSearchCV:
    """Standardisation + RBF-kernel SVM, tuned with 5-fold stratified grid search."""
    pipeline = Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="rbf"))])
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    return GridSearchCV(pipeline, PARAM_GRID, cv=inner_cv, n_jobs=-1)


def holdout_evaluation(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = RANDOM_STATE
) -> dict:
    """Tune on a stratified 80 % training split and evaluate once on the 20 % test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    model = build_model(random_state).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    best_params = {name.removeprefix("svc__"): value for name, value in model.best_params_.items()}
    return {
        "best_params": best_params,
        "train_accuracy": accuracy_score(y_train, model.predict(X_train)),
        "test_accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "report": classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True),
        "report_text": classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=3),
        "n_train": len(y_train),
        "n_test": len(y_test),
    }


def repeated_cv_evaluation(
    X: np.ndarray, y: np.ndarray, n_splits: int = 5, n_repeats: int = 10, random_state: int = RANDOM_STATE
) -> dict:
    """Nested cross-validation: grid search inside every outer fold.

    A single 20 % test split of a small dataset is noisy, so this estimates how
    much the scores vary from split to split.
    """
    outer_cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    scores = cross_validate(
        build_model(random_state), X, y, cv=outer_cv, scoring=("accuracy", "balanced_accuracy", "f1_macro")
    )
    summary = {}
    for metric in ("accuracy", "balanced_accuracy", "f1_macro"):
        values = scores[f"test_{metric}"]
        summary[metric] = {"mean": float(values.mean()), "std": float(values.std())}
    summary["n_folds"] = n_splits * n_repeats
    return summary
