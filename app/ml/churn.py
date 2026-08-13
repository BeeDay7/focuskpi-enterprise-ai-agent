from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


FEATURES = [
    "tenure_months",
    "monthly_spend",
    "support_tickets",
    "late_payments",
]

ARTIFACT_DIR = Path("data") / "models"
MODEL_PATH = ARTIFACT_DIR / "churn_model.joblib"

CHURN_THRESHOLD = 0.50
HIGH_RISK_THRESHOLD = 0.70


@dataclass
class ChurnModel:
    model: RandomForestClassifier
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


def train_model(df: pd.DataFrame) -> ChurnModel:
    """Train and evaluate the churn classification model."""

    X = df[FEATURES]
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    result = ChurnModel(
        model=model,
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        roc_auc=float(roc_auc),
    )

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        result,
        MODEL_PATH,
    )

    return result


def load_model() -> ChurnModel | None:
    """Load the persisted model artifact."""

    if not MODEL_PATH.exists():
        return None

    return joblib.load(
        MODEL_PATH
    )


def predict_customer(
    model: ChurnModel,
    row: dict,
) -> dict:
    """Generate a churn probability and risk category."""

    X = pd.DataFrame(
        [row]
    )[FEATURES]

    probabilities = model.model.predict_proba(
        X
    )

    classes = list(
        model.model.classes_
    )

    if 1 in classes:
        churn_index = classes.index(1)

        probability = float(
            probabilities[0][churn_index]
        )
    else:
        probability = 0.0

    if probability >= HIGH_RISK_THRESHOLD:
        risk = "high"

    elif probability >= CHURN_THRESHOLD:
        risk = "medium"

    else:
        risk = "low"

    return {
        "risk": risk,
        "churn_probability": round(
            probability,
            4,
        ),
        "model_accuracy": round(
            model.accuracy,
            4,
        ),
        "model_precision": round(
            model.precision,
            4,
        ),
        "model_recall": round(
            model.recall,
            4,
        ),
        "model_f1": round(
            model.f1,
            4,
        ),
        "model_roc_auc": round(
            model.roc_auc,
            4,
        ),
        "decision_threshold": CHURN_THRESHOLD,
        "high_risk_threshold": HIGH_RISK_THRESHOLD,
    }