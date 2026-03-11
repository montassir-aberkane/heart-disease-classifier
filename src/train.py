import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, roc_auc_score, confusion_matrix, classification_report
)


def train_logistic(X_train, y_train, C=1.0):
    model = LogisticRegression(C=C, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=100):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X, y, split_name="val"):
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    f1   = f1_score(y, preds)
    auroc = roc_auc_score(y, probs)
    cm   = confusion_matrix(y, preds)

    print(f"\n--- {split_name.upper()} RESULTS ---")
    print(f"F1:    {f1:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(classification_report(y, preds))

    return {"f1": f1, "auroc": auroc, "confusion_matrix": cm.tolist()}