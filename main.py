import numpy as np
from src.preprocess import load_and_clean, split_and_scale
from src.train import train_logistic, train_random_forest, evaluate

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ── Data ─────────────────────────────────────────────────────────────────────
df = load_and_clean("data/heart.csv")
X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale(df)

# ── Train ─────────────────────────────────────────────────────────────────────
print("Training Logistic Regression...")
lr_model = train_logistic(X_train, y_train, C=0.1)
lr_results = evaluate(lr_model, X_val, y_val, "val - logistic")

print("\nTraining Random Forest...")
rf_model = train_random_forest(X_train, y_train, n_estimators=100)
rf_results = evaluate(rf_model, X_val, y_val, "val - random forest")

# ── Final test eval (run once at the end) ────────────────────────────────────
print("\n=== FINAL TEST EVALUATION (Random Forest) ===")
test_results = evaluate(rf_model, X_test, y_test, "test")