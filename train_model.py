"""
train_model.py
---------------------------------------
Generates synthetic human vs. bot behavioral sessions and trains a
RandomForest classifier on the engineered features. In a real project,
replace `generate_synthetic_data()` with real logged sessions from
`collector.js` (label them via your existing auth/fraud pipeline,
manual review, or known-bot traffic sources).

Run:
    python train_model.py
Produces:
    model.pkl
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

FEATURE_COLUMNS = [
    "time_on_page_ms",
    "mouse_points",
    "mouse_path_length",
    "mouse_straightness",
    "mouse_avg_speed",
    "mouse_speed_variance",
    "key_avg_interval",
    "key_interval_variance",
    "key_count",
    "scroll_events",
    "click_count",
    "honeypot_triggered",
    "webdriver_flag",
]


def generate_synthetic_data(n_humans=2000, n_bots=2000, seed=42):
    """Simulate plausible feature distributions for humans vs bots."""
    rng = np.random.default_rng(seed)

    # --- Humans: slower, noisier, curved mouse paths, variable typing ---
    humans = pd.DataFrame({
        "time_on_page_ms": rng.normal(9000, 3500, n_humans).clip(1500, None),
        "mouse_points": rng.integers(80, 600, n_humans),
        "mouse_path_length": rng.normal(4000, 1500, n_humans).clip(200, None),
        "mouse_straightness": rng.beta(2, 5, n_humans),          # low straightness = curvy
        "mouse_avg_speed": rng.normal(0.8, 0.3, n_humans).clip(0.05, None),
        "mouse_speed_variance": rng.gamma(2, 0.4, n_humans),
        "key_avg_interval": rng.normal(180, 60, n_humans).clip(40, None),
        "key_interval_variance": rng.gamma(3, 800, n_humans),
        "key_count": rng.integers(10, 60, n_humans),
        "scroll_events": rng.integers(0, 12, n_humans),
        "click_count": rng.integers(1, 6, n_humans),
        "honeypot_triggered": np.zeros(n_humans, dtype=int),
        "webdriver_flag": np.zeros(n_humans, dtype=int),
        "label": np.zeros(n_humans, dtype=int),  # 0 = human
    })

    # --- Bots: fast, straight-line mouse (or none), uniform typing ---
    bots = pd.DataFrame({
        "time_on_page_ms": rng.normal(1200, 600, n_bots).clip(50, None),
        "mouse_points": rng.integers(0, 40, n_bots),
        "mouse_path_length": rng.normal(500, 300, n_bots).clip(0, None),
        "mouse_straightness": rng.beta(6, 2, n_bots),             # high straightness = linear
        "mouse_avg_speed": rng.normal(3.5, 1.2, n_bots).clip(0.05, None),
        "mouse_speed_variance": rng.gamma(1, 0.05, n_bots),
        "key_avg_interval": rng.normal(40, 8, n_bots).clip(5, None),   # near-constant, fast
        "key_interval_variance": rng.gamma(1, 20, n_bots),
        "key_count": rng.integers(10, 60, n_bots),
        "scroll_events": rng.integers(0, 2, n_bots),
        "click_count": rng.integers(1, 3, n_bots),
        "honeypot_triggered": rng.choice([0, 1], n_bots, p=[0.7, 0.3]),
        "webdriver_flag": rng.choice([0, 1], n_bots, p=[0.6, 0.4]),
        "label": np.ones(n_bots, dtype=int),  # 1 = bot
    })

    data = pd.concat([humans, bots], ignore_index=True).sample(frac=1, random_state=seed)
    return data


def main():
    data = generate_synthetic_data()
    X = data[FEATURE_COLUMNS]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42, class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    print("=== Classification Report ===")
    print(classification_report(y_test, preds, target_names=["human", "bot"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, probs):.4f}")

    print("\n=== Feature Importances ===")
    for name, imp in sorted(
        zip(FEATURE_COLUMNS, clf.feature_importances_), key=lambda x: -x[1]
    ):
        print(f"{name:25s} {imp:.4f}")

    joblib.dump(clf, "model.pkl")
    print("\nSaved model to model.pkl")


if __name__ == "__main__":
    main()
