import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import joblib


def train_all_models():
    df = pd.read_csv("dataset.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.drop(columns=["student_id"], errors="ignore")

    df["gender"] = df["gender"].str.strip().str.lower().map({"male": 0, "female": 1, "other": 2})
    df["stress_level"] = df["stress_level"].str.strip().str.lower().map({"low": 1, "medium": 2, "high": 3})
    df["physical_activity"] = df["physical_activity"].str.strip().str.lower().map({"yes": 1, "no": 0})
    df["academic_pressure"] = df["academic_pressure"].str.strip().str.lower().map({"low": 1, "medium": 2, "high": 3})

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()

    FEATURE_COLS = [
        "age", "gender", "sleep_hours", "screen_time_hours",
        "stress_level", "study_hours", "physical_activity", "caffeine_intake"
    ]
    X = df[FEATURE_COLS]

    # ── Engineered targets ────────────────────────────────────────────────────
    def burnout(r):
        raw = (r["stress_level"] * 22 + r["screen_time_hours"] * 5
               + r["caffeine_intake"] * 3 - r["sleep_hours"] * 9
               - r["physical_activity"] * 10 + 35)
        return float(np.clip(raw, 0, 100))

    def productivity(r):
        raw = (r["sleep_hours"] * 9 + r["study_hours"] * 7
               + r["physical_activity"] * 10 - r["stress_level"] * 14
               - r["screen_time_hours"] * 4 + 15)
        return float(np.clip(raw, 0, 100))

    def wellbeing(r):
        raw = (r["sleep_hours"] * 10 + r["physical_activity"] * 14
               - r["stress_level"] * 16 - r["caffeine_intake"] * 4
               - r["screen_time_hours"] * 2 + 20)
        return float(np.clip(raw, 0, 100))

    df["burnout_risk"] = df.apply(burnout, axis=1)
    df["productivity_score"] = df.apply(productivity, axis=1)
    df["wellbeing_score"] = df.apply(wellbeing, axis=1)

    # ── Train models ──────────────────────────────────────────────────────────
    clf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
    clf.fit(X, df["academic_pressure"])
    joblib.dump(clf, "model_pressure.pkl")

    configs = [
        ("burnout",      "burnout_risk"),
        ("productivity", "productivity_score"),
        ("wellbeing",    "wellbeing_score"),
    ]
    for name, target in configs:
        reg = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                        max_depth=4, random_state=42)
        reg.fit(X, df[target])
        joblib.dump(reg, f"model_{name}.pkl")

    # ── Metadata ──────────────────────────────────────────────────────────────
    importance = pd.DataFrame({
        "feature":    [f.replace("_", " ").title() for f in FEATURE_COLS],
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=True)
    joblib.dump(importance, "feature_importance.pkl")
    joblib.dump(FEATURE_COLS, "feature_cols.pkl")

    peer_data = {
        "n_students":          len(df),
        "burnout_values":      df["burnout_risk"].values,
        "productivity_values": df["productivity_score"].values,
        "wellbeing_values":    df["wellbeing_score"].values,
        "sleep_mean":          df["sleep_hours"].mean(),
        "study_mean":          df["study_hours"].mean(),
        "screen_mean":         df["screen_time_hours"].mean(),
        "stress_mean":         df["stress_level"].mean(),
    }
    joblib.dump(peer_data, "peer_data.pkl")

    acc = cross_val_score(clf, X, df["academic_pressure"], cv=5).mean()
    print(f"✅  All models trained on {len(df)} students.")
    print(f"    Academic-Pressure CV accuracy: {acc:.2%}")


# Streamlit will import this file and call train_all_models()
# Running directly also works
if __name__ == "__main__":
    train_all_models()
