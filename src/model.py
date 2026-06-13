"""
מודל ML לחיזוי Thermal Comfort Index (TCI).

החלטות M3 (שלבים 1-2):
  שלב 1 — סוג המשימה: רגרסיה. TCI הוא מספר רציף בטווח 1-10.
  שלב 2 — KPI: RMSE (ככל שנמוך יותר, טוב יותר). נמדד על test בלבד.
            R² מדווח כבונוס (R²=0 = רמת הניחוש-הממוצע).

הקובץ מממש כרגע:
  שלב 3 — חלוקת train/test  (load_train_test)
  שלב 5 — baseline טיפש: DummyRegressor  (run_baselines)
שלבים 4 (preprocessing) ו-6-7 (מודלים אמיתיים + הערכה) יתווספו בהמשך.

הרצה (מתיקיית השורש של הפרויקט):
    python -m src.model
"""
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.data import build_tci_df, FEATURE_COLS, TARGET_COL


def load_train_test(n: int = 5000, seed: int = 42, test_size: float = 0.2):
    """
    בונה את טבלת האימון ומפצל ל-train/test.

    מחזיר: X_train, X_test, y_train, y_test
    """
    df = build_tci_df(n=n, seed=seed)
    X = df[FEATURE_COLS]   # 7 הפיצ'רים שמנבאים TCI
    y = df[TARGET_COL]     # היעד הרציף (רגרסיה)

    # שלב 3: לא נוגעים ב-X_test / y_test עד ההערכה הסופית (שלב 7).
    # random_state מקבע את החלוקה — שחזוריות מלאה בין הרצות.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return X_train, X_test, y_train, y_test


def evaluate(y_true, y_pred) -> dict:
    """מחשב את מדדי שלב 2: RMSE (ראשי) ו-R² (בונוס)."""
    # np.sqrt(MSE) — תואם לכל גרסאות sklearn (ללא תלות ב-squared=/root_mean_squared_error)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "r2": r2}


def run_baselines(X_train, X_test, y_train, y_test) -> dict:
    """
    שלב 5: baseline טיפש שמתעלם מהפיצ'רים ומנבא קבוע.

    mean   — הרצפה הראשית: הקבוע שממזער RMSE (לכן R²=0 בדיוק).
    median — רצפה משנית: רלוונטית כי התפלגות ה-TCI מוטה ימינה (skew~0.78).

    מחזיר dict: {strategy: {"rmse", "r2", "const"}}
    """
    results = {}
    for strategy in ("mean", "median"):
        dummy = DummyRegressor(strategy=strategy)
        dummy.fit(X_train, y_train)          # "לומד" רק את הקבוע מתוך y_train
        y_pred = dummy.predict(X_test)       # אותו קבוע לכל שורה ב-test
        metrics = evaluate(y_test, y_pred)
        metrics["const"] = float(y_pred[0])  # הערך הקבוע שנובא
        results[strategy] = metrics
    return results


def build_models() -> dict:
    """
    שלב 6: המודלים המועמדים, כל אחד עטוף ב-Pipeline (מונע דליפה, נוח לשמירה).
    scaler רק ב-linear — עצים אדישים לסקאלה (מפצלים לפי סף על פיצ'ר בודד).
    """
    return {
        "linear": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", LinearRegression()),
        ]),
        "tree": Pipeline([
            ("reg", DecisionTreeRegressor(random_state=42)),
        ]),
        "forest": Pipeline([
            ("reg", RandomForestRegressor(random_state=42)),
        ]),
    }


def train_and_evaluate(X_train, X_test, y_train, y_test) -> dict:
    """
    שלב 6: מאמן כל מודל על train ומודד RMSE+R² על test.

    מחזיר: {name: {"rmse", "r2", "model"}}  (ה-model הוא ה-Pipeline המאומן)
    """
    results = {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)        # Pipeline: scaler.fit_transform → reg.fit
        y_pred = model.predict(X_test)     # Pipeline: scaler.transform → reg.predict
        metrics = evaluate(y_test, y_pred)
        metrics["model"] = model
        results[name] = metrics
    return results


def forest_importances(model, feature_cols=FEATURE_COLS):
    """מחזיר feature importances ממודל מבוסס-עצים בתוך Pipeline (או None)."""
    reg = model.named_steps["reg"]
    if not hasattr(reg, "feature_importances_"):
        return None
    return sorted(zip(feature_cols, reg.feature_importances_), key=lambda t: -t[1])


def select_winner(results: dict):
    """שלב 7: בוחר את המודל עם ה-RMSE הנמוך ביותר על test. מחזיר (name, metrics)."""
    name = min(results, key=lambda k: results[k]["rmse"])
    return name, results[name]


# נתיב ברירת מחדל למודל השמור — משותף בין model.py (שמירה) ל-app.py (טעינה)
MODEL_PATH = Path("data/tci_model.joblib")


def save_model(model, path=MODEL_PATH, **metadata) -> Path:
    """
    שלב 8: שומר את ה-Pipeline המאומן + metadata לקובץ joblib.

    שומר bundle (dict) עם המודל, סדר הפיצ'רים, והיעד — כדי שהאפליקציה תטען
    הכל יחד ותדע באיזה סדר להזין את הפיצ'רים ל-predict.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "features": FEATURE_COLS,
        "target": TARGET_COL,
        **metadata,
    }
    joblib.dump(bundle, path)
    return path


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # תמיכה בעברית בקונסול Windows

    X_train, X_test, y_train, y_test = load_train_test()
    print(f"train: {X_train.shape[0]:,} rows | test: {X_test.shape[0]:,} rows")
    print(f"features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"target: {TARGET_COL}  (range {y_train.min():.2f}-{y_train.max():.2f})\n")

    print("=== Step 5: baseline (DummyRegressor) — measured on TEST ===")
    print(f"{'strategy':<10}{'predicts':>10}{'RMSE':>10}{'R2':>10}")
    baselines = run_baselines(X_train, X_test, y_train, y_test)
    for strategy, m in baselines.items():
        print(f"{strategy:<10}{m['const']:>10.3f}{m['rmse']:>10.3f}{m['r2']:>10.4f}")
    floor = baselines["mean"]["rmse"]
    print(f"\nהרצפה: RMSE={floor:.3f} (mean) — כל מודל אמיתי צריך לנצח אותה.\n")

    # ── שלב 6+7: מאמן מודלים ומשווה הכל בטבלה אחת ──────────────────────────────
    models = train_and_evaluate(X_train, X_test, y_train, y_test)

    print("=== Step 7: comparison — RMSE / R² on TEST ===")
    print(f"{'model':<20}{'RMSE':>10}{'R2':>10}")
    print(f"{'baseline (mean)':<20}{floor:>10.3f}{baselines['mean']['r2']:>10.4f}")
    for name, m in models.items():
        print(f"{name:<20}{m['rmse']:>10.3f}{m['r2']:>10.4f}")

    winner, wm = select_winner(models)
    print(f"\n🏆 מנצח: {winner} — RMSE={wm['rmse']:.3f}, R²={wm['r2']:.4f}")
    print(f"   משפר את הרצפה (RMSE={floor:.3f}) פי {floor / wm['rmse']:.1f}.")

    imp = forest_importances(models["forest"]["model"])
    if imp:
        print("\nRandomForest feature importances:")
        for k, v in imp:
            print(f"  {k:<18}{v:.3f}")

    # שלב 8a: שמירת המודל המנצח (כפי שהוערך על train) — האפליקציה תטען אותו
    saved = save_model(wm["model"], name=winner, rmse=wm["rmse"])
    print(f"\n💾 נשמר המודל המנצח ({winner}) → {saved}")

    # תוצאות ההשוואה ל-JSON — מקור אמת אחד לתצוגה בסטרימליט (טאב אודות)
    import json
    _summary = {
        "rows": {"train": int(X_train.shape[0]), "test": int(X_test.shape[0])},
        "baseline_mean_rmse": round(floor, 3),
        "table": (
            [{"model": "baseline (mean)", "rmse": round(floor, 3),
              "r2": round(baselines["mean"]["r2"], 4)}]
            + [{"model": n, "rmse": round(m["rmse"], 3), "r2": round(m["r2"], 4)}
               for n, m in models.items()]
        ),
        "winner": winner,
        "winner_rmse": round(wm["rmse"], 3),
        "winner_r2": round(wm["r2"], 4),
        "importances": [[k, round(v, 3)] for k, v in (imp or [])],
    }
    Path("data/model_results.json").write_text(
        json.dumps(_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("📄 נשמרו תוצאות → data/model_results.json")
