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
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

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


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # תמיכה בעברית בקונסול Windows

    X_train, X_test, y_train, y_test = load_train_test()
    print(f"train: {X_train.shape[0]:,} rows | test: {X_test.shape[0]:,} rows")
    print(f"features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"target: {TARGET_COL}  (range {y_train.min():.2f}-{y_train.max():.2f})\n")

    print("=== Step 5: baseline (DummyRegressor) — measured on TEST ===")
    print(f"{'strategy':<10}{'predicts':>10}{'RMSE':>10}{'R2':>10}")
    for strategy, m in run_baselines(X_train, X_test, y_train, y_test).items():
        print(f"{strategy:<10}{m['const']:>10.3f}{m['rmse']:>10.3f}{m['r2']:>10.4f}")
    print("\nהרצפה: כל מודל אמיתי (שלב 6) יצטרך לנצח את ה-RMSE של mean.")
