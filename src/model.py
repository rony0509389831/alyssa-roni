"""
מודל ML לחיזוי Thermal Comfort Index (TCI).

החלטות M3 (שלבים 1-2):
  שלב 1 — סוג המשימה: רגרסיה. TCI הוא מספר רציף בטווח 1-10.
  שלב 2 — KPI: RMSE (ככל שנמוך יותר, טוב יותר). נמדד על test בלבד.
            R² מדווח כבונוס (R²=0 = רמת הניחוש-הממוצע).

הקובץ מממש:
  שלב 3 — חלוקת train/val/test 70/15/15  (load_train_val_test)
  שלב 5 — baseline טיפש: DummyRegressor  (run_baselines)
  שלב 6 — 3 מודלים מועמדים עטופי Pipeline  (train_and_evaluate)
  שלב 7 — בחירת מנצח לפי val, דיווח על test  (select_winner)
  שלב 8 — שמירת מנצח + ניתוח שגיאות  (save_model, analyze_errors)

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

# פיצ'רים שאינם חלק מנוסחת TCI — נכללו לבדיקת בחירת פיצ'רים
DECOY_COLS = ["temperature", "humidity"]


def load_train_val_test(n: int = 5000, seed: int = 42,
                        val_size: float = 0.15, test_size: float = 0.15):
    """
    בונה את טבלת האימון ומפצל ל-train/val/test (70/15/15).

    val  — לבחירת מנצח ולכיוון hyperparameters (מציצים בו חופשי).
    test — נגיעה אחת בסוף בלבד; המספר שמדווחים להצגה.

    מחזיר: X_train, X_val, X_test, y_train, y_val, y_test
    """
    df = build_tci_df(n=n, seed=seed)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # פיצול ראשון: train vs (val+test)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=val_size + test_size, random_state=seed
    )
    # פיצול שני: val vs test — 50/50 מתוך ה-30% שנשאר
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


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


def train_and_evaluate(X_train, y_train,
                        X_val=None, y_val=None) -> dict:
    """
    שלב 6: מאמן כל מודל על train.

    אם X_val/y_val מסופקים — מוסיף "rmse_val" לכל מודל.
    select_winner() ישתמש ב-rmse_val לבחירה; test נגיעה אחת בסוף בלבד (ב-__main__).

    מחזיר: {name: {"model", ["rmse_val"]}}
    """
    results = {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        metrics: dict = {}
        if X_val is not None:
            metrics["rmse_val"] = evaluate(y_val, model.predict(X_val))["rmse"]
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
    """
    שלב 7: בוחר את המודל עם ה-rmse_val הנמוך ביותר.

    מניח ש-train_and_evaluate הורץ עם X_val/y_val (rmse_val תמיד קיים).
    מחזיר (name, metrics).
    """
    name = min(results, key=lambda k: results[k].get("rmse_val", results[k].get("rmse", float("inf"))))
    return name, results[name]


# נתיב ברירת מחדל למודל השמור. app.py לא מייבא את הקבוע הזה — הוא מגדיר עצמאית
# אותה מחרוזת מילולית כברירת מחדל משלו; הסנכרון הוא בהסכמה בלבד, לא ע"י import.
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


def analyze_errors(model, X_test, y_test, feature_cols=FEATURE_COLS, n_worst: int = 10) -> dict:
    """
    בדיקה 5: ניתוח שגיאות על ה-test set.

    מחזיר dict עם:
      residuals  — מערך (y_true - y_pred) לכל שורה
      y_pred     — תחזיות המודל
      mae        — Mean Absolute Error ממוצע
      worst      — DataFrame של n_worst השורות עם השגיאה הגדולה ביותר
      slices     — RMSE לפי רבעוני כל פיצ'ר (מגלה אזורי חולשה)
    """
    import pandas as pd

    y_pred = model.predict(X_test)
    residuals = np.asarray(y_test) - y_pred
    abs_err = np.abs(residuals)

    df = X_test.copy().reset_index(drop=True)
    df["y_true"] = np.asarray(y_test)
    df["y_pred"] = y_pred
    df["abs_error"] = abs_err

    worst = df.nlargest(n_worst, "abs_error")[
        ["y_true", "y_pred", "abs_error"] + list(feature_cols)
    ].reset_index(drop=True)

    # RMSE לפי רבעוני כל פיצ'ר — חושף אם המודל גרוע על ערכים קיצוניים
    slices = {}
    for col in feature_cols:
        try:
            df["_q"] = pd.qcut(df[col], q=4,
                                labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"],
                                duplicates="drop")
            slices[col] = (
                df.groupby("_q", observed=True)["abs_error"]
                .agg(rmse=lambda x: float(np.sqrt((x**2).mean())))
                .to_dict()["rmse"]
            )
        except Exception:
            pass
    df.drop(columns=["_q"], errors="ignore", inplace=True)

    return {
        "residuals": residuals,
        "y_pred": y_pred,
        "mae": float(np.mean(abs_err)),
        "worst": worst,
        "slices": slices,
    }


if __name__ == "__main__":
    import json
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # תמיכה בעברית בקונסול Windows

    # ── שלב 3 (בדיקה 2): train/val/test נפרדים — val לבחירה, test לדיווח ──────
    X_train, X_val, X_test, y_train, y_val, y_test = load_train_val_test()
    print(f"train: {X_train.shape[0]:,} | val: {X_val.shape[0]:,} | test: {X_test.shape[0]:,} rows")
    print(f"features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"target: {TARGET_COL}  (range {y_train.min():.2f}-{y_train.max():.2f})\n")

    # ── שלב 5: baseline — נמדד על test ──────────────────────────────────────────
    print("=== Step 5: baseline (DummyRegressor) — measured on TEST ===")
    print(f"{'strategy':<10}{'predicts':>10}{'RMSE':>10}{'R2':>10}")
    baselines = run_baselines(X_train, X_test, y_train, y_test)
    for strategy, m in baselines.items():
        print(f"{strategy:<10}{m['const']:>10.3f}{m['rmse']:>10.3f}{m['r2']:>10.4f}")
    floor = baselines["mean"]["rmse"]
    print(f"\nהרצפה: RMSE={floor:.3f} (mean) — כל מודל אמיתי צריך לנצח אותה.\n")

    # ── שלבים 6+7: אימון עם val לבחירה — test נגיעה אחת בסוף בלבד ──────────────
    models = train_and_evaluate(X_train, y_train, X_val, y_val)

    print("=== Step 7: comparison — RMSE_val (choice) ===")
    print(f"{'model':<20}{'RMSE_val':>12}")
    print(f"{'baseline (mean)':<20}{'—':>12}")
    for name, m in models.items():
        rv = f"{m['rmse_val']:.3f}" if "rmse_val" in m else "—"
        print(f"{name:<20}{rv:>12}")

    winner, wm = select_winner(models)
    # test — נגיעה יחידה, למנצח בלבד
    _test_m = evaluate(y_test, wm["model"].predict(X_test))
    wm["rmse"] = _test_m["rmse"]
    wm["r2"]   = _test_m["r2"]
    print(f"\n🏆 מנצח (VAL): {winner} — RMSE_test={wm['rmse']:.3f}, R²={wm['r2']:.4f}")
    print(f"   משפר את הרצפה (RMSE={floor:.3f}) פי {floor / wm['rmse']:.1f}.")

    imp = forest_importances(models["forest"]["model"])
    if imp:
        print("\nRandomForest feature importances:")
        for k, v in imp:
            print(f"  {k:<18}{v:.3f}")

    # ── ניתוח Decoy Features ─────────────────────────────────────────────────
    _imp_dict = dict(imp) if imp else {}
    _decoy_imp = {col: round(_imp_dict.get(col, 0.0), 3) for col in DECOY_COLS}
    _total_decoy = sum(_decoy_imp.values())
    print("\n=== Decoy Feature Analysis (temperature & humidity) ===")
    print("שני הפיצ'רים אינם חלק מנוסחת TCI — אמורים להיות בעלי חשיבות ~0:")
    for col, v in _decoy_imp.items():
        flag = "⚠" if v > 0.01 else "✓"
        print(f"  {flag} {col:<12} {v:.3f}")
    print(f"  סה\"כ: {_total_decoy:.1%} מהחשיבות הכוללת")
    if _total_decoy > 0.05:
        print("  הסבר: קורלציה עונתית עקיפה — חודשי קיץ בנתוני האימון")
        print("         מתאמים טמפרטורה גבוהה עם sun_altitude גבוה (לא קשר סיבתי).")
        print("         בנתוני שטח אמיתיים, קורלציה זו צפויה להיחלש.")

    # ── שלב 8a: שמירת המודל המנצח ───────────────────────────────────────────────
    saved = save_model(wm["model"], name=winner, rmse=wm["rmse"])
    print(f"\n💾 נשמר המודל המנצח ({winner}) → {saved}")

    # ── בדיקה 5: ניתוח שגיאות על ה-test set ────────────────────────────────────
    print("\n=== Check 5: Error Analysis on TEST ===")
    err = analyze_errors(wm["model"], X_test, y_test)
    print(f"MAE={err['mae']:.3f}  |  max_error={err['worst']['abs_error'].iloc[0]:.3f}")

    print(f"\nTop-5 worst predictions (out of {X_test.shape[0]:,} test rows):")
    print(err["worst"][["y_true", "y_pred", "abs_error"]].head(5).to_string(index=False))

    print("\nRMSE by feature quartile (↑ = model struggles at this range):")
    for feat, rmse_by_q in err["slices"].items():
        row = "  ".join(f"{q}:{v:.3f}" for q, v in rmse_by_q.items())
        print(f"  {feat:<20} {row}")

    # test metrics לכל המודלים שלא חושבו (רק המנצח חושב בשלב 7)
    for _, _m in models.items():
        if "rmse" not in _m:
            _tm = evaluate(y_test, _m["model"].predict(X_test))
            _m["rmse"] = _tm["rmse"]
            _m["r2"] = _tm["r2"]

    # ── JSON לסטרימליט (טאב אודות) ──────────────────────────────────────────────
    _summary = {
        "rows": {
            "train": int(X_train.shape[0]),
            "val": int(X_val.shape[0]),
            "test": int(X_test.shape[0]),
        },
        "baseline_mean_rmse": round(floor, 3),
        "table": (
            [{"model": "baseline (mean)", "rmse": round(floor, 3),
              "rmse_val": None, "r2": round(baselines["mean"]["r2"], 4)}]
            + [{"model": n,
                "rmse_val": round(m["rmse_val"], 3) if "rmse_val" in m else None,
                "rmse": round(m["rmse"], 3) if "rmse" in m else None,
                "r2": round(m["r2"], 4) if "r2" in m else None}
               for n, m in models.items()]
        ),
        "winner": winner,
        "winner_rmse": round(wm["rmse"], 3),
        "winner_r2": round(wm["r2"], 4),
        "importances": [[k, round(v, 3)] for k, v in (imp or [])],
        "error_analysis": {
            "mae": round(err["mae"], 3),
            "max_error": round(float(err["worst"]["abs_error"].iloc[0]), 3),
            "slices": {
                feat: {q: round(v, 3) for q, v in qs.items()}
                for feat, qs in err["slices"].items()
            },
        },
        "decoy_analysis": {
            "note": "temperature ו-humidity אינם חלק מנוסחת TCI — נכללו לבדיקת בחירת פיצ'רים",
            "importances": _decoy_imp,
            "total": round(_total_decoy, 3),
            "explanation": (
                "קורלציה עונתית עקיפה: חודשי קיץ בנתוני האימון מתאמים "
                "טמפרטורה גבוהה עם sun_altitude גבוה — לא קשר סיבתי."
            ),
        },
    }
    Path("data/model_results.json").write_text(
        json.dumps(_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n📄 נשמרו תוצאות → data/model_results.json")
