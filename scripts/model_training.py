"""
=============================================================================
  model_training.py
  EV Charging Infrastructure Analysis — Step 3: ML Model Training
=============================================================================
  Run:  python scripts/model_training.py
=============================================================================
"""

import warnings, json, sys
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot  as plt
import matplotlib.patches as mpatches
import seaborn            as sns
import joblib
from pathlib import Path

from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics         import (classification_report, confusion_matrix,
                                     roc_auc_score, roc_curve,
                                     precision_recall_curve, average_precision_score,
                                     f1_score, precision_score, recall_score)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False})

BASE   = Path(__file__).resolve().parent.parent
DATA   = BASE / "data"
MODELS = BASE / "models"
CHARTS = BASE / "dashboard" / "images" / "charts"
OUT    = BASE / "dashboard" / "images" / "charts"

MODELS.mkdir(exist_ok=True)
CHARTS.mkdir(parents=True, exist_ok=True)

def savefig(name):
    path = CHARTS / f"{name}.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"   ✔ Saved → {path.name}")


FEATURE_COLS = [
    "log_station_count", "log_port_count", "ports_per_station",
    "fast_station_share", "fast_port_share", "fast_gap_ratio", "no_ultra_fast",
    "max_power_kw", "median_power_kw", "power_deficit", "relative_power",
    "country_fast_share", "country_median_kw", "country_max_kw",
    "country_tier", "city_infra_share",
    "latband_Tropical", "latband_Subtropical", "latband_Temperate",
    "latband_Cold", "latband_Arctic",
]


def validate_features(df):
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"   Warning: missing feature columns, filling zeros: {missing}")
        for col in missing:
            df[col] = 0

    if "priority" not in df.columns:
        raise ValueError("Missing required target column 'priority' in dataframe.")

    if df["priority"].nunique() < 2:
        raise ValueError("Target column 'priority' must contain at least two classes.")

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD FEATURES
# ─────────────────────────────────────────────────────────────────────────────
def load_features():
    path = DATA / "city_ml_features.csv"
    if not path.exists():
        print("   city_ml_features.csv not found. Running feature engineering first...")
        import sys
        sys.path.insert(0, str(BASE / "scripts"))
        from feature_engineering import main as fe_main
        fe_main()

    df = pd.read_csv(path)
    print(f"   Loaded: {len(df):,} rows  ×  {df.shape[1]} cols")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  TRAIN MODELS
# ─────────────────────────────────────────────────────────────────────────────
def train_models(df):
    print("\n▶  Preparing features & splitting data...")

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0

    X = df[FEATURE_COLS].fillna(0).astype(float)
    y = df["priority"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print("\n▶  Training models...")

    # Logistic Regression
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(X_train_sc, y_train)
    lr_proba = lr.predict_proba(X_test_sc)[:, 1]
    lr_pred  = lr.predict(X_test_sc)
    lr_auc   = roc_auc_score(y_test, lr_proba)
    lr_cv    = cross_val_score(lr, X_train_sc, y_train, cv=5, scoring="roc_auc").mean()
    print(f"   Logistic Regression  →  Test AUC: {lr_auc:.4f}  |  CV AUC: {lr_cv:.4f}")

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_split=10,
        min_samples_leaf=5, random_state=42, class_weight="balanced", n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_pred  = rf.predict(X_test)
    rf_auc   = roc_auc_score(y_test, rf_proba)
    rf_cv    = cross_val_score(rf, X_train, y_train, cv=5, scoring="roc_auc").mean()
    print(f"   Random Forest        →  Test AUC: {rf_auc:.4f}  |  CV AUC: {rf_cv:.4f}")

    # Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.08, max_depth=5,
        subsample=0.85, random_state=42
    )
    gb.fit(X_train, y_train)
    gb_proba = gb.predict_proba(X_test)[:, 1]
    gb_pred  = gb.predict(X_test)
    gb_auc   = roc_auc_score(y_test, gb_proba)
    gb_cv    = cross_val_score(gb, X_train, y_train, cv=5, scoring="roc_auc").mean()
    print(f"   Gradient Boosting    →  Test AUC: {gb_auc:.4f}  |  CV AUC: {gb_cv:.4f}")

    # Select best
    candidates = [("Logistic Regression", lr, lr_auc, lr_proba, lr_pred, lr_cv, X_test_sc),
                  ("Random Forest",       rf, rf_auc, rf_proba, rf_pred, rf_cv, X_test),
                  ("Gradient Boosting",   gb, gb_auc, gb_proba, gb_pred, gb_cv, X_test)]
    best = max(candidates, key=lambda x: x[2])
    best_name, best_model, best_auc, best_proba, best_pred, best_cv, _ = best
    print(f"\n   ★  Best Model: {best_name}  (AUC: {best_auc:.4f})")

    return {
        "models": {"lr": lr, "rf": rf, "gb": gb},
        "probas": {"lr": lr_proba, "rf": rf_proba, "gb": gb_proba},
        "preds":  {"lr": lr_pred,  "rf": rf_pred,  "gb": gb_pred},
        "aucs":   {"lr": lr_auc,   "rf": rf_auc,   "gb": gb_auc},
        "cvs":    {"lr": lr_cv,    "rf": rf_cv,    "gb": gb_cv},
        "best_name": best_name,
        "best_model": best_model,
        "best_proba": best_proba,
        "best_pred":  best_pred,
        "scaler": scaler,
        "X_train": X_train, "X_test": X_test,
        "X_train_sc": X_train_sc, "X_test_sc": X_test_sc,
        "y_train": y_train, "y_test": y_test,
        "X": X, "y": y,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  EVALUATION PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_and_plot(results):
    print("\n▶  Generating evaluation plots...")
    aucs   = results["aucs"]
    probas = results["probas"]
    preds  = results["preds"]
    y_test = results["y_test"]
    best_name  = results["best_name"]
    best_proba = results["best_proba"]
    best_pred  = results["best_pred"]

    # ROC Curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, key, color) in zip(axes, [
        ("Logistic Regression", "lr", "#42A5F5"),
        ("Random Forest",       "rf", "#FF5722"),
        ("Gradient Boosting",   "gb", "#4CAF50"),
    ]):
        fpr, tpr, _ = roc_curve(y_test, probas[key])
        auc = aucs[key]
        ax.plot(fpr, tpr, color=color, lw=2.2, label=f"AUC = {auc:.3f}")
        ax.plot([0,1],[0,1], "k--", lw=1, alpha=0.4)
        ax.fill_between(fpr, tpr, alpha=0.08, color=color)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC — {name}", fontweight="bold")
        ax.legend(loc="lower right")
    plt.tight_layout()
    savefig("ML_01_roc_curves")

    # Model comparison
    model_labels = ["Logistic\nRegression", "Random\nForest", "Gradient\nBoosting"]
    test_vals = [aucs["lr"], aucs["rf"], aucs["gb"]]
    cv_vals   = [results["cvs"]["lr"], results["cvs"]["rf"], results["cvs"]["gb"]]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - 0.2, test_vals, 0.35, label="Test AUC",  color="#2196F3", edgecolor="white")
    b2 = ax.bar(x + 0.2, cv_vals,   0.35, label="CV AUC",    color="#FF9800", edgecolor="white")
    ax.set_ylabel("ROC-AUC Score"); ax.set_ylim(0.5, 1.0)
    ax.set_xticks(x); ax.set_xticklabels(model_labels)
    ax.set_title("Model Comparison — Test vs CV AUC", fontweight="bold")
    ax.legend()
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    savefig("ML_02_model_comparison")

    # Confusion matrix
    cm = confusion_matrix(y_test, best_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Low","High"], yticklabels=["Low","High"],
                linewidths=0.5, ax=ax)
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    ax.set_title(f"Confusion Matrix — {best_name}", fontweight="bold")
    plt.tight_layout()
    savefig("ML_03_confusion_matrix")

    print(f"\n  {best_name} — Classification Report:")
    print(classification_report(y_test, best_pred, target_names=["Low Priority","High Priority"]))

    # Feature importance
    best_model = results["best_model"]
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    else:
        importances = np.abs(best_model.coef_[0])

    feat_imp = pd.Series(importances, index=FEATURE_COLS).sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#FF5722" if f in ["fast_station_share","fast_gap_ratio","no_ultra_fast"]
              else "#2196F3" if f in ["log_station_count","log_port_count","ports_per_station"]
              else "#4CAF50" for f in feat_imp.index]
    feat_imp.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top 15 Feature Importances — {best_name}", fontweight="bold")
    patches = [mpatches.Patch(color="#FF5722", label="Fast-Charge Gap"),
               mpatches.Patch(color="#2196F3", label="Demand/Supply"),
               mpatches.Patch(color="#4CAF50", label="Power & Country")]
    ax.legend(handles=patches, loc="lower right", fontsize=9)
    plt.tight_layout()
    savefig("ML_04_feature_importance")

    return importances


# ─────────────────────────────────────────────────────────────────────────────
#  SAVE MODEL & EXPORT PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────
def save_and_export(results, df, importances):
    print("\n▶  Saving model & exporting predictions...")

    best_name  = results["best_name"]
    best_model = results["best_model"]
    scaler     = results["scaler"]

    # Save model bundle
    bundle = {"model": best_model, "scaler": scaler,
              "feature_cols": FEATURE_COLS, "model_name": best_name}
    joblib.dump(bundle, MODELS / "trained_model.pkl")
    print(f"   Saved: models/trained_model.pkl  ({best_name})")

    # Full predictions
    X_full = df[FEATURE_COLS].fillna(0).astype(float)
    if best_name == "Logistic Regression":
        full_proba = best_model.predict_proba(scaler.transform(X_full))[:, 1]
    else:
        full_proba = best_model.predict_proba(X_full)[:, 1]

    # Threshold optimisation
    thresholds = np.arange(0.1, 0.91, 0.05)
    f1s = [f1_score(results["y_test"],
                    (results["best_proba"] >= t).astype(int),
                    zero_division=0) for t in thresholds]
    best_threshold = thresholds[np.argmax(f1s)]

    df["ml_priority_proba"] = full_proba.round(4)
    df["ml_priority_label"] = (full_proba >= best_threshold).astype(int)
    df["composite_score"]   = (
        0.60 * df["priority_score"] / 100 + 0.40 * df["ml_priority_proba"]
    ) * 100
    df["composite_score"] = df["composite_score"].round(1)

    def recommendation(row):
        s = row["composite_score"]
        if s >= 70:  return "Urgent — Install Immediately"
        if s >= 50:  return "High — Strong Candidate"
        if s >= 35:  return "Medium — Monitor Growth"
        return             "Low — Already Well-Served"

    df["recommendation"] = df.apply(recommendation, axis=1)

    result_cols = [
        "country_code", "city", "latitude", "longitude",
        "station_count", "fast_station_share", "median_power_kw",
        "no_ultra_fast", "composite_score", "ml_priority_proba",
        "ml_priority_label", "recommendation",
    ]
    result_df = df[result_cols].sort_values("composite_score", ascending=False).reset_index(drop=True)
    result_df.to_csv(DATA / "city_priority_predictions.csv", index=False)
    result_df.head(100).to_csv(DATA / "top100_priority_cities.csv", index=False)

    # Model performance CSV
    pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "Gradient Boosting"],
        "Test_AUC": [round(results["aucs"]["lr"], 4),
                     round(results["aucs"]["rf"], 4),
                     round(results["aucs"]["gb"], 4)],
    }).to_csv(DATA / "model_performance.csv", index=False)

    print(f"   Saved: city_priority_predictions.csv  ({len(result_df):,} rows)")
    print(f"   Saved: top100_priority_cities.csv")
    print(f"   Saved: model_performance.csv")

    # ML dashboard JSON
    feat_imp_js = {col: round(float(v), 5) for col, v in
                   sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True)[:15]}

    ml_json = {
        "model_used": best_name,
        "best_threshold": float(best_threshold),
        "model_performance": {
            "lr_auc": round(results["aucs"]["lr"], 4),
            "rf_auc": round(results["aucs"]["rf"], 4),
            "gb_auc": round(results["aucs"]["gb"], 4),
        },
        "dataset_stats": {
            "total_cities": len(df),
            "high_priority": int(df["ml_priority_label"].sum()),
            "urgent_cities": int((df["composite_score"] >= 70).sum()),
        },
        "top50_cities": result_df.head(50).to_dict(orient="records"),
        "feature_importance": feat_imp_js,
        "recommendation_counts": df["recommendation"].value_counts().to_dict(),
    }
    with open(DATA / "ml_dashboard_data.json", "w") as f:
        json.dump(ml_json, f, indent=2, default=str)
    print(f"   Saved: ml_dashboard_data.json")

    return result_df, best_threshold


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  MODEL TRAINING PIPELINE")
    print("=" * 60)

    df      = load_features()
    results = train_models(df)
    importances = evaluate_and_plot(results)
    result_df, threshold = save_and_export(results, df, importances)

    best_name = results["best_name"]
    best_auc  = results["aucs"][{"Logistic Regression":"lr","Random Forest":"rf","Gradient Boosting":"gb"}[best_name]]

    print(f"""
╔══════════════════════════════════════════════════════════╗
║  PIPELINE COMPLETE                                        ║
╚══════════════════════════════════════════════════════════╝
  Best Model    : {best_name}
  Test AUC      : {best_auc:.4f}
  Threshold     : {threshold:.2f}
  Cities scored : {len(result_df):,}
  Urgent (≥70)  : {int((result_df['composite_score']>=70).sum()):,}

  Top 5 Urgent Cities:""")

    for i, (_, r) in enumerate(result_df.head(5).iterrows(), 1):
        print(f"    {i}. {r['city']}, {r['country_code']}  "
              f"→ Score: {r['composite_score']:.1f}  |  "
              f"Fast: {r['fast_station_share']*100:.0f}%")

    print(f"\n  ✔  Charts saved → dashboard/images/charts/")
    print(f"  ✔  Model saved  → models/trained_model.pkl")


if __name__ == "__main__":
    main()
