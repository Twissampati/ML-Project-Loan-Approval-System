"""
ml_models.py
============
Implements and compares the four ML classification algorithms required
by the DA-1 project specification:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Naive Bayes

Evaluation metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC,
Confusion Matrix.

Also provides XAI (Explainable AI) via feature importance.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.naive_bayes     import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report,
)

# ── Constants ────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "age", "income", "credit_score", "existing_loans",
    "loan_amount", "assets", "liabilities", "loan_to_income_ratio",
]
LABEL_COL    = "loan_approved"   # 1 = approved, 0 = rejected
MODEL_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(MODEL_DIR, "ml_models.pkl")

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Naive Bayes":         GaussianNB(),
}


# ── Synthetic training data ──────────────────────────────────────────────────

def _generate_synthetic_data(n: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic loan applicant data for model training.
    Replace with real historical data for production.
    """
    np.random.seed(42)

    age            = np.random.randint(21, 61, n)
    income         = np.random.randint(15000, 150000, n)
    credit_score   = np.random.randint(300, 901, n)
    existing_loans = np.random.randint(0, 5, n)
    assets         = np.random.randint(50000, 2000000, n)
    liabilities    = (assets * np.random.uniform(0.1, 0.9, n)).astype(int)
    loan_amount    = np.random.randint(50000, 1500000, n)
    lti_ratio      = loan_amount / np.where(income > 0, income, 1)

    # Heuristic label: approved if credit ≥ 650, income ≥ 25k, LTI ≤ 10, lib/asset ≤ 0.6
    lib_ratio   = liabilities / np.where(assets > 0, assets, 1)
    loan_approved = (
        (credit_score >= 650) &
        (income >= 25000) &
        (lti_ratio <= 10) &
        (lib_ratio <= 0.6) &
        (existing_loans <= 2) &
        (age >= 21) & (age <= 60)
    ).astype(int)

    df = pd.DataFrame({
        "age":                  age,
        "income":               income,
        "credit_score":         credit_score,
        "existing_loans":       existing_loans,
        "loan_amount":          loan_amount,
        "assets":               assets,
        "liabilities":          liabilities,
        "loan_to_income_ratio": lti_ratio.round(2),
        LABEL_COL:              loan_approved,
    })
    return df


# ── Preprocessing ────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Clean and prepare features and labels from a DataFrame.
    Handles missing values and derived fields.
    Returns (X, y).
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Derive loan_to_income_ratio if not present
    if "loan_to_income_ratio" not in df.columns:
        df["loan_to_income_ratio"] = df["loan_amount"] / df["income"].replace(0, np.nan)

    # Default assets / liabilities to 0 if not uploaded
    for col in ["assets", "liabilities"]:
        if col not in df.columns:
            df[col] = 0

    # Fill any remaining NaN in feature cols
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())

    X = df[FEATURE_COLS]

    # Build label from rule-engine decision if no label column
    if LABEL_COL in df.columns:
        y = df[LABEL_COL].astype(int)
    else:
        y = None

    return X, y


# ── Train / evaluate ─────────────────────────────────────────────────────────

def train_and_evaluate(df: pd.DataFrame | None = None) -> dict:
    """
    Train all four models and return evaluation metrics + trained objects.

    Parameters
    ----------
    df : DataFrame with feature cols + LABEL_COL, or None (uses synthetic data).

    Returns
    -------
    dict {
        "results"  : { model_name: metrics_dict },
        "models"   : { model_name: fitted_model },
        "scaler"   : StandardScaler fitted on X_train,
        "best_model": name of model with highest F1,
        "X_test"   : test features,
        "y_test"   : true labels,
        "feature_cols": list of feature column names,
    }
    """
    if df is None:
        df = _generate_synthetic_data()

    X, y = preprocess(df)
    if y is None:
        raise ValueError("No label column found. DataFrame must contain 'loan_approved' (1/0).")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler  = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    results       = {}
    trained_models = {}

    for name, model in MODELS.items():
        # Logistic Regression needs scaling; tree-based don't (but scaling won't hurt)
        model.fit(X_train_sc, y_train)
        y_pred  = model.predict(X_test_sc)
        y_proba = model.predict_proba(X_test_sc)[:, 1] if hasattr(model, "predict_proba") else None

        results[name] = {
            "accuracy":  round(accuracy_score(y_test, y_pred),  4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0),    4),
            "f1":        round(f1_score(y_test, y_pred, zero_division=0),        4),
            "roc_auc":   round(roc_auc_score(y_test, y_proba), 4) if y_proba is not None else None,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "report":    classification_report(y_test, y_pred, output_dict=True),
        }
        trained_models[name] = model

    # Best model by F1
    best_name = max(results, key=lambda k: results[k]["f1"])

    # Save to disk
    bundle = {
        "models":  trained_models,
        "scaler":  scaler,
        "feature_cols": FEATURE_COLS,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)

    return {
        "results":      results,
        "models":       trained_models,
        "scaler":       scaler,
        "best_model":   best_name,
        "X_test":       X_test,
        "y_test":       y_test,
        "feature_cols": FEATURE_COLS,
    }


# ── Load saved models ────────────────────────────────────────────────────────

def load_ml_models() -> dict | None:
    """Load trained models bundle from disk. Returns None if not trained yet."""
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ── Single applicant ML prediction ──────────────────────────────────────────

def ml_predict(facts: dict, model_name: str = "Random Forest") -> dict:
    """
    Predict approval probability for one applicant using a saved ML model.

    Parameters
    ----------
    facts       : applicant feature dict (keys = FEATURE_COLS)
    model_name  : one of the four model names

    Returns
    -------
    dict {
        "approved_prob" : float 0–1,
        "prediction"    : int 0/1,
        "model_used"    : str,
    }
    """
    bundle = load_ml_models()
    if bundle is None:
        # Auto-train on synthetic data if no saved model
        result = train_and_evaluate()
        bundle = {
            "models":  result["models"],
            "scaler":  result["scaler"],
            "feature_cols": result["feature_cols"],
        }

    model  = bundle["models"].get(model_name, list(bundle["models"].values())[0])
    scaler = bundle["scaler"]

    # Build feature vector in the correct order
    features = [facts.get(col, 0) for col in FEATURE_COLS]
    X_scaled = scaler.transform([features])

    prob       = model.predict_proba(X_scaled)[0][1] if hasattr(model, "predict_proba") else None
    prediction = model.predict(X_scaled)[0]

    return {
        "approved_prob": round(float(prob), 4) if prob is not None else None,
        "prediction":    int(prediction),
        "model_used":    model_name,
    }


# ── XAI: Feature importance ──────────────────────────────────────────────────

def get_feature_importance(model_name: str = "Random Forest") -> dict | None:
    """
    Returns feature importance scores for tree-based models.
    Returns None for models without feature_importances_.

    Returns
    -------
    dict { feature_name: importance_score } sorted descending, or None.
    """
    bundle = load_ml_models()
    if bundle is None:
        return None

    model = bundle["models"].get(model_name)
    if model is None or not hasattr(model, "feature_importances_"):
        # For Logistic Regression use absolute coefficients
        if hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
            total = importances.sum()
            scores = {
                col: round(float(imp / total), 4)
                for col, imp in zip(FEATURE_COLS, importances)
            }
            return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        return None

    importances = model.feature_importances_
    scores = {
        col: round(float(imp), 4)
        for col, imp in zip(FEATURE_COLS, importances)
    }
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


def get_xai_explanation(facts: dict, model_name: str = "Random Forest") -> list[str]:
    """
    Generate human-readable XAI explanations for a prediction,
    based on feature importance and the applicant's values.

    Returns a list of explanation strings.
    """
    importance = get_feature_importance(model_name)
    if not importance:
        return ["XAI explanation not available for this model."]

    ml_result = ml_predict(facts, model_name)
    approved  = ml_result["prediction"] == 1
    prob      = ml_result["approved_prob"]

    explanations = []
    if prob is not None:
        explanations.append(
            f"The {model_name} model estimates a {prob*100:.1f}% probability of approval."
        )

    # Top 3 most important features — narrate their influence
    top_features = list(importance.items())[:3]
    for feat, imp in top_features:
        val = facts.get(feat, "N/A")
        pct = imp * 100

        if feat == "credit_score":
            status = "strong" if val >= 700 else "weak"
            explanations.append(
                f"Credit Score ({val}) — the most influential factor ({pct:.1f}% weight). "
                f"Your score is {status}."
            )
        elif feat == "income":
            status = "sufficient" if val >= 30000 else "below threshold"
            explanations.append(
                f"Monthly Income (₹{val:,.0f}) — {pct:.1f}% weight. Currently {status}."
            )
        elif feat == "loan_to_income_ratio":
            lti = facts.get("loan_to_income_ratio",
                            facts.get("loan_amount", 0) / facts.get("income", 1))
            status = "manageable" if lti <= 8 else "high"
            explanations.append(
                f"Loan-to-Income Ratio ({lti:.1f}×) — {pct:.1f}% weight. Currently {status}."
            )
        elif feat == "assets":
            explanations.append(
                f"Assets (₹{val:,.0f}) — {pct:.1f}% weight. Higher assets improve eligibility."
            )
        elif feat == "liabilities":
            explanations.append(
                f"Liabilities (₹{val:,.0f}) — {pct:.1f}% weight. Lower liabilities are preferred."
            )
        else:
            explanations.append(f"{feat.replace('_',' ').title()} — {pct:.1f}% importance weight.")

    return explanations
