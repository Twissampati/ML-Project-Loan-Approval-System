import numpy as np
from sklearn.ensemble import IsolationForest
import pickle
import os

MODEL_PATH = "anomaly_model.pkl"

def _get_training_data():
    """
    Synthetic training data representing normal loan applicants.
    Replace with real historical data for production use.
    """
    np.random.seed(42)
    n = 500
    return np.column_stack([
        np.random.randint(21,    60,      n),   # age
        np.random.randint(20000, 80000,   n),   # income
        np.random.randint(600,   850,     n),   # credit_score
        np.random.randint(0,     3,       n),   # existing_loans
        np.random.randint(50000, 500000,  n),   # loan_amount
    ])

def train_model():
    """Train and save the Isolation Forest model."""
    X     = _get_training_data()
    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,   # 5% expected anomalies
        random_state=42
    )
    model.fit(X)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"[AnomalyDetection] Model trained and saved to '{MODEL_PATH}'.")
    return model

def load_model():
    """Load saved model, or train a new one if not found."""
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return train_model()

def check_anomaly(facts: dict) -> dict:
    """
    Check whether an applicant's profile is anomalous.

    Parameters:
        facts (dict): Must contain keys:
                      age, income, credit_score, existing_loans, loan_amount

    Returns:
        dict with keys:
            is_anomaly (bool)   - True if anomalous
            score      (float)  - decision function score (lower = more anomalous)
            flags      (list)   - human-readable reasons for suspicion
    """
    model  = load_model()
    sample = np.array([[
        facts["age"],
        facts["income"],
        facts["credit_score"],
        facts["existing_loans"],
        facts["loan_amount"],
    ]])

    prediction = model.predict(sample)[0]          # -1 = anomaly, 1 = normal
    score      = model.decision_function(sample)[0] # lower = more anomalous

    # ── Rule-based flags for explainability ─────────────────────
    flags = []
    if facts["income"] > 150000 and facts["credit_score"] < 500:
        flags.append("Very high income combined with critically low credit score")
    if facts["loan_amount"] > facts["income"] * 15:
        flags.append("Loan amount is extremely high relative to income")
    if facts["age"] < 22 and facts["income"] > 100000:
        flags.append("Very young applicant with unusually high income")
    if facts["credit_score"] < 350 and facts["existing_loans"] == 0:
        flags.append("Critically low credit score with no prior loan history")

    return {
        "is_anomaly" : prediction == -1,
        "score"      : round(float(score), 4),
        "flags"      : flags,
    }