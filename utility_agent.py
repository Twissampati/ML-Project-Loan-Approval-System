"""
utility_agent.py
================
Utility-Based Agent for Loan Applicant Merit List Generation.

Updated to support extended fields from DA-1 spec:
  age, gender, occupation, income, loan_type, loan_to_income_ratio,
  credit_score, assets, liabilities, existing_loans, loan_amount
"""

import pandas as pd
import numpy as np
from inference         import calculate_risk_score, forward_chaining
from fuzzy_logic       import get_fuzzy_approval
from anomaly_detection import check_anomaly


# ── Utility weights (must sum to 1.0) ──────────────────────────────────────
WEIGHTS = {
    "credit_score_norm":  0.25,
    "income_norm":        0.20,
    "fuzzy_score_norm":   0.15,
    "risk_inverse_norm":  0.15,
    "loan_burden_norm":   0.10,
    "asset_health_norm":  0.15,   # new: assets vs liabilities
}

TIER_THRESHOLDS = {
    "Tier 1 – Premium":     0.78,
    "Tier 2 – Standard":    0.62,
    "Tier 3 – Basic":       0.50,
    "Tier 4 – Conditional": 0.38,
}

REQUIRED_COLUMNS = {
    "age", "income", "credit_score", "existing_loans", "loan_amount"
}

OPTIONAL_COLUMNS = {
    "assets", "liabilities", "gender", "occupation", "loan_type", "name"
}

OPTIONAL_NAME_COLS = ["name", "applicant_name", "full_name", "candidate"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _find_name_column(df: pd.DataFrame) -> str | None:
    for c in OPTIONAL_NAME_COLS:
        if c in df.columns.str.lower().tolist():
            return df.columns[df.columns.str.lower() == c][0]
    return None


def _normalise(value: float, low: float, high: float) -> float:
    return float(np.clip((value - low) / (high - low), 0, 1))


def _tier(utility: float) -> str:
    for label, threshold in TIER_THRESHOLDS.items():
        if utility >= threshold:
            return label
    return "Tier 5 – Rejected"


def _loan_burden_score(income: float, loan_amount: float) -> float:
    if income <= 0:
        return 0.0
    ratio = loan_amount / income
    if ratio <= 3:   return 1.0
    if ratio <= 6:   return 0.75
    if ratio <= 10:  return 0.50
    if ratio <= 15:  return 0.25
    return 0.0


def _asset_health_score(assets: float, liabilities: float) -> float:
    """Returns 0–1 based on net worth / asset ratio."""
    if assets <= 0:
        return 0.0
    net_worth = assets - liabilities
    ratio     = net_worth / assets
    return float(np.clip(ratio, 0, 1))


# ── Main agent function ───────────────────────────────────────────────────────

def run_utility_agent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perception → Utility computation → Action → Ranked merit list.

    Parameters
    ----------
    df : DataFrame with required columns (and optionally: assets, liabilities,
         gender, occupation, loan_type, name)

    Returns
    -------
    pd.DataFrame sorted by tier then utility_score descending.
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Fill optional columns with defaults
    for col in ["assets", "liabilities"]:
        if col not in df.columns:
            df[col] = 0
    for col in ["gender", "occupation", "loan_type"]:
        if col not in df.columns:
            df[col] = "—"

    for col in list(REQUIRED_COLUMNS) + ["assets", "liabilities"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Derived field
    df["loan_to_income_ratio"] = (
        df["loan_amount"] / df["income"].replace(0, np.nan)
    ).fillna(0).round(2)

    df.dropna(subset=list(REQUIRED_COLUMNS), inplace=True)
    df.reset_index(drop=True, inplace=True)

    name_col = _find_name_column(df)
    if name_col:
        df["applicant_name"] = df[name_col].astype(str)
    else:
        df["applicant_name"] = [f"Applicant {i+1}" for i in range(len(df))]

    records = []

    for _, row in df.iterrows():
        facts = {
            "age":                  int(row["age"]),
            "income":               float(row["income"]),
            "credit_score":         float(row["credit_score"]),
            "existing_loans":       int(row["existing_loans"]),
            "loan_amount":          float(row["loan_amount"]),
            "assets":               float(row["assets"]),
            "liabilities":          float(row["liabilities"]),
            "loan_to_income_ratio": float(row["loan_to_income_ratio"]),
            "gender":               str(row.get("gender", "—")),
            "occupation":           str(row.get("occupation", "—")),
            "loan_type":            str(row.get("loan_type", "—")),
        }

        # Sub-scores
        credit_norm   = _normalise(facts["credit_score"], 300, 900)
        income_norm   = _normalise(facts["income"], 0, 200_000)
        fuzzy_score, fuzzy_label = get_fuzzy_approval(
            facts["credit_score"], facts["income"], facts["existing_loans"]
        )
        fuzzy_norm    = fuzzy_score / 100.0
        risk          = calculate_risk_score(facts)
        risk_inv_norm = 1 - (risk / 100.0)
        burden_norm   = _loan_burden_score(facts["income"], facts["loan_amount"])
        asset_norm    = _asset_health_score(facts["assets"], facts["liabilities"])

        utility = (
            WEIGHTS["credit_score_norm"]  * credit_norm   +
            WEIGHTS["income_norm"]        * income_norm   +
            WEIGHTS["fuzzy_score_norm"]   * fuzzy_norm    +
            WEIGHTS["risk_inverse_norm"]  * risk_inv_norm +
            WEIGHTS["loan_burden_norm"]   * burden_norm   +
            WEIGHTS["asset_health_norm"]  * asset_norm
        )

        decision, rule_id, interest_rate, rule_label = forward_chaining(facts)
        anomaly = check_anomaly(facts)

        # Tier assignment — rule engine takes precedence over utility threshold
        if anomaly["is_anomaly"] or "Rejected" in decision:
            tier = "Tier 5 – Rejected"
        elif "Premium" in decision:
            tier = "Tier 1 – Premium"
        elif "Standard" in decision:
            tier = "Tier 2 – Standard"
        elif "Basic" in decision:
            tier = "Tier 3 – Basic"
        elif "Conditional" in decision:
            tier = "Tier 4 – Conditional"
        else:
            tier = "Tier 5 – Rejected"

        # Strengths & weaknesses
        strengths, weaknesses = [], []
        if facts["credit_score"] >= 700:  strengths.append("Good credit score")
        else:                             weaknesses.append("Low credit score")
        if facts["income"] >= 40_000:     strengths.append("Sufficient income")
        else:                             weaknesses.append("Low income")
        if facts["existing_loans"] <= 1:  strengths.append("Few existing loans")
        else:                             weaknesses.append("Multiple existing loans")
        if burden_norm >= 0.75:           strengths.append("Manageable loan amount")
        else:                             weaknesses.append("High loan-to-income ratio")
        if asset_norm >= 0.6:             strengths.append("Healthy asset-liability balance")
        elif asset_norm < 0.3:            weaknesses.append("High liabilities relative to assets")
        if anomaly["is_anomaly"]:         weaknesses.append("Anomalous profile detected")

        records.append({
            "applicant_name":       row["applicant_name"],
            "age":                  facts["age"],
            "gender":               facts["gender"],
            "occupation":           facts["occupation"],
            "income":               facts["income"],
            "credit_score":         facts["credit_score"],
            "existing_loans":       facts["existing_loans"],
            "loan_amount":          facts["loan_amount"],
            "loan_type":            facts["loan_type"],
            "loan_to_income_ratio": facts["loan_to_income_ratio"],
            "assets":               facts["assets"],
            "liabilities":          facts["liabilities"],
            # scores
            "utility_score":        round(utility, 4),
            "risk_score":           risk,
            "fuzzy_score":          fuzzy_score,
            "fuzzy_label":          fuzzy_label,
            # decisions
            "rule_decision":        decision,
            "rule_id":              rule_id,
            "interest_rate":        interest_rate,
            "tier":                 tier,
            # anomaly
            "is_anomaly":           anomaly["is_anomaly"],
            "anomaly_score":        anomaly["score"],
            "anomaly_flags":        "; ".join(anomaly["flags"]) if anomaly["flags"] else "None",
            # rationale
            "strengths":            ", ".join(strengths) if strengths else "—",
            "weaknesses":           ", ".join(weaknesses) if weaknesses else "—",
        })

    result_df = pd.DataFrame(records)

    tier_order = {
        "Tier 1 – Premium":     1,
        "Tier 2 – Standard":    2,
        "Tier 3 – Basic":       3,
        "Tier 4 – Conditional": 4,
        "Tier 5 – Rejected":    5,
    }
    result_df["tier_rank"] = result_df["tier"].map(tier_order)
    result_df.sort_values(["tier_rank", "utility_score"], ascending=[True, False], inplace=True)
    result_df.drop(columns=["tier_rank"], inplace=True)
    result_df.insert(0, "merit_rank", range(1, len(result_df) + 1))
    result_df.reset_index(drop=True, inplace=True)
    return result_df


def get_summary_stats(merit_df: pd.DataFrame) -> dict:
    tier_counts  = merit_df["tier"].value_counts().to_dict()
    rejected_mask = (
        merit_df["tier"].str.contains("Rejected") |
        (merit_df["is_anomaly"] == True) |
        merit_df["rule_decision"].str.contains("Rejected", na=False)
    )
    return {
        "total":       len(merit_df),
        "approved":    int((~rejected_mask).sum()),
        "rejected":    int(rejected_mask.sum()),
        "anomalies":   int(merit_df["is_anomaly"].sum()),
        "avg_utility": round(merit_df["utility_score"].mean(), 3),
        "avg_risk":    round(merit_df["risk_score"].mean(), 1),
        "avg_credit":  round(merit_df["credit_score"].mean(), 0),
        "tier_counts": tier_counts,
    }
