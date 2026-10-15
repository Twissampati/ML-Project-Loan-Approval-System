import json
import os

def load_rules(filepath=None):
    if filepath is None:
        base = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(base, "rules.json")
    with open(filepath, "r") as file:
        return json.load(file)

def check_condition(condition, facts):
    try:
        return eval(condition, {}, facts)
    except Exception as e:
        print(f"  [Warning] Could not evaluate condition '{condition}': {e}")
        return False

def calculate_risk_score(facts):
    """
    Computes a risk score from 0 (low risk) to 100 (high risk).
    Now includes assets and liabilities for a more complete picture.
    """
    score = 0

    # Credit score contribution (max 35 points)
    if facts["credit_score"] >= 750:
        score += 0
    elif facts["credit_score"] >= 700:
        score += 8
    elif facts["credit_score"] >= 650:
        score += 18
    elif facts["credit_score"] >= 580:
        score += 28
    else:
        score += 35

    # Income contribution (max 25 points)
    if facts["income"] >= 60000:
        score += 0
    elif facts["income"] >= 40000:
        score += 8
    elif facts["income"] >= 25000:
        score += 16
    else:
        score += 25

    # Existing loans (max 15 points)
    score += min(facts["existing_loans"] * 5, 15)

    # Loan-to-income ratio (max 10 points)
    ratio = facts.get("loan_to_income_ratio",
                       facts["loan_amount"] / facts["income"] if facts["income"] > 0 else 999)
    if ratio <= 4:
        score += 0
    elif ratio <= 8:
        score += 5
    else:
        score += 10

    # Liabilities-to-assets ratio (max 15 points) — new field
    assets      = facts.get("assets", 0)
    liabilities = facts.get("liabilities", 0)
    if assets > 0:
        lib_ratio = liabilities / assets
        if lib_ratio <= 0.2:
            score += 0
        elif lib_ratio <= 0.4:
            score += 5
        elif lib_ratio <= 0.6:
            score += 10
        else:
            score += 15
    else:
        score += 15   # no assets = maximum liability risk

    return min(score, 100)

def forward_chaining(facts):
    rules = load_rules()
    for rule in rules:
        if all(check_condition(c, facts) for c in rule["conditions"]):
            return rule["decision"], rule["id"], rule.get("interest_rate"), rule.get("label", "")
    return "Rejected", "No rule matched", None, "Default rejection"
