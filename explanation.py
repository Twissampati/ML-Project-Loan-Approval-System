EXPLANATIONS = {
    "R1":  "Premium Approval: Excellent credit score (750+), high income (₹60k+), no existing loans, loan within safe limits, and low liabilities relative to assets.",
    "R2":  "Standard Approval: Good credit score (700+), decent income (₹40k+), at most 1 existing loan, loan amount is reasonable, and liabilities are manageable.",
    "R3":  "Basic Approval: Acceptable credit score (650+), sufficient income (₹30k+), manageable existing loans, loan within limits, and reasonable asset-liability ratio.",
    "R4":  "Conditional Approval: Strong credit score compensates for lower income. Approved with conditions and higher interest rate.",
    "R5":  "Conditional Approval: Moderate credit score (600–650) but income is sufficient. Approved with stricter terms.",
    "R6":  "Conditional Approval: High income compensates for below-average credit score. Approved with high interest rate.",
    "R7":  "Rejected: Applicant is below the minimum age requirement of 21 years.",
    "R8":  "Rejected: Applicant exceeds the maximum eligible age of 60 years.",
    "R9":  "Rejected: Applicant already has 3 or more existing loans, indicating high financial burden.",
    "R10": "Rejected: Credit score is critically low (below 500), indicating very high default risk.",
    "R11": "Rejected: Requested loan amount is too high relative to monthly income (exceeds 12× income).",
    "R12": "Rejected: Monthly income is below the minimum threshold of ₹15,000.",
    "R13": "Rejected: Liabilities exceed 80% of total assets, indicating serious financial instability.",
    "No rule matched": "Rejected: The applicant's profile did not match any approval or specific rejection rule.",
}

def explain(rule_id):
    return EXPLANATIONS.get(rule_id, "No explanation available for this rule.")

def suggest_improvements(facts, decision):
    """
    Actionable suggestions for rejected or conditionally approved applicants.
    Now considers assets and liabilities too.
    """
    suggestions = []

    if "Rejected" in decision or "Conditional" in decision:
        if facts["credit_score"] < 650:
            suggestions.append("Improve your credit score to at least 650 by clearing existing dues.")
        if facts["income"] < 30000:
            suggestions.append("A higher monthly income (₹30,000+) significantly improves approval chances.")
        if facts["existing_loans"] >= 2:
            suggestions.append("Try closing existing loans before applying for a new one.")
        if facts["loan_amount"] > facts["income"] * 8:
            suggestions.append(f"Consider reducing your loan amount to below ₹{facts['income'] * 8:,.0f}.")
        if facts["age"] < 21:
            suggestions.append("You must be at least 21 years old to apply.")
        if facts["age"] > 60:
            suggestions.append("Loan applications are not accepted for applicants above 60 years of age.")

        assets      = facts.get("assets", 0)
        liabilities = facts.get("liabilities", 0)
        if assets > 0 and liabilities > assets * 0.6:
            suggestions.append(
                f"Reduce your liabilities. Current liabilities (₹{liabilities:,.0f}) are "
                f"{liabilities/assets*100:.0f}% of your assets (₹{assets:,.0f}). "
                "Aim to bring this below 60%."
            )
        elif assets == 0:
            suggestions.append("Building assets (savings, property) will significantly improve your eligibility.")

        lti = facts.get("loan_to_income_ratio",
                        facts["loan_amount"] / facts["income"] if facts["income"] > 0 else 0)
        if lti > 8:
            suggestions.append(f"Your loan-to-income ratio ({lti:.1f}×) is very high. Aim for below 8×.")

    return suggestions if suggestions else ["Your profile looks good. No major improvements needed."]
