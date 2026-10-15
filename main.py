from inference import forward_chaining, calculate_risk_score
from explanation import explain, suggest_improvements

def get_int_input(prompt, min_val=0, max_val=10000000):
    while True:
        try:
            value = int(input(prompt))
            if value < min_val or value > max_val:
                print(f"  Please enter a value between {min_val} and {max_val}.")
            else:
                return value
        except ValueError:
            print("  Invalid input. Please enter a number.")

print("=" * 50)
print("       LOAN APPROVAL EXPERT SYSTEM v2.0")
print("=" * 50)

age           = get_int_input("Enter Age                        : ", 0, 120)
income        = get_int_input("Enter Monthly Income (₹)         : ", 0)
credit_score  = get_int_input("Enter Credit Score (300-900)     : ", 300, 900)
existing_loans= get_int_input("Enter Number of Existing Loans   : ", 0, 20)
loan_amount   = get_int_input("Enter Requested Loan Amount (₹)  : ", 0)

facts = {
    "age": age,
    "income": income,
    "credit_score": credit_score,
    "existing_loans": existing_loans,
    "loan_amount": loan_amount
}

# Run inference
decision, rule_id, interest_rate = forward_chaining(facts)

# Risk score
risk = calculate_risk_score(facts)
risk_label = "Low" if risk < 30 else "Medium" if risk < 60 else "High"

print("\n" + "=" * 50)
print("               RESULT")
print("=" * 50)
print(f"  Decision      : {decision}")
print(f"  Rule Fired    : {rule_id}")
if interest_rate:
    print(f"  Interest Rate : {interest_rate}% per annum")
print(f"  Risk Score    : {risk}/100 ({risk_label} Risk)")
print(f"  Explanation   : {explain(rule_id)}")

# Suggestions
suggestions = suggest_improvements(facts, decision)
if suggestions:
    print("\n  Suggestions to Improve Eligibility:")
    for i, s in enumerate(suggestions, 1):
        print(f"    {i}. {s}")

print("=" * 50)