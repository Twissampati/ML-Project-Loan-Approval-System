"""
app.py  —  Loan Approval Expert System  v5.0
Four tabs:
  1. 🧾 Single Applicant   — full analysis + What-If + EMI + comparison
  2. 📋 Merit List         — batch upload + visual dashboard
  3. 🤖 ML Training        — train models, track versions
  4. 📊 ML Deep Dive       — per-model metrics, confusion matrix, XAI
"""

import streamlit as st
import pandas as pd
import numpy as np

from inference         import forward_chaining, calculate_risk_score
from explanation       import explain, suggest_improvements
from fuzzy_logic       import get_fuzzy_approval
from anomaly_detection import check_anomaly
from ml_models         import (ml_predict, get_feature_importance,
                                get_xai_explanation, train_and_evaluate,
                                load_ml_models, FEATURE_COLS)
from ml_session_state  import load_state, save_state, get_training_badge
from utility_agent     import run_utility_agent, get_summary_stats

st.set_page_config(
    page_title="Loan Approval Expert System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #f0f4f8; }
  .title-box {
      background: linear-gradient(135deg, #1e3a5f 0%, #2e6da4 60%, #1a8cba 100%);
      padding: 1.8rem 2rem; border-radius: 14px;
      text-align: center; color: white; margin-bottom: 1.5rem;
      box-shadow: 0 4px 16px rgba(30,58,95,0.18);
  }
  .title-box h1 { font-size: 2rem; margin: 0; letter-spacing:-0.5px; }
  .title-box p  { margin: 0.4rem 0 0; opacity: 0.88; font-size:0.95rem; }
  .card {
      background: white; border-radius: 12px; padding: 1.4rem;
      margin-bottom: 1.1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.07);
  }
  .card h3 { margin-top: 0; color: #1e3a5f;
             border-bottom: 2px solid #e0e8f0; padding-bottom: 0.4rem; }
  .badge { display:inline-block; padding:0.3rem 0.9rem;
           border-radius:20px; font-weight:bold; font-size:0.92rem; }
  .badge-green  { background:#d4edda; color:#155724; }
  .badge-yellow { background:#fff3cd; color:#856404; }
  .badge-red    { background:#f8d7da; color:#721c24; }
  .badge-blue   { background:#d1ecf1; color:#0c5460; }
  .flag-item  { background:#fff3cd; border-left:4px solid #ffc107;
                padding:0.45rem 0.9rem; border-radius:4px; margin:0.25rem 0; font-size:0.88rem; }
  .sugg-item  { background:#e8f4fd; border-left:4px solid #2e6da4;
                padding:0.45rem 0.9rem; border-radius:4px; margin:0.25rem 0; font-size:0.88rem; }
  .xai-item   { background:#f0f7e6; border-left:4px solid #28a745;
                padding:0.45rem 0.9rem; border-radius:4px; margin:0.25rem 0; font-size:0.88rem; }
  .step-header { font-size:0.75rem; font-weight:bold; text-transform:uppercase;
                 letter-spacing:1px; color:#2e6da4; margin-bottom:0.25rem; }
  .pbar-bg   { background:#e0e8f0; border-radius:10px; height:16px; width:100%; margin:0.35rem 0; }
  .pbar-fill { height:16px; border-radius:10px; transition: width 0.4s; }
  .emi-box   { background: linear-gradient(135deg,#e8f4fd,#d1ecf1);
               border-radius:10px; padding:1.1rem 1.4rem; margin-top:0.6rem; }
  .emi-box .big { font-size:2rem; font-weight:800; color:#1e3a5f; }
  .train-badge { background:#e8f5e9; border:1px solid #a5d6a7; border-radius:8px;
                 padding:0.4rem 0.8rem; font-size:0.82rem; color:#2e7d32;
                 margin-bottom:0.8rem; display:inline-block; }
  .model-card { background:white; border-radius:12px; padding:1.2rem;
                box-shadow:0 2px 10px rgba(0,0,0,0.08); margin-bottom:1rem; }
  .model-card h4 { margin:0 0 0.8rem 0; color:#1e3a5f; font-size:1.05rem; }
  .metric-pill { display:inline-block; background:#f0f4f8; border-radius:20px;
                 padding:0.2rem 0.7rem; margin:0.15rem; font-size:0.82rem;
                 font-weight:600; color:#1e3a5f; }
  .consensus-box { border-radius:10px; padding:0.9rem 1.2rem; margin-bottom:0.8rem;
                   font-weight:bold; font-size:1rem; }
  .eligibility-hero { text-align:center; padding:1.5rem 1rem; }
  .eligibility-hero .score-big { font-size:3.5rem; font-weight:900; }
  .eligibility-hero .score-lbl { font-size:0.9rem; color:#666; margin-top:0.2rem; }
  .timeline-step { display:flex; align-items:center; gap:0.6rem;
                   padding:0.4rem 0; font-size:0.88rem; }
  .ts-dot { width:22px; height:22px; border-radius:50%; display:flex;
            align-items:center; justify-content:center; font-size:0.7rem;
            font-weight:bold; flex-shrink:0; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────

def decision_badge(decision):
    if "Premium"     in decision: return f'<span class="badge badge-green">✅ {decision}</span>'
    if "Standard"    in decision: return f'<span class="badge badge-blue">✅ {decision}</span>'
    if "Basic"       in decision: return f'<span class="badge badge-blue">✅ {decision}</span>'
    if "Conditional" in decision: return f'<span class="badge badge-yellow">⚠️ {decision}</span>'
    return f'<span class="badge badge-red">❌ {decision}</span>'

def calc_emi(principal, annual_rate_pct, tenure_years):
    if annual_rate_pct is None or annual_rate_pct == 0:
        return principal / (tenure_years * 12) if tenure_years > 0 else 0
    r = (annual_rate_pct / 100) / 12
    n = tenure_years * 12
    return principal * r * (1 + r)**n / ((1 + r)**n - 1)

def eligibility_color(pct):
    if pct >= 75: return "#27ae60"
    if pct >= 55: return "#2e6da4"
    if pct >= 40: return "#f39c12"
    return "#e74c3c"

def pbar(val, mx, color):
    pct = int(min(val, mx) / mx * 100) if mx > 0 else 0
    return f'<div class="pbar-bg"><div class="pbar-fill" style="width:{pct}%;background:{color};"></div></div>'

def model_consensus(facts, models=("Random Forest","Logistic Regression","Decision Tree","Naive Bayes")):
    votes = []
    probs = []
    for m in models:
        r = ml_predict(facts, m)
        votes.append(r["prediction"])
        if r["approved_prob"] is not None:
            probs.append(r["approved_prob"])
    approved_count = sum(votes)
    avg_prob = round(np.mean(probs) * 100, 1) if probs else None
    return approved_count, len(models), avg_prob

def eligibility_score(facts):
    """0–100 composite eligibility, independent of hard rules."""
    credit_norm = np.clip((facts["credit_score"] - 300) / 600, 0, 1)
    income_norm = np.clip(facts["income"] / 120_000, 0, 1)
    lti         = facts.get("loan_to_income_ratio",
                             facts["loan_amount"] / facts["income"] if facts["income"] > 0 else 99)
    lti_norm    = np.clip(1 - lti / 20, 0, 1)
    assets      = facts.get("assets", 0)
    liabilities = facts.get("liabilities", 0)
    asset_norm  = np.clip((assets - liabilities) / max(assets, 1), 0, 1) if assets > 0 else 0
    score = (0.35 * credit_norm + 0.25 * income_norm +
             0.25 * lti_norm    + 0.15 * asset_norm)
    return round(score * 100, 1)


# ── Title + training status ───────────────────────────────────────
ml_state = load_state()

st.markdown("""
<div class="title-box">
  <h1>🏦 Loan Approval Expert System</h1>
  <p>Rule Engine · Fuzzy Logic · Anomaly Detection · ML (LR · DT · RF · NB) · XAI</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    f'<div class="train-badge">🤖 {get_training_badge(ml_state)}</div>',
    unsafe_allow_html=True
)

# ── Tabs ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🧾 Single Applicant",
    "📋 Merit List",
    "🤖 ML Training",
    "📊 ML Deep Dive",
])


# ════════════════════════════════════════════════════════════════════
#  TAB 1 — SINGLE APPLICANT
# ════════════════════════════════════════════════════════════════════
with tab1:

    # ── Sidebar ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 📋 Applicant Details")
        st.markdown("---")

        st.markdown("#### 👤 Personal")
        age        = st.number_input("Age", 1, 100, 30)
        gender     = st.selectbox("Gender", ["Male","Female","Other","Prefer not to say"])
        occupation = st.selectbox("Occupation", [
            "Salaried","Self-Employed","Business Owner",
            "Government Employee","Freelancer","Student","Other"
        ])

        st.markdown("#### 💰 Financial")
        income         = st.number_input("Monthly Income (₹)", 0, 500_000, 45_000, 1000)
        assets         = st.number_input("Total Assets (₹)",      0, 50_000_000, 500_000, 10_000)
        liabilities    = st.number_input("Total Liabilities (₹)", 0, 50_000_000, 100_000, 10_000)
        credit_score   = st.slider("Credit Score", 300, 900, 700)
        existing_loans = st.number_input("Existing Loans", 0, 20, 0)

        st.markdown("#### 🏠 Loan")
        loan_type   = st.selectbox("Loan Type", [
            "Home Loan","Personal Loan","Car Loan",
            "Education Loan","Business Loan","Gold Loan"
        ])
        loan_amount = st.number_input("Requested Loan (₹)", 0, 10_000_000, 200_000, 10_000)
        tenure_yrs  = st.slider("Tenure (years)", 1, 30, 10)

        lti_ratio = round(loan_amount / income, 2) if income > 0 else 0

        # Color hint
        cs_color = "green" if credit_score >= 700 else "orange" if credit_score >= 580 else "red"
        st.markdown(f"**LTI Ratio:** `{lti_ratio:.2f}×`")
        st.markdown(f"**Credit health:** :{cs_color}[{'Good ✅' if cs_color=='green' else 'Fair ⚠️' if cs_color=='orange' else 'Poor ❌'}]")

        st.markdown("---")
        ml_model_choice = st.selectbox(
            "Primary ML Model",
            ["Random Forest","Logistic Regression","Decision Tree","Naive Bayes"]
        )
        analyze = st.button("🔍 Analyze Application", use_container_width=True, type="primary")

    facts = {
        "age": age, "income": income, "credit_score": credit_score,
        "existing_loans": existing_loans, "loan_amount": loan_amount,
        "assets": assets, "liabilities": liabilities,
        "loan_to_income_ratio": lti_ratio,
        "gender": gender, "occupation": occupation, "loan_type": loan_type,
    }

    if analyze:
        with st.spinner("Running full analysis pipeline…"):
            anomaly                                      = check_anomaly(facts)
            fuzzy_score, fuzzy_label                     = get_fuzzy_approval(credit_score, income, existing_loans)
            decision, rule_id, interest_rate, rule_label = forward_chaining(facts)
            risk                                         = calculate_risk_score(facts)
            risk_label                                   = "Low" if risk < 30 else "Medium" if risk < 60 else "High"
            suggestions                                  = suggest_improvements(facts, decision)
            explanation_text                             = explain(rule_id)
            ml_result                                    = ml_predict(facts, ml_model_choice)
            xai_lines                                    = get_xai_explanation(facts, ml_model_choice)
            approved_votes, total_models, avg_prob       = model_consensus(facts)
            elig                                         = eligibility_score(facts)
        st.session_state["last_result"] = {
            "anomaly": anomaly, "fuzzy_score": fuzzy_score, "fuzzy_label": fuzzy_label,
            "decision": decision, "rule_id": rule_id, "interest_rate": interest_rate,
            "rule_label": rule_label, "risk": risk, "risk_label": risk_label,
            "suggestions": suggestions, "explanation_text": explanation_text,
            "ml_result": ml_result, "xai_lines": xai_lines,
            "approved_votes": approved_votes, "total_models": total_models,
            "avg_prob": avg_prob, "elig": elig,
            "credit_score": credit_score, "income": income, "loan_amount": loan_amount,
        }

    if "last_result" not in st.session_state:
        st.markdown("""
        <div class="card" style="text-align:center;padding:3rem;">
          <h2 style="color:#2e6da4;">👈 Fill in the details and click Analyze</h2>
          <p style="color:#888;font-size:1rem;">
            Runs Anomaly Detection → Fuzzy Logic → Rule Engine → ML Consensus → XAI
          </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        r              = st.session_state["last_result"]
        anomaly        = r["anomaly"]
        fuzzy_score    = r["fuzzy_score"]
        fuzzy_label    = r["fuzzy_label"]
        decision       = r["decision"]
        rule_id        = r["rule_id"]
        interest_rate  = r["interest_rate"]
        rule_label     = r["rule_label"]
        risk           = r["risk"]
        risk_label     = r["risk_label"]
        suggestions    = r["suggestions"]
        explanation_text = r["explanation_text"]
        ml_result      = r["ml_result"]
        xai_lines      = r["xai_lines"]
        approved_votes = r["approved_votes"]
        total_models   = r["total_models"]
        avg_prob       = r["avg_prob"]
        elig           = r["elig"]
        # Use saved values for What-If base (not live sidebar values, which reset on slider drag)
        base_credit    = r["credit_score"]
        base_income    = r["income"]
        base_loan      = r["loan_amount"]

        # ── Verdict banner ────────────────────────────────────────
        elig_color  = eligibility_color(elig)
        cons_color  = "#27ae60" if approved_votes >= 3 else "#f39c12" if approved_votes == 2 else "#e74c3c"
        r_color     = "#27ae60" if risk < 30 else "#f39c12" if risk < 60 else "#e74c3c"
        rate_str    = f"{interest_rate}% p.a." if interest_rate else "N/A"
        is_approved = "Approved" in decision

        verdict_bg    = "#e8f5e9" if is_approved else "#fdecea"
        verdict_border= "#27ae60" if is_approved else "#e74c3c"
        verdict_icon  = "✅" if is_approved else "❌"

        st.markdown(f"""
        <div style="background:{verdict_bg};border-left:6px solid {verdict_border};
                    border-radius:10px;padding:1.2rem 1.6rem;margin-bottom:1.2rem;
                    display:flex;align-items:center;gap:1.5rem;flex-wrap:wrap;">
          <div style="font-size:2.2rem;">{verdict_icon}</div>
          <div style="flex:1;">
            <div style="font-size:1.4rem;font-weight:800;color:{verdict_border};">{decision}</div>
            <div style="font-size:0.88rem;color:#555;margin-top:0.2rem;">
              Rule: <code>{rule_id}</code> · {rule_label}
              {f'· Interest Rate: <b>{rate_str}</b>' if interest_rate else ''}
            </div>
          </div>
          <div style="text-align:center;padding:0 1rem;">
            <div style="font-size:0.75rem;color:#888;text-transform:uppercase;letter-spacing:1px;">Eligibility</div>
            <div style="font-size:2rem;font-weight:900;color:{elig_color};">{elig}%</div>
          </div>
          <div style="text-align:center;padding:0 1rem;">
            <div style="font-size:0.75rem;color:#888;text-transform:uppercase;letter-spacing:1px;">ML Consensus</div>
            <div style="font-size:2rem;font-weight:900;color:{cons_color};">{approved_votes}/{total_models}</div>
          </div>
          <div style="text-align:center;padding:0 1rem;">
            <div style="font-size:0.75rem;color:#888;text-transform:uppercase;letter-spacing:1px;">Risk Score</div>
            <div style="font-size:2rem;font-weight:900;color:{r_color};">{risk}/100</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── 4 key metrics ─────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Credit Score",   base_credit,
                  delta="Good" if base_credit >= 700 else "Fair" if base_credit >= 580 else "Poor")
        m2.metric("Monthly Income", f"₹{base_income:,}")
        m3.metric("Fuzzy Approval", f"{fuzzy_score}%", delta=fuzzy_label)
        m4.metric("ML Probability",
                  f"{ml_result['approved_prob']*100:.1f}%" if ml_result['approved_prob'] else "N/A",
                  delta="Approved" if ml_result['prediction'] == 1 else "Rejected")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Analysis detail tabs ──────────────────────────────────
        dtab1, dtab2, dtab3, dtab4 = st.tabs([
            "⚖️ Rule Engine", "🤖 ML Models", "🌀 Fuzzy & Anomaly", "🔬 XAI & Tips"
        ])

        # ── Tab: Rule Engine ──────────────────────────────────────
        with dtab1:
            rc1, rc2 = st.columns([1, 1])
            with rc1:
                st.markdown(f"""
                <div class="card">
                  <div class="step-header">Decision</div>
                  <div style="margin:0.6rem 0;">{decision_badge(decision)}</div>
                  <div style="margin-top:0.5rem;font-size:0.9rem;">
                    <b>Rule fired:</b> <code>{rule_id}</code><br>
                    <b>Category:</b> {rule_label}<br>
                    <b>Interest Rate:</b> <span style="color:#27ae60;font-weight:700;">{rate_str}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            with rc2:
                st.markdown(f"""
                <div class="card">
                  <div class="step-header">Explanation</div>
                  <p style="color:#333;line-height:1.7;font-size:0.92rem;margin-top:0.5rem;">{explanation_text}</p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('<div class="card"><div class="step-header">Improvement Tips</div>', unsafe_allow_html=True)
            if suggestions:
                for s in suggestions:
                    st.markdown(f'<div class="sugg-item">→ {s}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="color:#155724;">✅ Profile meets all strong criteria.</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Tab: ML Models ────────────────────────────────────────
        with dtab2:
            ml_color  = "#27ae60" if ml_result["prediction"] == 1 else "#e74c3c"
            ml_lbl    = "Approved" if ml_result["prediction"] == 1 else "Rejected"
            prob_str2 = f"{ml_result['approved_prob']*100:.1f}%" if ml_result["approved_prob"] else "N/A"
            cons_text = ("Strong consensus: Approved" if approved_votes >= 3
                         else "Split decision — borderline case" if approved_votes == 2
                         else "Strong consensus: Rejected")
            prob_str  = f" · avg probability {avg_prob}%" if avg_prob else ""

            st.markdown(f"""
            <div class="card">
              <div class="step-header">ML Consensus — {approved_votes}/{total_models} models approve</div>
              <div class="consensus-box" style="background:{cons_color}22;border:2px solid {cons_color};
                           color:{cons_color};margin:0.6rem 0;">{cons_text}{prob_str}</div>
              <div style="font-size:0.88rem;color:#555;">
                Primary model (<b>{ml_model_choice}</b>):
                <span style="color:{ml_color};font-weight:700;"> {ml_lbl}</span>
                {f'· probability {prob_str2}' if prob_str2 != "N/A" else ""}
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Per-model mini cards
            model_results_grid = st.columns(4)
            for idx, mname in enumerate(["Random Forest","Logistic Regression","Decision Tree","Naive Bayes"]):
                mr = ml_predict(facts, mname)
                mc = "#27ae60" if mr["prediction"] == 1 else "#e74c3c"
                ml = "✅ Approved" if mr["prediction"] == 1 else "❌ Rejected"
                mp = f"{mr['approved_prob']*100:.0f}%" if mr["approved_prob"] else "—"
                with model_results_grid[idx]:
                    st.markdown(f"""
                    <div style="background:white;border-radius:10px;padding:0.9rem;
                                border-top:3px solid {mc};box-shadow:0 2px 8px rgba(0,0,0,0.07);
                                text-align:center;margin-bottom:0.5rem;">
                      <div style="font-size:0.75rem;color:#888;font-weight:600;margin-bottom:0.3rem;">
                        {mname}</div>
                      <div style="font-size:1rem;font-weight:800;color:{mc};">{ml}</div>
                      <div style="font-size:0.82rem;color:#555;margin-top:0.2rem;">prob: {mp}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # ── Tab: Fuzzy & Anomaly ──────────────────────────────────
        with dtab3:
            fa1, fa2 = st.columns(2)
            with fa1:
                f_color = "#27ae60" if fuzzy_label == "High" else "#f39c12" if fuzzy_label == "Moderate" else "#e74c3c"
                st.markdown(f"""
                <div class="card" style="text-align:center;">
                  <div class="step-header">Fuzzy Logic — Soft Assessment</div>
                  <div style="font-size:3rem;font-weight:900;color:{f_color};margin:0.5rem 0;">{fuzzy_score}%</div>
                  <div style="font-weight:bold;color:{f_color};margin-bottom:1rem;">{fuzzy_label} Approval Chance</div>
                </div>
                """, unsafe_allow_html=True)
                rows_f = [
                    ("Credit Score",        base_credit - 300, 600,
                     "#27ae60" if base_credit >= 700 else "#f39c12" if base_credit >= 580 else "#e74c3c"),
                    ("Income",              min(base_income, 200_000), 200_000,
                     "#27ae60" if base_income >= 60_000 else "#f39c12" if base_income >= 20_000 else "#e74c3c"),
                    ("Loan Burden (↓good)", max(0, 5 - existing_loans), 5,
                     "#27ae60" if existing_loans <= 1 else "#e74c3c"),
                    ("Asset Health",        max(0, assets - liabilities), max(assets, 1),
                     "#27ae60" if liabilities <= assets * 0.4 else "#f39c12" if liabilities <= assets * 0.6 else "#e74c3c"),
                ]
                st.markdown('<div class="card"><div class="step-header">Feature Memberships</div>', unsafe_allow_html=True)
                for fname, fval, fmx, fclr in rows_f:
                    fpct = int(fval / fmx * 100) if fmx > 0 else 0
                    st.markdown(f"""
                    <div style="margin-bottom:0.7rem;">
                      <span style="font-size:0.83rem;color:#555;">{fname}</span>
                      <span style="float:right;font-size:0.8rem;color:{fclr};font-weight:600;">{fpct}%</span>
                      {pbar(fval, fmx, fclr)}
                    </div>""", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with fa2:
                s_bg   = "#f8d7da" if anomaly["is_anomaly"] else "#d4edda"
                s_fg   = "#721c24" if anomaly["is_anomaly"] else "#155724"
                s_icon = "⚠️ Suspicious Profile" if anomaly["is_anomaly"] else "✅ Profile Normal"
                st.markdown(f"""
                <div class="card">
                  <div class="step-header">Anomaly Detection — Isolation Forest</div>
                  <div style="background:{s_bg};color:{s_fg};padding:0.6rem 1rem;border-radius:8px;
                              font-weight:bold;margin:0.6rem 0;">{s_icon}</div>
                  <div style="font-size:0.88rem;color:#555;">
                    Score: <b style="color:#1e3a5f;">{anomaly['score']}</b>
                    <span style="color:#aaa;"> (more negative = more unusual)</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('<div class="card"><div class="step-header">Profile Flags</div>', unsafe_allow_html=True)
                if anomaly["flags"]:
                    for fl in anomaly["flags"]:
                        st.markdown(f'<div class="flag-item">⚠️ {fl}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<p style="color:#155724;margin-top:0.5rem;">✅ No suspicious patterns detected.</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # ── Tab: XAI & Tips ──────────────────────────────────────
        with dtab4:
            st.markdown('<div class="card"><div class="step-header">Why did the ML model decide this?</div>', unsafe_allow_html=True)
            for line in xai_lines:
                st.markdown(f'<div class="xai-item">🧠 {line}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.caption(f"Explanation via: {ml_model_choice} · Go to ML Deep Dive tab for full feature importance charts.")

        st.markdown("---")

        # ── EMI Calculator ────────────────────────────────────────
        emi_col, whatif_col = st.columns(2)
        with emi_col:
            st.markdown('<div class="card"><h3>📅 EMI Calculator</h3>', unsafe_allow_html=True)
            rate_for_emi = interest_rate if interest_rate else 12.0
            emi_val = calc_emi(loan_amount, rate_for_emi, tenure_yrs)
            total_payable = emi_val * tenure_yrs * 12
            total_interest = total_payable - loan_amount
            dti_after = round((emi_val / income * 100), 1) if income > 0 else 0
            dti_color = "#27ae60" if dti_after <= 30 else "#f39c12" if dti_after <= 50 else "#e74c3c"
            dti_label = "Healthy" if dti_after <= 30 else "Moderate" if dti_after <= 50 else "High — risky"
            st.markdown(f"""
            <div class="emi-box">
              <div style="font-size:0.8rem;color:#555;margin-bottom:0.2rem;">Monthly EMI @ {rate_for_emi}% for {tenure_yrs} yrs</div>
              <div class="big">₹{emi_val:,.0f}</div>
            </div>
            <div style="margin-top:0.7rem;display:flex;gap:1.5rem;flex-wrap:wrap;">
              <div><b>Total Payable:</b><br>₹{total_payable:,.0f}</div>
              <div><b>Total Interest:</b><br>₹{total_interest:,.0f}</div>
              <div><b>Debt-to-Income:</b><br>
                <span style="color:{dti_color};font-weight:700;">{dti_after}% — {dti_label}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── What-If Simulator ────────────────────────────────────
        with whatif_col:
            st.markdown('<div class="card"><h3>🧪 What-If Simulator</h3>', unsafe_allow_html=True)
            st.caption("Adjust values to see how the decision would change:")
            wi_credit  = st.slider("Credit Score",      300, 900, base_credit,  key="wi_cs")
            wi_income  = st.slider("Monthly Income (₹)", 10_000, 300_000, base_income, 5000, key="wi_inc")
            wi_loan    = st.slider("Loan Amount (₹)",    50_000, 5_000_000, base_loan, 50_000, key="wi_loan")

            wi_facts = {**facts, "credit_score": wi_credit, "income": wi_income,
                        "loan_amount": wi_loan,
                        "loan_to_income_ratio": round(wi_loan / wi_income, 2) if wi_income > 0 else 0}
            wi_dec, wi_rid, wi_rate, _ = forward_chaining(wi_facts)
            wi_ml   = ml_predict(wi_facts, ml_model_choice)
            wi_elig = eligibility_score(wi_facts)
            wi_ec   = eligibility_color(wi_elig)

            changed = (wi_credit != base_credit or wi_income != base_income or wi_loan != base_loan)
            if changed:
                st.markdown(f"""
                <div style="margin-top:0.6rem;padding:0.7rem 1rem;border-radius:8px;
                     background:{'#d4edda' if 'Approved' in wi_dec else '#f8d7da'};
                     color:{'#155724' if 'Approved' in wi_dec else '#721c24'};font-weight:bold;">
                  {decision_badge(wi_dec)} &nbsp; Eligibility: <span style="color:{wi_ec}">{wi_elig}%</span>
                </div>
                """, unsafe_allow_html=True)
                if wi_rate:
                    wi_emi = calc_emi(wi_loan, wi_rate, tenure_yrs)
                    st.markdown(f"**New EMI:** ₹{wi_emi:,.0f}/month @ {wi_rate}%")
            else:
                st.info("Move any slider above to simulate a different scenario.")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Save to comparison ────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 💾 Save to Comparison (up to 3 profiles)")
        if "comparison_profiles" not in st.session_state:
            st.session_state.comparison_profiles = []

        save_label = st.text_input("Label for this profile (e.g. 'Applicant A')", value=f"Profile {len(st.session_state.comparison_profiles)+1}")
        if st.button("➕ Save Profile"):
            if len(st.session_state.comparison_profiles) >= 3:
                st.warning("Max 3 profiles. Clear one first.")
            else:
                st.session_state.comparison_profiles.append({
                    "label": save_label, "decision": decision, "elig": elig,
                    "credit": credit_score, "income": income, "loan": loan_amount,
                    "risk": risk, "fuzzy": fuzzy_score, "tier": decision,
                    "ml_prob": ml_result["approved_prob"],
                })
                st.success(f"Saved '{save_label}'")

        if st.session_state.comparison_profiles:
            comp_df = pd.DataFrame(st.session_state.comparison_profiles)
            st.dataframe(comp_df[["label","decision","elig","credit","income","loan","risk","fuzzy"]],
                         use_container_width=True, hide_index=True)
            if st.button("🗑 Clear All Profiles"):
                st.session_state.comparison_profiles = []
                st.rerun()


# ════════════════════════════════════════════════════════════════════
#  TAB 2 — MERIT LIST
# ════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📋 Batch Merit List — Upload Applicants")

    ml_state2 = load_state()
    if ml_state2["training_count"] > 0:
        st.markdown(
            f'<div class="train-badge">🤖 Using Model v{ml_state2["training_count"]} '
            f'({ml_state2["best_model"]}) — trained {ml_state2["last_trained"]}</div>',
            unsafe_allow_html=True
        )
    else:
        st.warning("⚠️ No trained model yet. Go to **ML Training** tab and train first for best results. Using rule engine only for now.")

    uploaded = st.file_uploader("Upload applicant Excel/CSV", type=["xlsx","csv"], key="merit_upload")
    if not uploaded:
        st.info("Upload a file with columns: name, age, income, credit_score, existing_loans, loan_amount (+ optional: gender, occupation, loan_type, assets, liabilities)")
    else:
        raw_df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
        st.success(f"Loaded {len(raw_df)} applicants")

        with st.spinner("Running merit list agent…"):
            result_df = run_utility_agent(raw_df)
            stats     = get_summary_stats(result_df)

        # ── Summary metrics ───────────────────────────────────────
        s1,s2,s3,s4,s5 = st.columns(5)
        s1.metric("Total",          stats["total"])
        s2.metric("Approved",       stats["approved"], delta=f"{stats['approved']/stats['total']*100:.0f}%")
        s3.metric("Rejected",       stats["rejected"])
        s4.metric("Anomalies",      stats["anomalies"])
        s5.metric("Avg Utility",    stats["avg_utility"])

        st.markdown("---")

        # ── Visual Dashboard ──────────────────────────────────────
        st.subheader("📊 Portfolio Dashboard")
        d1, d2 = st.columns(2)

        with d1:
            # Tier distribution
            tier_counts = result_df["tier"].value_counts().reset_index()
            tier_counts.columns = ["Tier","Count"]
            tier_order = ["Tier 1 – Premium","Tier 2 – Standard","Tier 3 – Basic",
                          "Tier 4 – Conditional","Tier 5 – Rejected"]
            tier_counts["Tier"] = pd.Categorical(tier_counts["Tier"], categories=tier_order, ordered=True)
            tier_counts = tier_counts.sort_values("Tier")
            colors_map = {
                "Tier 1 – Premium":     "#27ae60",
                "Tier 2 – Standard":    "#2e6da4",
                "Tier 3 – Basic":       "#17a2b8",
                "Tier 4 – Conditional": "#f39c12",
                "Tier 5 – Rejected":    "#e74c3c",
            }
            bars_html = ""
            total_c = tier_counts["Count"].sum()
            for _, row in tier_counts.iterrows():
                pct3 = int(row["Count"] / total_c * 100) if total_c > 0 else 0
                c = colors_map.get(row["Tier"], "#aaa")
                bars_html += f"""
                <div style="margin-bottom:0.7rem;">
                  <div style="display:flex;justify-content:space-between;font-size:0.85rem;">
                    <span style="color:#333;">{row['Tier']}</span>
                    <span style="color:{c};font-weight:700;">{row['Count']} ({pct3}%)</span>
                  </div>
                  <div class="pbar-bg"><div class="pbar-fill" style="width:{pct3}%;background:{c};"></div></div>
                </div>"""
            st.markdown(f'<div class="card"><h3>🎯 Tier Distribution</h3>{bars_html}</div>',
                        unsafe_allow_html=True)

        with d2:
            # Avg credit by tier
            avg_credit = result_df.groupby("tier")["credit_score"].mean().reset_index()
            avg_credit.columns = ["Tier","Avg Credit Score"]
            st.markdown('<div class="card"><h3>💳 Avg Credit Score by Tier</h3>', unsafe_allow_html=True)
            for _, row in avg_credit.iterrows():
                c = colors_map.get(row["Tier"], "#aaa")
                st.markdown(f"""
                <div style="margin-bottom:0.5rem;display:flex;justify-content:space-between;align-items:center;">
                  <span style="font-size:0.83rem;color:#555;flex:1;">{row['Tier']}</span>
                  <span style="font-weight:700;color:{c};margin:0 0.5rem;">{row['Avg Credit Score']:.0f}</span>
                  <div style="flex:2;">{pbar(row['Avg Credit Score']-300, 600, c)}</div>
                </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Scatter: income vs loan amount
        st.markdown('<div class="card"><h3>📈 Income vs Loan Amount</h3>', unsafe_allow_html=True)
        scatter_df = result_df[["applicant_name","income","loan_amount","tier","utility_score"]].copy()
        scatter_df["color"] = scatter_df["tier"].map(colors_map).fillna("#aaa")
        try:
            import plotly.express as px
            fig = px.scatter(
                scatter_df, x="income", y="loan_amount",
                color="tier", color_discrete_map=colors_map,
                hover_name="applicant_name",
                hover_data={"utility_score": True, "income": ":,.0f", "loan_amount": ":,.0f"},
                labels={"income": "Monthly Income (₹)", "loan_amount": "Loan Amount (₹)"},
                height=380,
            )
            fig.update_layout(
                paper_bgcolor="white", plot_bgcolor="#f8f9fa",
                margin=dict(l=20,r=20,t=20,b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.dataframe(scatter_df[["applicant_name","income","loan_amount","tier"]],
                         use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Filter + Table ────────────────────────────────────────
        st.markdown("---")
        st.subheader("🗂 Applicant List")
        filter_tier = st.multiselect(
            "Filter by Tier",
            options=result_df["tier"].unique().tolist(),
            default=result_df["tier"].unique().tolist()
        )
        show_anomaly = st.checkbox("Show only anomalous applicants", False)
        filtered_df = result_df[result_df["tier"].isin(filter_tier)]
        if show_anomaly:
            filtered_df = filtered_df[filtered_df["is_anomaly"] == True]

        st.dataframe(filtered_df[[
            "merit_rank","applicant_name","tier","utility_score","rule_decision",
            "credit_score","income","loan_amount","risk_score","fuzzy_score","is_anomaly"
        ]], use_container_width=True, hide_index=True)

        # ── Download ──────────────────────────────────────────────
        from io import BytesIO
        buf = BytesIO()
        filtered_df.to_excel(buf, index=False)
        st.download_button(
            "⬇️ Download Filtered Results (Excel)",
            data=buf.getvalue(),
            file_name="merit_list_filtered.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


# ════════════════════════════════════════════════════════════════════
#  TAB 3 — ML TRAINING
# ════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🤖 ML Model Training")

    ml_state3 = load_state()
    if ml_state3["training_count"] > 0:
        st.markdown(f"""
        <div class="card" style="background:linear-gradient(135deg,#e8f5e9,#f1f8e9);">
          <h3 style="color:#2e7d32;">📦 Current Model: v{ml_state3['training_count']}</h3>
          <div style="display:flex;gap:2rem;flex-wrap:wrap;font-size:0.9rem;">
            <div><b>Best Model:</b> {ml_state3['best_model']}</div>
            <div><b>Trained:</b> {ml_state3['last_trained']}</div>
            <div><b>Samples:</b> {ml_state3['n_samples']}</div>
            <div><b>Data Source:</b> {ml_state3['data_source']}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No model trained yet. Upload data or use synthetic data below.")

    # Training history
    if ml_state3["history"]:
        st.markdown("#### 📜 Training History")
        hist_df = pd.DataFrame(ml_state3["history"])
        st.dataframe(hist_df[["run","timestamp","best_model","data_source","n_samples","best_f1"]],
                     use_container_width=True, hide_index=True)

    st.markdown("---")

    # Upload training data
    uploaded_train = st.file_uploader(
        "Upload historical loan data (CSV/Excel) — must include `loan_approved` column",
        type=["csv","xlsx"], key="train_upload"
    )
    train_df = None
    data_src = "Synthetic"
    if uploaded_train:
        train_df = pd.read_csv(uploaded_train) if uploaded_train.name.endswith(".csv") \
                   else pd.read_excel(uploaded_train)
        data_src = uploaded_train.name
        st.success(f"Loaded {len(train_df)} records from {uploaded_train.name}")
        st.dataframe(train_df.head(5), use_container_width=True)

    col_btn, col_info = st.columns([1,3])
    with col_btn:
        train_btn = st.button("🚀 Train All 4 Models", type="primary", use_container_width=True)
    with col_info:
        st.info("Trains Logistic Regression, Decision Tree, Random Forest & Naive Bayes. Results instantly reflected in Single Applicant and Merit List tabs.")

    if train_btn:
        n_samples = len(train_df) if train_df is not None else 1000
        with st.spinner("Training all 4 models…"):
            train_result = train_and_evaluate(df=train_df)
            ml_state3    = save_state(train_result, data_src, n_samples)

        st.success(f"✅ Training complete! Best model by F1: **{train_result['best_model']}**")
        st.balloons()

        # Quick metrics table
        metrics_rows = []
        for mname, m in train_result["results"].items():
            metrics_rows.append({
                "Model":     mname,
                "Accuracy":  f"{m['accuracy']*100:.1f}%",
                "Precision": f"{m['precision']*100:.1f}%",
                "Recall":    f"{m['recall']*100:.1f}%",
                "F1 Score":  f"{m['f1']*100:.1f}%",
                "ROC-AUC":   f"{m['roc_auc']*100:.1f}%" if m['roc_auc'] else "N/A",
            })
        st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True, hide_index=True)
        st.info("👉 Go to **ML Deep Dive** tab for per-model confusion matrices, charts & XAI.")


# ════════════════════════════════════════════════════════════════════
#  TAB 4 — ML DEEP DIVE
# ════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📊 ML Deep Dive — Per-Model Analysis")

    ml_state4 = load_state()
    if not ml_state4["results"]:
        st.warning("No trained models found. Go to **ML Training** tab and train first.")
    else:
        MODEL_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest", "Naive Bayes"]
        MODEL_COLORS = {
            "Logistic Regression": "#2e6da4",
            "Decision Tree":       "#27ae60",
            "Random Forest":       "#8e44ad",
            "Naive Bayes":         "#e67e22",
        }
        MODEL_XAI_DESC = {
            "Logistic Regression": (
                "Logistic Regression is a linear model that learns a weighted combination of features. "
                "It draws a straight decision boundary — simple, fast, and highly interpretable. "
                "Feature importance is measured by the absolute value of each coefficient after scaling. "
                "It works best when relationships between features and outcome are roughly linear."
            ),
            "Decision Tree": (
                "A Decision Tree splits data by asking yes/no questions on features, building a tree of rules. "
                "Every prediction follows a path from root to leaf. It's fully transparent — you can trace exactly "
                "why any applicant was approved or rejected. Prone to overfitting on small datasets. "
                "Feature importance = total impurity reduction each feature provides across all splits."
            ),
            "Random Forest": (
                "Random Forest trains 100 independent decision trees on random subsets of data and features, "
                "then takes a majority vote. This 'wisdom of crowds' approach makes it robust and hard to overfit. "
                "It is generally the most accurate model here. Feature importance is averaged across all 100 trees."
            ),
            "Naive Bayes": (
                "Naive Bayes applies Bayes' theorem, assuming all features are independent of each other "
                "(the 'naive' assumption). Despite this simplification, it often performs surprisingly well. "
                "It is extremely fast and works well with limited data. It does not produce feature importances — "
                "it instead models each feature's probability distribution per class."
            ),
        }

        # ── Sliding model cards ───────────────────────────────────
        selected_model = st.radio(
            "Select model to inspect:",
            MODEL_NAMES,
            horizontal=True,
        )

        m = ml_state4["results"].get(selected_model, {})
        color = MODEL_COLORS[selected_model]

        if not m:
            st.warning(f"No results for {selected_model}. Retrain to include this model.")
        else:
            st.markdown(f"""
            <div class="model-card" style="border-top:4px solid {color};">
              <h4 style="color:{color};">🤖 {selected_model}</h4>
              <div>
                <span class="metric-pill">Accuracy: {m['accuracy']*100:.1f}%</span>
                <span class="metric-pill">Precision: {m['precision']*100:.1f}%</span>
                <span class="metric-pill">Recall: {m['recall']*100:.1f}%</span>
                <span class="metric-pill">F1 Score: {m['f1']*100:.1f}%</span>
                <span class="metric-pill">ROC-AUC: {m['roc_auc']*100:.1f}% if m['roc_auc'] else 'N/A'</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            left, right = st.columns(2)

            # ── Metric gauges ─────────────────────────────────────
            with left:
                st.markdown('<div class="card"><h3>📈 Evaluation Metrics</h3>', unsafe_allow_html=True)
                metric_items = [
                    ("Accuracy",  m["accuracy"],  "How often the model is correct overall"),
                    ("Precision", m["precision"], "Of those predicted Approved, how many actually were"),
                    ("Recall",    m["recall"],    "Of actual Approved, how many did it catch"),
                    ("F1 Score",  m["f1"],        "Balance of Precision and Recall"),
                ]
                if m["roc_auc"]:
                    metric_items.append(("ROC-AUC", m["roc_auc"], "Ability to rank approved above rejected"))

                for metric_name, val, tip in metric_items:
                    bar_c = "#27ae60" if val >= 0.85 else "#f39c12" if val >= 0.70 else "#e74c3c"
                    st.markdown(f"""
                    <div style="margin-bottom:0.8rem;">
                      <div style="display:flex;justify-content:space-between;">
                        <span style="font-weight:600;color:#1e3a5f;">{metric_name}</span>
                        <span style="color:{bar_c};font-weight:800;">{val*100:.1f}%</span>
                      </div>
                      <div style="font-size:0.75rem;color:#888;margin-bottom:0.2rem;">{tip}</div>
                      {pbar(val*100, 100, bar_c)}
                    </div>""", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # ── Confusion Matrix ──────────────────────────────────
            with right:
                st.markdown('<div class="card"><h3>🔲 Confusion Matrix</h3>', unsafe_allow_html=True)
                cm = m.get("confusion_matrix", [[0,0],[0,0]])
                tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
                total_cm = tn + fp + fn + tp

                cm_html = f"""
                <table style="width:100%;border-collapse:collapse;font-size:0.9rem;text-align:center;margin-top:0.5rem;">
                  <tr>
                    <th style="background:#f0f4f8;padding:0.5rem;"></th>
                    <th style="background:#f0f4f8;padding:0.5rem;color:#e74c3c;">Predicted: Rejected</th>
                    <th style="background:#f0f4f8;padding:0.5rem;color:#27ae60;">Predicted: Approved</th>
                  </tr>
                  <tr>
                    <td style="background:#f0f4f8;font-weight:700;padding:0.5rem;color:#e74c3c;">Actual: Rejected</td>
                    <td style="background:#d4edda;padding:0.8rem;font-size:1.4rem;font-weight:800;color:#155724;">{tn}<br><small style="font-size:0.7rem;font-weight:400;">True Negative</small></td>
                    <td style="background:#f8d7da;padding:0.8rem;font-size:1.4rem;font-weight:800;color:#721c24;">{fp}<br><small style="font-size:0.7rem;font-weight:400;">False Positive</small></td>
                  </tr>
                  <tr>
                    <td style="background:#f0f4f8;font-weight:700;padding:0.5rem;color:#27ae60;">Actual: Approved</td>
                    <td style="background:#f8d7da;padding:0.8rem;font-size:1.4rem;font-weight:800;color:#721c24;">{fn}<br><small style="font-size:0.7rem;font-weight:400;">False Negative</small></td>
                    <td style="background:#d4edda;padding:0.8rem;font-size:1.4rem;font-weight:800;color:#155724;">{tp}<br><small style="font-size:0.7rem;font-weight:400;">True Positive</small></td>
                  </tr>
                </table>
                <div style="margin-top:0.7rem;font-size:0.82rem;color:#555;line-height:1.7;">
                  <b>TN</b> = correctly rejected &nbsp;|&nbsp; <b>FP</b> = wrongly approved (risk!) &nbsp;|&nbsp;
                  <b>FN</b> = wrongly rejected (lost business) &nbsp;|&nbsp; <b>TP</b> = correctly approved
                </div>
                """
                st.markdown(cm_html, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # ── Feature Importance ────────────────────────────────
            st.markdown('<div class="card"><h3>🔬 Feature Importance / Coefficient Weights</h3>',
                        unsafe_allow_html=True)
            importance = get_feature_importance(selected_model)
            if importance:
                max_imp = max(importance.values()) if importance else 1
                for feat, imp in importance.items():
                    pct4 = int(imp / max_imp * 100)
                    bar_c2 = color if pct4 >= 60 else "#2e6da4" if pct4 >= 30 else "#adb5bd"
                    st.markdown(f"""
                    <div style="margin-bottom:0.55rem;">
                      <div style="display:flex;justify-content:space-between;">
                        <span style="font-weight:600;color:#1e3a5f;font-size:0.88rem;">
                          {feat.replace('_',' ').title()}</span>
                        <span style="color:{bar_c2};font-weight:700;font-size:0.88rem;">{imp*100:.1f}%</span>
                      </div>
                      {pbar(imp*100, 100, bar_c2)}
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("Feature importance not available for this model type.")
            st.markdown('</div>', unsafe_allow_html=True)

            # ── XAI Explanation paragraph ─────────────────────────
            st.markdown(f"""
            <div class="card" style="border-left:4px solid {color};">
              <h3>🧠 How {selected_model} Works — XAI Explanation</h3>
              <p style="color:#333;line-height:1.8;font-size:0.93rem;">
                {MODEL_XAI_DESC[selected_model]}
              </p>
            </div>
            """, unsafe_allow_html=True)

            # ── All models comparison summary ─────────────────────
            st.markdown("---")
            st.subheader("⚡ All Models Side-by-Side")
            comp_rows = []
            for mn in MODEL_NAMES:
                mr = ml_state4["results"].get(mn, {})
                if mr:
                    comp_rows.append({
                        "Model":     mn,
                        "Accuracy":  f"{mr['accuracy']*100:.1f}%",
                        "Precision": f"{mr['precision']*100:.1f}%",
                        "Recall":    f"{mr['recall']*100:.1f}%",
                        "F1 Score":  f"{mr['f1']*100:.1f}%",
                        "ROC-AUC":   f"{mr['roc_auc']*100:.1f}%" if mr["roc_auc"] else "N/A",
                        "Best?":     "⭐" if mn == ml_state4["best_model"] else "",
                    })
            if comp_rows:
                st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)