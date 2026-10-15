"""
merit_list_page.py
==================
Standalone Streamlit page for the Utility-Based Agent Merit List.
Run with:  streamlit run merit_list_page.py
Or integrate into your existing app.py as a page.
"""

import io
import pandas as pd
import streamlit as st
from utility_agent import run_utility_agent, get_summary_stats

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Merit List – Utility Agent",
    page_icon="📊",
    layout="wide",
)

# ── Styles ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

.stApp {
    background: #0c0f1a;
    font-family: 'DM Sans', sans-serif;
    color: #e8eaf0;
}

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0f1729 0%, #1a2340 50%, #0f1729 100%);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 20px;
    padding: 3rem 3.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(99,179,237,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute; bottom: -40px; left: 40%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(154,230,180,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 3px;
    color: #63b3ed;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    opacity: 0.8;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    color: #fff;
    line-height: 1.15;
    margin-bottom: 0.6rem;
}
.hero h1 span { color: #63b3ed; }
.hero p {
    color: #8892a4;
    font-size: 0.95rem;
    max-width: 520px;
    line-height: 1.7;
}

/* ── Upload zone ── */
.upload-zone {
    background: #111827;
    border: 2px dashed rgba(99,179,237,0.25);
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    transition: border-color 0.3s;
}
.upload-zone:hover { border-color: rgba(99,179,237,0.5); }

/* ── Stat cards ── */
.stats-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
.stat-card {
    flex: 1; min-width: 130px;
    background: #111827;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.stat-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 14px 14px 0 0;
}
.stat-card.blue::before   { background: linear-gradient(90deg,#63b3ed,#4299e1); }
.stat-card.green::before  { background: linear-gradient(90deg,#68d391,#48bb78); }
.stat-card.red::before    { background: linear-gradient(90deg,#fc8181,#f56565); }
.stat-card.yellow::before { background: linear-gradient(90deg,#f6e05e,#ed8936); }
.stat-card.purple::before { background: linear-gradient(90deg,#b794f4,#9f7aea); }
.stat-card.teal::before   { background: linear-gradient(90deg,#81e6d9,#4fd1c5); }

.stat-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #4a5568;
    margin-bottom: 0.4rem;
}
.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #e2e8f0;
    line-height: 1;
}
.stat-sub {
    font-size: 0.75rem;
    color: #4a5568;
    margin-top: 0.3rem;
}

/* ── Tier pills ── */
.tier-pill {
    display: inline-block;
    padding: 0.2rem 0.75rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.5px;
    white-space: nowrap;
}
.tier-1  { background: rgba(104,211,145,0.15); color: #68d391; border: 1px solid rgba(104,211,145,0.3); }
.tier-2  { background: rgba(99,179,237,0.15);  color: #63b3ed; border: 1px solid rgba(99,179,237,0.3); }
.tier-3  { background: rgba(144,205,244,0.15); color: #90cdf4; border: 1px solid rgba(144,205,244,0.3); }
.tier-4  { background: rgba(246,224,94,0.15);  color: #f6e05e; border: 1px solid rgba(246,224,94,0.3); }
.tier-5  { background: rgba(252,129,129,0.15); color: #fc8181; border: 1px solid rgba(252,129,129,0.3); }

/* ── Merit table card ── */
.table-card {
    background: #111827;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}
.table-card h3 {
    font-family: 'Syne', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 1rem;
}

/* ── Section header ── */
.section-header {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #63b3ed;
    margin-bottom: 0.8rem;
    opacity: 0.7;
}

/* ── Tier bar ── */
.tier-bar-wrap { margin-bottom: 0.6rem; }
.tier-bar-label {
    display: flex; justify-content: space-between;
    font-size: 0.82rem; color: #8892a4; margin-bottom: 0.25rem;
}
.tier-bar-bg {
    background: #1e2535; border-radius: 6px; height: 10px; width: 100%;
}
.tier-bar-fill { height: 10px; border-radius: 6px; }

/* ── Utility score badge ── */
.u-score {
    font-family: 'DM Mono', monospace;
    font-size: 0.88rem;
    font-weight: 500;
    padding: 0.15rem 0.5rem;
    border-radius: 6px;
    background: rgba(99,179,237,0.1);
    color: #63b3ed;
    border: 1px solid rgba(99,179,237,0.2);
}

/* ── Anomaly badge ── */
.anom-yes { color: #fc8181; font-weight: 600; }
.anom-no  { color: #68d391; }

/* ── Download button ── */
.dl-btn {
    display: inline-block;
    background: linear-gradient(135deg, #2b4a7a, #1e3a5f);
    color: #63b3ed !important;
    border: 1px solid rgba(99,179,237,0.3);
    border-radius: 10px;
    padding: 0.6rem 1.4rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 1px;
    cursor: pointer;
    text-decoration: none;
}

/* ── Sample format box ── */
.format-box {
    background: #0d1117;
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #63b3ed;
    line-height: 1.8;
    margin-top: 1rem;
}

/* ── Streamlit overrides ── */
div[data-testid="stFileUploader"] {
    background: #111827 !important;
    border: 2px dashed rgba(99,179,237,0.3) !important;
    border-radius: 14px !important;
}
div[data-testid="stFileUploader"]:hover {
    border-color: rgba(99,179,237,0.6) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #1a3a6b, #2e6da4) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(99,179,237,0.25) !important;
}
.stDataFrame { border-radius: 12px; overflow: hidden; }
div[data-testid="metric-container"] {
    background: #111827 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
    padding: 0.8rem 1rem !important;
}
[data-testid="stSidebar"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-tag">📊 Utility-Based Agent · Batch Processing</div>
  <h1>Loan <span>Merit List</span><br>Generator</h1>
  <p>Upload an Excel sheet of applicants. The utility agent scores, ranks, and tiers
     every candidate using weighted multi-criteria analysis, fuzzy logic, rule-based
     inference, and anomaly detection.</p>
</div>
""", unsafe_allow_html=True)


# ── Sample download + format guide ───────────────────────────────────────────
col_info, col_sample = st.columns([3, 2])

with col_info:
    st.markdown('<div class="section-header">Required Excel Columns</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="format-box">
        name            → Applicant full name (optional)<br>
        age             → Integer  (1 – 100)<br>
        income          → Monthly income in ₹<br>
        credit_score    → Integer  (300 – 900)<br>
        existing_loans  → Number of active loans<br>
        loan_amount     → Requested loan in ₹
    </div>
    """, unsafe_allow_html=True)

with col_sample:
    st.markdown('<div class="section-header">Download Sample File</div>', unsafe_allow_html=True)
    sample_data = pd.DataFrame({
        "name":           ["Arjun Sharma", "Priya Nair", "Ramesh Gupta", "Sneha Pillai",
                           "Vikram Mehta", "Divya Krishnan", "Suresh Babu", "Anita Das"],
        "age":            [32, 27, 45, 38, 55, 29, 41, 34],
        "income":         [65000, 28000, 80000, 42000, 35000, 72000, 18000, 55000],
        "credit_score":   [760, 590, 820, 680, 640, 780, 510, 720],
        "existing_loans": [0, 2, 1, 1, 3, 0, 1, 0],
        "loan_amount":    [200000, 150000, 500000, 300000, 250000, 400000, 100000, 350000],
    })
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        sample_data.to_excel(writer, index=False, sheet_name="Applicants")
    st.download_button(
        label="⬇ Download sample.xlsx",
        data=buf.getvalue(),
        file_name="sample_applicants.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

st.markdown("---")

# ── File uploader ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Upload Applicant Sheet</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drop your Excel (.xlsx / .xls) or CSV file here",
    type=["xlsx", "xls", "csv"],
    label_visibility="collapsed",
)

if uploaded is None:
    st.markdown("""
    <div style="text-align:center;padding:3rem;color:#4a5568;font-family:'DM Mono',monospace;font-size:0.85rem;">
      ↑ Upload a file to generate the merit list
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Load & run agent ──────────────────────────────────────────────────────────
try:
    if uploaded.name.endswith(".csv"):
        raw_df = pd.read_csv(uploaded)
    else:
        raw_df = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

with st.spinner("🤖 Agent is computing utility scores…"):
    try:
        merit_df = run_utility_agent(raw_df)
    except ValueError as e:
        st.error(str(e))
        st.stop()

stats = get_summary_stats(merit_df)

# ── Summary stats ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Summary Statistics</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="stats-row">
  <div class="stat-card blue">
    <div class="stat-label">Total</div>
    <div class="stat-value">{stats['total']}</div>
    <div class="stat-sub">applicants processed</div>
  </div>
  <div class="stat-card green">
    <div class="stat-label">Approved</div>
    <div class="stat-value">{stats['approved']}</div>
    <div class="stat-sub">eligible candidates</div>
  </div>
  <div class="stat-card red">
    <div class="stat-label">Rejected</div>
    <div class="stat-value">{stats['rejected']}</div>
    <div class="stat-sub">did not qualify</div>
  </div>
  <div class="stat-card yellow">
    <div class="stat-label">Anomalies</div>
    <div class="stat-value">{stats['anomalies']}</div>
    <div class="stat-sub">flagged profiles</div>
  </div>
  <div class="stat-card purple">
    <div class="stat-label">Avg Utility</div>
    <div class="stat-value">{stats['avg_utility']}</div>
    <div class="stat-sub">out of 1.00</div>
  </div>
  <div class="stat-card teal">
    <div class="stat-label">Avg Credit</div>
    <div class="stat-value">{int(stats['avg_credit'])}</div>
    <div class="stat-sub">credit score</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Tier distribution ──────────────────────────────────────────────────────────
tier_config = {
    "Tier 1 – Premium":     ("#68d391", "tier-1"),
    "Tier 2 – Standard":    ("#63b3ed", "tier-2"),
    "Tier 3 – Basic":       ("#90cdf4", "tier-3"),
    "Tier 4 – Conditional": ("#f6e05e", "tier-4"),
    "Tier 5 – Rejected":    ("#fc8181", "tier-5"),
}

bar_col, table_col = st.columns([1, 2])
with bar_col:
    st.markdown('<div class="section-header">Tier Distribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="table-card">', unsafe_allow_html=True)
    for tier_name, (color, css_class) in tier_config.items():
        count = stats["tier_counts"].get(tier_name, 0)
        pct   = int(count / stats["total"] * 100) if stats["total"] else 0
        st.markdown(f"""
        <div class="tier-bar-wrap">
          <div class="tier-bar-label">
            <span><span class="tier-pill {css_class}">{tier_name}</span></span>
            <span style="font-family:'DM Mono',monospace;">{count} ({pct}%)</span>
          </div>
          <div class="tier-bar-bg">
            <div class="tier-bar-fill" style="width:{pct}%;background:{color};opacity:0.8;"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── Full merit table ───────────────────────────────────────────────────────────
with table_col:
    st.markdown('<div class="section-header">Merit List (ranked by utility score)</div>', unsafe_allow_html=True)

    # Filter controls
    tier_filter = st.multiselect(
        "Filter by Tier",
        options=list(tier_config.keys()),
        default=list(tier_config.keys()),
        label_visibility="collapsed",
    )

filtered_df = merit_df[merit_df["tier"].isin(tier_filter)].copy()

# Render styled table
display_cols = [
    "merit_rank", "applicant_name", "credit_score", "income",
    "loan_amount", "utility_score", "risk_score", "fuzzy_score",
    "tier", "is_anomaly", "rule_decision", "interest_rate",
    "strengths", "weaknesses",
]
display_df = filtered_df[display_cols].copy()

# Format columns
display_df["income"]       = display_df["income"].apply(lambda x: f"₹{x:,.0f}")
display_df["loan_amount"]  = display_df["loan_amount"].apply(lambda x: f"₹{x:,.0f}")
display_df["interest_rate"]= display_df["interest_rate"].apply(
    lambda x: f"{x}% p.a." if pd.notna(x) and x else "—"
)
display_df["is_anomaly"]   = display_df["is_anomaly"].apply(
    lambda x: "⚠️ Yes" if x else "✅ No"
)
display_df.rename(columns={
    "merit_rank":     "Rank",
    "applicant_name": "Name",
    "credit_score":   "Credit",
    "income":         "Income",
    "loan_amount":    "Loan Amt",
    "utility_score":  "Utility",
    "risk_score":     "Risk",
    "fuzzy_score":    "Fuzzy %",
    "tier":           "Tier",
    "is_anomaly":     "Anomaly",
    "rule_decision":  "Decision",
    "interest_rate":  "Rate",
    "strengths":      "Strengths",
    "weaknesses":     "Weaknesses",
}, inplace=True)

st.dataframe(
    display_df,
    use_container_width=True,
    height=480,
    hide_index=True,
)

# ── Download merit list ────────────────────────────────────────────────────────
st.markdown("---")
dl_col1, dl_col2, _ = st.columns([2, 2, 3])

with dl_col1:
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        merit_df.to_excel(writer, index=False, sheet_name="Merit List")
    st.download_button(
        label="⬇ Download Merit List (.xlsx)",
        data=excel_buf.getvalue(),
        file_name="merit_list.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

with dl_col2:
    csv_buf = merit_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download Merit List (.csv)",
        data=csv_buf,
        file_name="merit_list.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ── Applicant deep-dive ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">Applicant Deep-Dive</div>', unsafe_allow_html=True)

names = merit_df["applicant_name"].tolist()
selected = st.selectbox("Select an applicant to inspect", names, label_visibility="collapsed")

if selected:
    row = merit_df[merit_df["applicant_name"] == selected].iloc[0]
    tier_name = row["tier"]
    color, css_class = tier_config.get(tier_name, ("#8892a4", "tier-5"))
    u_pct = int(row["utility_score"] * 100)
    r_pct = int(row["risk_score"])

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Utility Score",  f"{row['utility_score']:.3f}")
    d2.metric("Risk Score",     f"{int(row['risk_score'])}/100")
    d3.metric("Fuzzy Score",    f"{row['fuzzy_score']}%")
    d4.metric("Merit Rank",     f"#{int(row['merit_rank'])}")

    info_l, info_r = st.columns(2)
    with info_l:
        st.markdown(f"""
        <div class="table-card">
          <h3>📋 Decision Details</h3>
          <p><b>Tier:</b> <span class="tier-pill {css_class}">{tier_name}</span></p>
          <p style="margin-top:0.7rem;"><b>Rule Decision:</b> {row['rule_decision']}</p>
          <p style="margin-top:0.4rem;"><b>Rule ID:</b> <code style="background:#1a2535;padding:0.1rem 0.4rem;border-radius:4px;font-family:'DM Mono',monospace;">{row['rule_id']}</code></p>
          <p style="margin-top:0.4rem;"><b>Interest Rate:</b> {row['interest_rate'] if pd.notna(row['interest_rate']) and row['interest_rate'] else '—'}</p>
          <p style="margin-top:0.4rem;"><b>Anomaly:</b>
            {"<span class='anom-yes'>⚠️ Suspicious profile</span>" if row['is_anomaly'] else "<span class='anom-no'>✅ Normal</span>"}
          </p>
          {"<p style='margin-top:0.4rem;color:#fc8181;font-size:0.85rem;'><b>Flags:</b> " + str(row['anomaly_flags']) + "</p>" if row['is_anomaly'] else ""}
        </div>
        """, unsafe_allow_html=True)

    with info_r:
        st.markdown(f"""
        <div class="table-card">
          <h3>💡 Agent Rationale</h3>
          <p style="margin-bottom:0.5rem;font-size:0.85rem;color:#8892a4;">Utility bar</p>
          <div class="tier-bar-bg" style="margin-bottom:1rem;">
            <div class="tier-bar-fill" style="width:{u_pct}%;background:{color};"></div>
          </div>
          <p><b style="color:#68d391;">✅ Strengths</b></p>
          <p style="font-size:0.88rem;color:#8892a4;margin:0.3rem 0 0.8rem;">{row['strengths']}</p>
          <p><b style="color:#fc8181;">⚠️ Weaknesses</b></p>
          <p style="font-size:0.88rem;color:#8892a4;margin:0.3rem 0;">{row['weaknesses']}</p>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#2d3748;font-family:'DM Mono',monospace;
            font-size:0.72rem;letter-spacing:1px;margin-top:3rem;padding:1rem;">
  LOAN APPROVAL EXPERT SYSTEM · UTILITY-BASED AGENT v1.0 ·
  RULE ENGINE + FUZZY LOGIC + ISOLATION FOREST + UTILITY RANKING
</div>
""", unsafe_allow_html=True)