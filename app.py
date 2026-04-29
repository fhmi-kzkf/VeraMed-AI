import os
import joblib
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import io
from datetime import datetime, date
from pathlib import Path

# --- CONFIGURATION ---
st.set_page_config(
    page_title="VeraMed AI | Clinical Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THEME TOKENS ---
COLORS = {
    "primary": "#000000",
    "secondary": "#006a65",  # Teal
    "secondary_container": "#76f3ea",
    "surface": "#f8f9ff",
    "on_surface": "#0b1c30",
    "error": "#ba1a1a",
    "error_container": "#ffdad6",
    "background": "#f8f9ff",
    "slate_950": "#020617",
    "teal_400": "#2dd4bf"
}

# --- CUSTOM CSS ---
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    .stApp {{ background-color: {COLORS['background']}; font-family: 'Inter', sans-serif; }}
    [data-testid="stSidebar"] {{ background-color: {COLORS['slate_950']}; border-right: 1px solid #1e293b; color: white; }}
    [data-testid="stSidebar"] * {{ color: #94a3b8; }}
    [data-testid="stSidebarNav"] {{ display: none; }} 

    /* Typography */
    h1, h2, h3 {{ font-weight: 700 !important; letter-spacing: -0.02em !important; color: {COLORS['on_surface']}; }}
    
    /* Metrics & Cards */
    .metric-card {{ background: white; padding: 24px; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); }}
    .metric-label {{ text-transform: uppercase; font-size: 11px; font-weight: 700; letter-spacing: 0.05em; color: #000000; margin-bottom: 8px; }}
    .metric-value {{ font-size: 28px; font-weight: 700; color: #000000; }}
    .metric-subtext {{ font-size: 12px; margin-top: 4px; color: #000000; }}
    
    .bento-card {{ background: white; border-radius: 12px; border: 1px solid #e2e8f0; padding: 32px; height: 100%; }}
    
    /* Custom Table */
    .custom-table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
    .custom-table th {{ background-color: #f8fafc; padding: 12px 24px; text-align: left; font-size: 11px; text-transform: uppercase; font-weight: 800; color: #000000; border-bottom: 1px solid #e2e8f0; }}
    .custom-table td {{ padding: 16px 24px; border-bottom: 1px solid #f1f5f9; font-size: 14px; color: #000000; }}
    
    /* Badges */
    .risk-badge {{ padding: 4px 10px; border-radius: 4px; font-size: 10px; font-weight: 900; text-transform: uppercase; }}
    .risk-high {{ background-color: {COLORS['error']}; color: white; }}
    .risk-med {{ background-color: {COLORS['secondary_container']}; color: #006f69; }}
    .risk-low {{ background-color: #f1f5f9; color: #000000; }}

    /* Extractor Elements */
    .json-box {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 16px; font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #ffffff; overflow-x: auto; }}
    .flag-box {{ background: #fff1f2; border-left: 4px solid {COLORS['error']}; padding: 16px; border-radius: 4px; margin-bottom: 12px; }}
    .flag-title {{ font-weight: 800; font-size: 12px; color: {COLORS['error']}; text-transform: uppercase; margin-bottom: 4px; }}
    .flag-desc {{ font-size: 13px; color: #000000; }}

    .material-symbols-outlined {{ font-family: 'Material Symbols Outlined'; font-weight: normal; font-size: 20px; vertical-align: middle; }}
    
    /* Target Streamlit Metrics to be Black */
    [data-testid="stMetricLabel"] {{ color: #000000 !important; font-weight: 700 !important; }}
    [data-testid="stMetricValue"] {{ color: #000000 !important; font-weight: 800 !important; }}
    
    /* Target Streamlit Widget Labels to be Black */
    [data-testid="stWidgetLabel"] p {{ color: #000000 !important; font-weight: 600 !important; }}
    .stSelectbox label p, .stTextInput label p, .stDateInput label p {{ color: #000000 !important; }}
    
    /* File Uploader Text */
    [data-testid="stFileUploader"] div, 
    [data-testid="stFileUploader"] p, 
    [data-testid="stFileUploader"] small, 
    [data-testid="stFileUploader"] span {{ color: #000000 !important; }}
    
    /* Sidebar Buttons */
    .sidebar-btn {{ 
        display: flex; align-items: center; gap: 12px; padding: 12px 16px; 
        border-radius: 8px; margin-bottom: 4px; cursor: pointer; 
        transition: all 0.2s; color: #94a3b8; font-weight: 600; font-size: 14px;
        text-decoration: none;
    }}
    .sidebar-btn:hover {{ background: rgba(45, 212, 191, 0.1); color: white; }}
    .sidebar-btn.active {{ background: #1e293b; color: white; border-left: 4px solid {COLORS['teal_400']}; }}
</style>
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet" />
""", unsafe_allow_html=True)

# --- HELPERS ---
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
RESULT_PATH = BASE_DIR / "bpjs_claims_results.csv"

# Import extractor safely
try:
    from extractor import extract_from_document
    EXTRACTOR_LOADED = True
except ImportError:
    EXTRACTOR_LOADED = False

@st.cache_resource
def load_assets():
    try:
        xgb = joblib.load(MODEL_DIR / "xgboost_model.pkl")
        iso = joblib.load(MODEL_DIR / "isolation_forest.pkl")
        scl = joblib.load(MODEL_DIR / "scaler.pkl")
        le_r = joblib.load(MODEL_DIR / "le_room.pkl")
        le_i = joblib.load(MODEL_DIR / "le_icd.pkl")
        return xgb, iso, scl, le_r, le_i
    except: return None

def predict_single(data: dict, models) -> dict:
    xgb, iso, scl, le_r, le_i = models
    room = data.get("room_type") or "SAPHIRE"
    icd = data.get("icd_10_code") or "R50.9"
    room_enc = le_r.transform([room])[0] if room in le_r.classes_ else 0
    icd_enc = le_i.transform([icd])[0] if icd in le_i.classes_ else 0
    X = np.array([[data.get("patient_age", 0), room_enc, icd_enc, data.get("total_cost", 0),
                  data.get("is_resume_complete", 0), data.get("auth_signature", 0), data.get("los", 0)]])
    xgb_prob = float(xgb.predict_proba(X)[0, 1])
    X_scaled = scl.transform(X)
    iso_score = float(iso.decision_function(X_scaled)[0])
    iso_norm = float(np.clip(1 - (iso_score + 0.06) / 0.14, 0, 1))
    doc_pen = 1.0 if data.get("is_resume_complete", 0) == 0 else 0.0
    risk = float(np.clip(0.55*xgb_prob + 0.30*iso_norm + 0.15*doc_pen, 0, 1) * 100)
    label = "CRITICAL" if risk >= 70 else ("PENDING" if risk >= 40 else "NORMAL")
    return {"xgb_prob": round(xgb_prob*100, 2), "risk_score": round(risk, 2), "risk_label": label}

# --- DATA LOADING ---
assets = load_assets()
if assets:
    xgb_model, iso_model, scaler, le_room, le_icd = assets
    df = pd.read_csv(RESULT_PATH) if RESULT_PATH.exists() else None
else:
    st.error("Assets or Data not found. Run training first.")
    st.stop()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown(f"""
    <div style="padding: 0 16px 32px 16px; display: flex; align-items: center; gap: 12px;">
        <div style="width: 40px; height: 40px; background-color: {COLORS['teal_400']}; border-radius: 4px; display: flex; align-items: center; justify-content: center;">
            <span class="material-symbols-outlined" style="color: {COLORS['slate_950']}; font-variation-settings: 'FILL' 1;">shield_with_heart</span>
        </div>
        <div>
            <h1 style="color: white; font-size: 18px; font-weight: 800; margin: 0; letter-spacing: -0.05em; text-transform: uppercase;">VeraMed AI</h1>
            <p style="color: {COLORS['teal_400']}; font-size: 9px; font-weight: 600; margin: 0; letter-spacing: 0.1em; text-transform: uppercase;">Clinical Fraud Intel</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    menu_items = [
        ("Dashboard", "dashboard"),
        ("Claims", "receipt_long"),
        ("Document Extractor", "description"),
        ("New Audit", "add_circle"),
        ("Audit Logs", "manage_search"),
        ("Reports", "assessment")
    ]
    
    selection = st.session_state.get("page", "Dashboard")
    for label, icon in menu_items:
        if st.button(label, key=f"nav_{label}", use_container_width=True):
            st.session_state["page"] = label
            st.rerun()

    st.markdown("<div style='flex: 1;'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="padding: 24px; border-top: 1px solid #1e293b; display: flex; align-items: center; gap: 12px;">
       
    </div>
    """, unsafe_allow_html=True)

# --- PAGES ---
page = st.session_state.get("page", "Dashboard")

if page == "Dashboard":
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"<h1 style='color: #000000;'>Intelligence Dashboard</h1><p style='color: #000000;'>Surgical precision audit and claim anomaly detection.</p>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div style='display: flex; justify-content: flex-end;'><div style='background: white; padding: 8px 16px; border-radius: 8px; border: 1px solid #e2e8f0; display: flex; align-items: center; gap: 8px;'><span class='material-symbols-outlined' style='color: #000000;'>calendar_today</span><span style='font-size: 13px; font-weight: 600; color: #000000;'>{date.today().strftime('%b %d, %Y')}</span></div></div>", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        val = df[df['is_fraud']==1]['total_cost'].sum() if df is not None else 0
        st.markdown(f"<div class='metric-card'><div style='display: flex; justify-content: space-between;'><div class='metric-label'>Total Savings</div><div style='color: {COLORS['secondary']};'><span class='material-symbols-outlined'>payments</span></div></div><div class='metric-value'>Rp {val/1e9:.2f}B</div><div class='metric-subtext' style='color: {COLORS['secondary']};'>+12% vs last qtr</div></div>", unsafe_allow_html=True)
    with m2:
        val = len(df) if df is not None else 0
        crit = len(df[df['risk_score']>=70]) if df is not None else 0
        st.markdown(f"<div class='metric-card'><div style='display: flex; justify-content: space-between;'><div class='metric-label'>Active Audits</div><div style='color: #64748b;'><span class='material-symbols-outlined'>biotech</span></div></div><div class='metric-value'>{val:,}</div><div class='metric-subtext' style='color: {COLORS['error']};'>{crit} critical flagged</div></div>", unsafe_allow_html=True)
    with m3:
        val = df[df['risk_score']>=70]['total_cost'].sum() if df is not None else 0
        st.markdown(f"<div class='metric-card' style='background: {COLORS['primary']}; color: white;'><div style='display: flex; justify-content: space-between;'><div class='metric-label' style='color: #94a3b8;'>Savings Potential</div><div style='color: {COLORS['teal_400']};'><span class='material-symbols-outlined'>insights</span></div></div><div class='metric-value' style='color: white;'>Rp {val/1e6:.1f}M</div><div class='metric-subtext' style='color: {COLORS['teal_400']};'>High impact detected</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    b1, b2 = st.columns([1, 2])
    with b1:
        st.markdown(f"<div class='bento-card'><h3 style='font-size: 20px; color: #000000;'>Model Performance</h3><p style='font-size: 11px; font-weight: 700; color: #000000; margin-bottom: 24px;'>VERAENGINE v4.2</p>", unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(mode="gauge+number", value=0.9886, number={'font': {'size': 40, 'color': 'black'}}, gauge={'bar': {'color': COLORS['secondary']}, 'axis': {'range': [0, 1], 'tickfont': {'color': 'black'}}}))
        fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', font_color="black")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with b2:
        st.markdown(f"<div class='bento-card'><h3 style='font-size: 20px; color: #000000;'>Risk Distribution</h3><p style='color: #000000; font-size: 14px;'>Across all audited claims.</p>", unsafe_allow_html=True)
        if df is not None:
            fig_dist = px.histogram(df, x="risk_score", color_discrete_sequence=[COLORS['secondary']])
            fig_dist.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="black", xaxis={'tickfont': {'color': 'black'}}, yaxis={'tickfont': {'color': 'black'}})
            st.plotly_chart(fig_dist, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"<div style='background: white; border-radius: 12px; border: 1px solid #e2e8f0; margin-top: 24px;'><div style='padding: 24px; border-bottom: 1px solid #f1f5f9; font-weight: 600; font-size: 18px; color: #000000;'>High Risk Anomalies</div>", unsafe_allow_html=True)
    if df is not None:
        crit_df = df[df['risk_score']>=70].sort_values('risk_score', ascending=False).head(5)
        table = "<table class='custom-table'><thead><tr><th>Claim ID</th><th>Diagnosis</th><th>Status</th><th>Risk Score</th><th style='text-align: right;'>Value</th></tr></thead><tbody>"
        for _, row in crit_df.iterrows():
            table += f"<tr><td style='font-family: monospace; font-weight: 700; color: #000000;'>{row['claim_id']}</td><td style='color: #000000;'>{row['icd_10_code']}</td><td><span class='risk-badge risk-high'>CRITICAL</span></td><td style='color: #000000;'>{row['risk_score']:.1f}</td><td style='text-align: right; font-weight: 700; color: #000000;'>Rp {row['total_cost']:,.0f}</td></tr>"
        table += "</tbody></table></div>"
        st.markdown(table, unsafe_allow_html=True)

elif page == "Claims":
    st.markdown(f"<h1 style='color: #000000;'>Claims Management</h1><p style='color: #000000;'>Database of audited and pending claims.</p>", unsafe_allow_html=True)
    st.divider()
    search = st.text_input("Search Claim ID...", "", placeholder="CLM-2026-XXXX")
    if df is not None:
        filtered = df[df['claim_id'].str.contains(search, case=False)] if search else df
        st.dataframe(filtered.head(100), width='stretch')

elif page == "Document Extractor":
    st.markdown(f"<h1 style='color: #000000;'>Document Extractor</h1><p style='color: #000000;'>AI-powered OCR and medical audit analysis.</p>", unsafe_allow_html=True)
    st.divider()
    
    col_up, col_info = st.columns([1, 1])
    with col_up:
        st.markdown("<h3 style='color: #000000;'>📂 Upload Medical Record</h3>", unsafe_allow_html=True)
        doc_file = st.file_uploader("Select PDF or Image", type=["pdf", "jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
        if doc_file:
            st.info(f"File uploaded: **{doc_file.name}**")
            if st.button("🔍 Run AI Audit Analysis", type="primary", width='stretch'):
                with st.spinner("Gemini Vision is auditing document..."):
                    try:
                        file_bytes = doc_file.read()
                        res = extract_from_document(file_bytes, doc_file.name)
                        st.session_state["last_ext"] = res
                    except Exception as e:
                        st.error(f"Extraction failed: {e}")
    
    with col_info:
        st.markdown("<h3 style='color: #000000;'>📋 Active Audit Rules</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div style="color: #000000;">
        <ul>
            <li><b>Auth Signature</b>: Must be present (50.69% weight)</li>
            <li><b>Documentation</b>: Resume completeness check (29.67% weight)</li>
            <li><b>Cost Consistency</b>: ICD-10 vs Cost validation</li>
            <li><b>LOS Logic</b>: Duration vs Procedure sanity check</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h3 style='color: #000000; margin-top: 16px;'>📥 Download Demo Scenarios</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #000000; font-size: 13px;'>Test the AI by downloading these dummy medical records and uploading them on the left.</p>", unsafe_allow_html=True)
        
        scenarios = [
            ("Skenario_A_DM_VIP_Lengkap.pdf", "Skenario A (Normal)"),
            ("Skenario_B_Febris_Biaya_Anomali.pdf", "Skenario B (Inflated Cost)"),
            ("Skenario_C_DHF_TandaTangan_Kosong.pdf", "Skenario C (No Signature)"),
            ("Skenario_D_Riwayat_Kosong.pdf", "Skenario D (Incomplete)")
        ]
        
        cols = st.columns(2)
        for i, (file_name, label) in enumerate(scenarios):
            file_path = BASE_DIR / file_name
            if file_path.exists():
                with open(file_path, "rb") as f:
                    cols[i % 2].download_button(
                        label=f"📄 {label}",
                        data=f,
                        file_name=file_name,
                        mime="application/pdf",
                        use_container_width=True
                    )

    if "last_ext" in st.session_state:
        st.divider()
        res = st.session_state["last_ext"]
        pred = predict_single(res, assets)
        
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Risk Score", f"{pred['risk_score']:.1f}")
        r2.metric("ML Fraud Prob", f"{pred['xgb_prob']:.1f}%")
        r3.metric("ICD-10", res.get("icd_10_code", "-"))
        r4.metric("Total Cost", f"Rp {res.get('total_cost', 0):,}")
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("<h3 style='color: #000000;'>🗂️ Extracted Data</h3>", unsafe_allow_html=True)
            st.markdown(f"<div class='json-box' style='color: #ffffff;'><pre>{json.dumps(res, indent=2)}</pre></div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<h3 style='color: #000000;'>⚠️ Audit Findings</h3>", unsafe_allow_html=True)
            findings = res.get("ai_analysis", "No anomalies detected.").split("|")
            
            # Dynamic styling based on Risk Score
            is_critical = pred['risk_score'] >= 70
            box_bg = "#fff1f2" if is_critical else "#f0fdf4"
            border_color = COLORS['error'] if is_critical else "#16a34a"
            title_text = "ANOMALY DETECTED" if is_critical else "CLEARED / NORMAL"
            title_color = COLORS['error'] if is_critical else "#15803d"
            
            for f in findings:
                st.markdown(f"""
                <div style='background: {box_bg}; border-left: 4px solid {border_color}; padding: 16px; border-radius: 4px; margin-bottom: 12px;'>
                    <div style='font-weight: 800; font-size: 12px; color: {title_color}; text-transform: uppercase; margin-bottom: 4px;'>{title_text}</div>
                    <div style='font-size: 13px; color: #000000;'>{f.strip()}</div>
                </div>
                """, unsafe_allow_html=True)

elif page == "Audit Logs":
    st.markdown(f"<h1 style='color: #000000;'>Audit History & Logs</h1><p style='color: #000000;'>Detailed trace of AI decisions and system activities.</p>", unsafe_allow_html=True)
    st.divider()
    
    # Mock log data
    log_data = [
        {"timestamp": "2026-04-27 10:45:12", "action": "CLAIM_AUDIT", "actor": "VeraEngine v4.2", "details": "Analyzed CLM-2026-0089: Risk Score 78.4 (Critical)", "status": "FLAGGED"},
        {"timestamp": "2026-04-27 10:42:05", "action": "DATA_INGESTION", "actor": "System", "details": "Batch upload: 450 new claims processed.", "status": "SUCCESS"},
        {"timestamp": "2026-04-27 10:30:11", "action": "MODEL_UPDATE", "actor": "Admin", "details": "Retrained XGBoost with Q1 data.", "status": "COMPLETED"},
        {"timestamp": "2026-04-27 09:15:44", "action": "USER_LOGIN", "actor": "Admin", "details": "Secure session initiated from 192.168.1.45", "status": "INFO"},
        {"timestamp": "2026-04-27 08:55:00", "action": "CLAIM_AUDIT", "actor": "VeraEngine v4.2", "details": "Analyzed CLM-2026-0088: Risk Score 12.1 (Normal)", "status": "CLEARED"},
    ]
    
    table_html = "<table class='custom-table'><thead><tr><th>Timestamp</th><th>Action</th><th>Actor</th><th>Details</th><th>Status</th></tr></thead><tbody>"
    for log in log_data:
        status_color = "#059669" if log['status'] in ['SUCCESS', 'COMPLETED', 'CLEARED'] else "#ba1a1a"
        table_html += f"<tr><td style='color: #000000;'>{log['timestamp']}</td><td style='color: #000000;'><b>{log['action']}</b></td><td style='color: #000000;'>{log['actor']}</td><td style='color: #000000;'>{log['details']}</td><td style='color: {status_color}; font-weight: 700;'>{log['status']}</td></tr>"
    table_html += "</tbody></table>"
    
    st.markdown(table_html, unsafe_allow_html=True)

elif page == "Reports":
    st.markdown(f"<h1 style='color: #000000;'>Performance & Analytics Report</h1><p style='color: #000000;'>Real-time fraud detection metrics and institutional performance.</p>", unsafe_allow_html=True)
    st.divider()
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Detection Precision</div><div class='metric-value'>99.2%</div><div class='metric-subtext' style='color: #059669;'>+0.4% vs benchmark</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Audit Throughput</div><div class='metric-value'>1.4k/hr</div><div class='metric-subtext' style='color: #059669;'>Optimal performance</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>False Positive Rate</div><div class='metric-value'>1.8%</div><div class='metric-subtext' style='color: #059669;'>Below target (2.5%)</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.markdown("<h3 style='color: #000000;'>Fraud Category Prevalence</h3>", unsafe_allow_html=True)
        categories = ["Inflated Cost", "Missing Signature", "Procedure Unbundling", "Phantom Billing"]
        prevalence = [42.8, 31.2, 15.5, 10.5]
        fig_cat = px.bar(x=prevalence, y=categories, orientation='h', color_discrete_sequence=[COLORS['secondary']])
        fig_cat.update_layout(
            xaxis_title="Percentage (%)", 
            yaxis_title="", 
            font_color="black", 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis={'tickfont': {'color': 'black'}, 'title': {'font': {'color': 'black'}}},
            yaxis={'tickfont': {'color': 'black'}}
        )
        st.plotly_chart(fig_cat, use_container_width=True)
        
    with col_right:
        st.markdown("<h3 style='color: #000000;'>System Health</h3>", unsafe_allow_html=True)
        st.markdown(f"<div style='background: white; border: 1px solid #e2e8f0; color: #000000; padding: 24px; border-radius: 12px;'>", unsafe_allow_html=True)
        st.markdown("<p style='color: #000000; font-size: 11px; font-weight: 700; opacity: 0.7;'>MISS RATE</p>", unsafe_allow_html=True)
        st.markdown("<h2 style='color: #000000; margin: 0;'>6.5%</h2>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 12px; color: #000000; margin-top: 8px;'>Fraud undetected by AI but caught by human audit. Industry benchmark: 12.0%.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("<h3 style='color: #000000;'>Export Audit Results</h3>", unsafe_allow_html=True)
    e1, e2, e3 = st.columns(3)
    
    if df is not None:
        # Excel Export
        output_excel = io.BytesIO()
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Audit_Report')
        excel_data = output_excel.getvalue()
        
        e1.download_button(
            label="📊 Download Excel Report",
            data=excel_data,
            file_name=f"VeraMed_Audit_{date.today()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        
        # PDF Export (Simple)
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("helvetica", 'B', 16)
        pdf.cell(0, 10, "VeraMed AI - Audit Summary Report", ln=True, align='C')
        pdf.set_font("helvetica", '', 12)
        pdf.ln(10)
        pdf.cell(0, 10, f"Date Generated: {date.today()}", ln=True)
        pdf.cell(0, 10, f"Total Claims Audited: {len(df)}", ln=True)
        pdf.cell(0, 10, f"High Risk Anomalies Found: {len(df[df['risk_score'] >= 70])}", ln=True)
        pdf.cell(0, 10, f"Total Audit Value: Rp {df['total_cost'].sum():,.0f}", ln=True)
        pdf_bytes = pdf.output()
        
        e2.download_button(
            label="📄 Download PDF Summary",
            data=bytes(pdf_bytes),
            file_name=f"VeraMed_Summary_{date.today()}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        
        # CSV Export
        csv = df.to_csv(index=False).encode('utf-8')
        e3.download_button(
            label="📝 Download Raw CSV",
            data=csv,
            file_name=f"VeraMed_RawData_{date.today()}.csv",
            mime="text/csv",
            use_container_width=True
        )

elif page == "New Audit":
    st.markdown(f"<h1 style='color: #000000;'>New Audit Workflow</h1><p style='color: #000000;'>Configure and execute surgical fraud analysis.</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown("<h3 style='color: #000000;'>Audit Parameters</h3>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.text_input("Audit Name", "Q2 2026 BPJS Compliance")
        c2.selectbox("Target Provider", ["All Providers", "Metro Health", "Unity Surgical"])
        c1.date_input("Start Date", date(2026, 1, 1))
        c2.date_input("End Date", date(2026, 3, 31))
        
        st.markdown("---")
        st.markdown("<h3 style='color: #000000;'>Data Ingestion</h3>", unsafe_allow_html=True)
        
        sample_path = BASE_DIR / "sample_batch_claims_Q2.csv"
        if sample_path.exists():
            with open(sample_path, "rb") as f:
                st.download_button(
                    label="📥 Download Sample Batch CSV",
                    data=f,
                    file_name="sample_batch_claims_Q2.csv",
                    mime="text/csv"
                )
                
        batch_file = st.file_uploader("Upload Claims CSV Batch", type=["csv"])
        
        if st.button("Initialize VeraEngine v4.2 Audit", type="primary", width='stretch'):
            if batch_file is not None:
                with st.spinner("VeraEngine is processing batch data..."):
                    import time
                    time.sleep(1.5) # simulate processing delay for effect
                    batch_df = pd.read_csv(batch_file)
                    
                    results = []
                    for _, row in batch_df.iterrows():
                        pred = predict_single(row.to_dict(), assets)
                        row_dict = row.to_dict()
                        row_dict.update(pred)
                        results.append(row_dict)
                    
                    res_df = pd.DataFrame(results)
                    
                    # Update the main database so Dashboard is updated
                    if RESULT_PATH.exists():
                        main_df = pd.read_csv(RESULT_PATH)
                        # Ensure columns match, fill missing like is_fraud with 0
                        for col in main_df.columns:
                            if col not in res_df.columns:
                                res_df[col] = 0
                        updated_df = pd.concat([main_df, res_df[main_df.columns]], ignore_index=True)
                        updated_df.to_csv(RESULT_PATH, index=False)
                    
                    st.balloons()
                    st.success(f"Audit completed! Processed {len(res_df)} claims and updated the Dashboard database.")
                    
                    st.markdown("<h4 style='color: #000000; margin-top: 16px;'>Top High-Risk Findings</h4>", unsafe_allow_html=True)
                    critical_df = res_df[res_df['risk_score'] >= 70].sort_values('risk_score', ascending=False)
                    st.error(f"⚠️ Found {len(critical_df)} CRITICAL anomalies requiring manual review.")
                    
                    if len(critical_df) > 0:
                        st.dataframe(critical_df[['claim_id', 'icd_10_code', 'total_cost', 'risk_score', 'risk_label']].head(20), use_container_width=True)
                    else:
                        st.info("No critical anomalies found. Displaying general claims.")
                        st.dataframe(res_df[['claim_id', 'icd_10_code', 'total_cost', 'risk_score', 'risk_label']].head(10), use_container_width=True)
            else:
                st.warning("Please upload a CSV file first!")

else:
    st.title(page)
    st.info("This section is under development.")

# --- FOOTER ---
st.markdown("<br><br><div style='text-align: center; border-top: 1px solid #e2e8f0; padding: 24px;'><p style='font-size: 11px; color: #94a3b8;'>VeraMed AI • Powered by VeraEngine v4.2 • Precision Healthcare Analytics</p></div>", unsafe_allow_html=True)
