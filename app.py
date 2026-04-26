"""
app.py — Dashboard Deteksi Anomali Klaim BPJS Kesehatan
Jalankan: streamlit run app.py
"""

import os, joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ── Konfigurasi Halaman ────────────────────────────────────
st.set_page_config(
    page_title="VeraMed AI — BPJS Fraud Detector",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #0d1117; }
.metric-card {
    background: linear-gradient(135deg, #1a1f2e, #252d3d);
    border: 1px solid #30363d; border-radius: 12px;
    padding: 20px; text-align: center;
}
.metric-value { font-size: 2rem; font-weight: 700; }
.metric-label { font-size: 0.85rem; color: #8b949e; margin-top: 4px; }
.risk-high  { color: #ff4b4b; }
.risk-med   { color: #ffa500; }
.risk-low   { color: #00d26a; }
.stDataFrame { border-radius: 10px; }
.section-title {
    font-size: 1.1rem; font-weight: 600;
    color: #58a6ff; margin: 16px 0 8px 0;
    border-left: 3px solid #58a6ff; padding-left: 10px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# KONSTANTA & PATH
# ============================================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "models")
RESULT_PATH = os.path.join(BASE_DIR, "bpjs_claims_results.csv")
FEATURE_COLS = [
    "patient_age", "room_type_enc", "icd_10_code_enc",
    "total_cost", "is_resume_complete", "auth_signature", "los"
]
FEATURE_LABELS = {
    "patient_age": "Usia Pasien",
    "room_type_enc": "Tipe Kamar",
    "icd_10_code_enc": "Kode ICD-10",
    "total_cost": "Total Biaya",
    "is_resume_complete": "Kelengkapan Resume",
    "auth_signature": "Tanda Tangan DPJP",
    "los": "Length of Stay"
}

# ============================================================
# FUNGSI: MUAT MODEL & DATA
# ============================================================
@st.cache_resource(show_spinner="Memuat model AI...")
def load_models():
    """Muat semua model dan encoder yang sudah dilatih."""
    try:
        xgb  = joblib.load(os.path.join(MODEL_DIR, "xgboost_model.pkl"))
        iso  = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))
        scl  = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        le_r = joblib.load(os.path.join(MODEL_DIR, "le_room.pkl"))
        le_i = joblib.load(os.path.join(MODEL_DIR, "le_icd.pkl"))
        meta = joblib.load(os.path.join(MODEL_DIR, "metadata.pkl"))
        return xgb, iso, scl, le_r, le_i, meta
    except FileNotFoundError:
        return None

@st.cache_data(show_spinner="Memuat data...")
def load_data(path: str) -> pd.DataFrame:
    """Muat CSV hasil prediksi atau dataset mentah."""
    return pd.read_csv(path)

def predict_uploaded(df_raw, xgb, iso, scl, le_r, le_i):
    """Jalankan prediksi pada dataset yang diunggah pengguna."""
    df = df_raw.copy()
    for col in ["patient_age","total_cost","los"]:
        df[col].fillna(df[col].median(), inplace=True)
    for col in ["room_type","icd_10_code"]:
        df[col].fillna(df[col].mode()[0], inplace=True)
    for col in ["is_resume_complete","auth_signature"]:
        df[col].fillna(0, inplace=True)

    # Encoding kategorikal
    def safe_encode(encoder, series):
        known = set(encoder.classes_)
        return series.map(lambda x: encoder.transform([x])[0] if x in known else -1)

    df["room_type_enc"]   = safe_encode(le_r, df["room_type"])
    df["icd_10_code_enc"] = safe_encode(le_i, df["icd_10_code"])

    X = df[FEATURE_COLS].values
    xgb_proba  = xgb.predict_proba(X)[:, 1]
    X_scaled   = scl.transform(X)
    iso_scores = iso.decision_function(X_scaled)

    # Normalisasi IF score ke [0,1]
    s_min, s_max = iso_scores.min(), iso_scores.max()
    iso_norm = 1 - (iso_scores - s_min) / (s_max - s_min + 1e-9)

    doc_penalty = np.where(df["is_resume_complete"].values == 0, 1.0, 0.0)
    hybrid = np.clip(0.55*xgb_proba + 0.30*iso_norm + 0.15*doc_penalty, 0, 1) * 100

    df["xgb_fraud_prob"]    = np.round(xgb_proba * 100, 2)
    df["iso_anomaly_score"] = np.round(iso_scores, 4)
    df["risk_score"]        = np.round(hybrid, 2)
    df["risk_label"]        = pd.cut(
        df["risk_score"], bins=[0,40,70,100],
        labels=["🟢 Rendah","🟡 Sedang","🔴 Tinggi"], include_lowest=True
    )
    df["xgb_prediction"] = xgb.predict(X)
    return df

def explain_claim(row: pd.Series) -> list[str]:
    """Hasilkan daftar alasan mengapa klaim ditandai sebagai anomali."""
    reasons = []
    if row.get("icd_10_code") == "R50.9" and row.get("total_cost", 0) > 5_000_000:
        reasons.append("⚠️ Biaya melampaui batas normal untuk diagnosa **R50.9 (Demam)** — batas wajar Rp 5.000.000")
    if row.get("is_resume_complete", 1) == 0 and row.get("auth_signature", 1) == 0:
        reasons.append("⚠️ Resume medis **tidak lengkap** & **tidak ada tanda tangan DPJP/Perawat**")
    if row.get("los", 5) == 1 and row.get("total_cost", 0) > 7_000_000:
        reasons.append("⚠️ LOS hanya **1 hari** namun biaya sangat tinggi (**> Rp 7.000.000**)")
    if row.get("is_resume_complete", 1) == 0:
        reasons.append("📋 Dokumentasi **tidak lengkap** (is_resume_complete = 0) — risiko penolakan klaim")
    if row.get("auth_signature", 1) == 0:
        reasons.append("✍️ **Tidak ada tanda tangan** otorisasi DPJP/Perawat")
    if row.get("xgb_fraud_prob", 0) > 70:
        reasons.append(f"🤖 Model XGBoost memberi probabilitas fraud **{row['xgb_fraud_prob']:.1f}%**")
    if not reasons:
        reasons.append("ℹ️ Terdeteksi oleh IsolationForest sebagai pola statistik yang tidak lazim")
    return reasons

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/hospital.png", width=60)
    st.title("VeraMed AI")
    st.caption("BPJS Fraud Detection System")
    st.divider()

    uploaded = st.file_uploader("📂 Unggah Dataset CSV", type=["csv"])
    st.divider()

    search_id = st.text_input("🔍 Cari Claim ID", placeholder="CLM-2026-0001")
    st.divider()

    risk_filter = st.multiselect(
        "Filter Risk Level",
        ["🟢 Rendah", "🟡 Sedang", "🔴 Tinggi"],
        default=["🟢 Rendah", "🟡 Sedang", "🔴 Tinggi"]
    )
    st.divider()
    st.caption("© 2026 VeraMed AI — Healthcare Analytics")

# ============================================================
# MUAT DATA & MODEL
# ============================================================
models = load_models()

if models is None:
    st.error("❌ Model belum dilatih! Jalankan dulu: `python train_model.py`")
    st.stop()

xgb_model, iso_model, scaler, le_room, le_icd, metadata = models

# Muat data: prioritaskan file yang diunggah, lalu hasil prediksi tersimpan
if uploaded is not None:
    df_raw = pd.read_csv(uploaded)
    df = predict_uploaded(df_raw, xgb_model, iso_model, scaler, le_room, le_icd)
    st.sidebar.success(f"✅ {len(df)} klaim berhasil dianalisis")
elif os.path.exists(RESULT_PATH):
    df = load_data(RESULT_PATH)
else:
    st.warning("⚠️ Tidak ada data. Unggah CSV atau jalankan train_model.py terlebih dahulu.")
    st.stop()

# Konversi risk_label ke string jika berupa Categorical
df["risk_label"] = df["risk_label"].astype(str)

# Terapkan filter sidebar
df_filtered = df[df["risk_label"].isin(risk_filter)].copy()
if search_id:
    df_filtered = df_filtered[df_filtered["claim_id"].str.contains(search_id, case=False)]

# ============================================================
# HEADER & TAB NAVIGATION
# ============================================================
st.markdown("## 🏥 VeraMed AI — Deteksi Anomali Klaim BPJS Kesehatan")
st.caption("Sistem deteksi fraud berbasis Hybrid AI (IsolationForest + XGBoost)")
st.divider()

tab_dashboard, tab_extractor = st.tabs([
    "📊 Dashboard Analitik",
    "📄 Document Extractor (AI Audit)"
])

# ============================================================
# METRIC ROW
# ============================================================
total_klaim   = len(df_filtered)
high_risk     = (df_filtered["risk_label"] == "🔴 Tinggi").sum()
pct_high      = high_risk / total_klaim * 100 if total_klaim > 0 else 0
total_loss    = df_filtered.loc[df_filtered["risk_label"] == "🔴 Tinggi", "total_cost"].sum()
avg_risk      = df_filtered["risk_score"].mean() if total_klaim > 0 else 0

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("📋 Total Klaim", f"{total_klaim:,}")
with c2:
    st.metric("🔴 Berisiko Tinggi", f"{high_risk:,}", f"{pct_high:.1f}%")
with c3:
    st.metric("💸 Potensi Kerugian", f"Rp {total_loss/1e6:.1f}M")
with c4:
    st.metric("📊 Avg Risk Score", f"{avg_risk:.1f}/100")

st.divider()

# ============================================================
# VISUALISASI UTAMA — 2 KOLOM
# ============================================================
col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.markdown('<div class="section-title">📊 Distribusi Biaya per Kode ICD-10</div>', unsafe_allow_html=True)
    fig_box = px.box(
        df_filtered, x="icd_10_code", y="total_cost",
        color="risk_label",
        color_discrete_map={"🟢 Rendah":"#00d26a","🟡 Sedang":"#ffa500","🔴 Tinggi":"#ff4b4b"},
        template="plotly_dark",
        labels={"icd_10_code":"Kode ICD-10","total_cost":"Biaya (Rp)","risk_label":"Risk Level"},
    )
    fig_box.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        margin=dict(l=10,r=10,t=10,b=10), height=320,
        legend=dict(orientation="h", y=-0.25)
    )
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown('<div class="section-title">📈 Distribusi Risk Score</div>', unsafe_allow_html=True)
    fig_hist = px.histogram(
        df_filtered, x="risk_score", nbins=30,
        color="risk_label",
        color_discrete_map={"🟢 Rendah":"#00d26a","🟡 Sedang":"#ffa500","🔴 Tinggi":"#ff4b4b"},
        template="plotly_dark",
        labels={"risk_score":"Risk Score (0–100)","count":"Jumlah Klaim"}
    )
    fig_hist.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        margin=dict(l=10,r=10,t=10,b=10), height=260,
        bargap=0.05, showlegend=False
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col_right:
    st.markdown('<div class="section-title">🍩 Komposisi Risk Level</div>', unsafe_allow_html=True)
    risk_counts = df_filtered["risk_label"].value_counts().reset_index()
    risk_counts.columns = ["Risk Level","Jumlah"]
    fig_pie = px.pie(
        risk_counts, names="Risk Level", values="Jumlah",
        color="Risk Level",
        color_discrete_map={"🟢 Rendah":"#00d26a","🟡 Sedang":"#ffa500","🔴 Tinggi":"#ff4b4b"},
        hole=0.55, template="plotly_dark"
    )
    fig_pie.update_layout(
        paper_bgcolor="#0d1117", margin=dict(l=10,r=10,t=10,b=10), height=290,
        legend=dict(orientation="h", y=-0.1)
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown('<div class="section-title">🏨 Risk per Tipe Kamar</div>', unsafe_allow_html=True)
    room_risk = df_filtered.groupby(["room_type","risk_label"]).size().reset_index(name="count")
    fig_bar = px.bar(
        room_risk, x="room_type", y="count", color="risk_label",
        color_discrete_map={"🟢 Rendah":"#00d26a","🟡 Sedang":"#ffa500","🔴 Tinggi":"#ff4b4b"},
        template="plotly_dark", barmode="stack",
        labels={"room_type":"Tipe Kamar","count":"Jumlah","risk_label":"Risk Level"}
    )
    fig_bar.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        margin=dict(l=10,r=10,t=10,b=10), height=260, showlegend=False
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ============================================================
# TABEL INTERAKTIF HASIL PREDIKSI
# ============================================================
st.markdown('<div class="section-title">📋 Tabel Hasil Prediksi Klaim</div>', unsafe_allow_html=True)

cols_show = ["claim_id","patient_age","room_type","icd_10_code","total_cost",
             "los","is_resume_complete","auth_signature","risk_score","risk_label"]
cols_show = [c for c in cols_show if c in df_filtered.columns]

df_display = df_filtered[cols_show].copy()
df_display["total_cost"] = df_display["total_cost"].apply(lambda x: f"Rp {x:,.0f}")
df_display = df_display.sort_values("risk_score", ascending=False)

st.dataframe(
    df_display,
    use_container_width=True,
    height=320,
    column_config={
        "risk_score": st.column_config.ProgressColumn(
            "Risk Score", min_value=0, max_value=100, format="%.1f"
        ),
        "claim_id":   st.column_config.TextColumn("Claim ID"),
        "risk_label": st.column_config.TextColumn("Level Risiko"),
    }
)

st.divider()

# ============================================================
# EXPLAINABLE AI — FEATURE IMPORTANCE
# ============================================================
st.markdown('<div class="section-title">🧠 Feature Importance (Explainable AI)</div>', unsafe_allow_html=True)

importance = xgb_model.feature_importances_
fi_df = pd.DataFrame({
    "Fitur": [FEATURE_LABELS.get(f, f) for f in FEATURE_COLS],
    "Importance": importance
}).sort_values("Importance", ascending=True)

fig_fi = px.bar(
    fi_df, x="Importance", y="Fitur", orientation="h",
    template="plotly_dark",
    color="Importance",
    color_continuous_scale=["#1a3a5c","#58a6ff","#ff4b4b"],
    labels={"Importance":"Tingkat Pengaruh","Fitur":"Fitur Model"}
)
fig_fi.update_layout(
    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
    margin=dict(l=10,r=10,t=10,b=10), height=300,
    coloraxis_showscale=False
)
st.plotly_chart(fig_fi, use_container_width=True)

# Penjelasan bobot hybrid
with st.expander("ℹ️ Penjelasan Hybrid Risk Score"):
    st.markdown("""
    **Formula Risk Score (0–100):**
    ```
    Risk Score = (55% × XGBoost_Prob) + (30% × IsolationForest_Norm) + (15% × Penalti_Dokumentasi)
    ```
    | Komponen | Bobot | Keterangan |
    |---|---|---|
    | **XGBoost** | 55% | Prediksi berbasis pola historis berlabel |
    | **IsolationForest** | 30% | Deteksi statistik klaim "tidak lazim" |
    | **Penalti Dokumentasi** | 15% | +15 poin jika `is_resume_complete = 0` |

    > Bobot dokumentasi berdasarkan temuan riset: **59.43% masalah klaim BPJS** bersumber dari dokumentasi tidak lengkap.
    """)

st.divider()

# ============================================================
# AUDIT DETAIL — ANALISIS SATU KLAIM
# ============================================================
st.markdown('<div class="section-title">🔍 Audit Detail Klaim</div>', unsafe_allow_html=True)

claim_ids = df_filtered["claim_id"].tolist()
if claim_ids:
    default_idx = 0
    # Jika ada hasil pencarian, langsung ke klaim pertama
    selected_id = st.selectbox("Pilih Claim ID untuk diaudit:", claim_ids, index=default_idx)
    selected_row = df_filtered[df_filtered["claim_id"] == selected_id].iloc[0]

    a1, a2, a3 = st.columns(3)
    risk_color = {"🔴 Tinggi":"#ff4b4b","🟡 Sedang":"#ffa500","🟢 Rendah":"#00d26a"}
    rl = str(selected_row.get("risk_label",""))
    color = risk_color.get(rl, "#58a6ff")

    with a1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{color}">{selected_row.get('risk_score',0):.1f}</div>
            <div class="metric-label">Risk Score / 100</div>
        </div>""", unsafe_allow_html=True)
    with a2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#58a6ff">{selected_row.get('xgb_fraud_prob',0):.1f}%</div>
            <div class="metric-label">XGBoost Fraud Probability</div>
        </div>""", unsafe_allow_html=True)
    with a3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{color}">{rl}</div>
            <div class="metric-label">Risk Level</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("#### 📄 Detail Klaim")
    detail_cols = ["claim_id","patient_age","room_type","icd_10_code",
                   "total_cost","los","is_resume_complete","auth_signature"]
    detail_cols = [c for c in detail_cols if c in selected_row.index]
    detail_df = pd.DataFrame({
        "Field": detail_cols,
        "Nilai": [selected_row[c] for c in detail_cols]
    })
    st.table(detail_df.set_index("Field"))

    # Tampilkan alasan anomali
    if str(selected_row.get("risk_label","")) == "🔴 Tinggi" or selected_row.get("xgb_prediction",0) == 1:
        st.markdown("#### ⚠️ Alasan AI Menandai Klaim Ini")
        reasons = explain_claim(selected_row)
        for r in reasons:
            st.markdown(f"- {r}")

        # Gauge chart risk score
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(selected_row.get("risk_score", 0)),
            domain={"x":[0,1],"y":[0,1]},
            title={"text":"Risk Score","font":{"color":"white"}},
            gauge={
                "axis":{"range":[0,100],"tickcolor":"white"},
                "bar":{"color":color},
                "bgcolor":"#1a1f2e",
                "steps":[
                    {"range":[0,40],"color":"#0d2b1d"},
                    {"range":[40,70],"color":"#2b1f0d"},
                    {"range":[70,100],"color":"#2b0d0d"},
                ],
                "threshold":{
                    "line":{"color":"white","width":3},
                    "thickness":0.75,
                    "value": float(selected_row.get("risk_score",0))
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#0d1117", font_color="white",
            height=260, margin=dict(l=20,r=20,t=40,b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
    else:
        st.success("✅ Klaim ini tidak terindikasi sebagai anomali oleh model AI.")
else:
    st.info("Tidak ada klaim yang cocok dengan filter yang dipilih.")
