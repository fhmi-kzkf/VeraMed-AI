"""
pages/2_Document_Extractor.py
Tab AI Audit: Upload rekam medis (PDF/Gambar) → ekstrak JSON → prediksi risk score.
"""
import os, sys, json, joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Muat .env sebelum import extractor
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Tambahkan root ke path agar bisa import extractor.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extractor import extract_from_document

st.set_page_config(page_title="VeraMed AI — Document Extractor", page_icon="📄", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.json-box{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;font-family:monospace;font-size:0.85rem;color:#e6edf3;}
.flag-box{background:linear-gradient(135deg,#2d0f0f,#1a0a0a);border:1px solid #ff4b4b;border-radius:10px;padding:14px;}
.ok-box{background:linear-gradient(135deg,#0f2d14,#0a1a0f);border:1px solid #00d26a;border-radius:10px;padding:14px;}
.info-row{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0;}
.badge{padding:4px 12px;border-radius:20px;font-size:0.8rem;font-weight:600;}
.badge-red{background:#ff4b4b22;color:#ff4b4b;border:1px solid #ff4b4b;}
.badge-green{background:#00d26a22;color:#00d26a;border:1px solid #00d26a;}
.badge-yellow{background:#ffa50022;color:#ffa500;border:1px solid #ffa500;}
</style>
""", unsafe_allow_html=True)

# ── Konstanta ──────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR    = os.path.join(BASE_DIR, "models")
FEATURE_COLS = ["patient_age","room_type_enc","icd_10_code_enc",
                "total_cost","is_resume_complete","auth_signature","los"]

# ── Muat model ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    try:
        return (
            joblib.load(os.path.join(MODEL_DIR, "xgboost_model.pkl")),
            joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl")),
            joblib.load(os.path.join(MODEL_DIR, "scaler.pkl")),
            joblib.load(os.path.join(MODEL_DIR, "le_room.pkl")),
            joblib.load(os.path.join(MODEL_DIR, "le_icd.pkl")),
        )
    except FileNotFoundError:
        return None

def predict_single(data: dict, models) -> dict:
    """Jalankan prediksi hybrid pada satu klaim hasil ekstraksi."""
    xgb, iso, scl, le_r, le_i = models

    room = data.get("room_type") or "SAPHIRE"
    icd  = data.get("icd_10_code") or "R50.9"

    room_enc = le_r.transform([room])[0] if room in le_r.classes_ else 0
    icd_enc  = le_i.transform([icd])[0]  if icd  in le_i.classes_  else 0

    X = np.array([[
        data.get("patient_age", 0),
        room_enc, icd_enc,
        data.get("total_cost", 0),
        data.get("is_resume_complete", 0),
        data.get("auth_signature", 0),
        data.get("los", 0),
    ]])

    xgb_prob   = float(xgb.predict_proba(X)[0, 1])
    X_scaled   = scl.transform(X)
    iso_score  = float(iso.decision_function(X_scaled)[0])

    # Normalisasi IF ke [0,1] berdasarkan rentang training
    iso_norm = float(np.clip(1 - (iso_score + 0.06) / 0.14, 0, 1))
    doc_pen  = 1.0 if data.get("is_resume_complete", 0) == 0 else 0.0

    risk = float(np.clip(0.55*xgb_prob + 0.30*iso_norm + 0.15*doc_pen, 0, 1) * 100)

    if risk >= 70:   label = "🔴 Tinggi"
    elif risk >= 40: label = "🟡 Sedang"
    else:            label = "🟢 Rendah"

    return {"xgb_prob": round(xgb_prob*100,2), "risk_score": round(risk,2), "risk_label": label}

# ════════════════════════════════════════════════════════════
# UI UTAMA
# ════════════════════════════════════════════════════════════
st.markdown("## 📄 Document Extractor — AI Medical Auditor")
st.caption("Upload rekam medis BPJS (PDF/Gambar) → Ekstraksi JSON otomatis → Prediksi Risk Score")
st.divider()

models = load_models()
if models is None:
    st.error("❌ Model belum dilatih. Jalankan: `python train_model.py`")
    st.stop()

# ── Sidebar: Info & Format ────────────────────────────────
with st.sidebar:
    # Set default values internal (tidak ditampilkan di UI)
    ui_api_key = "" 
    use_mock   = False # Set ke True jika ingin tetap menggunakan mock secara default di produksi
    
    st.markdown("### 📄 Info Extractor")
    st.markdown("""
    **Format Didukung:**
    - 📑 PDF (rekam medis digital)
    - 🖼️ JPG / PNG / WEBP (foto dokumen)

    **Cara Kerja:**
    1. Upload dokumen rekam medis
    2. AI melakukan audit otomatis
    3. Data diekstrak ke JSON
    4. Prediksi skor risiko fraud
    """)
    st.divider()
    st.markdown("""
    **Format Didukung:**
    - 📑 PDF (rekam medis digital)
    - 🖼️ JPG / PNG / WEBP (foto dokumen)

    **Cara Kerja:**
    1. Upload dokumen
    2. Gemini Vision membaca & mengaudit
    3. Data diekstrak ke JSON
    4. Model ML prediksi risk score
    """)
    st.divider()
    st.caption("© 2026 VeraMed AI")

# ── Upload Area ────────────────────────────────────────────
col_up, col_info = st.columns([1, 1])

with col_up:
    st.markdown("### 📂 Upload Dokumen Rekam Medis")
    doc_file = st.file_uploader(
        "Pilih file rekam medis",
        type=["pdf", "jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    if doc_file:
        ext = os.path.splitext(doc_file.name)[1].lower()
        if ext in [".jpg",".jpeg",".png",".webp"]:
            st.image(doc_file, caption=f"Preview: {doc_file.name}", width="stretch")
        else:
            st.info(f"📑 File PDF diunggah: **{doc_file.name}** ({doc_file.size/1024:.1f} KB)")

with col_info:
    st.markdown("### 📋 Audit Rules Aktif")
    st.markdown("""
    | # | Aturan | Pengaruh |
    |---|---|---|
    | 1 | `auth_signature = 0` → Klaim tidak sah | **50.69%** |
    | 2 | `is_resume_complete = 0` → Risiko tinggi | **29.67%** |
    | 3 | R50.9 + biaya > Rp 5jt → Inflated Cost | Rule-based |
    | 4 | LOS = 1 hari + biaya > Rp 7jt → Anomali | Rule-based |
    | 5 | 59.43% masalah BPJS: dokumentasi | Riset |
    """)

st.divider()

# ── Tombol Ekstraksi ───────────────────────────────────────
if doc_file:
    if st.button("🔍 Analisis Dokumen dengan AI", type="primary", width="stretch"):
        with st.spinner("Gemini Vision sedang mengaudit dokumen rekam medis..."):
            try:
                file_bytes = doc_file.read()
                result = extract_from_document(
                    file_bytes=file_bytes,
                    filename=doc_file.name,
                    api_key=ui_api_key,  # extractor akan fallback ke GOOGLE_API_KEY di env
                    use_mock=use_mock
                )
                st.session_state["last_extraction"] = result
                st.session_state["last_filename"]   = doc_file.name
            except Exception as e:
                st.error(f"❌ Gagal mengekstrak dokumen: {e}")
                st.stop()
else:
    st.info("👆 Upload dokumen rekam medis untuk memulai analisis AI.")

# ── Tampilkan Hasil ────────────────────────────────────────
if "last_extraction" in st.session_state:
    result   = st.session_state["last_extraction"]
    filename = st.session_state.get("last_filename", "dokumen")

    st.success(f"✅ Dokumen **{filename}** berhasil diaudit oleh AI!")
    st.divider()

    # ── Prediksi Risk Score ────────────────────────────────
    pred = predict_single(result, models)

    r1, r2, r3, r4 = st.columns(4)
    risk_color = {"🔴 Tinggi":"#ff4b4b","🟡 Sedang":"#ffa500","🟢 Rendah":"#00d26a"}
    rc = risk_color.get(pred["risk_label"], "#58a6ff")

    with r1:
        st.metric("🎯 Risk Score", f"{pred['risk_score']:.1f} / 100")
    with r2:
        st.metric("🤖 XGBoost Fraud Prob", f"{pred['xgb_prob']:.1f}%")
    with r3:
        st.metric("📋 Resume Lengkap",
            "✅ Ya" if result.get("is_resume_complete") == 1 else "❌ Tidak")
    with r4:
        st.metric("✍️ Tanda Tangan DPJP",
            "✅ Ada" if result.get("auth_signature") == 1 else "❌ Tidak Ada")

    st.divider()

    # ── Dua Kolom: JSON + Gauge ────────────────────────────
    left, right = st.columns([1.1, 1])

    with left:
        st.markdown("### 🗂️ Hasil Ekstraksi JSON")

        # Tampilan JSON yang rapi
        json_display = json.dumps(result, ensure_ascii=False, indent=2)
        st.markdown(
            f'<div class="json-box"><pre>{json_display}</pre></div>',
            unsafe_allow_html=True
        )

        # Tombol download JSON
        st.download_button(
            label="⬇️ Download JSON",
            data=json_display,
            file_name=f"audit_{os.path.splitext(filename)[0]}.json",
            mime="application/json",
            width="stretch"
        )

    with right:
        st.markdown("### 📊 Risk Score Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred["risk_score"],
            delta={"reference": 40, "increasing": {"color": "#ff4b4b"}},
            domain={"x":[0,1],"y":[0,1]},
            title={"text": pred["risk_label"], "font": {"color": rc, "size": 18}},
            number={"font": {"color": rc, "size": 48}},
            gauge={
                "axis": {"range": [0,100], "tickcolor": "#8b949e"},
                "bar":  {"color": rc, "thickness": 0.3},
                "bgcolor": "#161b22",
                "bordercolor": "#30363d",
                "steps": [
                    {"range": [0,40],  "color": "#0d2b1d"},
                    {"range": [40,70], "color": "#2b1f0d"},
                    {"range": [70,100],"color": "#2b0d0d"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.8,
                    "value": pred["risk_score"]
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#0d1117", font_color="white",
            height=300, margin=dict(l=20,r=20,t=40,b=10)
        )
        st.plotly_chart(fig_gauge, width="stretch")

    st.divider()

    # ── Analisis Temuan & Flag ─────────────────────────────
    st.markdown("### ⚠️ Temuan Audit AI")

    flags = []
    if result.get("icd_10_code") == "R50.9" and result.get("total_cost",0) > 5_000_000:
        flags.append(("🔴 INFLATED COST", f"Diagnosa R50.9 (Demam) dengan biaya Rp {result['total_cost']:,} melebihi batas Rp 5.000.000", "badge-red"))
    if result.get("auth_signature") == 0:
        flags.append(("🔴 TIDAK SAH", "Tanda tangan DPJP tidak ditemukan — klaim tidak valid secara administrasi (pengaruh 50.69%)", "badge-red"))
    if result.get("is_resume_complete") == 0:
        flags.append(("🟡 DOKUMENTASI", "Resume medis tidak lengkap — 59.43% masalah klaim BPJS ada di bagian ini", "badge-yellow"))
    if result.get("los") == 1 and result.get("total_cost",0) > 7_000_000:
        flags.append(("🔴 LOS ANOMALI", f"LOS hanya 1 hari namun biaya Rp {result['total_cost']:,} sangat tinggi", "badge-red"))

    if flags:
        for badge_text, desc, badge_class in flags:
            st.markdown(f"""
            <div class="flag-box" style="margin-bottom:10px;">
                <span class="badge {badge_class}">{badge_text}</span>
                <p style="margin:8px 0 0 0;color:#e6edf3;font-size:0.9rem;">{desc}</p>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="ok-box">
            <span class="badge badge-green">✅ TIDAK ADA FLAG</span>
            <p style="margin:8px 0 0 0;color:#e6edf3;font-size:0.9rem;">
            Tidak ditemukan indikasi anomali berdasarkan aturan audit BPJS.</p>
        </div>""", unsafe_allow_html=True)

    # ai_analysis dari Gemini
    st.markdown("### 💬 Analisis Kognitif AI")
    st.info(result.get("ai_analysis", "-"))

    # ── Tabel Detail Klaim ─────────────────────────────────
    st.markdown("### 📋 Detail Data Terstruktur")
    field_map = {
        "patient_age":        "Usia Pasien (tahun)",
        "room_type":          "Tipe Kamar",
        "icd_10_code":        "Kode ICD-10",
        "total_cost":         "Total Biaya",
        "is_resume_complete": "Resume Lengkap (1=Ya)",
        "auth_signature":     "Tanda Tangan DPJP (1=Ada)",
        "los":                "Length of Stay (hari)",
    }
    rows = []
    for k, label in field_map.items():
        val = result.get(k)
        if val is None:
            val_str = "null"
        elif k == "total_cost":
            val_str = f"Rp {int(val):,}"
        else:
            val_str = str(val)
            
        rows.append({"Field": str(label), "Nilai": val_str})

    st.table(pd.DataFrame(rows).set_index("Field"))

    # ── Kirim ke Model Batch ───────────────────────────────
    st.divider()
    st.markdown("### 📤 Ekspor ke Dataset")
    if st.button("➕ Tambahkan ke Dataset & Re-Analyze", width="stretch"):
        result_path = os.path.join(BASE_DIR, "bpjs_claims_results.csv")
        if os.path.exists(result_path):
            df_existing = pd.read_csv(result_path)
            new_id = f"CLM-2026-{len(df_existing)+1:04d}"
            new_row = {
                "claim_id": new_id,
                "patient_age": result.get("patient_age", 0),
                "room_type": result.get("room_type", "SAPHIRE"),
                "icd_10_code": result.get("icd_10_code", "R50.9"),
                "total_cost": result.get("total_cost", 0),
                "is_resume_complete": result.get("is_resume_complete", 0),
                "auth_signature": result.get("auth_signature", 0),
                "los": result.get("los", 1),
                "is_fraud": 1 if pred["risk_score"] >= 70 else 0,
                "risk_score": pred["risk_score"],
                "risk_label": pred["risk_label"],
                "xgb_fraud_prob": pred["xgb_prob"],
            }
            df_existing = pd.concat([df_existing, pd.DataFrame([new_row])], ignore_index=True)
            df_existing.to_csv(result_path, index=False)
            st.success(f"✅ Klaim **{new_id}** berhasil ditambahkan ke dataset!")
        else:
            st.warning("Dataset tidak ditemukan. Jalankan train_model.py terlebih dahulu.")
