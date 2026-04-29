# train_model.py - BPJS Claims Anomaly Detection
# Hybrid Architecture: Unsupervised (IsolationForest) + Supervised (XGBoost)

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

# Constants & Paths
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "bpjs_claims_synthetic.csv")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Hybrid Risk Score Weights
W_SUPERVISED   = 0.55
W_UNSUPERVISED = 0.30
W_DOCUMENTATION = 0.15

# Kolom fitur yang akan digunakan oleh model
FEATURE_COLS = [
    "patient_age", "room_type_enc", "icd_10_code_enc",
    "total_cost", "is_resume_complete", "auth_signature", "los"
]

def load_and_prepare(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess dataset, handle missing values, and encode categoricals."""
    print(f"[INFO] Memuat dataset dari: {path}")
    df_raw = pd.read_csv(path)

    # ── Tangani Missing Values ──────────────────────────────
    # Numerik: isi dengan median (robust terhadap outlier)
    for col in ["patient_age", "total_cost", "los"]:
        if df_raw[col].isnull().any():
            df_raw[col].fillna(df_raw[col].median(), inplace=True)

    # Kategorikal: isi dengan modus
    for col in ["room_type", "icd_10_code"]:
        if df_raw[col].isnull().any():
            df_raw[col].fillna(df_raw[col].mode()[0], inplace=True)

    # Binary: isi dengan 0 (kasus terburuk / asumsi tidak ada)
    for col in ["is_resume_complete", "auth_signature", "is_fraud"]:
        if df_raw[col].isnull().any():
            df_raw[col].fillna(0, inplace=True)

    # ── Label Encoding untuk Kolom Kategorikal ──────────────
    df_proc = df_raw.copy()

    le_room = LabelEncoder()
    le_icd  = LabelEncoder()

    df_proc["room_type_enc"]    = le_room.fit_transform(df_proc["room_type"])
    df_proc["icd_10_code_enc"]  = le_icd.fit_transform(df_proc["icd_10_code"])

    # Simpan encoder agar bisa digunakan di aplikasi Streamlit
    joblib.dump(le_room, os.path.join(MODEL_DIR, "le_room.pkl"))
    joblib.dump(le_icd,  os.path.join(MODEL_DIR, "le_icd.pkl"))

    print(f"[INFO] Dataset dimuat: {len(df_raw)} baris, {df_raw.shape[1]} kolom")
    print(f"[INFO] Distribusi is_fraud:\n{df_raw['is_fraud'].value_counts()}")
    return df_raw, df_proc


def train_isolation_forest(X: np.ndarray) -> IsolationForest:
    """Train unsupervised IsolationForest model."""
    print("\n[TRAIN] Melatih IsolationForest (Unsupervised)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=0.25,
        max_samples="auto",
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_scaled)

    # Simpan scaler dan model
    joblib.dump(scaler,     os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(iso_forest, os.path.join(MODEL_DIR, "isolation_forest.pkl"))
    print("[INFO] IsolationForest berhasil dilatih & disimpan.")
    return iso_forest, scaler


def train_xgboost(X: np.ndarray, y: np.ndarray) -> XGBClassifier:
    """Train supervised XGBoost Classifier."""
    print("\n[TRAIN] Melatih XGBoost (Supervised)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Hitung rasio kelas untuk menangani imbalance
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos = neg_count / pos_count if pos_count > 0 else 1

    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # ── Evaluasi Model ──────────────────────────────────────
    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)[:, 1]

    print("\n[EVAL] Classification Report (XGBoost):")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))
    print(f"[EVAL] ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"[EVAL] Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # Simpan model XGBoost
    joblib.dump(xgb_model, os.path.join(MODEL_DIR, "xgboost_model.pkl"))
    print("[INFO] XGBoost berhasil dilatih & disimpan.")
    return xgb_model


def compute_hybrid_score(
    xgb_proba: np.ndarray,
    iso_scores: np.ndarray,
    is_resume_complete: np.ndarray
) -> np.ndarray:
    """Compute combined hybrid risk score [0-100]."""
    # Normalisasi anomaly score IF ke range [0, 1]
    # IsolationForest memberi nilai negatif untuk anomali (semakin negatif = semakin aneh)
    iso_min, iso_max = iso_scores.min(), iso_scores.max()
    if iso_max - iso_min > 0:
        iso_norm = 1 - (iso_scores - iso_min) / (iso_max - iso_min)
    else:
        iso_norm = np.zeros_like(iso_scores)

    # Penalti dokumentasi (sesuai temuan riset 59.43% masalah dokumentasi)
    doc_penalty = np.where(is_resume_complete == 0, 1.0, 0.0)

    hybrid_score = (
        W_SUPERVISED    * xgb_proba +
        W_UNSUPERVISED  * iso_norm  +
        W_DOCUMENTATION * doc_penalty
    )

    # Clamp ke [0, 1] lalu konversi ke [0, 100]
    hybrid_score = np.clip(hybrid_score, 0, 1) * 100
    return hybrid_score


def save_feature_metadata(feature_cols: list, icd_categories: list, room_categories: list):
    """Save metadata for UI inference."""
    metadata = {
        "feature_cols": feature_cols,
        "icd_categories": icd_categories,
        "room_categories": room_categories,
        "weights": {
            "supervised": W_SUPERVISED,
            "unsupervised": W_UNSUPERVISED,
            "documentation": W_DOCUMENTATION
        }
    }
    joblib.dump(metadata, os.path.join(MODEL_DIR, "metadata.pkl"))
    print("[INFO] Metadata fitur disimpan.")


def main():
    print("Starting Training Pipeline...")

    # ── Load & Prepare ──────────────────────────────────────
    df_raw, df_proc = load_and_prepare(DATA_PATH)

    X = df_proc[FEATURE_COLS].values
    y = df_proc["is_fraud"].values

    # ── Latih IsolationForest ───────────────────────────────
    iso_model, scaler = train_isolation_forest(X)

    # ── Latih XGBoost ───────────────────────────────────────
    xgb_model = train_xgboost(X, y)

    # ── Hitung Hybrid Risk Score untuk seluruh dataset ─────
    print("\n[INFO] Menghitung Hybrid Risk Score untuk seluruh dataset...")
    xgb_proba  = xgb_model.predict_proba(X)[:, 1]
    X_scaled   = scaler.transform(X)
    iso_scores = iso_model.decision_function(X_scaled)
    hybrid     = compute_hybrid_score(
        xgb_proba, iso_scores, df_proc["is_resume_complete"].values
    )

    # Simpan hasil ke CSV untuk digunakan Streamlit
    df_result = df_raw.copy()
    df_result["xgb_fraud_prob"]     = np.round(xgb_proba * 100, 2)
    df_result["iso_anomaly_score"]  = np.round(iso_scores, 4)
    df_result["risk_score"]         = np.round(hybrid, 2)
    df_result["risk_label"]         = pd.cut(
        df_result["risk_score"],
        bins=[0, 40, 70, 100],
        labels=["🟢 Rendah", "🟡 Sedang", "🔴 Tinggi"],
        include_lowest=True
    )
    df_result["xgb_prediction"]     = xgb_model.predict(X)

    result_path = os.path.join(BASE_DIR, "bpjs_claims_results.csv")
    df_result.to_csv(result_path, index=False)
    print(f"[INFO] Hasil prediksi disimpan ke: {result_path}")

    # ── Simpan Metadata ─────────────────────────────────────
    le_icd  = joblib.load(os.path.join(MODEL_DIR, "le_icd.pkl"))
    le_room = joblib.load(os.path.join(MODEL_DIR, "le_room.pkl"))
    save_feature_metadata(
        FEATURE_COLS,
        list(le_icd.classes_),
        list(le_room.classes_)
    )

    # ── Summary ─────────────────────────────────────────────
    high_risk = (df_result["risk_label"] == "🔴 Tinggi").sum()
    total_loss = df_result.loc[df_result["risk_label"] == "🔴 Tinggi", "total_cost"].sum()
    print("\n" + "=" * 60)
    print(f"  Total Klaim        : {len(df_result)}")
    print(f"  Klaim Berisiko Tinggi: {high_risk} ({high_risk/len(df_result)*100:.1f}%)")
    print(f"  Potensi Kerugian   : Rp {total_loss:,.0f}")
    print("=" * 60)
    print("\n[DONE] Pelatihan selesai. Jalankan: streamlit run app.py")


if __name__ == "__main__":
    main()
