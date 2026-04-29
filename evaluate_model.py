# evaluate_model.py - BPJS Fraud Detection Evaluation Script

import os, sys, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score, accuracy_score,
    average_precision_score
)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

FEATURE_COLS = [
    "patient_age", "room_type_enc", "icd_10_code_enc",
    "total_cost", "is_resume_complete", "auth_signature", "los"
]
FEATURE_LABELS = {
    "patient_age":      "Usia Pasien",
    "room_type_enc":    "Tipe Kamar",
    "icd_10_code_enc":  "Kode ICD-10",
    "total_cost":       "Total Biaya",
    "is_resume_complete": "Kelengkapan Resume",
    "auth_signature":   "Tanda Tangan DPJP",
    "los":              "Length of Stay"
}

def sep(char="=", n=62): print(char * n)
def title(text): sep(); print(f"  {text}"); sep()
def subtitle(text): print(f"\n{'─'*62}"); print(f"  {text}"); print(f"{'─'*62}")

# ── Muat model & data ─────────────────────────────────────────
title("VeraMed AI — Laporan Evaluasi Model Deteksi Fraud BPJS")

try:
    xgb     = joblib.load(os.path.join(MODEL_DIR, "xgboost_model.pkl"))
    iso     = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))
    scaler  = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    le_room = joblib.load(os.path.join(MODEL_DIR, "le_room.pkl"))
    le_icd  = joblib.load(os.path.join(MODEL_DIR, "le_icd.pkl"))
    print("  [OK] Semua model berhasil dimuat dari folder /models/")
except FileNotFoundError as e:
    print(f"  [ERROR] {e}"); sys.exit(1)

df = pd.read_csv(os.path.join(BASE_DIR, "bpjs_claims_synthetic.csv"))
df["room_type_enc"]   = le_room.transform(df["room_type"])
df["icd_10_code_enc"] = le_icd.transform(df["icd_10_code"])

X = df[FEATURE_COLS].values
y = df["is_fraud"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_test_scaled = scaler.transform(X_test)
X_all_scaled  = scaler.transform(X)

# ════════════════════════════════════════════════════════════
# 1. INFORMASI DATASET
# ════════════════════════════════════════════════════════════
subtitle("1. INFORMASI DATASET")
total = len(df)
n_fraud  = (y == 1).sum()
n_normal = (y == 0).sum()
print(f"  Total Baris          : {total:,}")
print(f"  Kolom Fitur          : {len(FEATURE_COLS)}")
print(f"  Label Normal (0)     : {n_normal:,}  ({n_normal/total*100:.1f}%)")
print(f"  Label Fraud  (1)     : {n_fraud:,}   ({n_fraud/total*100:.1f}%)")
print(f"  Rasio Imbalance      : 1 : {n_normal/n_fraud:.1f}")
print(f"\n  Split Train/Test     : 80% / 20%")
print(f"  Train set            : {len(X_train):,} baris")
print(f"  Test set             : {len(X_test):,} baris")

incomplete_doc = (df["is_resume_complete"] == 0).sum()
no_auth        = (df["auth_signature"] == 0).sum()
print(f"\n  Dokumentasi Tidak Lengkap (is_resume_complete=0): {incomplete_doc} ({incomplete_doc/total*100:.1f}%)")
print(f"  Tanpa Tanda Tangan (auth_signature=0)           : {no_auth} ({no_auth/total*100:.1f}%)")

# ════════════════════════════════════════════════════════════
# 2. XGBOOST — SUPERVISED CLASSIFICATION
# ════════════════════════════════════════════════════════════
subtitle("2. XGBOOST CLASSIFIER (Supervised)")

y_pred  = xgb.predict(X_test)
y_proba = xgb.predict_proba(X_test)[:, 1]

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
roc  = roc_auc_score(y_test, y_proba)
ap   = average_precision_score(y_test, y_proba)

print(f"\n  Accuracy             : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Precision (Fraud)    : {prec:.4f}")
print(f"  Recall    (Fraud)    : {rec:.4f}")
print(f"  F1-Score  (Fraud)    : {f1:.4f}")
print(f"  ROC-AUC Score        : {roc:.4f}")
print(f"  Average Precision    : {ap:.4f}")

print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal","Fraud"], digits=4))

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"  Confusion Matrix:")
print(f"  {'':20s}  Predicted Normal  Predicted Fraud")
print(f"  {'Actual Normal':20s}  {tn:^16}  {fp:^15}")
print(f"  {'Actual Fraud':20s}  {fn:^16}  {tp:^15}")
print(f"\n  True  Positives (TP) : {tp}")
print(f"  True  Negatives (TN) : {tn}")
print(f"  False Positives (FP) : {fp}")
print(f"  False Negatives (FN) : {fn}")
print(f"\n  Specificity          : {tn/(tn+fp):.4f}")
print(f"  Negative Pred. Value : {tn/(tn+fn):.4f}")
print(f"  Miss Rate (FN Rate)  : {fn/(fn+tp):.4f}  ({fn/(fn+tp)*100:.1f}%)")
print(f"  Fall-out  (FP Rate)  : {fp/(fp+tn):.4f}  ({fp/(fp+tn)*100:.1f}%)")

# Cross-Validation
print(f"\n  Cross-Validation (5-Fold Stratified):")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = cross_val_score(xgb, X, y, cv=cv, scoring="accuracy",  n_jobs=-1)
cv_f1  = cross_val_score(xgb, X, y, cv=cv, scoring="f1",        n_jobs=-1)
cv_roc = cross_val_score(xgb, X, y, cv=cv, scoring="roc_auc",   n_jobs=-1)

def fmt_cv(arr): return f"Mean={arr.mean():.4f}  Std={arr.std():.4f}  [{arr.min():.4f} – {arr.max():.4f}]"
print(f"    Accuracy  : {fmt_cv(cv_acc)}")
print(f"    F1-Score  : {fmt_cv(cv_f1)}")
print(f"    ROC-AUC   : {fmt_cv(cv_roc)}")

# ════════════════════════════════════════════════════════════
# 3. ISOLATION FOREST — UNSUPERVISED
# ════════════════════════════════════════════════════════════
subtitle("3. ISOLATION FOREST (Unsupervised Anomaly Detection)")

iso_pred   = iso.predict(X_all_scaled)          # -1=anomali, 1=normal
iso_scores = iso.decision_function(X_all_scaled) # lebih negatif = lebih aneh

# Konversi ke label binary (1=anomali)
iso_bin = np.where(iso_pred == -1, 1, 0)

iso_tp = ((iso_bin == 1) & (y == 1)).sum()
iso_fp = ((iso_bin == 1) & (y == 0)).sum()
iso_tn = ((iso_bin == 0) & (y == 0)).sum()
iso_fn = ((iso_bin == 0) & (y == 1)).sum()

iso_prec = iso_tp / (iso_tp + iso_fp) if (iso_tp + iso_fp) > 0 else 0
iso_rec  = iso_tp / (iso_tp + iso_fn) if (iso_tp + iso_fn) > 0 else 0
iso_f1   = 2 * iso_prec * iso_rec / (iso_prec + iso_rec) if (iso_prec + iso_rec) > 0 else 0

print(f"\n  Total Anomali Terdeteksi IF : {iso_bin.sum():,} ({iso_bin.sum()/len(y)*100:.1f}%)")
print(f"  Contamination Parameter    : 0.25 (25%)")
print(f"  n_estimators               : 200 trees")
print(f"\n  Perbandingan dengan Label Sebenarnya:")
print(f"    True  Positives (TP)     : {iso_tp}")
print(f"    False Positives (FP)     : {iso_fp}")
print(f"    True  Negatives (TN)     : {iso_tn}")
print(f"    False Negatives (FN)     : {iso_fn}")
print(f"\n    Precision                : {iso_prec:.4f}")
print(f"    Recall                   : {iso_rec:.4f}")
print(f"    F1-Score                 : {iso_f1:.4f}")
print(f"\n  Anomaly Score Stats:")
print(f"    Min Score                : {iso_scores.min():.4f}")
print(f"    Max Score                : {iso_scores.max():.4f}")
print(f"    Mean Score               : {iso_scores.mean():.4f}")
print(f"    Std  Score               : {iso_scores.std():.4f}")

# Score breakdown per label
fraud_scores  = iso_scores[y == 1]
normal_scores = iso_scores[y == 0]
print(f"\n  Anomaly Score (Fraud claims)  — Mean: {fraud_scores.mean():.4f}  Std: {fraud_scores.std():.4f}")
print(f"  Anomaly Score (Normal claims) — Mean: {normal_scores.mean():.4f}  Std: {normal_scores.std():.4f}")
print(f"  Separability (delta mean)     : {normal_scores.mean() - fraud_scores.mean():.4f}")

# ════════════════════════════════════════════════════════════
# 4. HYBRID RISK SCORE
# ════════════════════════════════════════════════════════════
subtitle("4. HYBRID RISK SCORE (Gabungan XGBoost + IF + Dokumentasi)")

results = pd.read_csv(os.path.join(BASE_DIR, "bpjs_claims_results.csv"))
rs = results["risk_score"]
rl = results["risk_label"].astype(str)

print(f"\n  Formula  : 55% x XGBoost_Prob + 30% x IF_Norm + 15% x Doc_Penalty")
print(f"  Range    : 0 (sangat aman) → 100 (sangat berisiko)")
print(f"\n  Statistik Risk Score Keseluruhan:")
print(f"    Mean   : {rs.mean():.2f}")
print(f"    Median : {rs.median():.2f}")
print(f"    Std    : {rs.std():.2f}")
print(f"    Min    : {rs.min():.2f}")
print(f"    Max    : {rs.max():.2f}")
print(f"    P25    : {rs.quantile(0.25):.2f}")
print(f"    P75    : {rs.quantile(0.75):.2f}")
print(f"    P90    : {rs.quantile(0.90):.2f}")

print(f"\n  Distribusi Risk Level:")
for label, color_tag in [("Rendah","[0–40]"), ("Sedang","[41–70]"), ("Tinggi","[71–100]")]:
    key  = f"Rendah" if "Rendah" in label else ("Sedang" if "Sedang" in label else "Tinggi")
    mask = rl.str.contains(label)
    cnt  = mask.sum()
    loss = results.loc[mask, "total_cost"].sum()
    print(f"    {label:7s} {color_tag:10s}: {cnt:4,} klaim  ({cnt/len(results)*100:.1f}%)  |  Biaya: Rp {loss:>15,.0f}")

total_high_loss = results.loc[rl.str.contains("Tinggi"), "total_cost"].sum()
print(f"\n  Total Potensi Kerugian (Risiko Tinggi): Rp {total_high_loss:,.0f}")
print(f"  Rata-rata Biaya Klaim Risiko Tinggi   : Rp {results.loc[rl.str.contains('Tinggi'), 'total_cost'].mean():,.0f}")

# ════════════════════════════════════════════════════════════
# 5. FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════
subtitle("5. FEATURE IMPORTANCE (XGBoost — gain)")

fi = xgb.feature_importances_
fi_pairs = sorted(zip(FEATURE_COLS, fi), key=lambda x: x[1], reverse=True)

print(f"\n  {'Rank':<6} {'Fitur':<30} {'Importance':>12}  Bar")
print(f"  {'─'*6} {'─'*30} {'─'*12}  {'─'*30}")
for rank, (feat, imp) in enumerate(fi_pairs, 1):
    label = FEATURE_LABELS.get(feat, feat)
    bar   = "█" * int(imp * 150)
    print(f"  {rank:<6} {label:<30} {imp:>12.6f}  {bar}")

top_feat = FEATURE_LABELS.get(fi_pairs[0][0], fi_pairs[0][0])
print(f"\n  Fitur paling berpengaruh: [{top_feat}] ({fi_pairs[0][1]:.4f})")

# ════════════════════════════════════════════════════════════
# 6. ANALISIS FRAUD RULE
# ════════════════════════════════════════════════════════════
subtitle("6. ANALISIS FRAUD RULES (Rule-based Breakdown)")

df_r = results.copy()
df_r["risk_label"] = df_r["risk_label"].astype(str)

# Rule 1: R50.9 + biaya > 5jt
r1 = df_r[(df_r["icd_10_code"]=="R50.9") & (df_r["total_cost"]>5_000_000)]
# Rule 2: resume=0 & auth=0
r2 = df_r[(df_r["is_resume_complete"]==0) & (df_r["auth_signature"]==0)]
# Rule 3: LOS=1 & biaya > 7jt
r3 = df_r[(df_r["los"]==1) & (df_r["total_cost"]>7_000_000)]

print(f"\n  Rule 1 — R50.9 + Biaya > Rp 5jt (Inflated Cost):")
print(f"    Total klaim terdampak  : {len(r1):,}")
print(f"    Terdeteksi fraud AI    : {(r1['xgb_prediction']==1).sum()}")
print(f"    Total biaya berisiko   : Rp {r1['total_cost'].sum():,.0f}")

print(f"\n  Rule 2 — Resume Tidak Lengkap & Tanpa Tanda Tangan:")
print(f"    Total klaim terdampak  : {len(r2):,}")
print(f"    Terdeteksi fraud AI    : {(r2['xgb_prediction']==1).sum()}")
print(f"    % dari seluruh dataset : {len(r2)/len(df_r)*100:.1f}%")

print(f"\n  Rule 3 — LOS = 1 hari & Biaya > Rp 7jt:")
print(f"    Total klaim terdampak  : {len(r3):,}")
print(f"    Terdeteksi fraud AI    : {(r3['xgb_prediction']==1).sum()}")
print(f"    Total biaya berisiko   : Rp {r3['total_cost'].sum():,.0f}")

# ════════════════════════════════════════════════════════════
# 7. RINGKASAN EKSEKUTIF
# ════════════════════════════════════════════════════════════
subtitle("7. RINGKASAN EKSEKUTIF")
print(f"""
  MODEL PERFORMANCE SUMMARY
  ─────────────────────────────────────────────────────────
  XGBoost Accuracy        : {acc*100:.2f}%
  XGBoost ROC-AUC         : {roc:.4f}   (Excellent > 0.90)
  XGBoost F1-Score        : {f1:.4f}   (Fraud class)
  XGBoost Avg Precision   : {ap:.4f}
  Cross-Val ROC-AUC Mean  : {cv_roc.mean():.4f}
  Cross-Val Accuracy Mean : {cv_acc.mean():.4f}

  BUSINESS IMPACT SUMMARY
  ─────────────────────────────────────────────────────────
  Klaim Risiko Tinggi     : {rl.str.contains('Tinggi').sum():,} klaim
  Potensi Kerugian        : Rp {total_high_loss:,.0f}
  Fraud Tidak Terdeteksi  : {fn} klaim (Miss Rate: {fn/(fn+tp)*100:.1f}%)
  False Alarm             : {fp} klaim (dari {(y==0).sum()} normal)

  KESIMPULAN
  ─────────────────────────────────────────────────────────
  ROC-AUC: {roc:.4f} | Accuracy: {acc*100:.2f}% | Top Feature: [{top_feat}]
""")
sep()
print("  Evaluasi selesai.")
sep()
