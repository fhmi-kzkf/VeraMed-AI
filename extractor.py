# extractor.py - Medical Document Extraction Engine
# Uses Google Gemini Vision API to parse PDF/images into structured JSON.

import os
import re
import json
import base64
from pathlib import Path

# ── Muat variabel dari .env (python-dotenv) ────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv opsional; env var bisa di-set langsung di server

# ── Cek ketersediaan library opsional ─────────────────────────
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import fitz  # PyMuPDF — konversi PDF ke gambar
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ============================================================
# PROMPT SYSTEM — Instruksi Audit Medis
# ============================================================
AUDIT_SYSTEM_PROMPT = """
Anda adalah Senior Medical Auditor & Data Extraction Engine khusus untuk 
ekosistem BPJS Kesehatan Indonesia. Tugas Anda: membaca dokumen rekam medis 
dan mengekstraknya menjadi JSON terstruktur.

SCHEMA JSON WAJIB (tanpa teks lain, hanya JSON mentah):
{
  "patient_age": <integer — usia pasien dalam tahun, atau 0 jika tidak ada>,
  "room_type": <string — pilih SATU dari: "SAPHIRE", "ZAMRUD", "BERLIAN", atau null>,
  "icd_10_code": <string — kode ICD-10 standar (contoh: E11.5, A91, R50.9), atau null>,
  "total_cost": <integer — total klaim tanpa simbol/titik/koma, atau 0>,
  "is_resume_complete": <integer 0 atau 1 — 0 jika ada bagian kosong/tidak terbaca>,
  "auth_signature": <integer 0 atau 1 — 0 jika tidak ada tanda tangan/nama DPJP>,
  "los": <integer — durasi rawat inap dalam hari, atau 0>,
  "ai_analysis": <string — temuan audit maksimal 2 kalimat>
}

ATURAN AUDIT:
1. is_resume_complete = 0 jika: identitas pasien tidak lengkap, diagnosa kosong,
   ringkasan penyakit tidak ada, atau tulisan tidak terbaca sama sekali.
   Ingat: 59.43% masalah klaim BPJS ada di pendokumentasian.
2. auth_signature = 0 jika: tanda tangan DPJP/nama jelas dokter tidak ditemukan.
   Faktor ini memiliki pengaruh 50.69% terhadap validitas klaim.
3. Jika icd_10_code = R50.9 dan total_cost > 5000000 → sebutkan "Indikasi Inflated Cost"
   di ai_analysis.
4. Jika auth_signature = 0 → tambahkan "Klaim tidak sah secara administrasi" di ai_analysis.
5. room_type mapping: Kelas 1/VIP/Utama → BERLIAN, Kelas 2 → ZAMRUD, Kelas 3 → SAPHIRE.
   Jika tidak jelas, gunakan konteks biaya atau keterangan ruangan.
6. Untuk tulisan tangan dokter yang sulit dibaca: gunakan interpretasi kontekstual.
   Jika benar-benar tidak dapat diinterpretasi → is_resume_complete = 0.

OUTPUT: Hanya JSON mentah. Tidak ada teks pembuka, penutup, atau markdown.
"""


# ============================================================
# FUNGSI: KONVERSI PDF KE GAMBAR
# ============================================================
def pdf_to_images(pdf_bytes: bytes) -> list[bytes]:
    """Convert PDF pages to PNG images."""
    if not PYMUPDF_AVAILABLE:
        raise ImportError("PyMuPDF tidak terinstall. Jalankan: pip install pymupdf")
    
    images = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        # Render dengan DPI tinggi untuk OCR yang akurat
        mat  = fitz.Matrix(2.0, 2.0)  # 2x zoom = ~144 DPI
        pix  = page.get_pixmap(matrix=mat, alpha=False)
        images.append(pix.tobytes("png"))
    doc.close()
    return images


# ============================================================
# FUNGSI: ENCODE GAMBAR KE BASE64
# ============================================================
def encode_image_base64(image_bytes: bytes, mime_type: str = "image/png") -> dict:
    """Mengubah bytes gambar menjadi format inline_data untuk Gemini API."""
    encoded = base64.standard_b64encode(image_bytes).decode("utf-8")
    return {"inline_data": {"mime_type": mime_type, "data": encoded}}


# ============================================================
# FUNGSI: EKSTRAKSI DENGAN GEMINI VISION
# ============================================================
def extract_with_gemini(
    file_bytes: bytes,
    file_ext: str,
    api_key: str = "",
    model_name: str = ""
) -> dict:
    """Extract JSON data from document using Gemini Vision API."""
    if not GEMINI_AVAILABLE:
        raise ImportError("google-generativeai tidak terinstall. Jalankan: pip install google-generativeai")

    # Prioritas: argumen → env var → default fallback
    resolved_key   = api_key   or os.getenv("GOOGLE_API_KEY", "")
    resolved_model = model_name or os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")

    if not resolved_key:
        raise ValueError("GOOGLE_API_KEY tidak ditemukan. Set di .env atau environment variable.")

    genai.configure(api_key=resolved_key)
    model = genai.GenerativeModel(resolved_model)
    
    # ── Siapkan konten gambar berdasarkan tipe file ─────────
    content_parts = [AUDIT_SYSTEM_PROMPT]
    
    if file_ext.lower() == ".pdf":
        # Konversi PDF ke gambar terlebih dahulu
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF diperlukan untuk membaca PDF. pip install pymupdf")
        
        pages = pdf_to_images(file_bytes)
        for i, page_img in enumerate(pages):
            content_parts.append(f"\n--- Halaman {i+1} ---")
            content_parts.append(encode_image_base64(page_img, "image/png"))
    
    elif file_ext.lower() in [".jpg", ".jpeg"]:
        content_parts.append(encode_image_base64(file_bytes, "image/jpeg"))
    
    elif file_ext.lower() == ".png":
        content_parts.append(encode_image_base64(file_bytes, "image/png"))
    
    elif file_ext.lower() == ".webp":
        content_parts.append(encode_image_base64(file_bytes, "image/webp"))
    
    else:
        raise ValueError(f"Format file tidak didukung: {file_ext}")
    
    # ── Kirim ke Gemini & terima respons ────────────────────
    response = model.generate_content(
        content_parts,
        generation_config=genai.types.GenerationConfig(
            temperature=0.1,       # rendah = deterministik & konsisten
            max_output_tokens=512,
        )
    )
    
    raw_text = response.text.strip()
    
    # ── Bersihkan markdown code block jika ada ──────────────
    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
    raw_text = re.sub(r"\s*```$",          "", raw_text)
    raw_text = raw_text.strip()
    
    # ── Parse JSON ──────────────────────────────────────────
    result = json.loads(raw_text)
    
    # ── Validasi & sanitasi tipe data ───────────────────────
    result = sanitize_extraction(result)
    return result


# ============================================================
# FUNGSI: SANITASI HASIL EKSTRAKSI
# ============================================================
def sanitize_extraction(data: dict) -> dict:
    """Validate and clean extracted JSON fields."""
    VALID_ROOMS = {"SAPHIRE", "ZAMRUD", "BERLIAN"}
    
    sanitized = {
        "patient_age":       int(data.get("patient_age") or 0),
        "room_type":         str(data.get("room_type") or "").upper() or None,
        "icd_10_code":       str(data.get("icd_10_code") or "").strip().upper() or None,
        "total_cost":        int(data.get("total_cost") or 0),
        "is_resume_complete": int(data.get("is_resume_complete") or 0),
        "auth_signature":    int(data.get("auth_signature") or 0),
        "los":               int(data.get("los") or 0),
        "ai_analysis":       str(data.get("ai_analysis") or "Tidak ada analisis.")
    }
    
    # Pastikan room_type valid
    if sanitized["room_type"] not in VALID_ROOMS:
        sanitized["room_type"] = None
    
    # Pastikan binary fields hanya 0 atau 1
    sanitized["is_resume_complete"] = min(1, max(0, sanitized["is_resume_complete"]))
    sanitized["auth_signature"]     = min(1, max(0, sanitized["auth_signature"]))
    
    # Terapkan aturan audit tambahan pada ai_analysis
    flags = []
    if sanitized["icd_10_code"] == "R50.9" and sanitized["total_cost"] > 5_000_000:
        flags.append("Indikasi Inflated Cost")
    if sanitized["auth_signature"] == 0:
        flags.append("Klaim tidak sah secara administrasi")
    if flags:
        prefix = " | ".join(flags) + ". "
        if prefix not in sanitized["ai_analysis"]:
            sanitized["ai_analysis"] = prefix + sanitized["ai_analysis"]
    
    return sanitized


# ============================================================
# FUNGSI: MOCK EXTRACTION (untuk demo tanpa API key)
# ============================================================
def mock_extraction(filename: str = "") -> dict:
    """Return dummy data for testing without API key."""
    import random
    rng = random.Random(hash(filename) % 1000)
    
    scenarios = [
        {
            "patient_age": 45,
            "room_type": "ZAMRUD",
            "icd_10_code": "R50.9",
            "total_cost": 7500000,
            "is_resume_complete": 0,
            "auth_signature": 0,
            "los": 1,
            "ai_analysis": "Indikasi Inflated Cost | Klaim tidak sah secara administrasi. Biaya Rp 7.500.000 tidak wajar untuk diagnosis R50.9 (Demam tidak spesifik) dengan LOS hanya 1 hari; tanda tangan DPJP tidak ditemukan."
        },
        {
            "patient_age": 62,
            "room_type": "BERLIAN",
            "icd_10_code": "E11.5",
            "total_cost": 4200000,
            "is_resume_complete": 1,
            "auth_signature": 1,
            "los": 5,
            "ai_analysis": "Dokumen lengkap dengan tanda tangan DPJP yang valid. Diagnosis Diabetes Mellitus tipe-2 dengan komplikasi perifer sesuai dengan LOS 5 hari dan total biaya klaim."
        },
        {
            "patient_age": 28,
            "room_type": "SAPHIRE",
            "icd_10_code": "A91",
            "total_cost": 2800000,
            "is_resume_complete": 0,
            "auth_signature": 1,
            "los": 4,
            "ai_analysis": "Resume medis tidak lengkap — bagian ringkasan riwayat penyakit kosong. Diagnosis Dengue Hemorrhagic Fever (DHF) dengan biaya klaim masih dalam batas wajar."
        }
    ]
    return rng.choice(scenarios)


# ============================================================
# FUNGSI UTAMA: ENTRY POINT
# ============================================================
def extract_from_document(
    file_bytes: bytes,
    filename: str,
    api_key: str = "",
    use_mock: bool = False
) -> dict:
    """Main entry point for document extraction."""
    if use_mock:
        return mock_extraction(filename)

    # Cek apakah ada key yang bisa digunakan
    resolved_key = api_key or os.getenv("GOOGLE_API_KEY", "")
    if not resolved_key:
        return mock_extraction(filename)  # fallback ke demo jika tidak ada key

    ext = Path(filename).suffix.lower()
    return extract_with_gemini(file_bytes, ext, resolved_key)
