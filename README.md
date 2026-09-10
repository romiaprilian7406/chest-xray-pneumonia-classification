# Chest X-Ray Pneumonia Classification

Klasifikasi biner (Normal vs Pneumonia) dari citra X-ray dada, menggunakan dataset [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (Paul Mooney, Kaggle) — 5,856 gambar.

## 1. Project Overview

| | |
|---|---|
| **Task** | Binary image classification (Normal / Pneumonia) |
| **Dataset** | Kaggle `paultimothymooney/chest-xray-pneumonia`, diunduh via `kagglehub` (anonim, tanpa API key) |
| **Model final** | **Custom CNN** (arsitektur dari referensi `chest-x-ray-images-pneumonia-using-cnn.ipynb`) |
| **Metrik prioritas** | **Recall** (sensitivitas kelas Pneumonia) — bukan accuracy |
| **Notebook** | [`notebooks/pneumonia-cls.ipynb`](notebooks/pneumonia-cls.ipynb) |
| **Deployment** | [`app/app.py`](app/app.py) — Streamlit |

Alasan recall diprioritaskan: ini konteks skrining medis, false negative (pneumonia terlewat) jauh lebih berbahaya daripada false positive, dan dataset timpang (~73% kelas Pneumonia) membuat accuracy menyesatkan.

## 2. Pipeline

```
Kaggle dataset (kagglehub, anonymous)
        │
        ▼
EDA  (distribusi kelas, resolusi/aspect ratio, sampling visual per kelas)
        │
        ▼
Patient-grouped split  (GroupShuffleSplit, 2x: test lalu val dari sisa train)
  + assert disjoint antar patient_id (train/val/test tidak boleh overlap)
        │
        ▼
tf.data pipeline  (image_dataset_from_directory, crop_to_aspect_ratio=True,
                    color_mode='rgb', augmentasi sebagai layer Keras di GPU)
        │
        ├──► Custom CNN        (grayscale, dari nol, 3x Conv-Pool blok)
        ├──► DenseNet121 TL    (freeze → fine-tune 30 layer terakhir)
        └──► EfficientNetB0 TL (freeze → fine-tune 20 layer terakhir)
        │
        ▼
Evaluation  (accuracy, precision, recall, specificity, F1, ROC-AUC + confusion matrix)
        │
        ▼
Comparison table (diurutkan berdasarkan recall) → export model terbaik → Streamlit
```

### Kenapa patient-grouped split, bukan flat split?

Nama file Pneumonia mengikuti pola `person<ID>_(bacteria|virus)_<n>.jpeg` — satu pasien bisa punya beberapa gambar. Split flat (`train_test_split(..., stratify=label)`, dipakai notebook-notebook Kaggle populer untuk dataset ini) berisiko menaruh gambar pasien yang sama di train **dan** test sekaligus → model "mengenali pasien", bukan pola penyakit → skor evaluasi bocor/terlalu optimis. Notebook ini split di level **pasien** (`GroupShuffleSplit`) plus assert disjointness yang gagal keras kalau ada overlap.

## 3. Model & Training Detail

| Model | Arsitektur | Training |
|---|---|---|
| **Custom CNN** | Grayscale, `Conv32→Pool→Conv64→Pool→Conv128→Pool→Flatten→Dense(64→128→256)→Dense(1,sigmoid)` | 1 tahap, Adam LR 1e-3 |
| **DenseNet121** | ImageNet backbone frozen → `GAP→BatchNorm→Dropout(0.3)→Dense(1,sigmoid)` | 2 tahap: freeze (LR 1e-4) → fine-tune 30 layer terakhir (LR 1e-5) |
| **EfficientNetB0** | sama seperti DenseNet121, tanpa `preprocess_input` (punya rescaling internal) | 2 tahap: freeze (LR 1e-4) → fine-tune 20 layer terakhir (LR 1e-5) |

Semua model: `class_weight` untuk imbalance, `EarlyStopping`+`ReduceLROnPlateau`+`ModelCheckpoint` (monitor `val_loss`), augmentasi via layer Keras (`RandomZoom`, `RandomRotation`, `RandomTranslation`).

## 4. Hasil (test set, patient-grouped split)

| Model | Accuracy | Precision | **Recall** | Specificity | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| 🏆 **Custom CNN** | 0.960 | 0.979 | **0.963** | 0.953 | 0.971 | 0.989 |
| EfficientNetB0 | 0.924 | 0.989 | 0.901 | 0.977 | 0.943 | 0.986 |
| DenseNet121 | 0.884 | 0.988 | 0.844 | 0.977 | 0.910 | 0.981 |

**Custom CNN** menang di hampir semua metrik, termasuk recall (metrik prioritas) — dan yang otomatis diekspor jadi `pneumonia_model.keras` oleh notebook (cell "Export Model" memilih model dengan recall tertinggi secara dinamis dari comparison table, bukan hardcode). Ini temuan yang sah, bukan anomali: dataset grayscale, task biner sederhana pada dataset relatif kecil (~5,800 gambar) ternyata lebih cocok ditangkap CNN kecil yang dilatih khusus untuk domain ini, dibanding backbone ImageNet besar yang fitur pretrained-nya dioptimalkan untuk 1000 kelas objek natural.

## 5. Struktur Proyek

```
chest-xray-pneumonia/
├── notebooks/
│   └── pneumonia-cls.ipynb     ← EDA, split, training 3 model, evaluasi, export
├── app/
│   ├── app.py                  ← Streamlit app (upload gambar → prediksi)
│   └── models/
│       └── pneumonia_model.keras
├── .streamlit/
│   └── config.toml             ← tema warna app (hijau medis)
├── requirements.txt
├── runtime.txt                 ← python-3.11 (samakan dengan environment training)
└── README.md
```

## 6. Cara Menjalankan

### Training (notebook)

1. Buka [`notebooks/pneumonia-cls.ipynb`](notebooks/pneumonia-cls.ipynb) di Google Colab atau Kaggle Notebook.
2. Aktifkan **GPU** (Colab: Runtime → Change runtime type; Kaggle: Settings → Accelerator, dan **Internet** harus ON untuk download bobot ImageNet).
3. Run All — dataset otomatis terunduh via `kagglehub` tanpa API key. Cell terakhir mengekspor model terbaik ke `pneumonia_model.keras`.
4. Download hasil export, taruh di `app/models/pneumonia_model.keras`.

### Streamlit app (lokal)

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

Upload citra X-ray (JPG/JPEG/PNG) → aplikasi menampilkan gambar dan hasil prediksi (Normal/Pneumonia + confidence, dengan progress bar) berdampingan dalam layout 2 kolom.

**Catatan teknis:** layer grayscale konversi di Custom CNN memakai `Lambda(tf.image.rgb_to_grayscale)`, yang diserialisasi Keras dengan referensi modul non-standar (`builtins`). Karena itu, load model **wajib** pakai:
```python
tf.keras.models.load_model(MODEL_PATH, safe_mode=False, custom_objects={"rgb_to_grayscale": tf.image.rgb_to_grayscale})
```
(sudah diterapkan di `app/app.py`). Pastikan juga versi Python environment deployment sama dengan `runtime.txt` (3.11) untuk kompatibilitas format model.
