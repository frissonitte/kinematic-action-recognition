# ISE446 Big Data Analysis — Industry 5.0 Kinematic Action Recognition

A full end-to-end pipeline for classifying factory operator actions from high-frequency motion-capture sensor data (~10 GB, 132 channels, 200 Hz). Covers out-of-core data engineering, time-series feature extraction, unsupervised and supervised learning, and real-time streaming with concept drift detection.

---

## Results

| Phase | Method                            | Key Result                                                            |
| ----- | --------------------------------- | --------------------------------------------------------------------- |
| 1     | Dask out-of-core ingestion        | 10 GB CSV → 6.86 GB Parquet (3× compression)                          |
| 2     | Scalable EDA                      | README said binary; EDA found 15 distinct classes                     |
| 3     | Rolling-window feature extraction | 9.7M rows → 97,612 windows × 528 features                             |
| 4A    | MiniBatchKMeans + PCA + Hungarian | Macro F1 ≈ 0.14 (curse of dimensionality baseline)                    |
| 4B    | RandomForest (binary, 0 vs 1)     | Macro F1 ≈ 0.9995 · 1.3s train · 8.7 MB RAM                           |
| 5     | ARF + ADWIN streaming             | 81 win/sec · drift detected in ~58 windows (~290 ms) · 59 MB peak RAM |

---

## Repository Structure

```
├── pipeline/
│ └── phase-1-data-architecture.py # CSV → Parquet via Dask (out-of-core)
│ └── phase-2-scalable-eda.py # Label distribution, stationarity checks, boxplots
│ └── phase-3-feature-engineering.py # Sliding-window extraction (200-row, 50% overlap)
│ └── phase-4a-unsupervised.py # MiniBatchKMeans + PCA + Hungarian matching
│ └── phase-4b-supervised.py # RandomForest on binary subset (class 0 vs 1)
│ └── phase-5-streaming.py # Prequential ARF + ADWIN drift detection
│ └── requirements.txt
├── competition/
│ └── ise446-competition.ipynb
├── plots/
└── README.md
```

---

## Data

The raw CSV (~10 GB) is proprietary motion-capture data and **cannot be redistributed**.

Phases 2–5 consume the cleaned Parquet produced by Phase 1. To reproduce from scratch you need the original CSV. To reproduce Phases 2–5 only, point `data/main_data_parquet` at any compatible Parquet directory with the same schema (134 columns: `Milliseconds`, 132 sensor channels, `LABEL`).

---

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.12. Tested on Windows with Ryzen 7 7800X3D / 32 GB DDR5. Phase 1 and 3 spin up a local Dask cluster (8 workers × 3 GB); adjust `n_workers` and `memory_limit` in those files to match your system.

---

## Running

Run phases in order. Each phase reads from `data/` and writes outputs to `data/`, `models/`, or `plots/`.

```bash
python phase-1-data-architecture.py   # ~5 min — produces data/main_data_parquet
python phase-2-scalable-eda.py        # ~1 min — produces plots/phase2_*.png
python phase-3-feature-engineering.py # ~10 min — produces data/main_data_features_parquet
python phase-4a-unsupervised.py       # ~2 min — produces models/kmeans_model.pkl, plots/phase4a_*.png
python phase-4b-supervised.py         # ~2 min — produces models/rf_supervised.pkl, plots/phase4b_*.png
python phase-5-streaming.py           # ~20 min — produces plots/phase5_accuracy_over_time.png
```

---

## Technical Notes

**Why not a random train/test split in Phase 4B?**
The 50% overlapping rolling windows mean adjacent windows share raw sensor rows. A random split lets a window's "twin" land in the training set while the original goes to test — the model memorises rather than generalises, producing a suspiciously perfect F1 of 1.0000. The fix is a systematic every-5th-window split, which preserves class balance while keeping adjacent windows strictly separated.

**Why K-Means underperforms (F1 ≈ 0.14) even after PCA?**
In a 528-dimensional space, Euclidean distances converge (curse of dimensionality) and the 15 action classes form dense overlapping clusters. PCA to 50 components retains 94.3% of variance and partially restores meaningful geometry, but the classes themselves are too fine-grained (e.g. "tighten" vs "loosen") for distance-based separation without label supervision.

**Why the "binary" framing in Phase 4B?**
The original project rubric required a classical supervised model on binary labels (0/1). Classes 3–15 (the additional actions discovered in Phase 2) were reserved for the unsupervised task. The RF treats only the binary subset.

**Concept drift simulation (Phase 5)**
The stream replays the feature matrix in temporal order. At the 50% mark, 30% of labels are randomly flipped to simulate a production-line fault. ADWIN detects the shift 58 windows (~290 ms at 200 Hz) after injection.

---

## Also in this project

A separate Kaggle competition notebook (`competition-notebook.ipynb`) was developed in parallel for the _Industry 5.0 Scalable Kinematic Action Recognition_ competition — same domain, but a self-contained LightGBM + RandomForest ensemble on a different (non-proprietary) competition dataset. It placed **1st on the private leaderboard (0.94169 accuracy)**.
