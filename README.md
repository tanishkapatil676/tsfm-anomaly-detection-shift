# Time-Series Foundation Models for Anomaly Detection under Distribution Shift

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build Status: Passing](https://img.shields.io/badge/Tests-52%2F52%20Passed-brightgreen.svg)](tests/)

An academic research and experimental benchmarking suite evaluating the robustness of pretrained **Time-Series Foundation Models (TSFMs)** compared to conventional baselines under controlled distribution shift on the **UCR Time Series Anomaly Archive**.

* **Student:** Tanishka Patil (TY IT, Division 3, PRN: 2405120)
* **Institute:** Institute of Information and Communication Technology (IICT), MGM University, Chh. Sambhajinagar
* **Academic Year:** 2026–2027
* **Project Guide:** `[To be confirmed: Guide Name, Designation]`

---

## 1. Project Overview & Research Context

### Research Question
> **"Can pretrained time-series foundation models remain reliable for anomaly detection when the test distribution changes?"**

### Research Gap
> **"Recent work shows that TSFMs can be adapted for anomaly detection, but evidence is fragmented across retrieval, adapters, embeddings, and reconstruction/forecasting-based methods. Their robustness and relative advantage over simple baselines under distribution shift remain an open experimental question."**

---

## 2. Current Implementation Status

* **Milestone 1 (Research Specification & Architecture):** **COMPLETED.** All 9 research planning documents locked and audited.
* **Milestone 2 (Project Foundation & Dataset Verification):** **COMPLETED.** Modular UCR data loader, 10-point quality validator, diagnostic profiler, master configuration, and dataset cards created.
* **Milestone 3 (Baseline Models & Initial Feasibility):** **COMPLETED.** `BaseAnomalyDetector` contract, leakage-safe windowing/scaling, Isolation Forest baseline, PyTorch LSTM Autoencoder baseline, and validation threshold selector.
* **Milestone 3A (Dataset Authenticity & TSFM Verification):** **COMPLETED.** Replaced placeholders with 5 authentic UCR series from Figshare release (`doi:10.6084/m9.figshare.26410744`). Documented MOMENT Python 3.13 blocker. Promoted `amazon/chronos-bolt-tiny` to active foundation model using official `chronos-forecasting` pipeline.
* **Milestone 4 (Controlled Shift Experiments & Evaluation Engine):** **SUPERSEDED & AUDITED.** Initial 45 runs archived into `results_superseded_m4/`.
* **Milestone 4A (Split, Threshold & Window-Alignment Audit & Clean Rerun):** **COMPLETED.** Disjoint train/val splits enforced, out-of-sample thresholds calibrated, causal window-end alignment verified, and clean 45-run benchmark executed with 0 failures.
* **Milestone 5 (Streamlit Dashboard & IICT Project Report):** **COMPLETED.** Interactive Streamlit dashboard implemented (`app.py`), complete 16-section individual IICT B.Tech Project Report generated (`report/Time_Series_Anomaly_Detection_Project_Report.docx`), and 52/52 automated tests passing.

---

## 3. Environment Setup & Execution Commands

### Prerequisites

```powershell
pip install -r requirements.txt
```

### Run Interactive Streamlit Dashboard (Milestone 5)
To launch the interactive demonstration dashboard on local browser:
```powershell
streamlit run app.py
```
Dashboard opens at `http://localhost:8501`. Loads pre-computed results — no inference or model download required.

### Generate Completed Academic Project Report (.docx)
To regenerate the full 16-section individual B.Tech project report with verified benchmark tables and embedded publication figures:
```powershell
python scripts/generate_report.py
```
Output document: `report/Time_Series_Anomaly_Detection_Project_Report.docx`

### Run Complete Automated Unit Test Suite
To verify the loader, baselines, causal Chronos adapter, metrics, shifts, persistence, dashboard logic, and report verification:
```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

### Run Controlled Distribution-Shift Benchmark
To re-execute the full 45-run evaluation matrix across 5 authentic UCR datasets, 3 models, and 3 shift levels:
```powershell
python -u experiments/run_experiment.py
```

### Run Model Smoke Test & Feasibility Runner
```powershell
python experiments/run_model_smoke_test.py
```

### Run Data Verification and Diagnostic Profiling
```powershell
python experiments/run_data_verification.py
```

---

## 4. Experiment Methodology

* **Datasets:** 5 authentic series from the UCR Time Series Anomaly Archive (`doi:10.6084/m9.figshare.26410744`): 3 ECG (sddb40), 1 respiratory (BIDMC1), 1 meteorological (CIMIS44).
* **Split protocol:** Disjoint train / validation / test splits. The integer prefix in UCR filenames is `t_split = train + val`; test begins at index `t_split`. No overlap.
* **Distribution shift:** Synthetic amplitude scaling — `x_test(t) ← a · x(t)` for `a ∈ {1.0 (ID), 1.25, 1.50}`. Scaler fitted on training split only.
* **Threshold:** 99th-percentile of validation-split anomaly scores, calibrated once on `ID` condition and **frozen** across all shift conditions for fair comparison.
* **Models:** Isolation Forest (sliding-window statistics), LSTM Autoencoder (reconstruction MSE), Chronos Bolt Tiny (zero-shot 1-step rolling forecast residual).
* **Metrics:** PR-AUC, ROC-AUC, F1, Precision, Recall, TP/FP/TN/FN, Δ-metric under shift.
* **Total runs:** 5 datasets × 3 models × 3 conditions = **45 runs**, all `status = SUCCESS`.

---

## 5. Primary Model Shortlist & Feasibility Status

1. **Isolation Forest Baseline:** Sliding window summary features + `sklearn.ensemble.IsolationForest`. **Status: VERIFIED ON CPU.**
2. **LSTM Autoencoder Baseline:** 2-layer sequence-to-sequence PyTorch autoencoder ($MSE$ loss). **Status: VERIFIED ON CPU.**
3. **Active TSFM (`amazon/chronos-bolt-tiny`):** Pretrained zero-shot 1-step rolling forecast absolute residual ($|x(t) - \hat{x}_{t, \text{median}}|$) via `chronos-forecasting` 2.3.2. **Status: VERIFIED & EVALUATED ON CPU.**
4. **Deferred TSFM Candidate (`AutonLab/MOMENT-1-small`):** Zero-shot masked reconstruction. **Status: DEFERRED** (Reproducible blocker: `momentfm 0.1.4` hard-pins `numpy==1.25.2` which lacks Python 3.13 wheels).

---

## 6. Dataset Citation

```
Wenzhao Chen, Yuxin Chen, Yucheng Lin, Ruoyin Qi, and Eamonn Keogh.
"UCR Time Series Anomaly Archive", 2021.
Figshare. doi:10.6084/m9.figshare.26410744
```

> The 5 authentic `.txt` series used in this project are committed to `data/raw/authentic/` for reproducibility. The full archive (94 MB ZIP) can be downloaded from the DOI above.

---

## 7. Project Structure

```
tanishka project/
├── app.py                        # Streamlit dashboard (main entry point)
├── requirements.txt              # Python dependencies
├── .gitignore
├── README.md
│
├── src/                          # Core library modules
│   ├── data/                     # UCR loader, profiler
│   ├── models/
│   │   ├── baselines/            # Isolation Forest, LSTM Autoencoder
│   │   └── tsfm/                 # Chronos adapter, MOMENT adapter (deferred)
│   ├── preprocessing/            # Causal windowing, scaling
│   ├── scoring/                  # Aligner, threshold selector
│   ├── shifts/                   # Amplitude shift implementation
│   ├── evaluation/               # Metrics (PR-AUC, ROC-AUC, F1 …)
│   └── utils/                    # Persistence helpers
│
├── experiments/                  # Experiment runner scripts
│   ├── run_experiment.py         # Full 45-run benchmark
│   ├── run_model_smoke_test.py
│   └── run_data_verification.py
│
├── tests/                        # 52 automated unit tests
├── configs/                      # default_config.json
├── data/
│   └── raw/
│       └── authentic/            # 5 canonical UCR .txt series (committed)
│
├── results/                      # Saved benchmark outputs
│   ├── experiment_summary.csv    # 45-row master results table
│   ├── experiment_summary.json
│   ├── json/                     # Per-run JSON records (45 + 5 profiles)
│   └── plots/                    # Publication PNG figures (gitignored)
│
├── report/                       # IICT B.Tech project report
│   └── Time_Series_Anomaly_Detection_Project_Report.docx
│
├── scripts/                      # Utility and generation scripts
│   └── generate_report.py
│
└── docs/                         # Additional documentation
    └── FINAL_AUDIT.md
```

---

## 8. Known Limitations

1. **MOMENT-1-small deferred:** `momentfm 0.1.4` hard-pins `numpy==1.25.2` which has no Python 3.13 wheels. MOMENT results are absent from the benchmark; all 45 runs use the three successfully evaluated models.
2. **Synthetic shift only:** Amplitude scaling is a controlled proxy for distribution shift; real-world covariate shift (seasonal drift, sensor recalibration, domain change) may exhibit different degradation patterns.
3. **Small dataset sample:** 5 UCR series cover ECG, respiratory, and meteorological domains. Generalisation claims should be treated cautiously; broader validation across the full 250-series UCR archive is recommended.
4. **Fixed threshold:** The 99th-percentile threshold is calibrated once on the in-distribution validation split and frozen across all shift conditions. Operational deployments require adaptive recalibration after any distribution change.
5. **Point-level anomaly scores not persisted:** Only aggregate metrics (TP/FP/TN/FN, PR-AUC, F1) are saved. Per-timestamp score arrays were not retained to avoid storage bloat.
6. **CPU-only benchmark:** All runs were executed on CPU. GPU-accelerated Chronos inference would reduce latency but does not affect metric values.
