# Dataset Card: 001_UCR_Anomaly_DISTORTED1k

### 1. General Metadata
* **Dataset Identifier:** `001_UCR_Anomaly_DISTORTED1k`
* **Full Filename Syntax:** `001_UCR_Anomaly_DISTORTED1k_10000_17700_17900.txt`
* **Source Archive:** UCR Time Series Anomaly Archive (Wu & Keogh, 2021)
* **Official Source URL:** https://doi.org/10.6084/m9.figshare.26410744
* **Access Date:** 2026-10-10
* **License / Usage Terms:** Publicly available academic research dataset (Free for non-commercial research; cite Wu & Keogh 2021).

### 2. Sequence Structure & Split Parameters
* **Total Sequence Length ($N$):** 20,000 timestamps
* **Training Prefix Parameter ($T_{\text{split}}$):** 10,000 timestamps (1-based line count)
* **Training Set ($D_{\text{train}}$):** Python slice `raw_data[0 : 10000]` (Length: 10,000, 0-based indices $0 \dots 9999$)
* **Validation Sub-split ($D_{\text{val}}$):** Python slice `train_data[8000 : 10000]` (Length: 2,000, 0-based indices $8000 \dots 9999$)
* **Test Set ($D_{\text{test}}$):** Python slice `raw_data[10000 : 20000]` (Length: 10,000, 0-based indices $10000 \dots 19999$)

### 3. Ground Truth Anomaly Annotation
* **Source Line Numbers (1-based):** Start = 17700, End = 17900
* **Global 0-Based Indices:** Start = 17699, End = 17899 (inclusive)
* **Test-Relative 0-Based Indices:** Start = 7699, End = 7899 (inclusive), Exclusive End = 7900
* **Anomaly Duration:** 201 timestamps
* **Test Anomaly Ratio:** $201 / 10000 = 2.01\%$

### 4. 10-Point Data Quality Verification Outcome
* [x] **File Existence:** Verified via candidate parser loader.
* [x] **Numeric Parseability:** 100% valid float64 observations.
* [x] **Non-empty Series:** $N = 20,000 > 0$.
* [x] **Missing / Non-finite Values:** 0 NaNs, 0 Infs verified.
* [x] **Training Prefix Length:** $T_{\text{split}} = 10,000 \ge 1000$.
* [x] **Test Segment Length:** $N_{\text{test}} = 10,000 \ge 100$.
* [x] **Anomaly Boundary Bounds:** $1 \le 17700 \le 17900 \le 20000$.
* [x] **Normal-Only Training Rule:** Anomaly start $17699 \ge T_{\text{split}} (10000)$.
* [x] **Validation Sub-split Size:** $N_{\text{val}} = 2000 \ge 200$.
* [x] **Training Variance:** $\sigma_{\text{train}} > 0$.

### 5. Baseline & Model Compatibility Notes
* **Isolation Forest:** Eligible ($W=100, S=1$).
* **LSTM Autoencoder:** Eligible ($W=100, S=1$).
* **MOMENT-1-small / Chronos-Bolt:** Eligible (context length $\ge 512$).
