# Dataset Card: 004_UCR_Anomaly_DISTORTED4k

### 1. General Metadata
* **Dataset Identifier:** `004_UCR_Anomaly_DISTORTED4k`
* **Full Filename Syntax:** `004_UCR_Anomaly_DISTORTED4k_10000_14000_14200.txt`
* **Source Archive:** UCR Time Series Anomaly Archive (Wu & Keogh, 2021)
* **Official Source URL:** https://doi.org/10.6084/m9.figshare.26410744
* **Access Date:** 2026-10-10
* **License / Usage Terms:** Publicly available academic research dataset.

### 2. Sequence Structure & Split Parameters
* **Total Sequence Length ($N$):** 20,000 timestamps
* **Training Prefix Parameter ($T_{\text{split}}$):** 10,000 timestamps
* **Training Set ($D_{\text{train}}$):** `raw_data[0 : 10000]` (Length: 10,000)
* **Validation Sub-split ($D_{\text{val}}$):** `train_data[8000 : 10000]` (Length: 2,000)
* **Test Set ($D_{\text{test}}$):** `raw_data[10000 : 20000]` (Length: 10,000)

### 3. Ground Truth Anomaly Annotation
* **Source Line Numbers (1-based):** Start = 14000, End = 14200
* **Test-Relative 0-Based Indices:** Start = 3999, End = 4199 (inclusive)
* **Anomaly Duration:** 201 timestamps
* **Test Anomaly Ratio:** $2.01\%$

### 4. 10-Point Data Quality Verification Outcome
* [x] Passed all 10 verification steps (0 NaNs, valid splits, normal training prefix).
