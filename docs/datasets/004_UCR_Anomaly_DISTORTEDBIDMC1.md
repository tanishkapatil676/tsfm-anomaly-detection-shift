# Dataset Card: 004_UCR_Anomaly_DISTORTEDBIDMC1

### 1. General Metadata & Provenance
* **Dataset Identifier:** `004_UCR_Anomaly_DISTORTEDBIDMC1`
* **Exact Authentic Filename:** `004_UCR_Anomaly_DISTORTEDBIDMC1_2500_5400_5600.txt`
* **Source Archive:** Official UCR Time Series Anomaly Archive (Wu & Keogh, 2021)
* **Official Source DOI / URL:** https://doi.org/10.6084/m9.figshare.26410744 (Direct File ID: 48036268)
* **Cryptographic Verification:** SHA-256: `f76d4cbc56328179183f3ce69a92f5a22389135a786d8f0bb1662bcfc748ede6` (198,000 bytes)
* **Provenance Status:** **VERIFIED AUTHENTIC** — Downloaded and cryptographically verified against official Figshare release.
* **License / Usage Terms:** Publicly available academic research dataset (Free for research; cite Wu & Keogh, 2021).

### 2. Sequence Structure & Split Parameters
* **Total Sequence Length ($N$):** 11,000 timestamps
* **Training Prefix Parameter ($T_{\text{split}}$):** 2,500 timestamps (1-based line count)
* **Training Set ($D_{\text{train}}$):** Python slice `raw_data[0 : 2500]` (Length: 2,500, 0-based indices $0 \dots 2499$)
* **Validation Sub-split ($D_{\text{val}}$):** Python slice `train_data[2000 : 2500]` (Length: 500, 0-based indices $2000 \dots 2499$)
* **Test Set ($D_{\text{test}}$):** Python slice `raw_data[2500 : 11000]` (Length: 8,500, 0-based indices $2500 \dots 10999$)

### 3. Ground Truth Anomaly Annotation & Mathematical Derivation
* **Source Filename Indices (1-based):** Anomaly Start $S = 5400$, Anomaly End $E = 5600$
* **Global 0-Based Indices:** Start $= S - 1 = 5399$, End (inclusive) $= E - 1 = 5599$
* **Test-Relative 0-Based Indices:**
  * Start: $S_{\text{test}} = S - 1 - T_{\text{split}} = 5400 - 1 - 2500 = 2899$
  * End (inclusive): $E_{\text{test\_inc}} = E - 1 - T_{\text{split}} = 5600 - 1 - 2500 = 3099$
  * End (exclusive): $E_{\text{test\_exc}} = 3100$
* **Anomaly Duration:** $E - S + 1 = 5600 - 5400 + 1 = 201$ timestamps
* **Test Anomaly Ratio:** $201 / 8500 = 2.365\%$

### 4. 10-Point Data Quality Verification Outcome
* [x] **File Existence:** Verified in `data/raw/`.
* [x] **Numeric Parseability:** 100% valid float64 observations (11,000 rows).
* [x] **Non-empty Series:** $N = 11,000 > 0$.
* [x] **Missing / Non-finite Values:** 0 NaNs, 0 Infs verified.
* [x] **Training Prefix Length:** $T_{\text{split}} = 2,500 \ge 1000$.
* [x] **Test Segment Length:** $N_{\text{test}} = 8,500 \ge 100$.
* [x] **Anomaly Boundary Bounds:** $1 \le 5400 \le 5600 \le 11000$.
* [x] **Normal-Only Training Rule:** Anomaly start $5399 \ge T_{\text{split}} (2500)$.
* [x] **Validation Sub-split Size:** $N_{\text{val}} = 500 \ge 200$.
* [x] **Training Variance:** $\sigma_{\text{train}} > 0$.

### 5. Historical Discrepancy Note
* **Previous Placeholder:** In Milestone 2, synthetic placeholder `DISTORTED4k` ($N=20000, T_{\text{split}}=10000, N_{\text{anom}}=201$) was recorded.
* **Correction in Milestone 3A:** Authenticated against official Figshare source `DISTORTEDBIDMC1` ($N=11000, T_{\text{split}}=2500, N_{\text{anom}}=201$).
