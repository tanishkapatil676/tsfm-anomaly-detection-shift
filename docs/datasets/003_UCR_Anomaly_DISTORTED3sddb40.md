# Dataset Card: 003_UCR_Anomaly_DISTORTED3sddb40

### 1. General Metadata & Provenance
* **Dataset Identifier:** `003_UCR_Anomaly_DISTORTED3sddb40`
* **Exact Authentic Filename:** `003_UCR_Anomaly_DISTORTED3sddb40_35000_46600_46900.txt`
* **Source Archive:** Official UCR Time Series Anomaly Archive (Wu & Keogh, 2021)
* **Official Source DOI / URL:** https://doi.org/10.6084/m9.figshare.26410744 (Direct File ID: 48036268)
* **Cryptographic Verification:** SHA-256: `557cbf5463ea3fc9cc1d17be60157c737de7a01810bc4907ab252e746eb6a7ae` (1,440,000 bytes)
* **Provenance Status:** **VERIFIED AUTHENTIC** — Downloaded and cryptographically verified against official Figshare release.
* **License / Usage Terms:** Publicly available academic research dataset (Free for research; cite Wu & Keogh, 2021).

### 2. Sequence Structure & Split Parameters
* **Total Sequence Length ($N$):** 80,000 timestamps
* **Training Prefix Parameter ($T_{\text{split}}$):** 35,000 timestamps (1-based line count)
* **Training Set ($D_{\text{train}}$):** Python slice `raw_data[0 : 35000]` (Length: 35,000, 0-based indices $0 \dots 34999$)
* **Validation Sub-split ($D_{\text{val}}$):** Python slice `train_data[28000 : 35000]` (Length: 7,000, 0-based indices $28000 \dots 34999$)
* **Test Set ($D_{\text{test}}$):** Python slice `raw_data[35000 : 80000]` (Length: 45,000, 0-based indices $35000 \dots 79999$)

### 3. Ground Truth Anomaly Annotation & Mathematical Derivation
* **Source Filename Indices (1-based):** Anomaly Start $S = 46600$, Anomaly End $E = 46900$
* **Global 0-Based Indices:** Start $= S - 1 = 46599$, End (inclusive) $= E - 1 = 46899$
* **Test-Relative 0-Based Indices:**
  * Start: $S_{\text{test}} = S - 1 - T_{\text{split}} = 46600 - 1 - 35000 = 11599$
  * End (inclusive): $E_{\text{test\_inc}} = E - 1 - T_{\text{split}} = 46900 - 1 - 35000 = 11899$
  * End (exclusive): $E_{\text{test\_exc}} = 11900$
* **Anomaly Duration:** $E - S + 1 = 46900 - 46600 + 1 = 301$ timestamps
* **Test Anomaly Ratio:** $301 / 45000 = 0.669\%$

### 4. 10-Point Data Quality Verification Outcome
* [x] **File Existence:** Verified in `data/raw/`.
* [x] **Numeric Parseability:** 100% valid float64 observations (80,000 rows).
* [x] **Non-empty Series:** $N = 80,000 > 0$.
* [x] **Missing / Non-finite Values:** 0 NaNs, 0 Infs verified.
* [x] **Training Prefix Length:** $T_{\text{split}} = 35,000 \ge 1000$.
* [x] **Test Segment Length:** $N_{\text{test}} = 45,000 \ge 100$.
* [x] **Anomaly Boundary Bounds:** $1 \le 46600 \le 46900 \le 80000$.
* [x] **Normal-Only Training Rule:** Anomaly start $46599 \ge T_{\text{split}} (35000)$.
* [x] **Validation Sub-split Size:** $N_{\text{val}} = 7000 \ge 200$.
* [x] **Training Variance:** $\sigma_{\text{train}} > 0$.

### 5. Historical Discrepancy Note
* **Previous Placeholder:** In Milestone 2, synthetic placeholder `DISTORTED3k` ($N=20000, T_{\text{split}}=10000, N_{\text{anom}}=201$) was recorded.
* **Correction in Milestone 3A:** Authenticated against official Figshare source `DISTORTED3sddb40` ($N=80000, T_{\text{split}}=35000, N_{\text{anom}}=301$).
