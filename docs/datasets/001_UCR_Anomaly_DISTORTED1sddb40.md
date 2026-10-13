# Dataset Card: 001_UCR_Anomaly_DISTORTED1sddb40

### 1. General Metadata & Provenance
* **Dataset Identifier:** `001_UCR_Anomaly_DISTORTED1sddb40`
* **Exact Authentic Filename:** `001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt`
* **Source Archive:** Official UCR Time Series Anomaly Archive (Wu & Keogh, 2021)
* **Official Source DOI / URL:** https://doi.org/10.6084/m9.figshare.26410744 (Direct File ID: 48036268)
* **Cryptographic Verification:** SHA-256: `e064e98e0b41d117f666945459671d3eec014ff9ac2457401befd8a244b9b93f` (1,436,310 bytes)
* **Provenance Status:** **VERIFIED AUTHENTIC** — Downloaded and cryptographically verified against official Figshare release.
* **License / Usage Terms:** Publicly available academic research dataset (Free for research; cite Wu & Keogh, 2021).

### 2. Sequence Structure & Split Parameters
* **Total Sequence Length ($N$):** 79,795 timestamps
* **Training Prefix Parameter ($T_{\text{split}}$):** 35,000 timestamps (1-based line count)
* **Training Set ($D_{\text{train}}$):** Python slice `raw_data[0 : 35000]` (Length: 35,000, 0-based indices $0 \dots 34999$)
* **Validation Sub-split ($D_{\text{val}}$):** Python slice `train_data[28000 : 35000]` (Length: 7,000, 0-based indices $28000 \dots 34999$)
* **Test Set ($D_{\text{test}}$):** Python slice `raw_data[35000 : 79795]` (Length: 44,795, 0-based indices $35000 \dots 79794$)

### 3. Ground Truth Anomaly Annotation & Mathematical Derivation
* **Source Filename Indices (1-based):** Anomaly Start $S = 52000$, Anomaly End $E = 52620$
* **Global 0-Based Indices:** Start $= S - 1 = 51999$, End (inclusive) $= E - 1 = 52619$
* **Test-Relative 0-Based Indices:**
  * Start: $S_{\text{test}} = S - 1 - T_{\text{split}} = 52000 - 1 - 35000 = 16999$
  * End (inclusive): $E_{\text{test\_inc}} = E - 1 - T_{\text{split}} = 52620 - 1 - 35000 = 17619$
  * End (exclusive): $E_{\text{test\_exc}} = 17620$
* **Anomaly Duration:** $E - S + 1 = 52620 - 52000 + 1 = 621$ timestamps
* **Test Anomaly Ratio:** $621 / 44795 = 1.386\%$

### 4. 10-Point Data Quality Verification Outcome
* [x] **File Existence:** Verified in `data/raw/`.
* [x] **Numeric Parseability:** 100% valid float64 observations (79,795 rows).
* [x] **Non-empty Series:** $N = 79,795 > 0$.
* [x] **Missing / Non-finite Values:** 0 NaNs, 0 Infs verified.
* [x] **Training Prefix Length:** $T_{\text{split}} = 35,000 \ge 1000$.
* [x] **Test Segment Length:** $N_{\text{test}} = 44,795 \ge 100$.
* [x] **Anomaly Boundary Bounds:** $1 \le 52000 \le 52620 \le 79795$.
* [x] **Normal-Only Training Rule:** Anomaly start $51999 \ge T_{\text{split}} (35000)$.
* [x] **Validation Sub-split Size:** $N_{\text{val}} = 7000 \ge 200$.
* [x] **Training Variance:** $\sigma_{\text{train}} > 0$.

### 5. Historical Discrepancy Note
* **Previous Placeholder:** In Milestone 2, a synthetic file `DISTORTED1k` with $N=20000, T_{\text{split}}=10000$ and 201 anomaly points was used prior to official archive extraction.
* **Correction in Milestone 3A:** Authenticated against official Figshare source `DISTORTED1sddb40` ($N=79795, T_{\text{split}}=35000, N_{\text{anom}}=621$).
