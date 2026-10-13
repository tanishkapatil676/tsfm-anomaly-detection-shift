# Dataset Card: 005_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature1

### 1. General Metadata & Provenance
* **Dataset Identifier:** `005_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature1`
* **Exact Authentic Filename:** `005_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature1_4000_5391_5392.txt`
* **Source Archive:** Official UCR Time Series Anomaly Archive (Wu & Keogh, 2021)
* **Official Source DOI / URL:** https://doi.org/10.6084/m9.figshare.26410744 (Direct File ID: 48036268)
* **Cryptographic Verification:** SHA-256: `076a847fcefaf023942db0324186aa976407c43a8d699beaee8b77527bb5ceae` (147,312 bytes)
* **Provenance Status:** **VERIFIED AUTHENTIC** — Downloaded and cryptographically verified against official Figshare release.
* **License / Usage Terms:** Publicly available academic research dataset (Free for research; cite Wu & Keogh, 2021).

### 2. Sequence Structure & Split Parameters
* **Total Sequence Length ($N$):** 8,184 timestamps
* **Training Prefix Parameter ($T_{\text{split}}$):** 4,000 timestamps (1-based line count)
* **Training Set ($D_{\text{train}}$):** Python slice `raw_data[0 : 4000]` (Length: 4,000, 0-based indices $0 \dots 3999$)
* **Validation Sub-split ($D_{\text{val}}$):** Python slice `train_data[3200 : 4000]` (Length: 800, 0-based indices $3200 \dots 3999$)
* **Test Set ($D_{\text{test}}$):** Python slice `raw_data[4000 : 8184]` (Length: 4,184, 0-based indices $4000 \dots 8183$)

### 3. Ground Truth Anomaly Annotation & Mathematical Derivation
* **Source Filename Indices (1-based):** Anomaly Start $S = 5391$, Anomaly End $E = 5392$
* **Global 0-Based Indices:** Start $= S - 1 = 5390$, End (inclusive) $= E - 1 = 5391$
* **Test-Relative 0-Based Indices:**
  * Start: $S_{\text{test}} = S - 1 - T_{\text{split}} = 5391 - 1 - 4000 = 1390$
  * End (inclusive): $E_{\text{test\_inc}} = E - 1 - T_{\text{split}} = 5392 - 1 - 4000 = 1391$
  * End (exclusive): $E_{\text{test\_exc}} = 1392$
* **Anomaly Duration:** $E - S + 1 = 5392 - 5391 + 1 = 2$ timestamps
* **Test Anomaly Ratio:** $2 / 4184 = 0.0478\%$

### 4. 10-Point Data Quality Verification Outcome
* [x] **File Existence:** Verified in `data/raw/`.
* [x] **Numeric Parseability:** 100% valid float64 observations (8,184 rows).
* [x] **Non-empty Series:** $N = 8,184 > 0$.
* [x] **Missing / Non-finite Values:** 0 NaNs, 0 Infs verified.
* [x] **Training Prefix Length:** $T_{\text{split}} = 4,000 \ge 1000$.
* [x] **Test Segment Length:** $N_{\text{test}} = 4,184 \ge 100$.
* [x] **Anomaly Boundary Bounds:** $1 \le 5391 \le 5392 \le 8184$.
* [x] **Normal-Only Training Rule:** Anomaly start $5390 \ge T_{\text{split}} (4000)$.
* [x] **Validation Sub-split Size:** $N_{\text{val}} = 800 \ge 200$.
* [x] **Training Variance:** $\sigma_{\text{train}} > 0$.

### 5. Historical Discrepancy Note
* **Previous Placeholder:** In Milestone 2, synthetic placeholder `DISTORTED5k` ($N=20000, T_{\text{split}}=10000, N_{\text{anom}}=201$) was recorded.
* **Correction in Milestone 3A:** Authenticated against official Figshare source `DISTORTEDCIMIS44AirTemperature1` ($N=8184, T_{\text{split}}=4000, N_{\text{anom}}=2$).
