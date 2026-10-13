# DATASET_AUTHENTICITY_AUDIT.md: UCR Anomaly Archive Authenticity & Indexing Audit

## 1. Executive Summary

This audit establishes the mathematical authenticity, cryptographic integrity, and indexing precision for the five primary benchmark time series used in the CA-4 research project (*Time-Series Foundation Models for Anomaly Detection under Distribution Shift*).

In Milestone 2, synthetic placeholder files with normalized 20,000-length arrays and uniform 10,000 splits were provisioned as development fixtures. This Milestone 3A audit resolves the provenance gap by:
1. Downloading the authentic authoritative UCR archive from the official Figshare repository (`doi:10.6084/m9.figshare.26410744`).
2. Cryptographically validating all five extracted series against authoritative SHA-256 hashes.
3. Independently deriving 0-based boundary conversions and testing for off-by-one errors.
4. Validating all five series against the 10-point data quality criteria with 100% pass rates.
5. Preserving synthetic edge-case fixtures under `data/raw/synthetic_fixtures/` for isolated unit testing.

---

## 2. Authoritative Source Provenance

* **Official Source Archive:** UCR Time Series Anomaly Archive (Wu & Keogh, 2021)
* **Authoritative DOI:** `https://doi.org/10.6084/m9.figshare.26410744` (Author: Daesoo Lee, *UCR Time Series Anomaly Detection datasets 2021*)
* **Archive Archive Zip:** `UCR_TimeSeriesAnomalyDatasets2021.zip`
* **Download URL:** `https://ndownloader.figshare.com/files/48036268`
* **Archive Size:** 99,019,161 bytes (~99 MB)
* **Archive Extraction Directory:** `data/raw/` (authentic target series)
* **Preserved Synthetic Fixtures Directory:** `data/raw/synthetic_fixtures/`

---

## 3. Per-Dataset Cryptographic & Structural Verification Table

| Metric / Attribute | Series 001 | Series 002 | Series 003 | Series 004 | Series 005 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Exact Source Filename** | `001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt` | `002_UCR_Anomaly_DISTORTED2sddb40_35000_56600_56900.txt` | `003_UCR_Anomaly_DISTORTED3sddb40_35000_46600_46900.txt` | `004_UCR_Anomaly_DISTORTEDBIDMC1_2500_5400_5600.txt` | `005_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature1_4000_5391_5392.txt` |
| **File Size (Bytes)** | 1,436,310 | 1,440,018 | 1,440,000 | 198,000 | 147,312 |
| **Cryptographic SHA-256** | `e064e98e0b41d117f666945459671d3eec014ff9ac2457401befd8a244b9b93f` | `1c730e485f186aa39859ac0fca707a7977860b8db49112489270990b7f240e84` | `557cbf5463ea3fc9cc1d17be60157c737de7a01810bc4907ab252e746eb6a7ae` | `f76d4cbc56328179183f3ce69a92f5a22389135a786d8f0bb1662bcfc748ede6` | `076a847fcefaf023942db0324186aa976407c43a8d699beaee8b77527bb5ceae` |
| **Cryptographic Status** | **MATCHED** | **MATCHED** | **MATCHED** | **MATCHED** | **MATCHED** |
| **Total Points ($N$)** | 79,795 | 80,001 | 80,000 | 11,000 | 8,184 |
| **Train Points ($T_{\text{split}}$)** | 35,000 | 35,000 | 35,000 | 2,500 | 4,000 |
| **Val Points ($N_{\text{val}}$)** | 7,000 | 7,000 | 7,000 | 500 | 800 |
| **Test Points ($N_{\text{test}}$)** | 44,795 | 45,001 | 45,000 | 8,500 | 4,184 |
| **1-Based Anomaly Start ($S$)** | 52,000 | 56,600 | 46,600 | 5,400 | 5,391 |
| **1-Based Anomaly End ($E$)** | 52,620 | 56,900 | 46,900 | 5,600 | 5,392 |
| **Global 0-Based Interval** | [51999, 52619] | [56599, 56899] | [46599, 46899] | [5399, 5599] | [5390, 5391] |
| **Test-Relative Interval** | [16999, 17619] | [21599, 21899] | [11599, 11899] | [2899, 3099] | [1390, 1391] |
| **Anomaly Length ($N_{\text{anom}}$)** | 621 | 301 | 301 | 201 | 2 |
| **Test Anomaly Ratio** | 1.386% | 0.669% | 0.669% | 2.365% | 0.0478% |
| **10-Point Quality Result** | `is_eligible = True` | `is_eligible = True` | `is_eligible = True` | `is_eligible = True` | `is_eligible = True` |

---

## 4. Mathematical Indexing Derivation & Inclusivity Verification

### 4.1. UCR Filename Encoding Syntax
The official UCR convention parses filenames as:
`{Index}_{ArchiveName}_{DatasetName}_{T_split}_{AnomalyStart}_{AnomalyEnd}.txt`

All filename numerical parameters ($T_{\text{split}}$, $S$, $E$) represent **1-based line positions**:
* $T_{\text{split}}$: 1-based count of normal training observations preceding test observations.
* $S$: 1-based line position of the initial anomalous observation.
* $E$: 1-based line position of the final anomalous observation (**inclusive**).

### 4.2. Mathematical Conversion to 0-Based Python Slices
For a raw time series array of length $N$:
1. **Training Partition:**
   $$D_{\text{train}} = \text{raw}[0 : T_{\text{split}}] \quad (\text{Length } T_{\text{split}})$$
2. **Validation Partition (Last 20% of Training Prefix):**
   $$D_{\text{val}} = D_{\text{train}}[T_{\text{split}} - \lfloor 0.20 \cdot T_{\text{split}} \rfloor : T_{\text{split}}]$$
3. **Test Partition:**
   $$D_{\text{test}} = \text{raw}[T_{\text{split}} : N] \quad (\text{Length } N_{\text{test}} = N - T_{\text{split}})$$
4. **Test-Relative Anomaly Slices:**
   * 0-based test start: $S_{\text{test}} = S - 1 - T_{\text{split}}$
   * 0-based test end (inclusive): $E_{\text{test\_inc}} = E - 1 - T_{\text{split}}$
   * 0-based Python half-open slice end (exclusive): $E_{\text{test\_exc}} = E - T_{\text{split}}$
   * Total anomalous point count:
     $$N_{\text{anom}} = E_{\text{test\_exc}} - S_{\text{test}} = (E - T_{\text{split}}) - (S - 1 - T_{\text{split}}) = E - S + 1$$

### 4.3. Empirical Boundary Verification
To ensure zero off-by-one errors:
* At index $S_{\text{test}} - 1$: `test_labels[S_test - 1] == 0`
* At index $S_{\text{test}}$: `test_labels[S_test] == 1`
* At index $E_{\text{test\_inc}}$: `test_labels[E_test_inc] == 1`
* At index $E_{\text{test\_inc}} + 1$: `test_labels[E_test_inc + 1] == 0`
* Exact label sum: $\sum_{t=0}^{N_{\text{test}}-1} y_t = N_{\text{anom}}$

All five series passed these assertions with zero discrepancies in `tests/test_data_loader.py`.

---

## 5. Discrepancies Between Previous Claims and Authentic Findings

| Series ID | Previously Recorded Placeholder Claim | Authentic Source Finding | Cause of Discrepancy & Corrective Action |
| :--- | :--- | :--- | :--- |
| **001** | `DISTORTED1k`: $N=20000, T=10000, N_{\text{anom}}=201$ | `DISTORTED1sddb40`: $N=79795, T=35000, N_{\text{anom}}=621$ | Placeholder generated prior to Figshare download; updated with authentic file and dataset card. |
| **002** | `DISTORTED2k`: $N=20000, T=10000, N_{\text{anom}}=201$ | `DISTORTED2sddb40`: $N=80001, T=35000, N_{\text{anom}}=301$ | Placeholder replaced with authentic source file. |
| **003** | `DISTORTED3k`: $N=20000, T=10000, N_{\text{anom}}=201$ | `DISTORTED3sddb40`: $N=80000, T=35000, N_{\text{anom}}=301$ | Placeholder replaced with authentic source file. |
| **004** | `DISTORTED4k`: $N=20000, T=10000, N_{\text{anom}}=201$ | `DISTORTEDBIDMC1`: $N=11000, T=2500, N_{\text{anom}}=201$ | Placeholder replaced with authentic source file. |
| **005** | `DISTORTED5k`: $N=20000, T=10000, N_{\text{anom}}=201$ | `DISTORTEDCIMIS44AirTemperature1`: $N=8184, T=4000, N_{\text{anom}}=2$ | Placeholder replaced with authentic source file (single-event micro-anomaly). |

---

## 6. Test Suite Execution & Validation Evidence

* **Command Executed:**
  ```powershell
  python -m unittest discover -s tests -p "test_*.py"
  ```
* **Execution Environment:** Windows AMD64, Python 3.13.14, PyTorch 2.11.0+cpu.
* **Test Results:** **20 Tests Ran, 20 Tests Passed (0 Failures, 0 Errors, 0 Skipped).**
* **Duration:** 49.495 seconds.
* **Specific Validations Included:**
  1. `test_authentic_ucr_files_validation` (in `tests/test_data_loader.py`): Checks SHA-256, shape, split math, label alignment, and off-by-one boundary safety for all 5 authentic files.
  2. 15 synthetic unit tests in `test_data_loader.py`: Evaluates edge cases (empty files, NaNs, Infs, boundary starts, boundary ends, training anomalies).
  3. 2 baseline smoke tests in `test_baseline_smoke.py`: Evaluates Isolation Forest and LSTM Autoencoder.
  4. 2 foundation model smoke tests in `test_tsfm_smoke.py`: Evaluates MOMENT blocker handling and Chronos Bolt Tiny pipeline execution.
