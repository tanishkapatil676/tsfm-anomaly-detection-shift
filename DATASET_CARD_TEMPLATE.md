# DATASET_CARD_TEMPLATE.md: Reusable UCR Dataset Documentation Card

This card template must be completed for every univariate time-series dataset selected from the UCR Time Series Anomaly Archive.

---

## Dataset Card: [DATASET_IDENTIFIER]

### 1. General Metadata
* **Dataset Identifier:** `[e.g., 001_UCR_Anomaly_DISTORTED1k]`
* **Full Filename:** `[e.g., 001_UCR_Anomaly_DISTORTED1k_10000_17700_17900.txt]`
* **Source Archive:** UCR Time Series Anomaly Archive (Wu & Keogh, 2021)
* **Official Source URL:** https://www.cs.ucr.edu/~eamonn/time_series_data_2018/ (or Figshare / Hexagon-ML link)
* **Access Date:** `[e.g., 2026-10-10]`
* **License / Terms of Use:** Publicly available academic research dataset (Free for non-commercial research use; cite Wu & Keogh 2021).

### 2. Sequence Structure & Split Parameters
* **Total Sequence Length ($N$):** `[e.g., 20000]`
* **Train / Test Split Index ($T_{\text{split}}$):** `[e.g., 10000]`
* **Training Set Length ($N_{\text{train}}$):** `[e.g., 10000]` (Indices $1 \dots 10000$)
* **Validation Set Sub-split ($N_{\text{val}}$):** `[e.g., 2000]` (Indices $8001 \dots 10000$)
* **Test Set Length ($N_{\text{test}}$):** `[e.g., 10000]` (Indices $10001 \dots 20000$)
* **Sampling Rate / Frequency Context:** `[Univariate, evenly sampled discrete timestamps]`

### 3. Ground Truth Anomaly Annotation
* **Anomaly Start Index ($\text{START}$):** `[e.g., 17700]`
* **Anomaly End Index ($\text{END}$):** `[e.g., 17900]`
* **Anomaly Duration ($\Delta t$):** `[e.g., 201 timestamps]`
* **Anomaly Ratio in Test Set ($R_{\text{anomaly}}$):** `[e.g., 201 / 10000 = 2.01%]`
* **Anomaly Characteristics:** `[e.g., Shapelet anomaly / Frequency spike / Amplitude surge / Flatline]`

### 4. Data Quality & Preprocessing Verification Checklist
* [ ] **Missing Values Check:** Zero NaN, null, or missing timestamps verified.
* [ ] **Infinite Values Check:** Zero `Inf` or `-Inf` values present.
* [ ] **Variance Verification:** Non-zero standard deviation in training and test splits ($\sigma_{\text{train}} > 0$).
* [ ] **Normal Training Assumption:** Training segment $1 \dots T_{\text{split}}$ confirmed normal (no annotated anomalies).
* [ ] **Label Range Check:** Anomaly range $[\text{START}, \text{END}]$ strictly within test range $T_{\text{split}} + 1 \dots N$.
* [ ] **Data Type:** Pytorch / NumPy float64 float vector.

### 5. Baseline & Model Compatibility Notes
* **Isolation Forest:** Suitable; 100-length sliding window produces standard 5-feature summary vectors.
* **LSTM Autoencoder:** Suitable; normal training set contains clean periodic structure for autoencoding.
* **MOMENT-1-small / Chronos-Bolt:** Window sequence compatibility verified (input context length $\ge 512$).

### 6. Known Limitations
* **Single Anomaly Benchmark:** Benchmark contains exactly one annotated anomaly sequence in test set.
* **Synthetic Distortion Note:** Series filename indicates whether synthetic distortion (`DISTORTED`) or natural series (`real`) was used by UCR creators.

---

## Example Filled Dataset Card: `001_UCR_Anomaly_DISTORTED1k`

```markdown
### 1. General Metadata
* **Dataset Identifier:** 001_UCR_Anomaly_DISTORTED1k
* **Full Filename:** 001_UCR_Anomaly_DISTORTED1k_10000_17700_17900.txt
* **Source Archive:** UCR Time Series Anomaly Archive
* **Official Source URL:** https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
* **Access Date:** 2026-10-10
* **License / Terms of Use:** Publicly available academic research dataset

### 2. Sequence Structure & Split Parameters
* **Total Sequence Length (N):** 20000
* **Train / Test Split Index (T_split):** 10000
* **Training Set Length (N_train):** 10000 (Indices 1 to 10000)
* **Validation Set Sub-split (N_val):** 2000 (Indices 8001 to 10000)
* **Test Set Length (N_test):** 10000 (Indices 10001 to 20000)

### 3. Ground Truth Anomaly Annotation
* **Anomaly Start Index (START):** 17700
* **Anomaly End Index (END):** 17900
* **Anomaly Duration:** 201 timestamps
* **Anomaly Ratio in Test Set:** 2.01%
* **Anomaly Type:** Local shapelet & amplitude distortion

### 4. Data Quality Status
* **Status:** Passed all 8 verification steps (0 NaNs, valid indices, normal train split verified).
```
