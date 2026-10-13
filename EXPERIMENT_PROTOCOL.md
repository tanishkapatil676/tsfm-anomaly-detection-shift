# EXPERIMENT_PROTOCOL.md: Experimental Protocol & Governance Rules

## 1. Overview & Data Integrity Principles
This document defines the strict experimental protocol governing dataset selection, data splitting, distribution shift generation, threshold selection, scoring alignment, and metric calculation for the CA-4 research project. 

### Core Integrity Rules
1. **Zero Data Leakage:** Preprocessing parameters (min/max, mean/std) and decision thresholds MUST be fitted strictly on the permitted normal training split.
2. **Strict Test Isolation:** Test set anomaly labels MUST NEVER be used for model training, hyperparameter tuning, scaler fitting, or threshold selection.
3. **Label Preservation:** Synthetic distribution shifts MUST alter sequence feature values only; anomaly label timestamps and interval bounds must remain 100% unchanged.

---

## 2. Dataset Selection & Verification Protocol (8-Point Checklist)

Before locking any univariate dataset from the UCR Time Series Anomaly Archive, it must pass this 8-point verification protocol:

1. **Source Authenticity:** Downloadable from legitimate, documented sources (UCR Archive website / Figshare / Hexagon-ML).
2. **Filename Parsing & Split Validity:** Filename follows standard syntax `<ID>_UCR_Anomaly_<NAME>_<SPLIT>_<START>_<END>.txt`. The split index dividing normal train data from test data must be valid.
3. **Training Normalcy Verification:** The training segment ($0 \dots \text{SPLIT}-1$ in 0-based indexing) contains exclusively normal behavior (or any minor noise is explicitly documented).
4. **Label Range Validity:** Ground truth anomaly indices ($\text{START} \dots \text{END}$) fall strictly within the test segment ($\text{SPLIT} \dots N-1$).
5. **Timestamp Alignment:** Point-wise binary labels $y_t \in \{0, 1\}$ match exact index locations where $y_t = 1$ for $t \in [\text{START}_{\text{test}}, \text{END}_{\text{test}}]$ and $y_t = 0$ otherwise.
6. **Class Imbalance & Observation Sufficiency:** Sequence length and anomaly duration provide sufficient samples for evaluation (e.g. $N \ge 5000$, anomaly length $\ge 50$).
7. **Data Quality Sanity Check:** Zero NaN values, zero Infinite values, constant non-zero variance, valid float64 data type.
8. **Model-Agnostic Eligibility:** Dataset candidate inclusion is decided purely on structural data quality before running any model benchmark. Exact series names remain candidates until source files are verified in Milestone 2.

---

## 3. Data Splitting & Index Parsing Protocol

### UCR Filename Metadata & 0-Based Python Indexing
Each UCR text file consists of a single univariate series of length $N$ (one floating-point value per line).
Filename syntax: `<ID>_UCR_Anomaly_<NAME>_<SPLIT>_<START>_<END>.txt`
Example: `001_UCR_Anomaly_DISTORTED1k_10000_17700_17900.txt`

The numerical metadata in UCR filenames use 1-based line numbering. In Python 0-based array indexing, parsing maps as follows:

* **Training Split Parameter ($T_{\text{split}}$):** Value `10000` denotes the line count of the normal training prefix.
  * **Training Set ($D_{\text{train}}$):** Python slice `series[0 : T_split]` (indices $0 \dots T_{\text{split}}-1$, length $N_{\text{train}} = T_{\text{split}}$). Contains strictly normal data.
  * **Validation Sub-split ($D_{\text{val}}$):** Obtained exclusively from $D_{\text{train}}$ as the final 20% of the training prefix: `series[int(0.8 * T_split) : T_split]` (length $N_{\text{val}} = \lceil 0.2 \cdot T_{\text{split}} \rceil$).
  * **Validation Data Checks:** Requires $N_{\text{train}} \ge 1000$ and $N_{\text{val}} \ge 200$.
* **Test Set ($D_{\text{test}}$):** Python slice `series[T_split : N]` (indices $T_{\text{split}} \dots N-1$, length $N_{\text{test}} = N - T_{\text{split}}$).
* **Ground-Truth Anomaly Parsing:**
  * 0-Based Global Indices: $\text{START}_{\text{global}} = \text{START} - 1$, $\text{END}_{\text{global}} = \text{END} - 1$.
  * Global Label Vector ($y_{\text{global}} \in \{0, 1\}^N$): $y_{\text{global}}[t] = 1$ for $t \in [\text{START}_{\text{global}}, \text{END}_{\text{global}}]$ and $0$ otherwise.
  * Test-Relative Anomaly Indices:
    $$\text{START}_{\text{test}} = \text{START} - 1 - T_{\text{split}}$$
    $$\text{END}_{\text{test}} = \text{END} - 1 - T_{\text{split}}$$
  * Test Label Vector ($y_{\text{test}} \in \{0, 1\}^{N_{\text{test}}}$): $y_{\text{test}} = y_{\text{global}}[T_{\text{split}} : N]$.

### Boundary Case Handling
1. **Anomaly at Test Boundary ($\text{START}_{\text{global}} = T_{\text{split}}$):** $\text{START}_{\text{test}} = 0$. Valid. The training split ($0 \dots T_{\text{split}}-1$) remains uncorrupted.
2. **Anomaly Extends Past Sequence End ($\text{END}_{\text{global}} \ge N$):** Truncate test-relative end index to $N_{\text{test}} - 1$ and log a warning.
3. **Anomaly Starts Inside Training Segment ($\text{START}_{\text{global}} < T_{\text{split}}$):** Violates the normal-only training assumption! Reject dataset under the 8-point checklist.

---

## 4. Preprocessing & Scale Shift Protocol

### Preprocessing & Scaling Rules
* Fit `StandardScaler` strictly on $D_{\text{train}}$ to compute training mean $\mu_{\text{train}}$ and standard deviation $\sigma_{\text{train}}$.
* Transform $D_{\text{train}}$, $D_{\text{val}}$, $D_{\text{test, ID}}$, $D_{\text{test, Shift1}}$, and $D_{\text{test, Shift2}}$ using $(\mu_{\text{train}}, \sigma_{\text{train}})$.
* **Strict Parameter Lock:** NEVER re-fit scaler parameters on test or shifted test data.

### Controlled Distribution-Shift Specifications
* **In-Distribution (ID):** Un-shifted test series $x_{\text{ID}}(t) = x_{\text{test}}(t)$.
* **Shift 1 (Mild Scale Shift):** $x_{\text{Shift1}}(t) = 1.25 \cdot x_{\text{test}}(t)$.
* **Shift 2 (Strong Scale Shift):** $x_{\text{Shift2}}(t) = 1.50 \cdot x_{\text{test}}(t)$.

### Mathematical Proof of Scale Preservation
When scaling shifted test data $x_{\text{Shift}}(t) = a \cdot x_{\text{test}}(t)$ using frozen training scaler parameters $(\mu_{\text{train}}, \sigma_{\text{train}})$:

$$z_{\text{Shift}}(t) = \frac{a \cdot x_{\text{test}}(t) - \mu_{\text{train}}}{\sigma_{\text{train}}} = a \cdot z_{\text{ID}}(t) + \frac{(a - 1) \mu_{\text{train}}}{\sigma_{\text{train}}}$$

Because $\mu_{\text{train}}$ and $\sigma_{\text{train}}$ are fixed constants, the amplitude scale factor $a$ is strictly preserved in the scaled feature space. Re-fitting the scaler on shifted test data would calculate $\mu_{\text{shift}} \approx a \cdot \mu_{\text{train}}$ and $\sigma_{\text{shift}} \approx a \cdot \sigma_{\text{train}}$, which would artificially cancel out the shift. Thus, strict freeze of training scaler parameters is mandatory.

---

## 5. Windowing, Scoring, & Timestamp Alignment Policy

### Sliding Window Parameters
* Window Size $W = 100$ (default, configurable per dataset). Stride $S = 1$ for point-wise evaluation.

### Continuous Score Direction & Timestamp Mapping
* **Score Direction Convention:** All models MUST output scores $s(t)$ where higher values indicate higher anomaly probability ($s(t) \to +\infty$). For reconstruction models, $s(t) = MSE(x(t), \hat{x}(t))$. For iForest, $s(t) = -\text{score\_samples}(x(t))$.
* **Window Alignment:** Window $[t - W + 1 \dots t]$ is assigned to window end timestamp $t$.
* **Padding:** Initial $W-1$ timestamps (indices $0 \dots W-2$) are padded with score $s(W-1)$ to maintain vector length $N_{\text{test}}$.

---

## 6. Threshold Selection & Freezing Policy

### Validation Threshold Calculation
Threshold $\tau$ is calculated ONCE using continuous validation scores $s(t \in D_{\text{val}})$:
1. **Validation Percentile:** $\tau = \text{Percentile}(s(D_{\text{val}}), 99.0)$.
2. **Mean + $3 \cdot \text{Std}$:** $\tau = \mu_{s,\text{val}} + 3 \cdot \sigma_{s,\text{val}}$.

### Strict Freeze Policy
Threshold $\tau$ is frozen and applied unchanged to ID test scores, Shift 1 test scores, and Shift 2 test scores. Test labels are NEVER used for threshold selection.

### Invalid Threshold Safety Protocol
If validation scores $s(D_{\text{val}})$ exhibit zero variance, contain NaNs/Infs, or fail percentile calculation:
1. Log an explicit warning detailing the anomaly score anomaly.
2. Attempt fallback quantile estimation on finite validation scores.
3. If scores remain invalid, flag threshold selection as `FAILED` and abort run evaluation with structured error log.

---

## 7. Evaluation Metrics & Robustness Degradation Formula

### Primary Point-Wise Metrics
Evaluated over all timestamps $t \in [0 \dots N_{\text{test}}-1]$:
* **Precision ($P$):** $\frac{TP}{TP + FP}$ (Zero-division handling: if $TP+FP=0$, $P = 0.0$).
* **Recall ($R$):** $\frac{TP}{TP + FN}$
* **F1-Score ($F1$):** $\frac{2 \cdot P \cdot R}{P + R}$
* **PR-AUC (Average Precision):** Primary un-thresholded metric across all operating points.
* **ROC-AUC:** Area under ROC curve.

### Secondary Event-Based Metric
* **Event Recall:** An anomaly sequence $[\text{START}_{\text{test}}, \text{END}_{\text{test}}]$ is detected if at least one timestamp $t \in [\text{START}_{\text{test}}, \text{END}_{\text{test}}]$ has $\hat{y}(t) = 1$.

### Metric Degradation Formula
$$\text{Degradation}(M, \text{Shift}) = M_{\text{ID}} - M_{\text{Shift}}$$
* $\text{Degradation} > 0$: Performance drop under shift.
* $\text{Degradation} \le 0$: Performance invariant or improved under shift.

---

## 8. Reproducibility & Persistence

All experiment outputs are saved in deterministic JSON files under `results/json/` recording environment details, random seeds (`seed=42`), frozen scaler parameters, frozen thresholds, point-wise metrics, event metrics, and degradation values.
