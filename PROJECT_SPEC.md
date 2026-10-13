# PROJECT_SPEC.md: CA-4 Research Specification & Project Scope

## 1. Project Title & Academic Context
* **Project Title:** Time-Series Foundation Models for Anomaly Detection under Distribution Shift
* **Academic Context:** B.Tech Information Technology, IICT, MGM University (Academic Year 2026–27)
* **Course Milestone:** CA-4 Implementation & Experimental Validation (Building on CA-3 Research Direction)

---

## 2. Fixed Research Question & Research Gap

### Research Question
> **"Can pretrained time-series foundation models remain reliable for anomaly detection when the test distribution changes?"**

### Research Gap
> **"Recent work shows that TSFMs can be adapted for anomaly detection, but evidence is fragmented across retrieval, adapters, embeddings, and reconstruction/forecasting-based methods. Their robustness and relative advantage over simple baselines under distribution shift remain an open experimental question."**

*Rule of Conduct:* This research question and research gap are fixed. The project does not claim that foundation models naturally outperform conventional baselines, nor does it assume zero existing research. Relative performance and robustness are treated strictly as empirical, experimental questions.

---

## 3. Research Hypotheses

The project will formulate and test three core empirical hypotheses:

* **H1 (Amplitude Shift Sensitivity):** Model anomaly detection performance changes significantly when the test-series amplitude deviates from the training distribution scale.
* **H2 (Differential Degradation):** Pretrained Time-Series Foundation Models (TSFMs) exhibit different degradation rates compared to conventional baselines (Isolation Forest, LSTM Autoencoder) when subjected to identical test set distribution shifts.
* **H3 (Relative Superiority Under Shift):** A TSFM may or may not maintain superior precision/recall/F1 compared to conventional baselines under in-distribution (ID) versus shifted conditions.

### Evaluation of Evidence
* **Support for H1/H2:** Observed drop in PR-AUC / F1 metrics under scale shifts $x_{\text{shifted}}(t) = a \cdot x(t)$ with statistical metric degradation.
* **Refutation/Neutral Evidence:** Metrics remain invariant under scale shift (e.g. due to robust z-score normalization) or both model families degrade at identical rates.
* **Budget-Aware Uncertainty:** Where hardware permits, multi-run seeds will provide mean and standard deviation bounds for baseline models to prevent over-interpreting minor metric fluctuations.

---

## 4. Development & System Constraints

| Constraint Category | Specification / Limitation | Impact on Implementation Design |
| :--- | :--- | :--- |
| **Primary Environment** | Windows 10/11 PC / Laptop | Native Python execution; CPU fallback mandatory |
| **Compute Hardware** | Unknown GPU / System RAM | Lightweight CPU baselines required; TSFM smoke tests mandatory |
| **Cloud Environment** | Google Colab (Free Tier) | Notebooks must execute within 12h session limit & ~12GB RAM |
| **AI Assistant** | Antigravity (Limited Free-Tier) | Small, incremental modules; no monolithic code dumps |
| **Core Libraries** | PyTorch, Hugging Face `transformers`, `scikit-learn`, `scipy` | Strictly open-source dependencies; pinned versions in `requirements.txt` |
| **Cost Constraint** | \$0.00 Budget (No paid APIs, GPUs, or Cloud) | Zero external API calls; models strictly local or free HF Hub |
| **Execution Safety** | Baseline independence | Conventional models (iForest, LSTM-AE) must run independently of TSFM availability |

---

## 5. Primary Scope & Model Shortlist

### A. Datasets
* **Primary Source:** UCR Time Series Anomaly Archive (250 univariate time series).
* **Initial Target:** 5 eligible univariate series selected via a predeclared 8-point data quality verification protocol.
* **Expansion Rule:** Scope expansion to additional series is permitted only after the full core pipeline passes end-to-end testing across the initial 5 series.

### B. Baseline Models (Mandatory)
1. **Isolation Forest (`iForest`):** Non-parametric tree ensemble applied over sliding window feature vectors (Mean, Std, Min, Max, Quantiles).
2. **Small LSTM Autoencoder (`LSTM-AE`):** Sequence-to-sequence neural network trained on normal data; anomaly score based on mean squared reconstruction error ($MSE$).

### C. Foundation Models (TSFM Candidate & Fallback)
1. **Primary TSFM Candidate:** `AutonLab/MOMENT-1-small` (Hugging Face Hub, MIT License, 125M parameters). Tested via reconstruction error / representation output.
2. **Conditional Fallback TSFM:** `amazon/chronos-bolt-tiny` (Hugging Face Hub, Apache-2.0 License, ~9M parameters). Evaluated via zero-shot forecasting error ($MAE$/$MSE$) if MOMENT-1-small cannot run under system RAM/CPU constraints.

### Excluded Scope (Out of Initial Scope)
* Large retrieval-augmented TSFMs (e.g., RATFM), complex adapters (STAR), ChronosAD, ensemble combinations, or multi-modal models.

---

## 6. Success & Acceptance Criteria

The CA-4 project milestone is successfully completed when the following criteria are satisfied:

1. **Complete Specification Artifacts:** 9 planning documents created, internally aligned, and locked.
2. **Reproducible Codebase:** Python codebase structured into clean modules under `src/` with working configuration files (`configs/`).
3. **Verified Datasets:** 5 UCR series verified, downloaded, and documented with complete dataset cards.
4. **Independent Baselines:** Isolation Forest and LSTM-AE execute cleanly on CPU, producing continuous anomaly scores.
5. **TSFM Compatibility Verified:** Smoke test documents execution outcome for `MOMENT-1-small` (or seamless fallback to `Chronos-Bolt-tiny`).
6. **Controlled Shift Experiments:** Shift engine generates ID, Shift 1 ($a=1.25$), and Shift 2 ($a=1.50$) test sets without data leakage or label corruption.
7. **Rigorous Evaluation:** Precision, Recall, F1, PR-AUC, and ROC-AUC calculated using strictly frozen validation thresholds.
8. **Interactive Dashboard:** Streamlit UI allows interactive dataset inspection, experiment execution, score visualization, and metric comparisons.
9. **IICT Academic Report:** Report sections fully supported by generated evidence, logs, tables, and figures.
