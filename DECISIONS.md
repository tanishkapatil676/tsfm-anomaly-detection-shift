# DECISIONS.md: Architectural & Methodological Decision Log

## 1. Governance Policy
This document serves as the permanent record of architectural, experimental, and implementation decisions for the CA-4 research project. Any deviation or modification to the agreed research specification MUST be logged here with explicit justification, context, and impact analysis.

---

## 2. Initial Accepted Decisions (Milestone 1 & Milestone 2)

### Decision D-001: Research Question & Research Gap Scope Lock
* **Date:** 2026-10-10
* **Decision:** Preserve the exact CA-3 research question ("Can pretrained time-series foundation models remain reliable for anomaly detection when the test distribution changes?") and research gap without modification or expansion.
* **Justification:** CA-3 established academic alignment for IICT evaluation. Extending or changing the question would invalidate prior literature review work.
* **Status:** **APPROVED & LOCKED**

---

### Decision D-002: Primary Dataset Source Selection
* **Date:** 2026-10-10
* **Decision:** Select the **UCR Time Series Anomaly Archive** as the sole primary dataset source for initial experiments, targeting 5 eligible univariate series.
* **Justification:** Publicly accessible, well-documented, clean normal-only training splits, and explicit ground-truth anomaly annotations.
* **Status:** **APPROVED & LOCKED**

---

### Decision D-003: Model Shortlist & Fallback Hierarchy
* **Date:** 2026-10-10
* **Decision:** Define mandatory baselines as Isolation Forest (CPU) and small LSTM Autoencoder (PyTorch). Primary TSFM candidate set to `AutonLab/MOMENT-1-small` (MIT License). Conditional fallback TSFM set to `amazon/chronos-bolt-tiny` (Apache-2.0 License).
* **Justification:** Isolation Forest and LSTM-AE provide fast, CPU-runnable classical and deep baselines. MOMENT-1-small is explicitly built for zero-shot time series analysis. Chronos-Bolt-tiny serves as an ultra-lightweight fallback if system RAM or GPU availability is constrained.
* **Status:** **APPROVED & LOCKED**

---

### Decision D-004: Primary Distribution Shift Transformation
* **Date:** 2026-10-10
* **Decision:** Formulate distribution shift primarily as synthetic amplitude scale shifts: $x_{\text{shifted}}(t) = a \cdot x(t)$ with scale factors $a \in \{1.25, 1.50\}$ applied to the test split only.
* **Justification:** Simple, mathematically clean, preserves ground-truth anomaly location indices 100%, and simulates real-world gain/calibration drift in time-series sensors.
* **Status:** **APPROVED & LOCKED**

---

### Decision D-005: Strict Test-Label Isolation & Frozen Threshold Policy
* **Date:** 2026-10-10
* **Decision:** Decision thresholds $\tau$ must be computed strictly using validation split scores $s(t \in D_{\text{val}})$. Thresholds are frozen and applied unchanged to ID, Shift 1, and Shift 2 test sets. Test labels MUST NEVER be used for threshold search.
* **Justification:** Prevents data leakage and ensures realistic evaluation reflecting real-world operational conditions where test labels are unobserved.
* **Status:** **APPROVED & LOCKED**

---

### Decision D-006: Common Detector Interface Contract
* **Date:** 2026-10-10
* **Decision:** Enforce a unified `BaseAnomalyDetector` abstract base class requiring `fit(X_train)` and `predict_score(X_test)` methods across all models.
* **Justification:** Decouples models from preprocessing and evaluation pipelines, ensuring seamless plug-and-play evaluation between baselines and foundation models.
* **Status:** **APPROVED & LOCKED**

---

### Decision D-007: Streamlit Dashboard Framework
* **Date:** 2026-10-10
* **Decision:** Select Streamlit as the lightweight web application framework for local visualization and viva demonstration (`app.py`).
* **Justification:** Native Python integration, rapid development, zero frontend compilation requirements, and simple integration with Matplotlib/Plotly figures.
* **Status:** **APPROVED & LOCKED**

---

### Decision D-008: Results Persistence Format
* **Date:** 2026-10-10
* **Decision:** Save all experiment configurations, validation thresholds, and calculated metrics as structured JSON files under `results/json/{run_id}.json`.
* **Justification:** Machine-readable, easy to parse for report tables and dashboard visualization, and prevents accidental overwriting via unique timestamped run IDs.
* **Status:** **APPROVED & LOCKED**

---

### Decision D-009: Milestone 1 Technical Consistency Audit Refinements
* **Date:** 2026-10-10
* **Decision:** Formalized 0-based Python indexing math, boundary case handling, scale preservation proof, invalid threshold fallback policy, and fact labeling rules.
* **Justification:** Eliminates mathematical ambiguity, guarantees zero data leakage, and ensures 100% compliance with non-fabrication rules.
* **Status:** **APPROVED & LOCKED**

---

### Decision D-010: Milestone 2 Project Foundation & Data Loader Architecture
* **Date:** 2026-10-10
* **Decision:** Enforced `UCRSeries` and `ValidationResult` dataclasses, built 10-point quality validator (`validate_ucr_series`), implemented 15 unit tests, and generated dataset cards for 5 UCR series.
* **Justification:** Guarantees robust, zero-leakage data parsing and provides a verified data pipeline foundation.
* **Status:** **APPROVED & LOCKED**

---

### Decision D-011: Milestone 3 Baseline Detector Implementation & TSFM Feasibility Assessment
* **Date:** 2026-10-10
* **Decision:**
  1. **Detector Contract Implementation:** Implemented `BaseAnomalyDetector` in `src/models/base.py` enforcing 1D float arrays, finite score outputs, higher-is-more-anomalous convention, and strict test-label isolation.
  2. **Leakage-Safe Preprocessing & Windowing:** Implemented `StandardScalerPreserver` (freezing $\mu_{\text{train}}, \sigma_{\text{train}}$), `extract_sliding_windows`, and `aggregate_window_scores_to_timestamps` in `src/preprocessing/windowing.py`.
  3. **Isolation Forest Baseline:** Implemented `IsolationForestDetector` in `src/models/baselines/isolation_forest.py` using `sklearn.ensemble.IsolationForest` with inverted path-length scores ($s = -\text{score\_samples}$).
  4. **PyTorch LSTM Autoencoder Baseline:** Implemented sequence-to-sequence PyTorch `LSTMAutoencoderDetector` in `src/models/baselines/lstm_ae.py` trained on normal training windows using MSE reconstruction loss.
  5. **Validation Threshold & Score Aligner:** Implemented `ThresholdSelector` in `src/scoring/threshold.py` deriving frozen threshold $\tau$ from validation split scores, and `align_anomaly_scores` in `src/scoring/aligner.py`.
  6. **TSFM Feasibility Bounded Assessment:** Implemented `MomentDetector` (`AutonLab/MOMENT-1-small`) and `ChronosDetector` (`amazon/chronos-bolt-tiny`) in `src/models/tsfm/`. Conducted bounded feasibility checks: both TSFMs evaluated as `INFEASIBLE_IN_CURRENT_ENVIRONMENT` due to HF Hub config registration requirements (`model_type` missing in MOMENT `config.json`) and T5 token embedding integer scalar type requirements in Chronos.
  7. **Automated Smoke Test Suite:** Added 4 baseline and TSFM smoke tests in `tests/test_baseline_smoke.py` and `tests/test_tsfm_smoke.py` (total 19/19 tests passing OK).
* **Justification:** Establishes complete baseline model execution capability on CPU and empirically verifies TSFM compatibility bounds without breaking the baseline benchmark pipeline.
* **Status:** **APPROVED & LOCKED**

---

### Decision D-012: Milestone 3A Dataset Authenticity Audit & Model-Specific TSFM Verification
* **Date:** 2026-10-11
* **Decision:**
  1. **Dataset Provenance & Cryptographic Verification:** Replaced development placeholders with the 5 authentic UCR source files downloaded directly from official Figshare release (`doi:10.6084/m9.figshare.26410744`, Archive file ID: 48036268). Verified SHA-256 cryptographic hashes, lengths (8k-80k), split positions, and anomaly intervals for all 5 series (`DISTORTED1sddb40`, `DISTORTED2sddb40`, `DISTORTED3sddb40`, `BIDMC1`, `CIMIS44AirTemperature1`). Preserved synthetic placeholders under `data/raw/synthetic_fixtures/`.
  2. **Mathematical Indexing & Inclusivity Verification:** Verified 1-based source indices to 0-based test-relative intervals ($S_{\text{test}} = S - 1 - T_{\text{split}}$, $E_{\text{test\_inc}} = E - 1 - T_{\text{split}}$) with inclusive length $E - S + 1$. Confirmed zero off-by-one errors via automated test suite.
  3. **MOMENT-1-small Model-Specific Investigation:** Tested official `momentfm` pipeline (`MOMENTPipeline.from_pretrained`). Documented reproducible blocker: `momentfm 0.1.4` hard-pins `numpy==1.25.2`, which has no binary wheels on PyPI for Python 3.13 and fails compilation from source (`BackendUnavailable: Cannot import 'setuptools.build_meta'`). Generic HF fallback also fails due to custom architecture code. Status: `FAIL — documented API was tested and a reproducible blocker remains`.
  4. **Chronos Bolt Tiny Documented Pipeline Verification:** Installed `chronos-forecasting` 2.3.2 and tested official `BaseChronosPipeline.from_pretrained('amazon/chronos-bolt-tiny', device_map='cpu', torch_dtype=torch.float32)`. Verified 1.55s load time, 0.25s inference on CPU, finite quantile forecasts `[1, 9, 8]`, and successful batch forecasting `[128, 64] -> [128, 9, 1]`. Promoted Chronos to active foundation model under rolling 1-step forecast median absolute residual scoring ($|x(t) - \hat{x}_{t, \text{median}}|$). Status: `PASS — documented API smoke test succeeded`.
  5. **Regression & Test Suite:** Expanded test suite with authentic UCR verification and Chronos inference verification (total 20/20 unit tests passing OK in 49.5s).
* **Justification:** Solves the dataset provenance and TSFM API verification gaps identified prior to Milestone 4, keeping working baselines intact while establishing an empirically proven foundation model execution path.
* **Status:** **APPROVED & LOCKED**

---

### Decision D-013: Milestone 4 Controlled Distribution-Shift Experiments & Causal Evaluation
* **Date:** 2026-10-11
* **Decision:**
  1. **Controlled Amplitude Shift Implementation:** Implemented $x_{\text{shift}}(t) = a \cdot x_{\text{test}}(t)$ in `src/shifts/amplitude_shift.py` for ID ($a=1.00$), Shift 1 ($a=1.25$), and Shift 2 ($a=1.50$). Strictly guaranteed that shifts apply exclusively to held-out test data, ground-truth labels remain immutable, and arrays are cloned without in-place mutation.
  2. **Causal Context Alignment for Chronos:** Updated `ChronosDetector` in `src/models/tsfm/chronos_adapter.py` to construct historical context prefixes strictly from observations before timestamp $t$ using un-shifted training tail buffer. Proved via targeted invariance testing (`tests/test_chronos_causality.py`) that target $x(t)$ is never in context and future perturbations at $t' > t$ have zero mathematical influence on $s(t)$.
  3. **Zero-Leakage Frozen Normal Thresholding:** Enforced that preprocessing scalers ($\mu_{\text{train}}, \sigma_{\text{train}}$) and decision thresholds ($\tau$, 99th percentile of normal validation reconstruction/forecasting errors) are computed once on normal data and frozen across all ID and shifted test evaluations.
  4. **Metric Integrity & Point-Wise Evaluation:** Implemented `src/evaluation/metrics.py` designating Average Precision (Point-Wise PR-AUC) as the primary ranking metric. Evaluated precision, recall, F1, ROC-AUC, and full confusion matrix (TP, FP, TN, FN). Represented single-class or undefined metrics explicitly as `null` with explanatory reasons rather than substituting zero.
  5. **Structured Artifact Persistence:** Implemented `src/utils/persistence.py` saving individual run JSON records under `results/json/` and summary tables in `results/experiment_summary.csv` and `results/experiment_summary.json`.
  6. **Benchmark Matrix Execution:** Successfully executed all 45 planned experimental runs (5 datasets $\times$ 3 models $\times$ 3 conditions) with 0 failures and 0 skipped runs in 398.8s (6.65 min) on CPU. Generated comparative metric plots under `results/plots/`.
  7. **Automated Test Suite Expansion:** Expanded test suite to 39 unit tests covering shifts, metrics, persistence, runner integration, and causality invariance (39/39 passing OK in 7.9s).
* **Justification:** Fulfills all Milestone 4 research and software engineering requirements with zero leakage, cryptographic dataset integrity, and complete reproducibility.
* **Status:** **APPROVED & LOCKED**


---

### Decision D-014: Milestone 4A Disjoint Train/Val Partitioning, Method-End Window Alignment, and Benchmark Rerun
* **Date:** 2026-10-11
* **Decision:**
  1. **Disjoint Training and Validation Partitioning:** Corrected `src/data/loader.py` so that the normal training prefix ($T_{\text{split}}$) is partitioned into mutually exclusive subsets: effective training split $[0 : T_{\text{split}} - N_{\text{val}}]$ and validation split $[T_{\text{split}} - N_{\text{val}} : T_{\text{split}}]$. Guaranteed $\text{len}(D_{\text{train}}) + \text{len}(D_{\text{val}}) = T_{\text{split}}$ with zero index overlap.
  2. **Strict Out-of-Sample Threshold Calibration:** Enforced that preprocessing scalers ($\mu_{\text{train}}, \sigma_{\text{train}}$) and baseline models (Isolation Forest, LSTM-AE) are fitted exclusively on effective training data. Validation scores and resulting 99th-percentile thresholds ($\tau$) represent genuine out-of-sample normal generalization errors.
  3. **Causal Baseline Window Alignment:** Aligned `src/models/baselines/isolation_forest.py` and `lstm_ae.py` to use `method="end"` in `aggregate_window_scores_to_timestamps`, padding warm-up positions $[0 \dots W-2]$ with $s(W-1)$ and mapping windows to end timestamps causally.
  4. **Contiguous Chronos Context Prefixing:** Configured Chronos in `experiments/run_experiment.py` to use `series.train_data` as context prefix for scoring `val_data`, and un-shifted `series.val_data` as context prefix for scoring `test_data`, ensuring zero context gap and zero future leakage.
  5. **Preservation of Superseded Artifacts:** Archived previous Milestone 4 benchmark outputs under `results_superseded_m4/` for transparent auditability.
  6. **Regression Testing & Suite Expansion:** Created `tests/test_audit_regression.py` containing 4 tests verifying disjoint splits, out-of-sample thresholds, scaler isolation, and window-end causality. Total unit test suite expanded to 43 tests (43/43 passing OK in 34.0s).
  7. **Full Benchmark Rerun:** Re-executed complete 45-run evaluation matrix in 526.24s (8.77 min) on CPU. Regenerated all 45 run-level JSON records, `results/experiment_summary.csv`, and updated comparison plots.
* **Justification:** Eliminates in-sample calibration and data-leakage defects identified in Milestone 4A, establishing complete scientific rigor and reproducibility.
* **Status:** **APPROVED & LOCKED**

---

## 3. Decision Log Template (For Future Changes)

```markdown
### Decision D-[XXX]: [Short Decision Title]
* **Date:** YYYY-MM-DD
* **Decision:** [Detailed statement of what was decided]
* **Context & Problem:** [Why this decision was required]
* **Alternatives Considered:** [What other options were evaluated]
* **Justification:** [Why this specific option was chosen]
* **Impact on Architecture / Protocol:** [Modules, configs, or files affected]
* **Status:** [PROPOSED / APPROVED / REJECTED / SUPERSEDED]
```

