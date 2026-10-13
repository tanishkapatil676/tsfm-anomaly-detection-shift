# IMPLEMENTATION_ROADMAP.md: Milestone Roadmap & Risk Governance

## 1. Executive Summary & Strategy
To maximize AI-assisted efficiency under free-tier constraints (Antigravity session budget, Windows CPU development environment, Google Colab free GPU), implementation is divided into five incremental, modular milestones. Each milestone has strictly defined entry criteria, deliverables, verification commands, exit criteria, and risk fallback mechanisms.

---

## 2. Five-Milestone Detailed Roadmap

### Milestone 1: Research Specification & Project Architecture
* **Status:** **COMPLETED / IN VERIFICATION (CURRENT)**
* **Focus:** Rigorous research planning, protocol definition, schema design, and baseline feasibility assessment.
* **Deliverables:**
  * `PROJECT_SPEC.md`
  * `EXPERIMENT_PROTOCOL.md`
  * `DATASET_CARD_TEMPLATE.md`
  * `MODEL_FEASIBILITY.md`
  * `SYSTEM_ARCHITECTURE.md`
  * `IMPLEMENTATION_ROADMAP.md`
  * `REPORT_EVIDENCE.md`
  * `PROJECT_STATUS.md`
  * `DECISIONS.md`
* **Exit Criteria:** All 9 specification documents created, internally aligned, and verified against IICT requirements.

---

### Milestone 2: Project Foundation & Dataset Verification
* **Status:** **UPCOMING (NEXT)**
* **Focus:** Environment setup, core directory initialization, dataset acquisition, dataset verification, profiling script, and base unit tests.
* **Deliverables:**
  * Directory layout creation (`src/`, `configs/`, `tests/`, `results/`, `docs/`).
  * `requirements.txt` with pinned, compatible versions.
  * `.gitignore` configured to exclude large raw data files and cache directories.
  * `configs/default_config.json` master configuration.
  * `src/data/loader.py`: Download, parse UCR filename, split datasets, execute 8-point checklist.
  * `src/data/profiler.py`: Dataset profiling and visual representation.
  * `tests/test_data_loader.py`: Unit test verifying split parsing and data quality on dummy series.
  * Dataset Cards completed for initial 5 UCR series.
* **Verification Command:** `pytest tests/test_data_loader.py`
* **Exit Criteria:** 5 target UCR datasets downloaded, parsed, verified against 8-point checklist, and logged into dataset cards.

---

### Milestone 3: Model Implementation & TSFM Smoke Test
* **Status:** **PLANNED**
* **Focus:** Common detector interface implementation, baseline models, TSFM adapters, windowing engine, scoring alignment, and isolated smoke tests.
* **Deliverables:**
  * `src/models/base.py`: Abstract `BaseAnomalyDetector` contract.
  * `src/preprocessing/windowing.py`: Sliding window engine and scaler parameter preservation.
  * `src/models/baselines/isolation_forest.py`: Windowed Isolation Forest detector.
  * `src/models/baselines/lstm_ae.py`: PyTorch LSTM Autoencoder detector.
  * `src/models/tsfm/moment_adapter.py`: `MOMENT-1-small` adapter (reconstruction error).
  * `src/models/tsfm/chronos_adapter.py`: `Chronos-Bolt-tiny` adapter (forecasting error fallback).
  * `src/scoring/aligner.py`: Window-to-timestamp score mapper.
  * `tests/test_baseline_smoke.py`: Baseline CPU smoke test suite.
  * `tests/test_tsfm_smoke.py`: Isolated TSFM model loading and inference smoke test.
* **Verification Command:** `pytest tests/test_baseline_smoke.py tests/test_tsfm_smoke.py`
* **Exit Criteria:** Baseline models execute cleanly on CPU; TSFM compatibility confirmed or fallback path cleanly activated.

---

### Milestone 4: Controlled Shift Experiments & Evaluation Engine
* **Status:** **PLANNED**
* **Focus:** Amplitude shift generator, validation-only threshold selector, metric evaluator, results persistence, and result plotting.
* **Deliverables:**
  * `src/shifts/amplitude_shift.py`: Scale transformation engine ($a=1.25, 1.50$).
  * `src/scoring/threshold.py`: Validation split threshold selector ($\tau$).
  * `src/evaluation/metrics.py`: Precision, Recall, F1, PR-AUC, ROC-AUC, and Degradation engine.
  * `src/utils/persistence.py`: Experiment JSON saver and run registry.
  * `experiments/run_experiment.py`: Main CLI experiment script executing full pipeline.
  * `tests/test_shifts.py`: Test ensuring shift preserves label indices and locations.
  * `tests/test_metrics.py`: Test verifying degradation formula ($M_{\text{ID}} - M_{\text{Shift}}$).
* **Verification Command:** `python experiments/run_experiment.py --config configs/default_config.json`
* **Exit Criteria:** End-to-end experiment pipeline executes across all 5 series for all models; JSON results saved under `results/json/`.

---

### Milestone 5: Streamlit Dashboard, Report Evidence & GitHub Release
* **Status:** **PLANNED**
* **Focus:** Interactive web application, IICT academic report evidence compilation, README documentation, and final repository verification.
* **Deliverables:**
  * `app.py`: Streamlit dashboard featuring dataset inspection, experiment execution, score charts, and metric comparison tables.
  * `src/utils/report_generator.py`: Automated generator producing IICT report tables and figures.
  * `README.md`: Professional GitHub repository README with research questions, setup, reproduction guide, and citations.
  * `REPORT_EVIDENCE.md`: Final verified mapping connecting IICT report sections to experiment artifacts.
* **Verification Command:** `streamlit run app.py`
* **Exit Criteria:** Streamlit dashboard operates flawlessly using real saved results; repository passes clean linting/testing; all IICT report evidence exported.

---

## 3. Dependency Matrix & Critical Path

```
Milestone 1 (Specification)
       │
       ▼
Milestone 2 (Data Loader & Profiler)
       │
       ▼
Milestone 3 (Baselines & TSFM Smoke Test)
       │
       ▼
Milestone 4 (Shift Generator & Evaluation Pipeline)
       │
       ▼
Milestone 5 (Streamlit Dashboard & IICT Report Evidence)
```

---

## 4. Risk Matrix & Fallback Mechanisms

| Risk Event | Severity | Probability | Trigger Condition | Automated / Protocol Fallback Action |
| :--- | :--- | :--- | :--- | :--- |
| **MOMENT-1-small OOM / Load Error** | Medium | Medium | PyTorch fails to load model on CPU/RAM | Execute fallback protocol in `MODEL_FEASIBILITY.md`: switch immediately to `amazon/chronos-bolt-tiny`. |
| **Windows PyTorch DLL Failure** | High | Low | `Cuda/DLL` loading error on Windows PC | Run heavy TSFM experiments on Google Colab free tier; develop baselines locally on Windows CPU. |
| **Class Imbalance Metric Undefined** | Low | Medium | Precision is undefined ($TP+FP=0$) at high $\tau$ | Set zero-division behavior to $0.0$; evaluate PR-AUC as primary ranking metric. |
| **Colab Session Timeout** | Low | Medium | Colab disconnects after 12h max limit | Save modular experiment state to Google Drive/JSON after every individual series run. |
| **AI Token / Request Limit** | Medium | Low | Antigravity quota exceeded | Work in small, isolated modules; never regenerate full codebase in single prompts. |
