# MODEL_FEASIBILITY.md: Model Comparison, Compatibility & Smoke Test Plan

## 1. Overview & Comparison Matrix

This document provides the technical feasibility assessment, execution constraints, scoring mechanisms, and fallback decision protocols for all baseline and foundation models in the CA-4 research project.

| Model Name | Category | Exact Model ID / Repository | License | Hardware Requirement | Primary Anomaly Scoring Method |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Isolation Forest** | Non-parametric Baseline | `sklearn.ensemble.IsolationForest` | BSD-3-Clause | CPU (< 100 MB RAM) | Negative Average Path Length on Sliding Window Features |
| **LSTM Autoencoder** | Deep Learning Baseline | Custom PyTorch `LSTMAutoencoder` | MIT / Project | CPU or GPU (< 500 MB RAM) | Mean Squared Reconstruction Error ($MSE$) per Window |
| **MOMENT-1-small** | Primary TSFM Candidate | `AutonLab/MOMENT-1-small` | MIT | CPU / Free Colab T4 GPU (~500M RAM) | Zero-shot Masked Reconstruction Mean Squared Error ($MSE$) |
| **Chronos-Bolt-tiny** | Conditional TSFM Fallback | `amazon/chronos-bolt-tiny` | Apache-2.0 | CPU / Free Colab T4 GPU (~50MB RAM) | Zero-shot Rolling 1-Step Forecasting Absolute Error ($MAE$) |

---

## 2. Detailed Model Technical Specifications

### Model A: Isolation Forest (`iForest`)
* **Role:** Required non-parametric baseline.
* **Input Representation:** Sliding window feature extraction (Window size $W=100$, stride $S=1$). Each window is summarized into 5 summary statistics: $[\text{Mean}, \text{Std}, \text{Min}, \text{Max}, \text{Median}]$.
* **Training Protocol:** Fit strictly on training window features $X_{\text{train\_win}} \in \mathbb{R}^{N_{\text{train\_win}} \times 5}$.
* **Scoring Method:** `score_samples()` outputs negative path length. Score is negated so that higher values indicate higher anomaly likelihood: $s_{\text{iForest}} = - \text{score\_samples}(X_{\text{test\_win}})$.
* **Verification Status:** **PASS — Standard library baseline verified.** `sklearn.ensemble.IsolationForest` verified on CPU; smoke test completes in $< 1$s.

### Model B: Small LSTM Autoencoder (`LSTM-AE`)
* **Role:** Required parametric deep learning baseline.
* **Architecture:** 2-layer LSTM Encoder (hidden dim: 16), Bottleneck representation (8), 2-layer LSTM Decoder, Linear Projection.
* **Input Representation:** Raw normalized sliding window sequence $X_{\text{win}} \in \mathbb{R}^{B \times 1 \times W}$ ($W=50$).
* **Training Protocol:** Trained using Adam optimizer ($\text{lr}=10^{-3}$) with $MSE$ loss on $D_{\text{train}}$ windows for 2-20 epochs. Early stopping monitored on $D_{\text{val}}$.
* **Scoring Method:** Reconstruction error $s_{\text{LSTM-AE}}(t) = \| x_{t} - \hat{x}_{t} \|^2$.
* **Verification Status:** **PASS — PyTorch CPU baseline verified.** Custom PyTorch module executes cleanly on CPU; 2-epoch smoke test completes in ~15s.

### Model C: MOMENT-1-small (Primary TSFM Candidate)
* **Role:** Primary pretrained Time-Series Foundation Model candidate.
* **Hugging Face Model Tag:** `AutonLab/MOMENT-1-small`
* **Documented Interface:** `from momentfm import MOMENTPipeline; MOMENTPipeline.from_pretrained('AutonLab/MOMENT-1-small', ...)`
* **Architecture & Size:** Masked Autoencoder based on T5 transformer architecture (~125M parameters).
* **Verification Status:** **FAIL — documented API was tested and a reproducible blocker remains.**
  * *Documented Blocker Details:* `momentfm 0.1.4` hard-pins `numpy==1.25.2`, which has no pre-compiled binary wheels on PyPI for Python 3.13 on Windows/Linux and fails compilation from sdist (`BackendUnavailable: Cannot import 'setuptools.build_meta'`). Generic HF `AutoModel` loading also fails due to custom architecture code dependencies.
  * *Outcome:* MOMENT cannot be instantiated in this environment without downgrading Python to 3.10/3.11, which would conflict with the local Python 3.13 installation.

### Model D: Chronos-Bolt-tiny (Conditional TSFM Fallback)
* **Role:** Conditional fallback TSFM promoted to active foundation model.
* **Hugging Face Model Tag:** `amazon/chronos-bolt-tiny`
* **Documented Interface:** `from chronos import BaseChronosPipeline; BaseChronosPipeline.from_pretrained('amazon/chronos-bolt-tiny', device_map='cpu', torch_dtype=torch.float32)`
* **Architecture & Size:** Direct quantile forecasting model based on T5 (~9M parameters, 101 weight tensors).
* **Verification Status:** **PASS — documented API smoke test succeeded.**
  * *Empirical Smoke Test Evidence:* Weights loaded in 1.55s. CPU inference on context shape `[1, 32]` completed in 0.25s, returning finite quantile forecasts `[1, 9, 8]`. Batch processing `[128, 64]` returns `[128, 9, 1]` without memory or numerical issues.
  * *Anomaly Scoring Formulation:* Rolling 1-step forecast residual $|x(t) - \hat{x}_{t, \text{median}}|$ aggregated to timestamps. It is methodologically defensible as a forecasting prediction error baseline, but differs conceptually from bidirectional reconstruction.

---

## 3. Fact Classification & Verification Log

* **Verified Open Specifications & Licenses:**
  * UCR dataset structure specifies 1-based filename split index and ground-truth anomaly start/end boundaries.
  * 5 authentic UCR files cryptographically matched against Figshare DOI `10.6084/m9.figshare.26410744`.
  * `amazon/chronos-bolt-tiny` (Apache-2.0 License) verified functional via `chronos-forecasting` 2.3.2.
* **Empirical Feasibility Findings:**
  * Isolation Forest: FEASIBLE on CPU.
  * LSTM Autoencoder: FEASIBLE on CPU.
  * MOMENT-1-small: FAIL (package pins legacy `numpy==1.25.2` incompatible with Python 3.13).
  * Chronos-Bolt-tiny: PASS (official `BaseChronosPipeline` executes cleanly in 0.25s).
* **Approved Methodological Decisions Requiring Tracking:**
  * Chronos Bolt Tiny uses rolling 1-step forecast median absolute residual $|x(t) - \hat{x}_{\text{median}}(t)|$ as anomaly score. Decision D-012 records this formalization.

---

## 4. Smoke Test Procedure

To ensure project progress is never blocked by model initialization issues, two independent smoke test scripts are specified:

### Baseline Smoke Test (`tests/test_baseline_smoke.py`)
1. Create a synthetic univariate series of length $N=1000$ (float64).
2. Instantiate `iForest` and `LSTM-AE`.
3. Fit models on training split ($N_{\text{train}}=500$).
4. Generate scores on test split ($N_{\text{test}}=500$).
5. Assert output score shape equals $N_{\text{test}}$ and contains zero NaNs.
6. **Pass Criterion:** Execution completes on CPU in $< 10$ seconds.

### TSFM Smoke Test (`tests/test_tsfm_smoke.py`)
1. Attempt loading `AutonLab/MOMENT-1-small` with `eval()` mode.
2. Pass a dummy tensor batch $[1, 1, 512]$ through model forward pass.
3. Verify output reconstruction shape and memory footprint.
4. **Pass Criterion:** Model loads and computes forward pass without `OutOfMemoryError` or `ImportError` within 60 seconds.

---

## 5. TSFM Fallback Decision Protocol

```
                  +-----------------------------------+
                  | Attempt MOMENT-1-small Smoke Test |
                  +-----------------------------------+
                                    |
                    Is test successful on CPU/Colab?
                                   / \
                                  /   \
                             YES /     \ NO (ImportError / OOM)
                                /       \
                               v         v
             +-----------------------+ +------------------------------------+
             | Use MOMENT-1-small as | | Attempt Chronos-Bolt-tiny Fallback |
             | Primary TSFM          | +------------------------------------+
             +-----------------------+                   |
                                                 Is test successful?
                                                        / \
                                                   YES /   \ NO
                                                      /     \
                                                     v       v
                                     +---------------------+ +----------------------------+
                                     | Use Chronos-Bolt as | | Proceed with iForest vs.   |
                                     | Fallback TSFM       | | LSTM-AE Baseline Comparison|
                                     +---------------------+ | (Document TSFM Limitations)|
                                                             +----------------------------+
```
