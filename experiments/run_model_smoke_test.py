"""Milestone 3 Model Execution and TSFM Feasibility Runner.

Executes baseline Isolation Forest, PyTorch LSTM Autoencoder, and TSFM feasibility smoke tests,
derives validation thresholds, and prints model status summaries.
"""

import json
from pathlib import Path
import sys
import numpy as np

# Ensure src modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_ucr_file, discover_candidate_files
from src.models.baselines.isolation_forest import IsolationForestDetector
from src.models.baselines.lstm_ae import LSTMAutoencoderDetector
from src.models.tsfm.moment_adapter import MomentDetector
from src.models.tsfm.chronos_adapter import ChronosDetector
from src.scoring.threshold import ThresholdSelector


def run_model_smoke_tests():
    """Run baseline detectors and TSFM feasibility checks."""
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw"
    candidates = discover_candidate_files(raw_dir)

    print("\n=======================================================")
    print(" MILESTONE 3: MODEL EXECUTION & TSFM SMOKE TEST RUNNER ")
    print("=======================================================\n")

    if len(candidates) > 0:
        target_file = candidates[0]
        print(f"[INFO] Using dataset file: {target_file.name}")
        series = load_ucr_file(target_file)
        train_data = series.train_data
        val_data = series.val_data
        test_data = series.test_data
    else:
        print("[INFO] No raw files found. Using synthetic benchmark series...")
        np.random.seed(42)
        train_data = np.sin(np.linspace(0, 40, 10000)) + np.random.normal(0, 0.1, 10000)
        val_data = train_data[-2000:]
        test_data = np.sin(np.linspace(40, 80, 10000)) + np.random.normal(0, 0.1, 10000)
        test_data[7000:7200] += 4.0

    print(f"Train samples: {len(train_data)} | Val samples: {len(val_data)} | Test samples: {len(test_data)}\n")

    # 1. Isolation Forest Baseline Smoke Test
    print("1. Testing Isolation Forest Baseline Detector...")
    iforest = IsolationForestDetector(config={"n_estimators": 50, "window_len": 100, "stride": 1})
    iforest.fit(train_data)
    val_scores_if = iforest.predict_score(val_data)
    test_scores_if = iforest.predict_score(test_data)

    thresh_if = ThresholdSelector(method="percentile", percentile=99.0).fit(val_scores_if)
    print(f"   -> Fit Status: SUCCESS | Val Threshold (tau): {thresh_if:.6f}")
    print(f"   -> Test Scores: min={test_scores_if.min():.4f}, max={test_scores_if.max():.4f}, mean={test_scores_if.mean():.4f}\n")

    # 2. PyTorch LSTM Autoencoder Baseline Smoke Test
    print("2. Testing PyTorch LSTM Autoencoder Baseline Detector...")
    lstm_ae = LSTMAutoencoderDetector(config={"window_len": 100, "hidden_dim": 16, "epochs": 2, "batch_size": 64})
    lstm_ae.fit(train_data, val_data)
    val_scores_ae = lstm_ae.predict_score(val_data)
    test_scores_ae = lstm_ae.predict_score(test_data)

    thresh_ae = ThresholdSelector(method="percentile", percentile=99.0).fit(val_scores_ae)
    print(f"   -> Fit Status: SUCCESS | Val Threshold (tau): {thresh_ae:.6f}")
    print(f"   -> Test Scores: min={test_scores_ae.min():.4f}, max={test_scores_ae.max():.4f}, mean={test_scores_ae.mean():.4f}\n")

    # 3. MOMENT-1-small Feasibility Check
    print("3. Bounded Feasibility Check: AutonLab/MOMENT-1-small...")
    moment_detector = MomentDetector()
    status_m, details_m = moment_detector.check_feasibility()
    print(f"   -> Feasibility Status: {status_m}")
    for k, v in details_m.items():
        print(f"      * {k}: {v}")
    print()

    # 4. Chronos-Bolt-tiny Feasibility Check
    print("4. Bounded Feasibility Check: amazon/chronos-bolt-tiny (Conditional Fallback)...")
    chronos_detector = ChronosDetector()
    status_c, details_c = chronos_detector.check_feasibility()
    print(f"   -> Feasibility Status: {status_c}")
    for k, v in details_c.items():
        print(f"      * {k}: {v}")
    print()

    print("=======================================================")
    print(" MILESTONE 3 SUMMARY: ALL BASELINES VERIFIED ")
    print("=======================================================\n")


if __name__ == "__main__":
    run_model_smoke_tests()
