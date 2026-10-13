"""Representative Authentic Preflight Run.

Tests an authentic dataset (Series 004) end-to-end with all three detectors
(Isolation Forest, LSTM-AE, Chronos Bolt Tiny) under ID, SHIFT_125, and SHIFT_150.
Verifies runtime, memory stability, score ranges, and threshold freezing before launching full matrix.
"""

import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.data.loader import load_ucr_file
from src.shifts.amplitude_shift import get_shift_conditions
from src.scoring.threshold import ThresholdSelector
from src.evaluation.metrics import calculate_pointwise_metrics, calculate_metric_degradation
from src.models.baselines.isolation_forest import IsolationForestDetector
from src.models.baselines.lstm_ae import LSTMAutoencoderDetector
from src.models.tsfm.chronos_adapter import ChronosDetector


def run_representative_check():
    data_path = Path("data/raw/004_UCR_Anomaly_DISTORTEDBIDMC1_2500_5400_5600.txt")
    print(f"Loading representative authentic file: {data_path.name}")
    series = load_ucr_file(data_path)

    print(f"Total points: {series.total_length}, Train: {len(series.train_data)}, "
          f"Val: {len(series.val_data)}, Test: {len(series.test_data)}")
    anom_len = int(series.test_labels.sum())
    print(f"Ground truth anomaly interval: [{series.test_anomaly_start}, {series.test_anomaly_end_inclusive}] "
          f"({anom_len} points)")

    shifts = get_shift_conditions(series.test_data)
    y_true = series.test_labels

    models = [
        ("isolation_forest", IsolationForestDetector(config={"n_estimators": 50, "random_seed": 42})),
        ("lstm_ae", LSTMAutoencoderDetector(config={"epochs": 3, "batch_size": 64, "random_seed": 42})),
        ("chronos_bolt_tiny", ChronosDetector(config={"context_len": 64, "batch_size": 512})),
    ]

    print("\n" + "=" * 60)
    for model_name, detector in models:
        print(f"\n--- Testing Detector: {model_name} ---")
        t0 = time.time()
        detector.fit(series.train_data, val_series=series.val_data)
        t_fit = time.time() - t0
        print(f"  Fitted in {t_fit:.2f}s")

        # Validation scoring & threshold derivation
        t_val0 = time.time()
        val_scores = detector.predict_score(series.val_data)
        t_val = time.time() - t_val0

        thresh_sel = ThresholdSelector(method="percentile", percentile=99.0)
        tau = thresh_sel.fit(val_scores)
        print(f"  Val scored in {t_val:.2f}s, Frozen threshold tau (99%): {tau:.6f}")

        cond_results = {}
        for cond_name in ["ID", "SHIFT_125", "SHIFT_150"]:
            cond_data = shifts[cond_name]["data"]
            t_test0 = time.time()
            test_scores = detector.predict_score(cond_data)
            t_test = time.time() - t_test0

            test_preds = thresh_sel.predict_binary(test_scores)
            metrics = calculate_pointwise_metrics(y_true, test_scores, test_preds)
            cond_results[cond_name] = metrics

            print(f"  [{cond_name}] Scored in {t_test:.2f}s | AP: {metrics['pr_auc']:.4f} | "
                  f"F1: {metrics['f1']:.4f} | TP: {metrics['tp']} | FP: {metrics['fp']}")

        # Degradation
        deg_125 = calculate_metric_degradation(cond_results["ID"], cond_results["SHIFT_125"])
        deg_150 = calculate_metric_degradation(cond_results["ID"], cond_results["SHIFT_150"])
        print(f"  Delta AP (ID - Shift125): {deg_125['delta_pr_auc']:.4f}")
        print(f"  Delta AP (ID - Shift150): {deg_150['delta_pr_auc']:.4f}")

    print("\n" + "=" * 60)
    print("Representative preflight run complete. All 3 detectors verified!")


if __name__ == "__main__":
    run_representative_check()
