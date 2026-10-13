"""Milestone 4 Experiment Runner: Controlled Distribution-Shift Benchmark.

Executes the complete experimental matrix across 5 authentic UCR datasets,
3 anomaly detectors (Isolation Forest, LSTM-AE, Chronos Bolt Tiny),
and 3 test conditions (ID, SHIFT_125, SHIFT_150) yielding up to 45 runs.

Strictly follows EXPERIMENT_PROTOCOL.md:
- Preprocessing and scalers fit strictly on training data.
- Decision threshold tau derived from validation split and frozen.
- No test-label leakage.
- Saves structured JSON run records and generates publication-ready summary plots.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.loader import discover_candidate_files, load_ucr_file
from src.shifts.amplitude_shift import get_shift_conditions
from src.scoring.threshold import ThresholdSelector
from src.evaluation.metrics import calculate_pointwise_metrics, calculate_metric_degradation
from src.models.baselines.isolation_forest import IsolationForestDetector
from src.models.baselines.lstm_ae import LSTMAutoencoderDetector
from src.models.tsfm.chronos_adapter import ChronosDetector
from src.utils.persistence import save_run_record, save_summary_table


def instantiate_detector(model_name: str, config: Optional[Dict] = None):
    """Instantiate detector by model identifier string."""
    cfg = config or {}
    if model_name == "isolation_forest":
        return IsolationForestDetector(config=cfg)
    elif model_name == "lstm_ae":
        return LSTMAutoencoderDetector(config=cfg)
    elif model_name == "chronos_bolt_tiny":
        return ChronosDetector(config=cfg)
    else:
        raise ValueError(f"Unknown model identifier '{model_name}'.")


def run_single_dataset_benchmark(
    file_path: Path,
    model_names: List[str],
    shift_conditions: List[str],
    results_dir: Path,
    smoke_mode: bool = False,
) -> List[Dict]:
    """Execute all models and shift conditions for one authentic dataset."""
    series = load_ucr_file(file_path)
    file_bytes = file_path.read_bytes()
    file_sha256 = hashlib.sha256(file_bytes).hexdigest()

    from src.data.loader import parse_ucr_filename
    meta = parse_ucr_filename(file_path.name)
    dataset_id = meta["dataset_id"]
    dataset_name = meta["dataset_name"]
    anom_count = int(series.test_labels.sum())

    print(f"\n========================================================")
    print(f"Dataset [{dataset_id}]: {dataset_name} ({file_path.name})")
    print(f"  Length: {series.total_length} | Train: {len(series.train_data)} | "
          f"Val: {len(series.val_data)} | Test: {len(series.test_data)}")
    print(f"  Test Anomaly Pts: {anom_count} ({anom_count / len(series.test_data) * 100:.2f}%)")
    print(f"  SHA-256: {file_sha256[:16]}...")
    print(f"========================================================")

    shifts = get_shift_conditions(series.test_data)
    y_true = series.test_labels

    dataset_records = []

    for model_name in model_names:
        print(f"\n--- Model: {model_name} ---")

        # Configuration tailored for environment and smoke mode
        model_cfg = {"random_seed": 42}
        if model_name == "lstm_ae":
            model_cfg["epochs"] = 2 if smoke_mode else 3
            model_cfg["batch_size"] = 256
        elif model_name == "chronos_bolt_tiny":
            model_cfg["context_len"] = 64
            model_cfg["batch_size"] = 512

        detector = instantiate_detector(model_name, model_cfg)

        # 1. Fit strictly on training split
        t_fit_start = time.time()
        try:
            detector.fit(series.train_data, val_series=series.val_data)
            fit_duration = time.time() - t_fit_start
            print(f"  [Fit] Completed in {fit_duration:.2f}s")
        except Exception as exc:
            print(f"  [Fit FAILED] {exc}")
            for cond in shift_conditions:
                dataset_records.append({
                    "run_id": f"{dataset_id}_{model_name}_{cond}_{int(time.time())}",
                    "dataset_id": dataset_id,
                    "dataset_name": dataset_name,
                    "model_id": model_name,
                    "condition": cond,
                    "status": "FAILED",
                    "error": str(exc),
                })
            continue

        # 2. Score validation split & derive frozen threshold
        t_val_start = time.time()
        # For Chronos, validation uses the strictly preceding training observations as prior context
        if model_name == "chronos_bolt_tiny":
            val_scores = detector.predict_score(series.val_data, context_prefix=series.train_data)
        else:
            val_scores = detector.predict_score(series.val_data)
        val_score_duration = time.time() - t_val_start

        thresh_selector = ThresholdSelector(method="percentile", percentile=99.0)
        tau = thresh_selector.fit(val_scores)
        print(f"  [Validation] Scored in {val_score_duration:.2f}s | Frozen threshold tau (99th pct): {tau:.6f}")

        # 3. Evaluate each test condition with the frozen threshold
        condition_results = {}
        for cond_name in shift_conditions:
            shift_info = shifts[cond_name]
            test_input = shift_info["data"]
            factor = shift_info["factor"]

            t_test_start = time.time()
            # For Chronos, test scoring uses the un-shifted validation observations preceding test as prior context
            if model_name == "chronos_bolt_tiny":
                test_scores = detector.predict_score(test_input, context_prefix=series.val_data)
            else:
                test_scores = detector.predict_score(test_input)
            test_score_duration = time.time() - t_test_start

            test_preds = thresh_selector.predict_binary(test_scores)
            metrics = calculate_pointwise_metrics(y_true, test_scores, test_preds)
            condition_results[cond_name] = (test_scores, test_preds, metrics)

            run_id = f"{dataset_id}_{model_name}_{cond_name}"
            timestamp_iso = datetime.now(timezone.utc).isoformat()

            print(f"  [{cond_name} (a={factor:.2f})] Score time: {test_score_duration:.2f}s | "
                  f"AP: {metrics['pr_auc']:.4f} | F1: {metrics['f1']:.4f} | "
                  f"TP: {metrics['tp']} | FP: {metrics['fp']}")

            run_record = {
                "run_id": run_id,
                "timestamp": timestamp_iso,
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "source_filename": file_path.name,
                "file_sha256": file_sha256,
                "model_id": model_name,
                "condition": cond_name,
                "shift_factor": factor,
                "shift_formula": shift_info["formula"],
                "total_points": series.total_length,
                "train_points": len(series.train_data),
                "val_points": len(series.val_data),
                "test_points": len(series.test_data),
                "anomaly_start_global": series.source_anomaly_start,
                "anomaly_end_global": series.source_anomaly_end,
                "anomaly_start_test": series.test_anomaly_start,
                "anomaly_end_test": series.test_anomaly_end_inclusive,
                "scaler_mean": float(detector.scaler.mean_),
                "scaler_scale": float(detector.scaler.scale_),
                "threshold_value": float(tau),
                "threshold_method": "percentile_99",
                "fit_duration_seconds": round(fit_duration, 3),
                "test_score_duration_seconds": round(test_score_duration, 3),
                "status": "SUCCESS",
                "metrics": metrics,
            }

            # If shifted condition, compute degradation from ID
            if cond_name != "ID" and "ID" in condition_results:
                id_metrics = condition_results["ID"][2]
                deg = calculate_metric_degradation(id_metrics, metrics)
                run_record["degradation"] = deg
                print(f"    -> Delta AP (ID - {cond_name}): {deg['delta_pr_auc']:.4f} | "
                      f"Delta F1: {deg['delta_f1']:.4f}")
            else:
                run_record["degradation"] = {
                    "delta_pr_auc": 0.0,
                    "delta_roc_auc": 0.0,
                    "delta_f1": 0.0,
                    "delta_precision": 0.0,
                    "delta_recall": 0.0,
                }

            # Persist individual run record
            save_run_record(run_record, results_dir / "json")

            # Flat summary dictionary for CSV table
            flat_summary = {
                "run_id": run_id,
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "model_id": model_name,
                "condition": cond_name,
                "factor": factor,
                "pr_auc": metrics["pr_auc"],
                "roc_auc": metrics["roc_auc"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "delta_pr_auc": run_record["degradation"]["delta_pr_auc"],
                "delta_f1": run_record["degradation"]["delta_f1"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "tn": metrics["tn"],
                "fn": metrics["fn"],
                "threshold": round(tau, 6),
                "fit_duration_s": round(fit_duration, 2),
                "score_duration_s": round(test_score_duration, 2),
                "status": "SUCCESS",
            }
            dataset_records.append(flat_summary)

    return dataset_records


def generate_experiment_plots(
    summary_records: List[Dict],
    plots_dir: Path,
):
    """Generate publication-ready comparison and degradation charts."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    models = sorted(list(set(r["model_id"] for r in summary_records if r["status"] == "SUCCESS")))
    conditions = ["ID", "SHIFT_125", "SHIFT_150"]

    if not models:
        print("No successful records to plot.")
        return

    # Plot 1: Average Precision (AP) across models and shift conditions (Aggregated mean across datasets)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    width = 0.25

    for idx, cond in enumerate(conditions):
        means = []
        for m in models:
            vals = [
                r["pr_auc"] for r in summary_records
                if r["model_id"] == m and r["condition"] == cond and r["pr_auc"] is not None
            ]
            means.append(np.mean(vals) if vals else 0.0)
        ax.bar(x + (idx - 1) * width, means, width, label=cond)

    ax.set_ylabel("Average Precision (PR-AUC)")
    ax.set_title("Average Precision across Detectors and Shift Conditions")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plot1_path = plots_dir / "metric_comparison_ap.png"
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"Saved AP comparison plot: {plot1_path}")

    # Plot 2: F1 Score across models and shift conditions
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, cond in enumerate(conditions):
        means = []
        for m in models:
            vals = [
                r["f1"] for r in summary_records
                if r["model_id"] == m and r["condition"] == cond and r["f1"] is not None
            ]
            means.append(np.mean(vals) if vals else 0.0)
        ax.bar(x + (idx - 1) * width, means, width, label=cond)

    ax.set_ylabel("Point-Wise F1-Score (Frozen Threshold)")
    ax.set_title("F1-Score across Detectors and Shift Conditions")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plot2_path = plots_dir / "metric_comparison_f1.png"
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Saved F1 comparison plot: {plot2_path}")

    # Plot 3: Performance Degradation Delta AP (ID - Shift)
    fig, ax = plt.subplots(figsize=(8, 5))
    x_shifts = np.arange(len(models))
    shift_cols = ["SHIFT_125", "SHIFT_150"]
    width_deg = 0.35

    for idx, cond in enumerate(shift_cols):
        delta_means = []
        for m in models:
            vals = [
                r["delta_pr_auc"] for r in summary_records
                if r["model_id"] == m and r["condition"] == cond and r["delta_pr_auc"] is not None
            ]
            delta_means.append(np.mean(vals) if vals else 0.0)
        ax.bar(x_shifts + (idx - 0.5) * width_deg, delta_means, width_deg, label=f"Delta ({cond})")

    ax.set_ylabel("Delta AP (M_ID - M_shift, positive = degradation)")
    ax.set_title("Average Precision Degradation under Amplitude Scale Shift")
    ax.set_xticks(x_shifts)
    ax.set_xticklabels(models)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plot3_path = plots_dir / "degradation_summary_ap.png"
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"Saved Degradation plot: {plot3_path}")


def main():
    parser = argparse.ArgumentParser(description="Run CA-4 Controlled Distribution Shift Experiments.")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Directory containing authentic UCR files.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save experiment results.")
    parser.add_argument(
        "--models",
        type=str,
        default="isolation_forest,lstm_ae,chronos_bolt_tiny",
        help="Comma-separated model identifiers.",
    )
    parser.add_argument(
        "--shifts",
        type=str,
        default="ID,SHIFT_125,SHIFT_150",
        help="Comma-separated shift condition identifiers.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help="Comma-separated dataset prefixes/IDs to evaluate (or 'all').",
    )
    parser.add_argument("--smoke", action="store_true", help="Fast smoke mode (fewer epochs).")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    shift_conditions = [s.strip() for s in args.shifts.split(",") if s.strip()]

    candidate_files = discover_candidate_files(data_dir)
    if not candidate_files:
        print(f"No candidate files found in {data_dir.resolve()}.")
        sys.exit(1)

    if args.datasets != "all":
        selected_ids = [d.strip() for d in args.datasets.split(",") if d.strip()]
        files_to_run = [
            f for f in candidate_files
            if any(f.name.startswith(sid) for sid in selected_ids)
        ]
    else:
        files_to_run = candidate_files[:5]  # The 5 primary benchmark series

    print(f"\n========================================================")
    print(f"CA-4 DISTRIBUTION SHIFT BENCHMARK EXPERIMENT RUNNER")
    print(f"  Target Datasets: {len(files_to_run)}")
    print(f"  Target Models: {model_names}")
    print(f"  Target Shifts: {shift_conditions}")
    print(f"  Expected Run Count: {len(files_to_run) * len(model_names) * len(shift_conditions)}")
    print(f"  Results Directory: {results_dir.resolve()}")
    print(f"========================================================")

    all_records = []
    total_start = time.time()

    for fpath in files_to_run:
        records = run_single_dataset_benchmark(
            file_path=fpath,
            model_names=model_names,
            shift_conditions=shift_conditions,
            results_dir=results_dir,
            smoke_mode=args.smoke,
        )
        all_records.extend(records)

    total_duration = time.time() - total_start

    # Save summary tables
    csv_path = results_dir / "experiment_summary.csv"
    json_path = results_dir / "experiment_summary.json"
    save_summary_table(all_records, csv_path, json_path)

    # Generate plots
    plots_dir = results_dir / "plots"
    generate_experiment_plots(all_records, plots_dir)

    successful_runs = sum(1 for r in all_records if r.get("status") == "SUCCESS")
    failed_runs = sum(1 for r in all_records if r.get("status") != "SUCCESS")

    print(f"\n========================================================")
    print(f"BENCHMARK COMPLETE in {total_duration:.2f}s ({total_duration / 60:.2f} min)")
    print(f"  Total Runs Attempted: {len(all_records)}")
    print(f"  Successful Runs: {successful_runs}")
    print(f"  Failed Runs: {failed_runs}")
    print(f"  Summary CSV: {csv_path.resolve()}")
    print(f"  Summary JSON: {json_path.resolve()}")
    print(f"  Plots: {plots_dir.resolve()}")
    print(f"========================================================")


if __name__ == "__main__":
    main()
