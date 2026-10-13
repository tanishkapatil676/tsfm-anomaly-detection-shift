"""Milestone 2 End-to-End Data Verification and Profiling Runner.

Discovers UCR candidate series in data/raw/, executes data quality validation,
computes profiling statistics, saves JSON reports under results/json/, and
saves diagnostic plots under results/plots/.
"""

import json
from pathlib import Path
import sys
import numpy as np

# Ensure src modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import (
    discover_candidate_files,
    load_ucr_file,
    UCRSeries,
)
from src.data.profiler import (
    profile_ucr_series,
    save_profile_report,
    plot_ucr_series,
)


def run_verification(
    data_dir: str = "data/raw",
    config_path: str = "configs/default_config.json",
    create_synthetic_if_empty: bool = True,
):
    """Run verification and profiling pipeline across candidate series.

    Args:
        data_dir: Directory containing raw UCR text files.
        config_path: Path to master configuration JSON.
        create_synthetic_if_empty: If data_dir has no files, generate 5 candidate fixtures.
    """
    project_root = Path(__file__).resolve().parent.parent
    raw_path = project_root / data_dir
    cfg_file = project_root / config_path

    raw_path.mkdir(parents=True, exist_ok=True)

    config = {}
    if cfg_file.is_file():
        with open(cfg_file, "r", encoding="utf-8") as f:
            config = json.load(f)

    candidates = discover_candidate_files(raw_path)

    # If raw_data_dir is empty, generate 5 synthetic candidate series for local validation
    if len(candidates) == 0 and create_synthetic_if_empty:
        print(f"[INFO] No candidate files found in '{data_dir}'. Generating 5 synthetic test series...")
        series_specs = [
            ("001_UCR_Anomaly_DISTORTED1k_10000_17700_17900.txt", 20000, 10000, 17700, 17900),
            ("002_UCR_Anomaly_DISTORTED2k_10000_15000_15200.txt", 20000, 10000, 15000, 15200),
            ("003_UCR_Anomaly_DISTORTED3k_10000_16500_16700.txt", 20000, 10000, 16500, 16700),
            ("004_UCR_Anomaly_DISTORTED4k_10000_14000_14200.txt", 20000, 10000, 14000, 14200),
            ("005_UCR_Anomaly_DISTORTED5k_10000_18000_18200.txt", 20000, 10000, 18000, 18200),
        ]
        for fname, n_total, t_split, start, end in series_specs:
            f_out = raw_path / fname
            np.random.seed(42)
            train_part = np.sin(np.linspace(0, 50, t_split)) + np.random.normal(0, 0.1, t_split)
            test_len = n_total - t_split
            test_part = np.sin(np.linspace(50, 100, test_len)) + np.random.normal(0, 0.1, test_len)
            # Inject anomaly shapelet distortion
            anom_start_rel = start - 1 - t_split
            anom_end_rel = end - 1 - t_split
            test_part[anom_start_rel : anom_end_rel + 1] += 3.5
            full_series = np.concatenate([train_part, test_part])
            np.savetxt(f_out, full_series, fmt="%.6f")

        candidates = discover_candidate_files(raw_path)

    print(f"\n=======================================================")
    print(f" UCR DATASET VERIFICATION & PROFILING RUNNER ")
    print(f" Found Candidates: {len(candidates)}")
    print(f"=======================================================\n")

    results = []
    results_json_dir = project_root / config.get("results_json_dir", "results/json")
    results_plots_dir = project_root / config.get("results_plots_dir", "results/plots")

    results_json_dir.mkdir(parents=True, exist_ok=True)
    results_plots_dir.mkdir(parents=True, exist_ok=True)

    for idx, cpath in enumerate(candidates, start=1):
        print(f"[{idx}/{len(candidates)}] Processing: {cpath.name} ...")
        series = load_ucr_file(cpath, config)
        profile = profile_ucr_series(series)

        # Save JSON profile
        json_file = results_json_dir / f"profile_{series.series_id}.json"
        save_profile_report(profile, json_file)

        # Save Diagnostic Plot
        plot_file = results_plots_dir / f"plot_{series.series_id}.png"
        plot_ucr_series(series, plot_file)

        status = "ELIGIBLE" if series.validation_result.is_eligible else "REJECTED"
        print(f"   -> Status: {status} | Length: {series.total_length} | Test Anomaly Points: {int(series.test_labels.sum())}")
        if not series.validation_result.is_eligible:
            print(f"   -> Rejection Reasons: {series.validation_result.rejection_reasons}")

        results.append(profile)

    print(f"\n[SUCCESS] Completed verification for {len(results)} series.")
    print(f"JSON reports saved to: {results_json_dir}")
    print(f"Diagnostic plots saved to: {results_plots_dir}\n")


if __name__ == "__main__":
    run_verification()
