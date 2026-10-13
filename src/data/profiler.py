"""UCR Time Series Anomaly Archive Profiler.

Computes data quality summaries, descriptive statistics, anomaly metrics,
and generates headless-safe diagnostic plots for candidate time series.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np

import matplotlib
matplotlib.use("Agg")  # Headless execution safe for Colab / Windows background tasks
import matplotlib.pyplot as plt

from src.data.loader import UCRSeries


def profile_ucr_series(series: UCRSeries) -> Dict:
    """Generate a comprehensive profiling report for a loaded UCR series.

    Args:
        series: UCRSeries dataclass instance.

    Returns:
        Structured dictionary containing descriptive statistics, label counts,
        split information, and eligibility quality findings.
    """
    raw = series.raw_data
    train = series.train_data
    test = series.test_data

    anomaly_count = int(series.test_labels.sum())
    test_len = len(test)
    anomaly_ratio = float(anomaly_count / test_len) if test_len > 0 else 0.0

    profile = {
        "series_id": series.series_id,
        "filename": series.filename,
        "filepath": str(series.filepath),
        "split_structure": {
            "total_length": series.total_length,
            "t_split_parameter": series.t_split,
            "train_length": len(train),
            "val_length": len(series.val_data),
            "test_length": test_len,
        },
        "descriptive_statistics": {
            "full_series": {
                "mean": float(np.mean(raw)) if len(raw) > 0 else 0.0,
                "std": float(np.std(raw)) if len(raw) > 0 else 0.0,
                "min": float(np.min(raw)) if len(raw) > 0 else 0.0,
                "max": float(np.max(raw)) if len(raw) > 0 else 0.0,
            },
            "training_prefix": {
                "mean": float(np.mean(train)) if len(train) > 0 else 0.0,
                "std": float(np.std(train)) if len(train) > 0 else 0.0,
                "min": float(np.min(train)) if len(train) > 0 else 0.0,
                "max": float(np.max(train)) if len(train) > 0 else 0.0,
            },
            "test_segment": {
                "mean": float(np.mean(test)) if len(test) > 0 else 0.0,
                "std": float(np.std(test)) if len(test) > 0 else 0.0,
                "min": float(np.min(test)) if len(test) > 0 else 0.0,
                "max": float(np.max(test)) if len(test) > 0 else 0.0,
            },
        },
        "anomaly_annotation": {
            "source_anomaly_start": series.source_anomaly_start,
            "source_anomaly_end": series.source_anomaly_end,
            "test_anomaly_start": series.test_anomaly_start,
            "test_anomaly_end_inclusive": series.test_anomaly_end_inclusive,
            "anomaly_duration_timestamps": int(
                series.test_anomaly_end_exclusive - series.test_anomaly_start
            ),
            "test_anomaly_positive_timestamps": anomaly_count,
            "test_anomaly_ratio": anomaly_ratio,
        },
        "data_quality_evaluation": {
            "is_eligible": series.validation_result.is_eligible,
            "rejection_reasons": series.validation_result.rejection_reasons,
            "warnings": series.validation_result.warnings,
            "check_details": series.validation_result.check_details,
        },
    }

    return profile


def save_profile_report(profile: Dict, output_json_path: Union[str, Path]) -> Path:
    """Save profiling report dictionary to a machine-readable JSON file.

    Args:
        profile: Profiling report dictionary.
        output_json_path: Destination file path.

    Returns:
        Resolved Path object of output JSON.
    """
    out_path = Path(output_json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    return out_path.resolve()


def plot_ucr_series(
    series: UCRSeries,
    output_img_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> Optional[Path]:
    """Generate diagnostic time-series plot with labeled split and anomaly regions.

    Args:
        series: UCRSeries dataclass instance.
        output_img_path: Optional destination file path to save plot image.
        show: Whether to display plot interactively (False by default for headless safety).

    Returns:
        Path object if saved, else None.
    """
    fig, ax = plt.subplots(figsize=(12, 4), dpi=120)

    x = np.arange(series.total_length)
    y = series.raw_data

    # Plot full sequence
    ax.plot(x, y, color="#2B3A42", linewidth=1.0, label="Time Series Signal", alpha=0.9)

    # Highlight training prefix
    t_split = series.t_split
    ax.axvspan(
        0,
        t_split,
        color="#3498DB",
        alpha=0.15,
        label=f"Normal Train Prefix (0..{t_split})",
    )

    # Highlight ground truth anomaly in global coordinates
    global_anom_start = series.source_anomaly_start - 1
    global_anom_end = series.source_anomaly_end - 1
    ax.axvspan(
        global_anom_start,
        global_anom_end,
        color="#E74C3C",
        alpha=0.35,
        label=f"Annotated Anomaly ({series.source_anomaly_start}..{series.source_anomaly_end})",
    )

    ax.set_title(
        f"UCR Series Diagnostic Profile: {series.series_id}\n"
        f"Status: {'ELIGIBLE' if series.validation_result.is_eligible else 'REJECTED'}",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Global Timestamp Index", fontsize=10)
    ax.set_ylabel("Observation Value", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()

    saved_path = None
    if output_img_path:
        out_p = Path(output_img_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=150)
        saved_path = out_p.resolve()

    if show:
        plt.show()

    plt.close(fig)
    return saved_path
