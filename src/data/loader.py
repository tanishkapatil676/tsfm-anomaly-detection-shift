"""UCR Time Series Anomaly Archive Data Loader and Validator.

This module provides verified, deterministic loading, parsing, validation,
and label alignment for univariate time series from the UCR Anomaly Archive.
"""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class ValidationResult:
    """Structure storing eligibility status and data quality checks for a series."""
    is_eligible: bool = True
    rejection_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    check_details: Dict[str, Union[bool, int, float, str]] = field(default_factory=dict)


@dataclass
class UCRSeries:
    """Structure holding loaded UCR time-series observations, metadata, and labels."""
    series_id: str
    filename: str
    filepath: Path
    total_length: int
    t_split: int
    source_anomaly_start: int
    source_anomaly_end: int
    raw_data: np.ndarray
    train_data: np.ndarray
    val_data: np.ndarray
    test_data: np.ndarray
    test_anomaly_start: int
    test_anomaly_end_inclusive: int
    test_anomaly_end_exclusive: int
    test_labels: np.ndarray
    validation_result: ValidationResult


FILENAME_PATTERN = re.compile(
    r"^(\d+)_UCR_Anomaly_([A-Za-z0-9_-]+)_(\d+)_(\d+)_(\d+)\.txt$"
)


def parse_ucr_filename(filename: str) -> Dict[str, Union[str, int]]:
    """Parse metadata embedded in the standard UCR Anomaly Archive filename.

    Syntax: <ID>_UCR_Anomaly_<NAME>_<SPLIT>_<START>_<END>.txt
    Example: 001_UCR_Anomaly_DISTORTED1k_10000_17700_17900.txt

    Args:
        filename: Basename of the file (e.g. '001_UCR_Anomaly_DISTORTED1k_10000_17700_17900.txt')

    Returns:
        Dict containing:
            - dataset_id (str): Numerical ID string (e.g., '001')
            - dataset_name (str): Series name descriptor (e.g., 'DISTORTED1k')
            - series_id (str): Full identifier string
            - t_split (int): 1-based training-prefix length parameter
            - source_anomaly_start (int): 1-based start line number of anomaly
            - source_anomaly_end (int): 1-based end line number of anomaly

    Raises:
        ValueError: If filename does not match expected UCR syntax.
    """
    match = FILENAME_PATTERN.match(filename)
    if not match:
        raise ValueError(
            f"Filename '{filename}' does not match UCR Anomaly Archive syntax: "
            f"<ID>_UCR_Anomaly_<NAME>_<SPLIT>_<START>_<END>.txt"
        )
    
    dataset_id, dataset_name, split_str, start_str, end_str = match.groups()
    t_split = int(split_str)
    source_start = int(start_str)
    source_end = int(end_str)

    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "series_id": f"{dataset_id}_UCR_Anomaly_{dataset_name}",
        "t_split": t_split,
        "source_anomaly_start": source_start,
        "source_anomaly_end": source_end,
    }


def validate_ucr_series(
    raw_data: np.ndarray,
    t_split: int,
    source_anomaly_start: int,
    source_anomaly_end: int,
    config: Optional[dict] = None,
) -> ValidationResult:
    """Perform 10-point data quality and protocol validation on a time series.

    Args:
        raw_data: 1D numpy array of observations.
        t_split: Training prefix length (1-based from filename).
        source_anomaly_start: 1-based line number of anomaly start.
        source_anomaly_end: 1-based line number of anomaly end.
        config: Optional configuration dictionary with threshold parameters.

    Returns:
        ValidationResult detailing eligibility and quality findings.
    """
    result = ValidationResult()
    cfg = config or {}

    min_train_len = cfg.get("min_train_len", 1000)
    min_val_len = cfg.get("min_val_len", 200)
    min_window_len = cfg.get("window_len", 100)
    val_fraction = cfg.get("val_split_fraction", 0.20)

    total_len = len(raw_data)
    result.check_details["total_length"] = total_len
    result.check_details["t_split"] = t_split

    # Check 1: Non-empty series
    if total_len == 0:
        result.is_eligible = False
        result.rejection_reasons.append("Series is empty (0 observations).")
        return result

    # Check 2: Missing or non-finite values
    nan_count = int(np.isnan(raw_data).sum())
    inf_count = int(np.isinf(raw_data).sum())
    result.check_details["nan_count"] = nan_count
    result.check_details["inf_count"] = inf_count

    if nan_count > 0 or inf_count > 0:
        result.is_eligible = False
        result.rejection_reasons.append(
            f"Series contains invalid non-finite values (NaN: {nan_count}, Inf: {inf_count})."
        )

    # Check 3: Valid training-prefix length
    if t_split < min_train_len:
        result.is_eligible = False
        result.rejection_reasons.append(
            f"Training prefix length {t_split} is below minimum requirement {min_train_len}."
        )

    if t_split >= total_len:
        result.is_eligible = False
        result.rejection_reasons.append(
            f"Training split index {t_split} exceeds total sequence length {total_len} (empty test segment)."
        )

    # Check 4: Valid 1-based source anomaly boundaries
    if source_anomaly_start > source_anomaly_end:
        result.is_eligible = False
        result.rejection_reasons.append(
            f"Source anomaly start index {source_anomaly_start} is greater than end index {source_anomaly_end}."
        )

    if source_anomaly_start < 1 or source_anomaly_end > total_len:
        result.is_eligible = False
        result.rejection_reasons.append(
            f"Source anomaly boundaries [{source_anomaly_start}, {source_anomaly_end}] fall outside sequence range [1, {total_len}]."
        )

    # Check 5: Anomaly located in test segment (Normal-only training assumption)
    # Convert source 1-based to 0-based global index:
    global_anomaly_start_0based = source_anomaly_start - 1
    if global_anomaly_start_0based < t_split:
        result.is_eligible = False
        result.rejection_reasons.append(
            f"Anomaly start index {source_anomaly_start} falls inside training prefix (0..{t_split-1}), "
            f"violating normal-only training protocol."
        )

    # Check 6: Test segment length
    test_len = total_len - t_split
    result.check_details["test_length"] = test_len
    if test_len < min_window_len:
        result.is_eligible = False
        result.rejection_reasons.append(
            f"Test segment length {test_len} is shorter than minimum window length {min_window_len}."
        )

    # Check 7: Validation sub-split sufficiency
    val_len = int(val_fraction * t_split)
    result.check_details["val_length"] = val_len
    if val_len < min_val_len:
        result.warnings.append(
            f"Validation subset length {val_len} is below recommended {min_val_len}."
        )

    # Check 8: Non-zero variance in training prefix
    if t_split > 0 and total_len >= t_split:
        train_std = float(np.std(raw_data[0:t_split]))
        result.check_details["train_std"] = train_std
        if train_std == 0.0:
            result.is_eligible = False
            result.rejection_reasons.append("Training prefix has zero variance (constant signal).")

    return result


def load_ucr_file(
    filepath: Union[str, Path],
    config: Optional[dict] = None,
) -> UCRSeries:
    """Load a UCR Time Series Anomaly Archive text file, validate, and extract splits.

    Args:
        filepath: Path to the raw .txt file.
        config: Configuration parameters dictionary.

    Returns:
        UCRSeries instance containing raw data, splits, test labels, and validation details.

    Raises:
        FileNotFoundError: If filepath does not exist.
        ValueError: If file is unparseable or metadata is invalid.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path.resolve()}")

    meta = parse_ucr_filename(path.name)
    t_split = meta["t_split"]
    source_start = meta["source_anomaly_start"]
    source_end = meta["source_anomaly_end"]

    # Read numeric series (one float per line)
    try:
        raw_data = np.loadtxt(path, dtype=np.float64)
        if raw_data.ndim > 1:
            raw_data = raw_data.flatten()
    except Exception as exc:
        raise ValueError(f"Failed to parse numeric observations from '{path.name}': {exc}") from exc

    if raw_data.size == 0:
        raise ValueError(f"Dataset file '{path.name}' is empty (0 observations).")

    # Perform validation checklist
    val_res = validate_ucr_series(
        raw_data=raw_data,
        t_split=t_split,
        source_anomaly_start=source_start,
        source_anomaly_end=source_end,
        config=config,
    )

    total_len = len(raw_data)
    cfg = config or {}
    val_fraction = cfg.get("val_split_fraction", 0.20)

    # Indexing math & split creation
    # Normal training prefix length is t_split.
    # Validation split is partitioned strictly from the end of the normal prefix:
    # Effective training split: 0 .. (t_split - val_size - 1)
    # Independent validation split: (t_split - val_size) .. (t_split - 1)
    # Test split: t_split .. N - 1
    val_size = max(1, int(val_fraction * t_split)) if t_split > 0 else 0
    effective_train_len = max(0, t_split - val_size)
    
    train_data = raw_data[0:effective_train_len] if t_split <= total_len else raw_data[:effective_train_len]
    val_data = raw_data[effective_train_len:t_split] if t_split <= total_len else np.array([], dtype=np.float64)
    
    # Test split: t_split .. N - 1
    test_data = raw_data[t_split:] if t_split < total_len else np.array([], dtype=np.float64)
    test_len = len(test_data)

    # Convert 1-based source indices to 0-based test-relative indices:
    # test_start = source_start - 1 - t_split
    # test_end_inclusive = source_end - 1 - t_split
    test_anomaly_start = max(0, source_start - 1 - t_split)
    test_anomaly_end_inclusive = min(test_len - 1, source_end - 1 - t_split) if test_len > 0 else 0
    test_anomaly_end_exclusive = max(test_anomaly_start, test_anomaly_end_inclusive + 1)

    # Construct binary pointwise test label vector
    test_labels = np.zeros(test_len, dtype=np.int8)
    if val_res.is_eligible and test_len > 0:
        test_labels[test_anomaly_start:test_anomaly_end_exclusive] = 1

    return UCRSeries(
        series_id=meta["series_id"],
        filename=path.name,
        filepath=path.resolve(),
        total_length=total_len,
        t_split=t_split,
        source_anomaly_start=source_start,
        source_anomaly_end=source_end,
        raw_data=raw_data,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        test_anomaly_start=test_anomaly_start,
        test_anomaly_end_inclusive=test_anomaly_end_inclusive,
        test_anomaly_end_exclusive=test_anomaly_end_exclusive,
        test_labels=test_labels,
        validation_result=val_res,
    )


def discover_candidate_files(data_dir: Union[str, Path]) -> List[Path]:
    """Search directory for candidate UCR Anomaly Archive text files.

    Args:
        data_dir: Directory path to scan.

    Returns:
        List of Path objects matching standard UCR filename syntax.
    """
    p = Path(data_dir)
    if not p.is_dir():
        return []
    
    candidates = []
    for f in p.glob("*.txt"):
        if FILENAME_PATTERN.match(f.name):
            candidates.append(f.resolve())

    return sorted(candidates, key=lambda x: x.name)
