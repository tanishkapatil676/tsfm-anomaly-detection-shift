"""Results Persistence Utility for Reproducible Experiments.

Saves structured experiment run records, handles JSON serialization of NumPy types,
preserves null representations for undefined metrics, and generates summary CSV/JSON tables.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np


def convert_numpy_to_python(obj: Any) -> Any:
    """Recursively convert NumPy scalars, arrays, and non-finite floats to Python standard types.

    Ensures JSON compliance: NaNs and Infs are converted to None (JSON null).
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return [convert_numpy_to_python(x) for x in obj.tolist()]
    elif isinstance(obj, dict):
        return {str(k): convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_python(x) for x in obj]
    return obj


def save_run_record(
    record: Dict[str, Any],
    results_dir: Path = Path("results/json"),
) -> Path:
    """Save an experiment run record to a structured JSON file.

    Args:
        record: Experiment record dictionary containing run_id and all metadata.
        results_dir: Directory path for saving JSON records (default 'results/json').

    Returns:
        Path to the saved JSON file.

    Raises:
        ValueError: If record is missing 'run_id'.
    """
    if "run_id" not in record:
        raise ValueError("Record must contain a unique 'run_id' key.")

    run_id = str(record["run_id"])
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    file_path = results_dir / f"{run_id}.json"
    clean_record = convert_numpy_to_python(record)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(clean_record, f, indent=2)

    return file_path


def load_run_record(file_path: Path) -> Dict[str, Any]:
    """Load a persisted experiment run record from a JSON file.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Dictionary containing experiment record.
    """
    path = Path(file_path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_summary_table(
    summary_records: List[Dict[str, Any]],
    output_csv_path: Path = Path("results/experiment_summary.csv"),
    output_json_path: Optional[Path] = Path("results/experiment_summary.json"),
) -> Path:
    """Save aggregated experiment summary records to CSV and JSON formats.

    Args:
        summary_records: List of flat metric dictionaries.
        output_csv_path: Destination path for summary CSV.
        output_json_path: Optional destination path for summary JSON.

    Returns:
        Path to the saved summary CSV file.
    """
    import csv

    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not summary_records:
        output_csv_path.touch()
        return output_csv_path

    fieldnames = list(summary_records[0].keys())
    clean_records = [convert_numpy_to_python(r) for r in summary_records]

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in clean_records:
            # Format None as empty string in CSV
            row = {k: ("" if v is None else v) for k, v in r.items()}
            writer.writerow(row)

    if output_json_path:
        output_json_path = Path(output_json_path)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(clean_records, f, indent=2)

    return output_csv_path
