"""Audit script for project results, datasets, and structure."""
import csv, json, os, collections, pathlib, sys

PROJECT = pathlib.Path(r"R:\GIS_Emergency_Healthcare_CSMC\tanishka project")

print("=" * 60)
print("TASK C: RESULTS AUDIT")
print("=" * 60)

# --- CSV audit ---
csv_path = PROJECT / "results" / "experiment_summary.csv"
with open(csv_path) as f:
    rows = list(csv.DictReader(f))

print(f"\nCSV total rows: {len(rows)}")
datasets = sorted(set(r["dataset_id"] for r in rows))
models   = sorted(set(r["model_id"] for r in rows))
conditions = sorted(set(r["condition"] for r in rows))
print(f"Datasets  ({len(datasets)}): {datasets}")
print(f"Models    ({len(models)}): {models}")
print(f"Conditions({len(conditions)}): {conditions}")

failures = [r for r in rows if r.get("status","").upper() != "SUCCESS"]
print(f"\nNon-SUCCESS rows: {len(failures)}")
for row in failures:
    print(f"  FAIL: {row['run_id']} status={row['status']}")

missing_metrics = []
for r in rows:
    for m in ["pr_auc","roc_auc","f1","precision","recall"]:
        val = r.get(m,"").strip()
        if val in ("","None","nan","NaN"):
            missing_metrics.append((r["run_id"], m))
print(f"\nMissing metric values: {len(missing_metrics)}")
for x in missing_metrics[:10]:
    print(f"  {x}")

run_ids = [r["run_id"] for r in rows]
dupes = [k for k,v in collections.Counter(run_ids).items() if v > 1]
print(f"\nDuplicate run_ids: {len(dupes)} -> {dupes}")

expected = len(datasets) * len(models) * len(conditions)
print(f"\nExpected runs: {expected} | Actual CSV rows: {len(rows)}")
if len(rows) != expected:
    # find missing combos
    found = set(run_ids)
    all_combos = set()
    for d in datasets:
        for mo in models:
            for c in conditions:
                all_combos.add(f"{d}_{mo}_{c}")
    # try to infer run_id structure
    print("  Sample run_ids:", run_ids[:3])

# JSON individual files
json_dir = PROJECT / "results" / "json"
json_files = sorted([f for f in os.listdir(json_dir)
                     if f.endswith(".json")
                     and not f.startswith("profile_")
                     and not f.startswith("experiment_")])
print(f"\nIndividual experiment JSONs: {len(json_files)}")

# Check one JSON for structure
if json_files:
    sample_path = json_dir / json_files[0]
    with open(sample_path) as f:
        sample = json.load(f)
    print(f"Sample JSON keys ({json_files[0]}): {list(sample.keys())}")

print("\n" + "=" * 60)
print("TASK B: DATASET VERIFICATION")
print("=" * 60)

authentic_dir = PROJECT / "data" / "raw" / "authentic"
auth_files = sorted(authentic_dir.glob("*.txt"))

# Parse filename metadata: NNN_UCR_Anomaly_NAME_totalLen_anomalyStart_anomalyEnd.txt
for fp in auth_files:
    parts = fp.stem.split("_")
    # last 3 parts are totalLen, anomaly_start, anomaly_end
    try:
        total_len   = int(parts[-3])
        anom_start  = int(parts[-2])
        anom_end    = int(parts[-1])
    except (ValueError, IndexError):
        total_len = anom_start = anom_end = -1

    # Count actual lines
    with open(fp) as f:
        actual_lines = sum(1 for _ in f)

    match = "OK" if actual_lines == total_len else f"MISMATCH (expected {total_len})"
    label_len = anom_end - anom_start
    print(f"\n{fp.name}")
    print(f"  Filename total_len={total_len}, actual lines={actual_lines} [{match}]")
    print(f"  Anomaly window: [{anom_start}, {anom_end}] => {label_len} anomaly points")

print("\n" + "=" * 60)
print("TASK A: STRUCTURE CHECK — app.py and key files")
print("=" * 60)

app_py = PROJECT / "app.py"
print(f"app.py exists: {app_py.exists()}, size: {app_py.stat().st_size} bytes")

# Check what app.py imports
imports = []
with open(app_py) as f:
    for line in f:
        l = line.strip()
        if l.startswith("import ") or l.startswith("from "):
            imports.append(l)
print(f"app.py imports ({len(imports)} lines):")
for imp in imports:
    print(f"  {imp}")

# Check streamlit installed
try:
    import importlib.util
    spec = importlib.util.find_spec("streamlit")
    print(f"\nstreamlit installed: {spec is not None}")
    if spec:
        import streamlit
        print(f"  streamlit version: {streamlit.__version__}")
except Exception as e:
    print(f"streamlit check error: {e}")

print("\nDone.")
