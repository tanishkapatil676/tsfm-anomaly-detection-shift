"""JSON spot-check for results consistency."""
import json, pathlib, os

rdir = pathlib.Path(r"R:\GIS_Emergency_Healthcare_CSMC\tanishka project\results\json")

def show(fname):
    with open(rdir / fname) as f:
        d = json.load(f)
    m = d.get("metrics", {})
    deg = d.get("degradation", {})
    print(f"\n--- {fname} ---")
    print(f"  run_id={d['run_id']}  status={d['status']}")
    print(f"  dataset={d['dataset_name']}  model={d['model_id']}  condition={d['condition']}")
    print(f"  total={d.get('total_points')} train={d.get('train_points')} val={d.get('val_points')} test={d.get('test_points')}")
    print(f"  anom_global=[{d.get('anomaly_start_global')},{d.get('anomaly_end_global')}]  anom_test=[{d.get('anomaly_start_test')},{d.get('anomaly_end_test')}]")
    print(f"  pr_auc={m.get('pr_auc','?')}  roc_auc={m.get('roc_auc','?')}  f1={m.get('f1','?')}  P={m.get('precision','?')}  R={m.get('recall','?')}")
    print(f"  tp={m.get('tp')}  fp={m.get('fp')}  tn={m.get('tn')}  fn={m.get('fn')}")
    print(f"  threshold={d.get('threshold_value')}  method={d.get('threshold_method')}")
    print(f"  degradation_keys={list(deg.keys())}")

show("001_chronos_bolt_tiny_ID.json")
show("001_chronos_bolt_tiny_SHIFT_150.json")
show("004_lstm_ae_SHIFT_125.json")
show("005_isolation_forest_ID.json")

print("\n--- SPLIT MATH CHECK (all 45 JSONs) ---")
all_ok = True
for fname in sorted(os.listdir(rdir)):
    if not fname.endswith(".json") or fname.startswith("profile_") or fname.startswith("experiment_"):
        continue
    with open(rdir / fname) as f:
        d = json.load(f)
    tot = d.get("total_points", 0)
    tr = d.get("train_points", 0)
    va = d.get("val_points", 0)
    te = d.get("test_points", 0)
    if tr + va + te != tot:
        print(f"  SPLIT MISMATCH: {fname}  {tr}+{va}+{te}={tr+va+te} != {tot}")
        all_ok = False
if all_ok:
    print("  All 45 JSON split sums verified OK (train+val+test==total)")

# CSV vs JSON consistency check
import csv
csv_path = pathlib.Path(r"R:\GIS_Emergency_Healthcare_CSMC\tanishka project\results\experiment_summary.csv")
with open(csv_path) as f:
    csv_rows = {r["run_id"]: r for r in csv.DictReader(f)}

print("\n--- CSV vs JSON consistency (pr_auc) ---")
mismatches = 0
for fname in sorted(os.listdir(rdir)):
    if not fname.endswith(".json") or fname.startswith("profile_") or fname.startswith("experiment_"):
        continue
    with open(rdir / fname) as f:
        d = json.load(f)
    rid = d["run_id"]
    if rid not in csv_rows:
        print(f"  run_id {rid} in JSON but NOT in CSV")
        mismatches += 1
        continue
    csv_pr = float(csv_rows[rid].get("pr_auc", -1))
    json_pr = float(d.get("metrics", {}).get("pr_auc", -2))
    if abs(csv_pr - json_pr) > 1e-6:
        print(f"  MISMATCH {rid}: CSV pr_auc={csv_pr}  JSON pr_auc={json_pr}")
        mismatches += 1
if mismatches == 0:
    print("  All 45 run pr_auc values match between CSV and individual JSONs")
