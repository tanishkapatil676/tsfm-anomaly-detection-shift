"""Verify dataset filename metadata vs JSON experiment records."""
import json, pathlib, os

rdir = pathlib.Path(r"R:\GIS_Emergency_Healthcare_CSMC\tanishka project\results\json")
adir = pathlib.Path(r"R:\GIS_Emergency_Healthcare_CSMC\tanishka project\data\raw\authentic")

files = {
    "001": "001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt",
    "002": "002_UCR_Anomaly_DISTORTED2sddb40_35000_56600_56900.txt",
    "003": "003_UCR_Anomaly_DISTORTED3sddb40_35000_46600_46900.txt",
    "004": "004_UCR_Anomaly_DISTORTEDBIDMC1_2500_5400_5600.txt",
    "005": "005_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature1_4000_5391_5392.txt",
}

print("Dataset metadata verification:")
print("(Note: filename integer before anom_start = t_split/training_prefix, NOT total length)")
for did, fname in sorted(files.items()):
    parts = fname[:-4].split("_")
    t_split_fn = int(parts[-3])
    anom_s_fn  = int(parts[-2])
    anom_e_fn  = int(parts[-1])
    actual_lines = sum(1 for _ in open(adir / fname))

    jfname = did + "_chronos_bolt_tiny_ID.json"
    with open(rdir / jfname) as f:
        d = json.load(f)

    tot = d["total_points"]
    tr  = d["train_points"]
    va  = d["val_points"]
    te  = d["test_points"]
    as_ = d["anomaly_start_global"]
    ae_ = d["anomaly_end_global"]

    lines_match = "OK" if actual_lines == tot else "MISMATCH(file=%d vs json=%d)" % (actual_lines, tot)
    anom_match  = "OK" if (anom_s_fn == as_ and anom_e_fn == ae_) else "MISMATCH"
    # t_split_fn should equal tr+va (training prefix = all data before test)
    split_match = "OK" if t_split_fn == (tr + va) else ("MISMATCH(fn=%d vs tr+va=%d)" % (t_split_fn, tr + va))

    print("\n  [%s] %s" % (did, fname))
    print("    Filename: t_split=%d  anom=[%d,%d]" % (t_split_fn, anom_s_fn, anom_e_fn))
    print("    JSON: total=%d  train=%d  val=%d  test=%d" % (tot, tr, va, te))
    print("    JSON: anom_global=[%d,%d]  anom_test=[%d,%d]" % (as_, ae_, d["anomaly_start_test"], d["anomaly_end_test"]))
    print("    lines_match=%s | anom_match=%s | split_match=%s" % (lines_match, anom_match, split_match))

print("\nDone.")
