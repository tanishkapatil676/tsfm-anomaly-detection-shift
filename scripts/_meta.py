import json, pathlib

proj = pathlib.Path(r"R:\GIS_Emergency_Healthcare_CSMC\tanishka project")
rdir = proj / "results" / "json"

datasets = ["001","002","003","004","005"]
print("Dataset metadata from JSON:")
for d in datasets:
    fname = d + "_chronos_bolt_tiny_ID.json"
    with open(rdir / fname) as f:
        j = json.load(f)
    m = j.get("metrics", {})
    print("  [%s] name=%s" % (d, j["dataset_name"]))
    print("       total=%d train=%d val=%d test=%d" % (j["total_points"], j["train_points"], j["val_points"], j["test_points"]))
    print("       anom_global=[%d,%d]  anom_test=[%d,%d]" % (j["anomaly_start_global"], j["anomaly_end_global"], j["anomaly_start_test"], j["anomaly_end_test"]))
    print("       anomalous_pts=%s  prevalence=%s" % (m.get("anomalous_points"), m.get("anomaly_prevalence")))
    print("       source=%s" % j["source_filename"])
    print()
