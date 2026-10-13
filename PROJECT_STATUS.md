# PROJECT_STATUS.md: Current Project State & Tracking Log

## 1. Project Overview & Current Milestone
* **Project Title:** Time-Series Foundation Models for Anomaly Detection under Distribution Shift
* **Current Stage:** **Milestone 4A Completed / Milestone 5 Ready to Begin**
* **Last Updated:** 2026-10-11
* **Overall Status:** On Track (Milestone 4A audit completed, disjoint train/val splits verified, out-of-sample thresholds calibrated, method='end' window alignment implemented, 43/43 unit tests passing, 45/45 benchmark runs re-executed with zero leakage)

---

## 2. Milestone Progress Summary

| Milestone Number & Title | Status | Completion % | Completion Date |
| :--- | :--- | :--- | :--- |
| **Milestone 1:** Research Specification & Technical Consistency Audit | **COMPLETED** | 100% | 2026-10-10 |
| **Milestone 2:** Project Foundation & Dataset Verification | **COMPLETED** | 100% | 2026-10-10 |
| **Milestone 3:** Baselines & Initial TSFM Feasibility | **COMPLETED** | 100% | 2026-10-10 |
| **Milestone 3A:** Dataset Authenticity Audit & TSFM Feasibility Verification | **COMPLETED** | 100% | 2026-10-11 |
| **Milestone 4:** Controlled Shift Experiments & Evaluation | **SUPERSEDED** | 100% | 2026-10-11 (Archived) |
| **Milestone 4A:** Split, Threshold & Window-Alignment Audit & Clean Rerun | **COMPLETED** | 100% | 2026-10-11 |
| **Milestone 5:** Streamlit Dashboard & IICT Individual Project Report | **COMPLETED** | 100% | 2026-10-11 |

---

## 3. Completed Artifacts Log (Milestones 4A & 5)

All Milestone 5 deliverables, verification tests, and academic artifacts are complete and verified:

1. `app.py` — Interactive, CPU-friendly Streamlit dashboard loading strictly verified benchmark outputs from `results/` across 5 datasets, 3 models, and 3 conditions.
2. `report/Time_Series_Anomaly_Detection_Project_Report.docx` — Complete, publication-quality academic project report formatted according to the 16-section IICT B.Tech template with embedded figures, tables, authentic results, and single-student metadata (Tanishka Patil, PRN: 2405120).
3. `scripts/generate_report.py` — Deterministic python-docx script assembling the complete academic report directly from verified artifacts.
4. `tests/test_dashboard_logic.py` — 5 unit tests verifying dataset loading, dropdown coverage, publication plot presence, and metric integrity.
5. `tests/test_report_verification.py` — 6 unit tests verifying report structure, mandatory headings, guide placeholders, table row counts, and reconciling sample metrics with CSV.
6. `src/data/loader.py` — Disjoint partitioning of normal prefix into effective training $[0 : T_{\text{split}} - N_{\text{val}}]$ and validation $[T_{\text{split}} - N_{\text{val}} : T_{\text{split}}]$.
7. `src/models/baselines/isolation_forest.py` — Window-end causal score alignment (`method="end"`).
8. `src/models/baselines/lstm_ae.py` — Window-end causal score alignment (`method="end"`).
9. `src/models/tsfm/chronos_adapter.py` — Causal 1-step forecast scoring with contiguous historical context prefix.
10. `experiments/run_experiment.py` — Updated runner with out-of-sample validation scoring and contiguous context prefixing.
11. `results_superseded_m4/` — Preserved archive of initial Milestone 4 run outputs prior to split audit.
12. `results/json/` — Exactly 45 individual run JSON records under verified out-of-sample calibration.
13. `results/experiment_summary.csv` — Regenerated 45-row experiment summary table (10,205 bytes).
14. `results/plots/` — Standardized publication comparison plots (`metric_comparison_ap.png`, `metric_comparison_f1.png`, `degradation_summary_ap.png`).

---

## 4. Verification & Command Log

* [x] **Master Test Suite:** `python -m unittest discover -s tests -p "test_*.py" -v` $\implies$ **54/54 PASSED OK** (Exit code: 0, Execution time: 9.137s).
* [x] **Dashboard Backend Logic:** `python tests/test_dashboard_logic.py` $\implies$ **5/5 PASSED OK** (Exit code: 0).
* [x] **Report Structure & Value Reconciliation:** `python tests/test_report_verification.py` $\implies$ **6/6 PASSED OK** (Exit code: 0).
* [x] **Report Generation:** `python scripts/generate_report.py` $\implies$ Generated `report/Time_Series_Anomaly_Detection_Project_Report.docx`.
* [x] **Benchmark Matrix Execution:** **45/45 runs SUCCEEDED** (0 failed, 0 skipped, verified in `results/experiment_summary.csv`).

---

## 5. Academic Metadata & Placeholders Log

* **Student:** Tanishka Patil (Sole student)
* **Class & Division:** TY IT, Division 3
* **PRN:** 2405120
* **Institute:** Institute of Information and Communication Technology (IICT), MGM University (MGMU), Chh. Sambhajinagar
* **Academic Year:** 2026–2027
* **Guide Placeholder:** `[To be confirmed: Guide Name, Designation]` (Clearly preserved across Cover Page, Certificate, Acknowledgement, and Metadata Table)
* **Submission Date Placeholder:** `[To be confirmed: Date]` (Preserved on Certificate signature block)
* **Blank DOCX Template Note:** The file `Case Study Implementation Report Template (IICT IT MGMU).docx` was confirmed absent from the repository and workspace. If official submission requires direct in-place editing of the university template file, it must be copied into the repository. The complete report has been generated at `report/Time_Series_Anomaly_Detection_Project_Report.docx`.

---

## 6. How to Launch & Demonstrate Locally

1. **Launch Streamlit Dashboard:**
   ```powershell
   streamlit run app.py
   ```
2. **Re-generate DOCX Report:**
   ```powershell
   python scripts/generate_report.py
   ```
3. **Execute Full Automated Verification:**
   ```powershell
   python -m unittest discover -s tests -p "test_*.py" -v
   ```

