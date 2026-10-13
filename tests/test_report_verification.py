"""Verification suite for the generated IICT Academic Project Report DOCX document.

Validates document structure, student details, placeholders, 16 mandatory sections,
tables, and reconciliation of empirical benchmark metrics against CSV artifacts.
"""

from pathlib import Path
import unittest
import docx
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
REPORT_PATH = PROJECT_ROOT / "report" / "Time_Series_Anomaly_Detection_Project_Report.docx"
CSV_PATH = PROJECT_ROOT / "results" / "experiment_summary.csv"


class TestReportVerification(unittest.TestCase):
    """Test suite verifying structural and empirical fidelity of the project report."""

    @classmethod
    def setUpClass(cls):
        cls.assertTrue(REPORT_PATH.is_file(), f"Report DOCX not found at {REPORT_PATH}")
        cls.doc = docx.Document(str(REPORT_PATH))
        cls.df = pd.read_csv(CSV_PATH)
        cls.all_text = "\n".join([p.text for p in cls.doc.paragraphs])

    def test_student_and_institutional_metadata(self):
        """Verify student personal details and institutional metadata."""
        self.assertIn("TANISHKA PATIL", self.all_text.upper())
        self.assertIn("2405120", self.all_text)
        self.assertIn("TY IT", self.all_text)
        self.assertIn("Division: 3", self.all_text)
        self.assertIn("MGM University", self.all_text)
        self.assertIn("Institute of Information and Communication Technology", self.all_text)
        self.assertIn("2026–2027", self.all_text)

    def test_guide_placeholders_preserved(self):
        """Verify guide and submission date placeholders are explicitly preserved."""
        self.assertIn("[To be confirmed: Guide Name, Designation]", self.all_text)
        self.assertIn("[To be confirmed: Head of Department]", self.all_text)
        self.assertIn("[To be confirmed: Director / Principal]", self.all_text)

    def test_sixteen_mandatory_sections_present(self):
        """Verify all 16 mandatory IICT template sections are present in sequence."""
        required_sections = [
            "1. Abstract and Keywords",
            "2. Introduction",
            "3. Problem Statement",
            "4. Objectives",
            "5. Scope",
            "6. Literature Survey",
            "7. Methodology",
            "8. System Architecture and Design",
            "9. System Specifications",
            "10. Implementation Details",
            "11. Results and Analysis",
            "12. Performance Analysis",
            "13. Conclusion",
            "14. Applications",
            "15. Future Scope",
            "16. References",
        ]
        for sec in required_sections:
            self.assertIn(sec, self.all_text, f"Missing section heading: {sec}")

    def test_tables_present(self):
        """Verify required tables (Metadata, Literature Survey, Benchmark Results) are present."""
        tables = self.doc.tables
        # We have at least 4 tables: Signature table, Metadata table, Literature Survey, Results table
        self.assertGreaterEqual(len(tables), 4, f"Expected at least 4 tables, got {len(tables)}")

        # Verify Results table has 46 rows (1 header + 45 runs)
        results_table = tables[-1]
        self.assertEqual(len(results_table.rows), 46, f"Expected 46 rows in Results table, got {len(results_table.rows)}")

    def test_reconcile_sample_benchmark_metrics(self):
        """Reconcile reported metrics in Table 2 with ground-truth experiment_summary.csv."""
        results_table = self.doc.tables[-1]

        # Map row index to CSV row (rows are 1-indexed in table, skipping header)
        for r_idx in [1, 10, 20, 30, 44, 45]:
            csv_idx = r_idx - 1
            csv_row = self.df.iloc[csv_idx]
            cells = results_table.rows[r_idx].cells

            dataset_reported = cells[0].text.strip()
            pr_auc_reported = float(cells[3].text.strip())
            f1_reported = float(cells[5].text.strip())
            fp_reported = int(cells[8].text.strip())

            self.assertEqual(dataset_reported, str(csv_row["dataset_id"]).zfill(3))
            self.assertAlmostEqual(pr_auc_reported, float(csv_row["pr_auc"]), places=4)
            self.assertAlmostEqual(f1_reported, float(csv_row["f1"]), places=4)
            self.assertEqual(fp_reported, int(csv_row["fp"]))

    def test_embedded_figures_present(self):
        """Verify figures are embedded in the docx package."""
        rels = self.doc.part.rels
        image_parts = [rel.target_ref for rel in rels.values() if "image" in rel.target_ref]
        self.assertGreaterEqual(len(image_parts), 3, f"Expected at least 3 embedded figures, got {len(image_parts)}")


if __name__ == "__main__":
    unittest.main()
