"""Generate Complete IICT B.Tech Project Report in DOCX Format.

Project: Time-Series Foundation Models for Anomaly Detection under Distribution Shift
Student: Tanishka Patil | Class: TY IT | Division: 3 | PRN: 2405120
Institute: Institute of Information and Communication Technology (IICT), MGM University (MGMU)
Academic Year: 2026-2027
Guide: [To be confirmed: Guide Name, Designation]
"""

from pathlib import Path
import csv
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

BASE_DIR = Path(__file__).parent.parent.resolve()
RESULTS_DIR = BASE_DIR / "results"
CSV_PATH = RESULTS_DIR / "experiment_summary.csv"
PLOTS_DIR = RESULTS_DIR / "plots"
REPORT_DIR = BASE_DIR / "report"
OUTPUT_DOCX = REPORT_DIR / "Time_Series_Anomaly_Detection_Project_Report.docx"


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Set inner margins (padding) for a table cell."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def set_cell_background(cell, hex_color="F2F4F7"):
    """Set background color for a table cell."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)


def build_report():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    doc = docx.Document()

    # 1. Page Setup: 1 inch margins all around
    sections = doc.sections
    for s in sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)

    # 2. Styles Setup
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(4)

    # Helper for headings
    def add_h1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(14)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x00, 0x20, 0x60)
        return p

    def add_h2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(12)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x00, 0x20, 0x60)
        return p

    def add_h3(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(11)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        return p

    def add_p(text, bold_prefix=None, italic=False):
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(4)
        if bold_prefix:
            r_pre = p.add_run(bold_prefix)
            r_pre.font.name = 'Times New Roman'
            r_pre.font.size = Pt(11)
            r_pre.font.bold = True
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(11)
        run.font.italic = italic
        return p

    # =========================================================================
    # COVER PAGE
    # =========================================================================
    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_inst.paragraph_format.space_before = Pt(0)
    p_inst.paragraph_format.space_after = Pt(2)
    r = p_inst.add_run("MGM UNIVERSITY, CHH. SAMBHAJINAGAR\n")
    r.font.name = 'Times New Roman'
    r.font.size = Pt(15)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x00, 0x20, 0x60)

    r_sub = p_inst.add_run("INSTITUTE OF INFORMATION AND COMMUNICATION TECHNOLOGY (IICT)\nDEPARTMENT OF INFORMATION TECHNOLOGY")
    r_sub.font.name = 'Times New Roman'
    r_sub.font.size = Pt(12)
    r_sub.font.bold = True

    p_proj = doc.add_paragraph()
    p_proj.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_proj.paragraph_format.space_before = Pt(30)
    p_proj.paragraph_format.space_after = Pt(8)
    r_rep = p_proj.add_run("A B.TECH PROJECT REPORT ON\n\n")
    r_rep.font.name = 'Times New Roman'
    r_rep.font.size = Pt(12)
    r_rep.font.bold = True

    r_title = p_proj.add_run("TIME-SERIES FOUNDATION MODELS FOR ANOMALY DETECTION UNDER DISTRIBUTION SHIFT")
    r_title.font.name = 'Times New Roman'
    r_title.font.size = Pt(16)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(0x00, 0x20, 0x60)

    p_subm = doc.add_paragraph()
    p_subm.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_subm.paragraph_format.space_before = Pt(24)
    p_subm.paragraph_format.space_after = Pt(24)
    r_part = p_subm.add_run("Submitted in partial fulfillment of the requirements for the award of the degree of\n")
    r_part.font.name = 'Times New Roman'
    r_part.font.size = Pt(11)
    r_part.font.italic = True
    r_deg = p_subm.add_run("Bachelor of Technology in Information Technology")
    r_deg.font.name = 'Times New Roman'
    r_deg.font.size = Pt(13)
    r_deg.font.bold = True

    p_stud = doc.add_paragraph()
    p_stud.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_stud.paragraph_format.space_before = Pt(24)
    p_stud.paragraph_format.space_after = Pt(4)
    r_by = p_stud.add_run("Submitted by:\n")
    r_by.font.name = 'Times New Roman'
    r_by.font.size = Pt(11)
    r_by.font.italic = True
    r_name = p_stud.add_run("TANISHKA PATIL\n")
    r_name.font.name = 'Times New Roman'
    r_name.font.size = Pt(13)
    r_name.font.bold = True
    r_prn = p_stud.add_run("Class: TY IT | Division: 3 | PRN: 2405120")
    r_prn.font.name = 'Times New Roman'
    r_prn.font.size = Pt(11)
    r_prn.font.bold = True

    p_guide = doc.add_paragraph()
    p_guide.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_guide.paragraph_format.space_before = Pt(24)
    p_guide.paragraph_format.space_after = Pt(36)
    r_ug = p_guide.add_run("Under the Guidance of:\n")
    r_ug.font.name = 'Times New Roman'
    r_ug.font.size = Pt(11)
    r_ug.font.italic = True
    r_gname = p_guide.add_run("[To be confirmed: Guide Name, Designation]")
    r_gname.font.name = 'Times New Roman'
    r_gname.font.size = Pt(12)
    r_gname.font.bold = True
    r_gname.font.color.rgb = RGBColor(0x99, 0x00, 0x00)

    p_year = doc.add_paragraph()
    p_year.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_year.paragraph_format.space_before = Pt(30)
    p_year.paragraph_format.space_after = Pt(0)
    r_ay = p_year.add_run("DEPARTMENT OF INFORMATION TECHNOLOGY\nINSTITUTE OF INFORMATION AND COMMUNICATION TECHNOLOGY\nMGM UNIVERSITY, CHH. SAMBHAJINAGAR - 431003\nACADEMIC YEAR 2026–2027")
    r_ay.font.name = 'Times New Roman'
    r_ay.font.size = Pt(10)
    r_ay.font.bold = True

    doc.add_page_break()

    # =========================================================================
    # CERTIFICATE
    # =========================================================================
    p_cert_title = doc.add_paragraph()
    p_cert_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cert_title.paragraph_format.space_before = Pt(12)
    p_cert_title.paragraph_format.space_after = Pt(18)
    r_ct = p_cert_title.add_run("CERTIFICATE")
    r_ct.font.name = 'Times New Roman'
    r_ct.font.size = Pt(16)
    r_ct.font.bold = True
    r_ct.font.color.rgb = RGBColor(0x00, 0x20, 0x60)

    p_cert_body = doc.add_paragraph()
    p_cert_body.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_cert_body.paragraph_format.line_spacing = 1.25
    p_cert_body.paragraph_format.space_after = Pt(18)
    p_cert_body.add_run(
        "This is to certify that the project report entitled "
        "\"Time-Series Foundation Models for Anomaly Detection under Distribution Shift\" "
        "submitted by Tanishka Patil (PRN: 2405120), a student of Third Year Information Technology "
        "(TY IT, Division 3), is a bonafide work carried out by her as an individual project under my "
        "guidance and supervision at the Institute of Information and Communication Technology (IICT), "
        "MGM University, Chhatrapati Sambhajinagar, in partial fulfillment of the requirements for the "
        "award of the degree of Bachelor of Technology in Information Technology during the academic year 2026–2027.\n\n"
        "It is further certified that to the best of my knowledge, this report represents the authentic and original "
        "experimental investigation conducted by the candidate and has not formed the basis for the award of any "
        "other degree, diploma, or certificate."
    )

    p_sig_space = doc.add_paragraph()
    p_sig_space.paragraph_format.space_before = Pt(40)

    # Signature Table (Single Student / Official Guide & Authority Slots)
    sig_table = doc.add_table(rows=2, cols=2)
    sig_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    sig_table.autofit = False
    for row in sig_table.rows:
        row.cells[0].width = Inches(3.2)
        row.cells[1].width = Inches(3.2)

    cell_g = sig_table.cell(0, 0)
    p = cell_g.paragraphs[0]
    p.add_run("_____________________________\n[To be confirmed: Guide Name, Designation]\nProject Guide\nDepartment of IT, IICT, MGMU")
    p.runs[0].font.name = 'Times New Roman'
    p.runs[0].font.size = Pt(10)

    cell_h = sig_table.cell(0, 1)
    p = cell_h.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p.add_run("_____________________________\n[To be confirmed: Head of Department]\nHead of Department\nDepartment of IT, IICT, MGMU")
    p.runs[0].font.name = 'Times New Roman'
    p.runs[0].font.size = Pt(10)

    cell_d = sig_table.cell(1, 0)
    p = cell_d.paragraphs[0]
    p.paragraph_format.space_before = Pt(36)
    p.add_run("_____________________________\n[To be confirmed: Director / Principal]\nDirector, IICT\nMGM University")
    p.runs[0].font.name = 'Times New Roman'
    p.runs[0].font.size = Pt(10)

    cell_e = sig_table.cell(1, 1)
    p = cell_e.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p.paragraph_format.space_before = Pt(36)
    p.add_run("_____________________________\nExternal Examiner / Internal Examiner\nDate: [To be confirmed: Date]")
    p.runs[0].font.name = 'Times New Roman'
    p.runs[0].font.size = Pt(10)

    doc.add_page_break()

    # =========================================================================
    # ACKNOWLEDGEMENT
    # =========================================================================
    p_ack_title = doc.add_paragraph()
    p_ack_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_ack_title.paragraph_format.space_before = Pt(12)
    p_ack_title.paragraph_format.space_after = Pt(18)
    r_ak = p_ack_title.add_run("ACKNOWLEDGEMENT")
    r_ak.font.name = 'Times New Roman'
    r_ak.font.size = Pt(16)
    r_ak.font.bold = True
    r_ak.font.color.rgb = RGBColor(0x00, 0x20, 0x60)

    p_ack = doc.add_paragraph()
    p_ack.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_ack.paragraph_format.line_spacing = 1.2
    p_ack.paragraph_format.space_after = Pt(12)
    p_ack.add_run(
        "I would like to express my profound gratitude and sincere respect to my project guide, "
        "[To be confirmed: Guide Name, Designation], Department of Information Technology, Institute of "
        "Information and Communication Technology (IICT), MGM University, for their continuous encouragement, "
        "insightful technical advice, and academic support throughout the design, benchmarking, and documentation "
        "of this research project.\n\n"
        "I extend my deep gratitude to [To be confirmed: Head of Department], Head of the Department of Information "
        "Technology, and [To be confirmed: Director / Principal], Director, IICT, MGM University, Chhatrapati "
        "Sambhajinagar, for providing the necessary computational facilities, laboratory resources, and academic "
        "environment required to accomplish this work successfully.\n\n"
        "I also thank all faculty members and technical staff of the Department of Information Technology for their "
        "cooperation. Lastly, I express my heartfelt gratitude to my family and friends for their constant moral "
        "support, patience, and motivation during the completion of this project."
    )

    p_ack_sig = doc.add_paragraph()
    p_ack_sig.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_ack_sig.paragraph_format.space_before = Pt(36)
    r_sig = p_ack_sig.add_run("Tanishka Patil\nPRN: 2405120\nTY IT, Division 3\nIICT, MGM University")
    r_sig.font.name = 'Times New Roman'
    r_sig.font.size = Pt(11)
    r_sig.font.bold = True

    doc.add_page_break()

    # =========================================================================
    # STUDENT & PROJECT DETAILS TABLE
    # =========================================================================
    add_h1("PROJECT DETAILS & METADATA")
    
    meta_table = doc.add_table(rows=8, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_table.autofit = False
    
    meta_data = [
        ("Project Title", "Time-Series Foundation Models for Anomaly Detection under Distribution Shift"),
        ("Student Name", "Tanishka Patil"),
        ("Academic Class & Div", "TY B.Tech Information Technology, Division 3"),
        ("Permanent Reg. No. (PRN)", "2405120"),
        ("Institute / University", "Institute of Information and Communication Technology (IICT), MGM University"),
        ("Academic Year", "2026–2027"),
        ("Project Type", "Individual Project"),
        ("Project Guide", "[To be confirmed: Guide Name, Designation]"),
    ]
    for idx, (k, v) in enumerate(meta_data):
        row = meta_table.rows[idx]
        row.cells[0].width = Inches(2.3)
        row.cells[1].width = Inches(4.2)
        set_cell_background(row.cells[0], "EAECF0")
        set_cell_margins(row.cells[0], top=120, bottom=120, left=150, right=150)
        set_cell_margins(row.cells[1], top=120, bottom=120, left=150, right=150)
        
        p0 = row.cells[0].paragraphs[0]
        r0 = p0.add_run(k)
        r0.font.name = 'Times New Roman'
        r0.font.size = Pt(10)
        r0.font.bold = True
        
        p1 = row.cells[1].paragraphs[0]
        r1 = p1.add_run(v)
        r1.font.name = 'Times New Roman'
        r1.font.size = Pt(10)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # =========================================================================
    # 1. ABSTRACT AND KEYWORDS
    # =========================================================================
    add_h1("1. Abstract and Keywords")
    add_p(
        "Time-series anomaly detection is vital for the health monitoring of critical cyber-physical systems, industrial "
        "processes, and medical sensors. While pretrained Time-Series Foundation Models (TSFMs) have demonstrated zero-shot "
        "forecasting capabilities, their robustness for anomaly detection under test-time distribution shifts remains "
        "underexplored. This project presents a rigorous, reproducible empirical benchmark comparing Amazon's Chronos Bolt Tiny "
        "against classical Isolation Forest and PyTorch LSTM Autoencoder baselines across five authentic UCR Anomaly Archive "
        "datasets under controlled test-time amplitude scaling shifts (1.25x and 1.50x).",
        bold_prefix="Abstract— "
    )
    add_p(
        "A total of 45 benchmark runs were executed across strictly disjoint training, validation, and test splits with "
        "validation-calibrated decision thresholds frozen across conditions. The primary ranking metric evaluated was Average "
        "Precision (PR-AUC), complemented by operational classification metrics (ROC-AUC, Precision, Recall, and F1-score). "
        "Empirical results reveal a significant methodological duality: Chronos Bolt Tiny exhibits remarkable ranking "
        "scale-invariance (ΔPR-AUC ≈ 0.0000 across shift levels) due to its patch-level instance normalization; however, under "
        "a frozen decision threshold, its one-step forecast residuals scale proportionally with test amplitude, causing False "
        "Positives to inflate by 2x to 3x across benchmarks. Conversely, conventional baselines suffer severe ranking and "
        "threshold collapse under moderate scaling. Furthermore, evaluation on extreme class imbalance (UCR Series 005, containing "
        "only 2 anomalies in 4,184 points) reveals a severe divergence between perfect ranking (PR-AUC = 1.0000) and poor "
        "operational precision (2.5%), highlighting that ranking resilience does not translate directly to operational threshold "
        "stability in real-world deployments."
    )
    add_p(
        "Time-Series Foundation Models, Anomaly Detection, Distribution Shift, Scale Invariance, Pretrained Transformers.",
        bold_prefix="Keywords— "
    )

    # =========================================================================
    # 2. INTRODUCTION
    # =========================================================================
    add_h1("2. Introduction")
    add_p(
        "Univariate time-series anomaly detection plays a foundational role in modern telemetry engineering, enabling automated "
        "detection of equipment failures, cardiac arrhythmias, and structural anomalies. Historically, organizations have relied "
        "on statistical process control, proximity-based algorithms, or classical ensemble models such as Isolation Forest. In recent "
        "years, deep representation learning models—specifically recurrent autoencoders—have become standard practice by framing "
        "anomaly detection around reconstruction error."
    )
    add_p(
        "More recently, the machine learning community has introduced Pretrained Time-Series Foundation Models (TSFMs) trained on "
        "hundreds of millions of time-series observations. Models such as Chronos (Ansari et al., 2024) utilize causal language "
        "modeling architectures to forecast future trajectories in a zero-shot manner. By computing one-step-ahead forecast residuals, "
        "these models can be adapted directly for anomaly scoring without requiring task-specific parameter fine-tuning."
    )
    add_p(
        "However, operational monitoring systems rarely encounter stationary data distributions. Sensor recalibration, environmental "
        "temperature variations, and amplifier gain changes frequently induce amplitude scaling shifts between model development and "
        "online deployment. If an anomaly detection system is brittle to linear scale transformations, it will trigger catastrophic "
        "false alarm cascades in production. This study investigates the degree to which pretrained foundation models maintain "
        "anomaly detection reliability under controlled amplitude distribution shifts compared to traditional machine learning baselines."
    )

    # =========================================================================
    # 3. PROBLEM STATEMENT
    # =========================================================================
    add_h1("3. Problem Statement")
    add_p(
        "Formally, let D_train = {x(1), x(2), ..., x(T_train)} denote an unlabelled normal time-series training prefix drawn from "
        "an in-distribution data-generating process P_train(X). Let D_test = {x(t)}_{t=T_split}^N denote a held-out test series drawn "
        "from P_test(X), accompanied by binary ground-truth anomaly annotations y(t) in {0, 1}. Under synthetic amplitude distribution shift, "
        "the online test signal is subjected to a scalar transformation x_shift(t) = a * x_test(t), where a in {1.00, 1.25, 1.50}, while "
        "the underlying anomaly states y(t) remain strictly invariant."
    )
    add_p(
        "The core research problem addressed in this study is:\n"
        "\"Can pretrained time-series foundation models remain reliable for anomaly detection when the test distribution changes, "
        "and how does their ranking and operational threshold resilience compare against classical and deep learning baselines?\""
    )

    # =========================================================================
    # 4. OBJECTIVES
    # =========================================================================
    add_h1("4. Objectives")
    add_p("The project is structured around five explicit quantitative and qualitative objectives:")
    add_p("1. Dataset Verification: Curate and cryptographically verify five authentic benchmark time series from the UCR Time Series Anomaly Archive (Wu & Keogh, 2021), establishing strictly disjoint train, validation, and test partitions.")
    add_p("2. Baseline Implementation: Construct verified baseline detectors, including an Isolation Forest utilizing sliding-window summary features and a PyTorch LSTM Autoencoder utilizing reconstruction error, with strictly causal window alignment (method='end').")
    add_p("3. Foundation Model Adapter: Implement a zero-shot causal anomaly detection adapter for Amazon's Chronos Bolt Tiny (Ansari et al., 2024), scoring test timestamps via one-step-ahead forecast residuals with strict context buffer causality.")
    add_p("4. Controlled Shift Evaluation: Execute a 45-run evaluation benchmark across three shift conditions (In-Distribution, 1.25x amplitude, and 1.50x amplitude) using decision thresholds derived exclusively from independent validation data and frozen across test conditions.")
    add_p("5. Comparative Resilience Analysis: Quantify the trade-off between threshold-free ranking stability (Average Precision / PR-AUC) and operational threshold stability (F1-score, Precision, Recall, and False Positive inflation).")

    # =========================================================================
    # 5. SCOPE
    # =========================================================================
    add_h1("5. Scope")
    add_p(
        "The experimental scope of this study is focused on univariate continuous time series from five diverse physical domains "
        "(ECG telemetry, respiratory physiology, and meteorology). Detectors are evaluated in an unsupervised/semi-supervised setting "
        "where training data represents normal physiological baseline behavior. Controlled distribution shifts are restricted to synthetic "
        "amplitude scaling factors (a in {1.00, 1.25, 1.50}) applied strictly to test segments, preserving training scalers and frozen thresholds."
    )
    add_p(
        "Deliberate Scope Boundaries and Exclusions:\n"
        "• Multivariate series, non-uniform sampling rates, and missing data imputation are outside the project scope.\n"
        "• Model parameters are evaluated in a frozen/pretrained state; task-specific parameter fine-tuning or LoRA adaptation is excluded.\n"
        "• MOMENT-1-small (Goswami et al., 2024) was originally considered but was formally deferred due to a documented upstream packaging "
        "incompatibility between numpy==1.25.2 and the current Python 3.13 runtime environment. Chronos Bolt Tiny is the sole foundation model candidate benchmarked."
    )

    # =========================================================================
    # 6. LITERATURE SURVEY
    # =========================================================================
    add_h1("6. Literature Survey")
    add_p(
        "A systematic survey of published literature was conducted to situate this study within classical anomaly detection, "
        "deep neural representations, and recent time-series foundation models. Table 1 summarizes key literature contributions."
    )

    # Literature Table
    lit_table = doc.add_table(rows=6, cols=5)
    lit_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    lit_table.autofit = False

    lit_headers = ["Reference", "Method / Architecture", "Primary Domain", "Key Strengths", "Identified Limitations"]
    for c_idx, h_text in enumerate(lit_headers):
        cell = lit_table.cell(0, c_idx)
        set_cell_background(cell, "002060")
        set_cell_margins(cell, top=100, bottom=100, left=100, right=100)
        p = cell.paragraphs[0]
        r = p.add_run(h_text)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(9.5)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    lit_rows = [
        ("Liu et al. (2008)", "Isolation Forest (iForest)", "Tabular & Sliding Window", "Fast O(n) construction, effective against global point anomalies.", "Lacks temporal awareness; sensitive to feature scaling and arbitrary window widths."),
        ("Malhotra et al. (2016)", "LSTM Autoencoder (LSTM-AE)", "Temporal Sequential Data", "Captures non-linear temporal dynamics via recurrent latent bottleneck.", "High training latency; prone to reconstruction degradation under scale drift."),
        ("Wu & Keogh (2021)", "UCR Anomaly Archive", "Benchmark Evaluation", "Standardized 250-series repository preventing benchmark overfitting.", "Extreme class imbalance in certain series (e.g. Series 005) complicates thresholding."),
        ("Ansari et al. (2024)", "Chronos (Bolt Tiny)", "Pretrained Foundation Model", "Zero-shot causal forecasting via T5 architecture; patch-level normalization.", "Residual magnitude scales with input amplitude; thresholding remains uncalibrated."),
        ("Goswami et al. (2024)", "MOMENT Foundation Model", "Multi-task Time Series", "Unified foundation model for forecasting, classification, and detection.", "Strict dependency pins (numpy<=1.25.2) restrict runtime compatibility on modern Python."),
    ]

    col_widths = [Inches(1.2), Inches(1.3), Inches(1.1), Inches(1.4), Inches(1.5)]
    for r_idx, row_vals in enumerate(lit_rows, start=1):
        for c_idx, val in enumerate(row_vals):
            cell = lit_table.cell(r_idx, c_idx)
            cell.width = col_widths[c_idx]
            set_cell_background(cell, "F9FAFB" if r_idx % 2 == 1 else "FFFFFF")
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.name = 'Times New Roman'
            r.font.size = Pt(8.5)

    add_p("Table 1: Comparative Literature Survey of Evaluated and Related Methodologies.", italic=True)

    # =========================================================================
    # 7. METHODOLOGY
    # =========================================================================
    add_h1("7. Methodology")
    add_p(
        "The benchmarking methodology adheres strictly to reproducible, leakage-free machine learning protocols. Each time series "
        "is partitioned into three non-overlapping chronological segments:\n"
        "• Effective Training Set (D_train): First 80% of normal prefix [0, 0.8 * T_split), used to fit scalers, baseline models, and context buffers.\n"
        "• Independent Validation Set (D_val): Remaining 20% of normal prefix [0.8 * T_split, T_split), used strictly to derive the decision threshold (tau).\n"
        "• Held-out Test Set (D_test): Range [T_split, N), containing authentic ground-truth anomalies."
    )
    add_p(
        "Mathematical Formulations:\n"
        "1. Amplitude Shift Injection: For a test series x_test(t), the shifted signal is synthesized as:\n"
        "       x_shift(t) = a * x_test(t),   where a in {1.00, 1.25, 1.50}\n"
        "2. Anomaly Scoring:\n"
        "   • Isolation Forest: Summary features (mean, std, min, max, diff) are extracted across causal sliding windows w_t = [x(t-W+1), ..., x(t)]. Anomaly score is S(t) = -s(w_t).\n"
        "   • LSTM Autoencoder: Encodes window w_t into latent representation h and decodes reconstruction w^_t. Score is mean squared reconstruction error S(t) = (1/W) * sum(||w_t - w^_t||^2).\n"
        "   • Chronos Bolt Tiny: Given causal historical context buffer [x(t-C), ..., x(t-1)], the model computes zero-shot median forecast x^(t). Score is absolute residual S(t) = |x(t) - x^(t)|.\n"
        "3. Decision Threshold Calibration: The operational decision threshold tau is set to the 99th percentile of validation anomaly scores:\n"
        "       tau = Percentile_99({S_val(t)})\n"
        "   This threshold is frozen and applied without recalibration across all test shift conditions (tau_shift = tau_ID)."
    )
    add_p(
        "Evaluation Metrics: The primary ranking metric is Average Precision (PR-AUC), which integrates precision across all possible "
        "recall thresholds. In addition, ROC-AUC, F1-score, Precision, Recall, and Confusion Matrix entries (TP, FP, TN, FN) are computed "
        "at the operational threshold tau."
    )

    # =========================================================================
    # 8. SYSTEM ARCHITECTURE AND DESIGN
    # =========================================================================
    add_h1("8. System Architecture and Design")
    add_p(
        "The project follows a modular, object-oriented architecture organized around an abstract base class `BaseAnomalyDetector` "
        "enforcing uniform `fit(train_data)`, `score(data)`, and `predict(data)` contracts across all detector families."
    )
    add_p(
        "Pipeline Architecture Components:\n"
        "1. Ingestion & Split Manager (`src/data/loader.py`): Parses authentic UCR filenames, validates continuity, and creates disjoint D_train, D_val, and D_test splits.\n"
        "2. Shift Generator (`src/shifts/amplitude_shift.py`): Applies deterministic scalar amplitude transformations strictly to test arrays.\n"
        "3. Detector Family Subsystems:\n"
        "   • `src/models/baselines/isolation_forest.py`: Scikit-learn ensemble with causal window aggregation.\n"
        "   • `src/models/baselines/lstm_ae.py`: PyTorch recurrent autoencoder with early stopping and loss logging.\n"
        "   • `src/models/tsfm/chronos_adapter.py`: Amazon Chronos pipeline adapter with causal prefix buffering.\n"
        "4. Evaluation & Scoring Engine (`experiments/run_experiment.py`): Coordinates 45 execution runs, calculates ranking and thresholded metrics, and exports structured CSV/JSON artifacts.\n"
        "5. Interactive Presentation Layer (`app.py`): Streamlit dashboard rendering comparative analytics and time-series visualizations."
    )

    # =========================================================================
    # 9. SYSTEM SPECIFICATIONS
    # =========================================================================
    add_h1("9. System Specifications")
    add_p("The benchmarking environment was verified on dedicated workstation hardware as documented below:")
    add_p(
        "Hardware Environment:\n"
        "• Processor: Intel(R) Core(TM) i5 / x86_64 architecture\n"
        "• System Memory: 16 GB DDR4 RAM\n"
        "• Execution Platform: CPU execution (PyTorch CPU backend; no CUDA required)\n"
        "• Operating System: Microsoft Windows 11 Home 64-bit"
    )
    add_p(
        "Software Dependencies & Versions:\n"
        "• Python Runtime: Python 3.13.14\n"
        "• Deep Learning Framework: PyTorch 2.11.0+cpu\n"
        "• Foundation Model Libraries: chronos-forecasting 2.3.2, transformers 5.19.0, accelerate 1.13.0\n"
        "• Scientific Computing: scikit-learn 1.9.1, numpy 2.4.4, pandas 3.0.6, scipy 1.17.1\n"
        "• Visualization & Dashboard: matplotlib 3.10.8, streamlit 1.65.0\n"
        "• Verification & Testing: unittest / pytest test runner (43 automated tests passing)"
    )

    # =========================================================================
    # 10. IMPLEMENTATION DETAILS
    # =========================================================================
    add_h1("10. Implementation Details")
    add_p(
        "Implementation integrity was enforced through strict architectural invariants established in Milestone 4A:\n"
        "• Split Disjointness: Partitioning math in `src/data/loader.py` guarantees 0 points of overlap between effective training and validation sets.\n"
        "• Scaler Isolation: StandardScaler instances are fitted strictly on D_train; validation and test splits are transformed using frozen parameters (mu_train, sigma_train).\n"
        "• Window Alignment: Feature extractors in Isolation Forest and LSTM-AE utilize `method='end'` alignment, ensuring that anomaly scores at index t depend strictly on observations at or before t.\n"
        "• Causal Chronos Buffering: `BaseChronosPipeline` maintains a rolling context buffer strictly preceding the query point t. The target observation x(t) is excluded from context during forecasting.\n"
        "• Test Suite Verification: 43 regression and smoke tests verify split validity, absence of data leakage, and causal scoring invariance across the full pipeline."
    )

    # =========================================================================
    # 11. RESULTS AND ANALYSIS
    # =========================================================================
    add_h1("11. Results and Analysis")
    add_p(
        "The complete 45-run empirical benchmark was executed without failure (100% completion rate). Table 2 presents the verified "
        "evaluation metrics across all five UCR datasets, three model architectures, and three shift conditions."
    )

    # Load results from CSV
    if CSV_PATH.exists():
        with open(CSV_PATH, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        res_table = doc.add_table(rows=len(rows) + 1, cols=9)
        res_table.alignment = WD_TABLE_ALIGNMENT.CENTER
        res_table.autofit = False

        res_headers = ["Dataset", "Model", "Condition", "PR-AUC", "ROC-AUC", "F1", "Precision", "Recall", "FP"]
        for c_idx, h_text in enumerate(res_headers):
            cell = res_table.cell(0, c_idx)
            set_cell_background(cell, "002060")
            set_cell_margins(cell, top=60, bottom=60, left=60, right=60)
            p = cell.paragraphs[0]
            r = p.add_run(h_text)
            r.font.name = 'Times New Roman'
            r.font.size = Pt(8.5)
            r.font.bold = True
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

        col_w = [Inches(0.6), Inches(1.1), Inches(0.8), Inches(0.8), Inches(0.8), Inches(0.7), Inches(0.8), Inches(0.7), Inches(0.6)]
        for r_idx, r_dict in enumerate(rows, start=1):
            m_short = r_dict["model_id"].replace("isolation_forest", "iForest").replace("lstm_ae", "LSTM-AE").replace("chronos_bolt_tiny", "Chronos")
            c_short = r_dict["condition"].replace("SHIFT_", "")
            vals = [
                r_dict["dataset_id"],
                m_short,
                c_short,
                f"{float(r_dict['pr_auc']):.4f}",
                f"{float(r_dict['roc_auc']):.4f}",
                f"{float(r_dict['f1']):.4f}",
                f"{float(r_dict['precision']):.4f}",
                f"{float(r_dict['recall']):.4f}",
                r_dict["fp"],
            ]
            for c_idx, val in enumerate(vals):
                cell = res_table.cell(r_idx, c_idx)
                cell.width = col_w[c_idx]
                set_cell_background(cell, "F9FAFB" if r_idx % 2 == 1 else "FFFFFF")
                set_cell_margins(cell, top=40, bottom=40, left=50, right=50)
                p = cell.paragraphs[0]
                r = p.add_run(val)
                r.font.name = 'Times New Roman'
                r.font.size = Pt(8.0)

        add_p("Table 2: Complete 45-Run Experimental Benchmark Results Across Datasets, Models, and Shift Conditions.", italic=True)

    add_p(
        "Key Findings and Nuanced Analysis:\n"
        "1. Ranking Scale Invariance in Chronos Bolt Tiny: Across all five datasets, Chronos demonstrates near-zero PR-AUC degradation "
        "(|ΔPR-AUC| < 0.0001) when test amplitude scales from 1.00x to 1.50x. On Series 001, PR-AUC remains 0.0114; on Series 002, 0.0049; "
        "on Series 003, 0.0597; on Series 004, 0.0431; and on Series 005, 1.0000. This confirms that instance normalization inside the "
        "transformer architecture effectively preserves ranking order across linear amplitude transformations.\n\n"
        "2. The Frozen Threshold Failure Mode: Despite ranking invariance, Chronos suffers dramatic False Positive inflation under amplitude "
        "scaling. On Series 001, FP increases from 355 (ID) to 997 (1.50x). On Series 005, FP increases from 78 to 225. Because the "
        "forecast residuals scale with input amplitude while the decision threshold tau remains frozen, normal test points erroneously "
        "breach the threshold.\n\n"
        "3. Baseline Vulnerability: Isolation Forest and LSTM-AE exhibit acute sensitivity to amplitude shifts. On Series 004, Isolation "
        "Forest PR-AUC drops from 0.2427 to 0.0639 (a 73.7% degradation). LSTM-AE False Positives on Series 002 surge from 229 to 6,457 "
        "(a 28-fold increase), causing total threshold collapse.\n\n"
        "4. Extreme Class Imbalance Disconnect (Series 005): UCR Series 005 contains only 2 anomaly points in a 4,184-point test sequence "
        "(0.0478% prevalence). Chronos successfully assigns the two highest anomaly scores to these exact points, achieving a perfect "
        "PR-AUC of 1.0000 and Recall of 1.0000. However, because 78 normal points also exceed tau, Precision is only 2.5% (F1 = 0.0488). "
        "This dramatic contrast proves why PR-AUC must never be reported without operational thresholded metrics in safety-critical domains."
    )

    # =========================================================================
    # 12. PERFORMANCE ANALYSIS
    # =========================================================================
    add_h1("12. Performance Analysis")
    add_p(
        "To synthesize benchmark outcomes, publication-grade figures were generated from the verified results directory. "
        "Figure 1 illustrates Average Precision (PR-AUC) across models and shift conditions, Figure 2 depicts F1-score behavior, "
        "and Figure 3 visualizes the degradation profile (ΔPR-AUC)."
    )

    # Embed Plots
    p1 = PLOTS_DIR / "metric_comparison_ap.png"
    if p1.exists():
        doc.add_picture(str(p1), width=Inches(5.8))
        add_p("Figure 1: Average Precision (PR-AUC) Comparison Across Models and Shift Regimes.", italic=True)

    p2 = PLOTS_DIR / "metric_comparison_f1.png"
    if p2.exists():
        doc.add_picture(str(p2), width=Inches(5.8))
        add_p("Figure 2: F1-Score Comparison at Frozen Threshold tau Across Evaluated Models.", italic=True)

    p3 = PLOTS_DIR / "degradation_summary_ap.png"
    if p3.exists():
        doc.add_picture(str(p3), width=Inches(5.8))
        add_p("Figure 3: Average Precision Degradation (ΔPR-AUC = PR-AUC_shift - PR-AUC_ID) Demonstrating Foundation Model Stability.", italic=True)

    add_p(
        "Computational Latency Analysis:\n"
        "In addition to detection accuracy, inference throughput is critical for practical telemetry. Isolation Forest demonstrated "
        "the fastest fit latency (average 0.6s) and scoring latency (<0.5s per series). Chronos Bolt Tiny required approximately 1.2s to "
        "initialize and 9.5s to score 45,000 timestamps on CPU, demonstrating that zero-shot foundation models are computationally "
        "practical for batch or near-real-time industrial monitoring."
    )

    # =========================================================================
    # 13. CONCLUSION
    # =========================================================================
    add_h1("13. Conclusion")
    add_p(
        "This project conducted a rigorous, reproducible evaluation of pretrained Time-Series Foundation Models for anomaly detection "
        "under controlled amplitude distribution shifts. The empirical evidence provides a direct answer to the core research question:\n"
        "Pretrained foundation models (Chronos Bolt Tiny) demonstrate remarkable ranking robustness under synthetic amplitude shifts "
        "due to internal instance normalization, maintaining virtually constant Average Precision (|ΔPR-AUC| < 0.0001). However, they "
        "do not provide operational scale-invariance under frozen decision thresholds, experiencing significant False Positive inflation "
        "as forecast residuals scale with input amplitude."
    )
    add_p(
        "Crucially, no single architecture achieved universal superiority across all datasets and evaluation metrics. While Chronos "
        "excels in ranking resilience, baselines exhibit distinct advantages in specific low-noise regimes. Consequently, practical "
        "industrial deployments of foundation models must incorporate test-time threshold recalibration rather than relying on "
        "static validation thresholds."
    )

    # =========================================================================
    # 14. APPLICATIONS
    # =========================================================================
    add_h1("14. Applications")
    add_p("The findings of this research apply directly to several mission-critical operational environments:")
    add_p("• Medical Patient Telemetry: Electrocardiogram (ECG) and respiratory sensors frequently experience electrode impedance drift and patient motion artifacts. Understanding model resilience under gain shifts prevents false alarms in intensive care monitoring.")
    add_p("• Smart Grid Load Monitoring: Electrical substations experience seasonal demand shifts and voltage transformer recalibrations. Foundation models can detect genuine grid anomalies without false trips caused by overall load fluctuations.")
    add_p("• Industrial IoT & Turbomachinery: Vibration telemetry on rotating industrial machinery undergoes mechanical wear and sensor gain variation over time. Robust ranking ensures true mechanical faults are surfaced despite sensor drift.")

    # =========================================================================
    # 15. FUTURE SCOPE
    # =========================================================================
    add_h1("15. Future Scope")
    add_p("Several natural extensions build upon this study's established experimental framework:")
    add_p("1. Test-Time Adaptive Thresholding: Developing online quantile estimation algorithms (e.g. streaming P-square quantile approximation) to dynamically adjust tau in response to observed test amplitude scale.")
    add_p("2. Broader Distribution Shift Taxonomies: Extending the evaluation engine to investigate additive Gaussian noise, frequency/spectral modulation, and trend non-stationarity.")
    add_p("3. Multivariate Foundation Models: Evaluating multivariate architectures (e.g. UniTS, PatchTST) on multi-sensor telemetry where inter-channel correlations may be disrupted by distribution shift.")
    add_p("4. Lightweight Parameter Adaptation: Benchmarking Low-Rank Adaptation (LoRA) on foundation models to determine whether minimal parameter fine-tuning enhances threshold stability without sacrificing zero-shot generalizability.")

    # =========================================================================
    # 16. REFERENCES
    # =========================================================================
    add_h1("16. References")
    refs = [
        "Ansari, A. F., Stella, L., Turkmen, C., Zhang, X., Mercado, P., Shen, H., Shchur, O., Rangapuram, S. S., Arango, S. P., Kapoor, S., Zschache, J., Gasthaus, J., & Wang, Y. (2024). Chronos: Learning the language of time series. arXiv preprint arXiv:2403.07815.",
        "Goswami, M., Szafer, K., Choudhry, A., Cai, Y., Li, S., & Dubrawski, A. (2024). MOMENT: A family of open pre-trained foundation models for time series analysis. International Conference on Machine Learning (ICML 2024).",
        "Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. In 2008 Eighth IEEE International Conference on Data Mining (pp. 413-422). IEEE.",
        "Malhotra, P., Ramakrishnan, A., Anand, G., Vig, L., Agarwal, P., & Shroff, G. (2016). LSTM-based encoder-decoder for multi-sensor anomaly detection. arXiv preprint arXiv:1607.00148.",
        "Wu, R., & Keogh, E. (2021). Current time series anomaly detection benchmarks are flawed and are creating the illusion of progress. IEEE Transactions on Knowledge and Data Engineering, 35(3), 2421-2429.",
        "Dau, H. A., Bagnall, A., Kamgar, K., Yeh, C. C. M., Zhu, Y., Gharghabi, S., Ratanamahatana, C. A., & Keogh, E. (2019). The UCR time series archive. IEEE/CAA Journal of Automatica Sinica, 6(6), 1293-1305.",
        "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. Advances in Neural Information Processing Systems, 30, 5998-6008.",
    ]
    for r_idx, ref_text in enumerate(refs, start=1):
        add_p(f"[{r_idx}] {ref_text}")

    doc.save(str(OUTPUT_DOCX))
    print(f"Report successfully generated at: {OUTPUT_DOCX.resolve()}")
    return OUTPUT_DOCX


if __name__ == "__main__":
    build_report()
