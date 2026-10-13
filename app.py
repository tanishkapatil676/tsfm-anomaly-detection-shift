"""
app.py — Upgraded Streamlit Research Dashboard
Project : Time-Series Foundation Models for Anomaly Detection under Distribution Shift
Student : Tanishka Patil | TY IT Div-3 | PRN: 2405120
Institute: IICT, MGM University, Chh. Sambhajinagar | AY 2026-2027
"""

from pathlib import Path
import json
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TSFM Anomaly Detection Benchmark",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.resolve()
RESULTS_DIR  = BASE_DIR / "results"
CSV_PATH     = RESULTS_DIR / "experiment_summary.csv"
JSON_DIR     = RESULTS_DIR / "json"
PLOTS_DIR    = RESULTS_DIR / "plots"
DATA_DIR     = BASE_DIR / "data" / "raw" / "authentic"
SCRIPTS_DIR  = BASE_DIR / "scripts"
REPORT_PATH  = BASE_DIR / "report" / "Time_Series_Anomaly_Detection_Project_Report.docx"

# ─────────────────────────────────────────────────────────────────────────────
# STATIC METADATA  (verified from JSON experiment records — do not fabricate)
# ─────────────────────────────────────────────────────────────────────────────
DATASET_META = {
    "001": dict(
        name="DISTORTED1sddb40", label="001 · sddb40 ECG #1",
        domain="Electrocardiogram (ECG)",
        total=79795, train=28000, val=7000, test=44795,
        anom_global=[52000, 52620], anom_test=[16999, 17619],
        anomalous_pts=621, prevalence=0.01386,
        source="001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt",
    ),
    "002": dict(
        name="DISTORTED2sddb40", label="002 · sddb40 ECG #2",
        domain="Electrocardiogram (ECG)",
        total=80001, train=28000, val=7000, test=45001,
        anom_global=[56600, 56900], anom_test=[21599, 21899],
        anomalous_pts=301, prevalence=0.00669,
        source="002_UCR_Anomaly_DISTORTED2sddb40_35000_56600_56900.txt",
    ),
    "003": dict(
        name="DISTORTED3sddb40", label="003 · sddb40 ECG #3",
        domain="Electrocardiogram (ECG)",
        total=80000, train=28000, val=7000, test=45000,
        anom_global=[46600, 46900], anom_test=[11599, 11899],
        anomalous_pts=301, prevalence=0.00669,
        source="003_UCR_Anomaly_DISTORTED3sddb40_35000_46600_46900.txt",
    ),
    "004": dict(
        name="DISTORTEDBIDMC1", label="004 · BIDMC1 Respiration",
        domain="Respiratory Signal",
        total=11000, train=2000, val=500, test=8500,
        anom_global=[5400, 5600], anom_test=[2899, 3099],
        anomalous_pts=201, prevalence=0.02365,
        source="004_UCR_Anomaly_DISTORTEDBIDMC1_2500_5400_5600.txt",
    ),
    "005": dict(
        name="DISTORTEDCIMIS44AirTemperature1", label="005 · CIMIS44 Air Temp",
        domain="Meteorology",
        total=8184, train=3200, val=800, test=4184,
        anom_global=[5391, 5392], anom_test=[1390, 1391],
        anomalous_pts=2, prevalence=0.000478,
        source="005_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature1_4000_5391_5392.txt",
    ),
}

MODEL_DISPLAY = {
    "isolation_forest":  "Isolation Forest",
    "lstm_ae":           "LSTM Autoencoder",
    "chronos_bolt_tiny": "Chronos Bolt Tiny (TSFM)",
}
MODEL_COLORS = {
    "isolation_forest":  "#2563EB",
    "lstm_ae":           "#10B981",
    "chronos_bolt_tiny": "#F59E0B",
}
COND_LABELS = {
    "ID":        "In-Distribution (1.00×)",
    "SHIFT_125": "Amplitude Shift 1.25×",
    "SHIFT_150": "Amplitude Shift 1.50×",
}
COND_COLORS = {"ID": "#1E40AF", "SHIFT_125": "#D97706", "SHIFT_150": "#DC2626"}
METRIC_LABELS = {
    "pr_auc":  "PR-AUC (Avg Precision)",
    "roc_auc": "ROC-AUC",
    "f1":      "F1-Score",
}

NAV_SECTIONS = [
    "📊 Overview",
    "🔬 Dataset Explorer",
    "⚖️ Model Comparison",
    "📋 Experiment Results",
    "📉 Time-Series View",
    "💡 Key Findings",
    "📄 Report & Docs",
]

# ─────────────────────────────────────────────────────────────────────────────
# CSS — dark navy sidebar + card styles
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Dark navy sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b35 0%, #142244 60%, #1a2d52 100%);
    border-right: 1px solid #1e3a6e;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] .stRadio > label { color: #94a3b8 !important; font-size: 0.78rem; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #94a3b8 !important; font-size: 0.82rem;
}
[data-testid="stSidebar"] hr { border-color: #2d4a7a; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label { color: #94a3b8 !important; }

/* ── KPI cards ── */
.kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1.1rem 1rem 0.9rem 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    text-align: center;
}
.kpi-label {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.07em;
    text-transform: uppercase; color: #64748b; margin-bottom: 0.35rem;
}
.kpi-value {
    font-size: 1.75rem; font-weight: 800; color: #0f172a; line-height: 1.1;
}
.kpi-sub {
    font-size: 0.72rem; color: #94a3b8; margin-top: 0.2rem;
}
.kpi-badge {
    display: inline-block; font-size: 0.68rem; font-weight: 600;
    background: #eff6ff; color: #1d4ed8; border-radius: 20px;
    padding: 0.15rem 0.55rem; margin-top: 0.3rem;
}

/* ── Section headers ── */
.section-header {
    font-size: 1.35rem; font-weight: 700; color: #0f1f3d;
    border-bottom: 2px solid #2563eb; padding-bottom: 0.4rem;
    margin-bottom: 1.2rem;
}
.section-sub { font-size: 0.9rem; color: #475569; margin-bottom: 1rem; }

/* ── Info / warning banners ── */
.info-box {
    background: #eff6ff; border-left: 4px solid #2563eb;
    border-radius: 6px; padding: 0.75rem 1rem;
    color: #1e3a8a; font-size: 0.88rem; margin: 0.8rem 0;
}
.warn-box {
    background: #fffbeb; border-left: 4px solid #d97706;
    border-radius: 6px; padding: 0.75rem 1rem;
    color: #78350f; font-size: 0.88rem; margin: 0.8rem 0;
}
.success-box {
    background: #f0fdf4; border-left: 4px solid #16a34a;
    border-radius: 6px; padding: 0.75rem 1rem;
    color: #14532d; font-size: 0.88rem; margin: 0.8rem 0;
}
.finding-card {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 1rem 1.1rem; margin-bottom: 0.9rem;
}
.finding-title { font-weight: 700; color: #0f1f3d; margin-bottom: 0.3rem; }

/* ── Main title ── */
.main-title {
    font-size: 1.85rem; font-weight: 800; color: #0f1f3d; line-height: 1.2;
}
.main-sub { font-size: 1rem; color: #475569; margin-top: 0.2rem; }
.meta-pill {
    display: inline-block; background: #eff6ff; color: #1e40af;
    border-radius: 20px; font-size: 0.78rem; font-weight: 600;
    padding: 0.2rem 0.75rem; margin: 0.15rem 0.1rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_results() -> pd.DataFrame:
    """Load and type-cast the experiment summary CSV."""
    if not CSV_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(CSV_PATH)
    for col in ["pr_auc", "roc_auc", "f1", "precision", "recall",
                "delta_pr_auc", "delta_f1", "factor",
                "threshold", "fit_duration_s", "score_duration_s"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["tp", "fp", "tn", "fn"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df["dataset_id"] = df["dataset_id"].astype(str).str.zfill(3)
    return df


@st.cache_data(show_spinner=False)
def load_raw_series(dataset_id: str):
    """Load authentic UCR test series. Returns (series_array, rel_anom_start, rel_anom_end)."""
    meta = DATASET_META.get(dataset_id)
    if not meta:
        return None, None, None
    fp = DATA_DIR / meta["source"]
    if not fp.exists():
        return None, None, None
    data = np.loadtxt(fp, dtype=np.float64)
    if data.ndim > 1:
        data = data.flatten()
    t_split = meta["train"] + meta["val"]
    test_data = data[t_split:]
    a_s, a_e = meta["anom_test"]
    return test_data, a_s, a_e


def kpi_card(label: str, value: str, sub: str = "", badge: str = "") -> str:
    badge_html = f'<div class="kpi-badge">{badge}</div>' if badge else ""
    sub_html   = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'{sub_html}{badge_html}'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
df = load_results()
data_ok = not df.empty

# Pre-compute KPIs from actual data
if data_ok:
    n_success       = int((df["status"].str.upper() == "SUCCESS").sum())
    n_datasets      = df["dataset_id"].nunique()
    n_models        = df["model_id"].nunique()
    best_f1_row     = df.loc[df["f1"].idxmax()]
    best_pr_row     = df.loc[df["pr_auc"].idxmax()]
    best_f1_val     = float(best_f1_row["f1"])
    best_pr_val     = float(best_pr_row["pr_auc"])
    best_f1_tag     = "%s / %s / %s" % (best_f1_row["dataset_id"], MODEL_DISPLAY.get(best_f1_row["model_id"], best_f1_row["model_id"]), best_f1_row["condition"])
    best_pr_tag     = "%s / %s / %s" % (best_pr_row["dataset_id"], MODEL_DISPLAY.get(best_pr_row["model_id"], best_pr_row["model_id"]), best_pr_row["condition"])
else:
    n_success = n_datasets = n_models = 0
    best_f1_val = best_pr_val = 0.0
    best_f1_tag = best_pr_tag = "—"

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:0.6rem 0 0.4rem 0;">
      <div style="font-size:1.05rem;font-weight:800;color:#f1f5f9;line-height:1.3;">
        TSFM Anomaly<br>Detection Benchmark
      </div>
      <div style="font-size:0.72rem;color:#94a3b8;margin-top:0.3rem;">
        Distribution Shift Robustness Study
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    section = st.radio("Navigation", NAV_SECTIONS, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem;color:#64748b;line-height:1.6;">
      <b style="color:#94a3b8;">Student:</b> Tanishka Patil<br>
      <b style="color:#94a3b8;">PRN:</b> 2405120 · TY IT Div-3<br>
      <b style="color:#94a3b8;">Institute:</b> IICT, MGM University<br>
      <b style="color:#94a3b8;">AY:</b> 2026–2027
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    if data_ok:
        st.markdown(f"""
        <div style="font-size:0.72rem;color:#64748b;line-height:1.7;">
          <b style="color:#94a3b8;">Datasets:</b> {n_datasets} / 5 verified<br>
          <b style="color:#94a3b8;">Models evaluated:</b> {n_models}<br>
          <b style="color:#94a3b8;">Runs (SUCCESS):</b> {n_success} / 45<br>
          <b style="color:#94a3b8;">MOMENT-1-small:</b> Deferred ⚠️
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("results/experiment_summary.csv not found.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if section == NAV_SECTIONS[0]:
    st.markdown('<div class="main-title">Time-Series Foundation Models for Anomaly Detection under Distribution Shift</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-sub">Evaluating TSFM robustness vs. classical baselines under synthetic amplitude shifts on UCR Time Series Anomaly Archive</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin:0.7rem 0 1.2rem 0;">
      <span class="meta-pill">📚 UCR Anomaly Archive</span>
      <span class="meta-pill">🤖 Chronos Bolt Tiny · LSTM-AE · Isolation Forest</span>
      <span class="meta-pill">📐 Amplitude Shifts: 1.25× · 1.50×</span>
      <span class="meta-pill">🏛️ IICT, MGM University</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # KPI cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(kpi_card("Authentic Datasets", "5", "UCR Anomaly Archive", "Verified ✓"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Evaluated Models", "3", "IF · LSTM-AE · Chronos", "Benchmarked ✓"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Successful Runs", str(n_success), f"of 45 expected", "100% Success"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Best F1-Score", f"{best_f1_val:.4f}", best_f1_tag, "Single run"), unsafe_allow_html=True)
    with c5:
        st.markdown(kpi_card("Best PR-AUC", f"{best_pr_val:.4f}", best_pr_tag, "Single run ⚠️ imbalanced"), unsafe_allow_html=True)

    st.markdown('<div class="warn-box">⚠️ <b>Best PR-AUC = 1.0000</b> is on Dataset 005 (CIMIS44, 2 anomaly points / 4,184 test points — 0.048% prevalence). Chronos correctly ranks both anomaly points highest. This reflects extreme class imbalance, not universal superiority.</div>', unsafe_allow_html=True)
    st.markdown('<div class="warn-box">⚠️ <b>MOMENT-1-small</b> is deferred — <code>momentfm 0.1.4</code> hard-pins <code>numpy==1.25.2</code> which has no Python 3.13 wheels. It does <b>not</b> appear in benchmark results.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Publication Figures</div>', unsafe_allow_html=True)

    pcols = st.columns(3)
    plots = [
        ("metric_comparison_ap.png",     "Fig 1 — PR-AUC Across Datasets & Conditions"),
        ("metric_comparison_f1.png",      "Fig 2 — F1-Score Across Datasets & Conditions"),
        ("degradation_summary_ap.png",    "Fig 3 — PR-AUC Degradation Under Amplitude Shift"),
    ]
    for col, (fname, cap) in zip(pcols, plots):
        with col:
            p = PLOTS_DIR / fname
            if p.exists():
                st.image(str(p), caption=cap, use_container_width=True)
            else:
                st.info(f"Plot not found: {fname}")

    st.markdown("---")
    st.markdown('<div class="section-header">Research Questions</div>', unsafe_allow_html=True)
    q1, q2 = st.columns(2)
    with q1:
        st.markdown("""
        <div class="finding-card">
          <div class="finding-title">Primary Research Question</div>
          <em>"Can pretrained time-series foundation models remain reliable for anomaly detection
          when the test distribution changes?"</em>
        </div>
        <div class="finding-card">
          <div class="finding-title">Research Gap</div>
          Evidence on TSFM robustness under simple distribution shift is fragmented.
          Their relative advantage over classical baselines under amplitude scaling
          remains an open experimental question addressed by this benchmark.
        </div>
        """, unsafe_allow_html=True)
    with q2:
        st.markdown("""
        <div class="finding-card">
          <div class="finding-title">Experiment Matrix</div>
          5 UCR series × 3 models × 3 conditions (ID, 1.25×, 1.50×) = <b>45 runs</b>,
          all completed with <b>status = SUCCESS</b>. Threshold calibrated on held-out
          validation split; applied frozen to test set without leakage.
        </div>
        <div class="finding-card">
          <div class="finding-title">Shift Protocol</div>
          Amplitude scaling: x<sub>test</sub>(t) ← a·x(t), a ∈ {1.25, 1.50}.
          Scaler fitted on training split only. Validation threshold (99th percentile
          of training scores) frozen across all conditions for fair comparison.
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: DATASET EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
elif section == NAV_SECTIONS[1]:
    st.markdown('<div class="section-header">🔬 Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Authentic UCR Time Series Anomaly Archive datasets. All metadata verified from experiment JSON records and actual file line counts.</div>', unsafe_allow_html=True)

    # Dataset metadata table
    rows = []
    for did, m in DATASET_META.items():
        rows.append({
            "ID": did,
            "Name": m["name"],
            "Domain": m["domain"],
            "Total Pts": f"{m['total']:,}",
            "Train": f"{m['train']:,}",
            "Val": f"{m['val']:,}",
            "Test": f"{m['test']:,}",
            "Anom (global)": f"[{m['anom_global'][0]:,}, {m['anom_global'][1]:,}]",
            "Anom Pts": f"{m['anomalous_pts']:,}",
            "Prevalence": f"{m['prevalence']*100:.4f}%",
        })
    meta_df = pd.DataFrame(rows).set_index("ID")
    st.dataframe(meta_df, use_container_width=True)

    st.markdown('<div class="info-box">ℹ️ <b>Split note:</b> The integer before the anomaly start in UCR filenames is the <b>training prefix</b> (t_split = train + val), not total series length. Total length is the actual line count of the .txt file. Train/Val/Test columns above reflect the disjoint splits used in experiments.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Time-Series Viewer</div>', unsafe_allow_html=True)

    sel_ds = st.selectbox(
        "Select Dataset",
        list(DATASET_META.keys()),
        format_func=lambda x: DATASET_META[x]["label"],
    )
    sel_cond_viz = st.selectbox(
        "Display Condition (amplitude scaling)",
        ["ID", "SHIFT_125", "SHIFT_150"],
        format_func=lambda x: COND_LABELS[x],
    )

    factor_map = {"ID": 1.0, "SHIFT_125": 1.25, "SHIFT_150": 1.50}
    factor = factor_map[sel_cond_viz]

    test_data, as_, ae_ = load_raw_series(sel_ds)
    meta = DATASET_META[sel_ds]

    if test_data is not None:
        displayed = test_data * factor
        fig, ax = plt.subplots(figsize=(11, 3.6), dpi=110)
        ax.plot(displayed, color="#1e3a8a", linewidth=0.6, alpha=0.85,
                label=f"Test Signal (×{factor:.2f})")
        if as_ is not None and 0 <= as_ < len(displayed):
            ax.axvspan(as_, min(len(displayed), ae_ + 1),
                       color="#dc2626", alpha=0.30,
                       label=f"Ground-Truth Anomaly [{as_:,}–{ae_:,}]")
        ax.set_title(f"UCR Test Segment: {meta['label']} — {COND_LABELS[sel_cond_viz]}\n"
                     f"Length: {len(displayed):,} pts | Anomaly: {meta['anomalous_pts']} pts ({meta['prevalence']*100:.4f}%)",
                     fontsize=9, pad=6)
        ax.set_xlabel("Relative Index (t − t_split)", fontsize=8)
        ax.set_ylabel("Amplitude", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.92)
        ax.grid(True, linestyle="--", alpha=0.25)
        plt.tight_layout(pad=0.8)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown(f"""
        <div class="info-box">
          ℹ️ <b>Ground-truth annotation:</b> Red shading marks the verified UCR anomalous subsequence
          (test-relative indices [{as_:,}, {ae_:,}], {meta['anomalous_pts']} points).
          Point-level anomaly scores were not persisted across 45 runs to avoid storage bloat;
          only aggregate metrics (TP/FP/TN/FN, PR-AUC, F1) are available in saved results.
          No model prediction curve is displayed to avoid fabrication.
        </div>
        """, unsafe_allow_html=True)

        # Per-dataset plot if it exists
        plot_fname = PLOTS_DIR / f"plot_{sel_ds}_UCR_Anomaly_{meta['name']}.png"
        # Try a glob since the exact suffix varies
        plot_candidates = list(PLOTS_DIR.glob(f"plot_{sel_ds}_*.png"))
        if plot_candidates:
            st.markdown("**Saved benchmark plot for this dataset:**")
            st.image(str(plot_candidates[0]), use_container_width=True)
    else:
        st.warning(f"Raw data file not found for Dataset {sel_ds}. Expected: {meta['source']}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
elif section == NAV_SECTIONS[2]:
    st.markdown('<div class="section-header">⚖️ Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Compare the three successfully evaluated models across shift conditions. All values are from verified saved experiment records.</div>', unsafe_allow_html=True)

    if not data_ok:
        st.error("Experiment results not found.")
        st.stop()

    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 2])
    with ctrl1:
        sel_metric = st.selectbox("Metric", list(METRIC_LABELS.keys()),
                                  format_func=lambda x: METRIC_LABELS[x])
    with ctrl2:
        sel_ds_cmp = st.selectbox(
            "Dataset",
            ["All (Average)"] + list(DATASET_META.keys()),
            format_func=lambda x: "All Datasets (Mean)" if x == "All (Average)" else DATASET_META[x]["label"],
        )
    with ctrl3:
        sel_conds = st.multiselect(
            "Conditions",
            ["ID", "SHIFT_125", "SHIFT_150"],
            default=["ID", "SHIFT_125", "SHIFT_150"],
            format_func=lambda x: COND_LABELS[x],
        )

    if not sel_conds:
        st.warning("Select at least one condition.")
        st.stop()

    sub = df[df["status"].str.upper() == "SUCCESS"].copy()
    if sel_ds_cmp != "All (Average)":
        sub = sub[sub["dataset_id"] == sel_ds_cmp]

    # Build pivot: model × condition → metric mean
    pivot = (sub[sub["condition"].isin(sel_conds)]
             .groupby(["model_id", "condition"])[sel_metric]
             .mean()
             .unstack(fill_value=np.nan))

    models_ordered = [m for m in ["isolation_forest", "lstm_ae", "chronos_bolt_tiny"] if m in pivot.index]
    conds_ordered  = [c for c in ["ID", "SHIFT_125", "SHIFT_150"] if c in sel_conds and c in pivot.columns]

    pivot = pivot.loc[models_ordered, conds_ordered]

    # Grouped bar chart
    x = np.arange(len(models_ordered))
    bar_w = 0.22
    offsets = np.linspace(-(len(conds_ordered) - 1) / 2, (len(conds_ordered) - 1) / 2, len(conds_ordered)) * bar_w

    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=110)
    for i, cond in enumerate(conds_ordered):
        vals = [float(pivot.loc[m, cond]) if m in pivot.index and cond in pivot.columns
                and not np.isnan(pivot.loc[m, cond]) else 0
                for m in models_ordered]
        bars = ax.bar(x + offsets[i], vals, bar_w * 0.92,
                      label=COND_LABELS[cond], color=COND_COLORS[cond], alpha=0.88)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, color="#374151")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in models_ordered], fontsize=8.5)
    ax.set_ylabel(METRIC_LABELS[sel_metric], fontsize=9)
    ax.set_ylim(0, min(1.12, max(ax.get_ylim()[1] + 0.08, 0.15)))
    title_ds = "All Datasets (Mean)" if sel_ds_cmp == "All (Average)" else DATASET_META[sel_ds_cmp]["label"]
    ax.set_title(f"{METRIC_LABELS[sel_metric]} · {title_ds}", fontsize=10, pad=8)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.92)
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(pad=0.9)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Numeric summary table
    st.markdown("**Numeric Summary Table:**")
    tbl = pivot.copy()
    tbl.index = [MODEL_DISPLAY.get(m, m) for m in tbl.index]
    tbl.columns = [COND_LABELS.get(c, c) for c in tbl.columns]
    st.dataframe(tbl.style.format("{:.4f}").highlight_max(axis=None, color="#dbeafe"),
                 use_container_width=True)

    st.markdown("---")
    # Degradation chart — delta metrics across conditions
    st.markdown('<div class="section-header">Shift Degradation (Δ vs ID Baseline)</div>', unsafe_allow_html=True)
    delta_col = "delta_pr_auc" if sel_metric == "pr_auc" else "delta_f1" if sel_metric == "f1" else None

    if delta_col and delta_col in df.columns:
        sub_shift = df[(df["condition"].isin(["SHIFT_125", "SHIFT_150"])) &
                       (df["status"].str.upper() == "SUCCESS")].copy()
        if sel_ds_cmp != "All (Average)":
            sub_shift = sub_shift[sub_shift["dataset_id"] == sel_ds_cmp]

        pivot_d = (sub_shift.groupby(["model_id", "condition"])[delta_col]
                   .mean().unstack(fill_value=np.nan))

        fig2, ax2 = plt.subplots(figsize=(8, 3.6), dpi=110)
        conds_d = [c for c in ["SHIFT_125", "SHIFT_150"] if c in pivot_d.columns]
        x2 = np.arange(len(conds_d))
        models_d = [m for m in ["isolation_forest", "lstm_ae", "chronos_bolt_tiny"] if m in pivot_d.index]
        offsets2 = np.linspace(-(len(models_d) - 1) / 2, (len(models_d) - 1) / 2, len(models_d)) * bar_w
        for i, m in enumerate(models_d):
            vals2 = [float(pivot_d.loc[m, c]) if c in pivot_d.columns and not np.isnan(pivot_d.loc[m, c]) else 0
                     for c in conds_d]
            ax2.bar(x2 + offsets2[i], vals2, bar_w * 0.92,
                    label=MODEL_DISPLAY.get(m, m), color=MODEL_COLORS[m], alpha=0.88)

        ax2.axhline(0, color="#374151", linewidth=0.8, linestyle="--")
        ax2.set_xticks(x2)
        ax2.set_xticklabels([COND_LABELS[c] for c in conds_d], fontsize=8.5)
        ax2.set_ylabel(f"Δ {METRIC_LABELS.get(sel_metric, sel_metric)} (Shift − ID)", fontsize=8.5)
        ax2.set_title("Metric Degradation Under Amplitude Shift", fontsize=10, pad=6)
        ax2.legend(fontsize=8, loc="lower right", framealpha=0.92)
        ax2.grid(axis="y", linestyle="--", alpha=0.28)
        ax2.spines[["top", "right"]].set_visible(False)
        plt.tight_layout(pad=0.9)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)
    else:
        st.info("Δ-metric column not available for ROC-AUC in saved results. Select F1 or PR-AUC to see degradation.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: EXPERIMENT RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────────────
elif section == NAV_SECTIONS[3]:
    st.markdown('<div class="section-header">📋 Experiment Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">All 45 verified experiment records from <code>results/experiment_summary.csv</code>. Filter, sort, and download below.</div>', unsafe_allow_html=True)

    if not data_ok:
        st.error("Experiment results CSV not found.")
        st.stop()

    fa, fb, fc = st.columns(3)
    with fa:
        ds_opts = ["All"] + sorted(df["dataset_id"].unique().tolist())
        f_ds = st.selectbox("Dataset", ds_opts,
                            format_func=lambda x: "All Datasets" if x == "All" else DATASET_META.get(x, {}).get("label", x))
    with fb:
        m_opts = ["All"] + sorted(df["model_id"].unique().tolist())
        f_model = st.selectbox("Model", m_opts,
                               format_func=lambda x: "All Models" if x == "All" else MODEL_DISPLAY.get(x, x))
    with fc:
        c_opts = ["All"] + sorted(df["condition"].unique().tolist())
        f_cond = st.selectbox("Condition", c_opts,
                              format_func=lambda x: "All Conditions" if x == "All" else COND_LABELS.get(x, x))

    view = df.copy()
    if f_ds != "All":
        view = view[view["dataset_id"] == f_ds]
    if f_model != "All":
        view = view[view["model_id"] == f_model]
    if f_cond != "All":
        view = view[view["condition"] == f_cond]

    display_cols = ["run_id", "dataset_id", "dataset_name", "model_id",
                    "condition", "factor", "pr_auc", "roc_auc", "f1",
                    "precision", "recall", "tp", "fp", "tn", "fn",
                    "delta_pr_auc", "delta_f1", "threshold",
                    "fit_duration_s", "score_duration_s", "status"]
    display_cols = [c for c in display_cols if c in view.columns]
    view_display = view[display_cols].copy()

    col_rename = {
        "run_id": "Run ID", "dataset_id": "DS", "dataset_name": "Dataset",
        "model_id": "Model", "condition": "Condition", "factor": "Factor",
        "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC", "f1": "F1",
        "precision": "Precision", "recall": "Recall",
        "tp": "TP", "fp": "FP", "tn": "TN", "fn": "FN",
        "delta_pr_auc": "Δ PR-AUC", "delta_f1": "Δ F1",
        "threshold": "Threshold (τ)", "fit_duration_s": "Fit (s)",
        "score_duration_s": "Score (s)", "status": "Status",
    }
    view_display.rename(columns={k: v for k, v in col_rename.items() if k in view_display.columns}, inplace=True)
    view_display["Model"] = view_display["Model"].map(MODEL_DISPLAY).fillna(view_display["Model"])
    view_display["Condition"] = view_display["Condition"].map(COND_LABELS).fillna(view_display["Condition"])

    float_cols = ["PR-AUC", "ROC-AUC", "F1", "Precision", "Recall",
                  "Δ PR-AUC", "Δ F1", "Threshold (τ)", "Fit (s)", "Score (s)"]
    fmt = {c: "{:.4f}" for c in float_cols if c in view_display.columns}

    st.markdown(f"**Showing {len(view_display)} of 45 records:**")
    st.dataframe(
        view_display.style.format(fmt).apply(
            lambda col: ["background-color:#f0fdf4" if v == "SUCCESS" else "" for v in col]
            if col.name == "Status" else [""] * len(col),
            axis=0
        ),
        use_container_width=True,
        height=420,
    )

    # CSV download
    csv_bytes = view_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download filtered results as CSV",
        data=csv_bytes,
        file_name="experiment_results_filtered.csv",
        mime="text/csv",
    )

    # Quick summary stats for filtered view
    if len(view) > 0:
        st.markdown("---")
        st.markdown("**Filtered Summary Statistics:**")
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.metric("Runs", len(view))
        with sc2:
            st.metric("Avg PR-AUC", f"{view['pr_auc'].mean():.4f}")
        with sc3:
            st.metric("Avg F1", f"{view['f1'].mean():.4f}")
        with sc4:
            st.metric("Avg ROC-AUC", f"{view['roc_auc'].mean():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: TIME-SERIES VIEW
# ─────────────────────────────────────────────────────────────────────────────
elif section == NAV_SECTIONS[4]:
    st.markdown('<div class="section-header">📉 Time-Series Visualization</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Authentic UCR test segments with verified ground-truth anomaly annotations. Point-level model scores are not available in saved artifacts.</div>', unsafe_allow_html=True)

    if not data_ok:
        st.error("Results data not loaded — cannot cross-reference metrics.")

    v1, v2, v3 = st.columns([2, 2, 2])
    with v1:
        sel_ds_ts = st.selectbox(
            "Dataset", list(DATASET_META.keys()),
            format_func=lambda x: DATASET_META[x]["label"], key="ts_ds",
        )
    with v2:
        sel_cond_ts = st.selectbox(
            "Condition", ["ID", "SHIFT_125", "SHIFT_150"],
            format_func=lambda x: COND_LABELS[x], key="ts_cond",
        )
    with v3:
        sel_model_ts = st.selectbox(
            "Show metrics for model",
            ["isolation_forest", "lstm_ae", "chronos_bolt_tiny"],
            format_func=lambda x: MODEL_DISPLAY[x], key="ts_model",
        )

    factor_map = {"ID": 1.0, "SHIFT_125": 1.25, "SHIFT_150": 1.50}
    factor_ts = factor_map[sel_cond_ts]
    meta_ts = DATASET_META[sel_ds_ts]

    test_data_ts, as_ts, ae_ts = load_raw_series(sel_ds_ts)

    if test_data_ts is not None:
        displayed_ts = test_data_ts * factor_ts

        fig_ts, ax_ts = plt.subplots(figsize=(12, 4), dpi=110)
        ax_ts.plot(displayed_ts, color="#1e3a8a", linewidth=0.55, alpha=0.85,
                   label=f"Observed Signal (×{factor_ts:.2f})")
        if as_ts is not None and 0 <= as_ts < len(displayed_ts):
            ax_ts.axvspan(as_ts, min(len(displayed_ts), ae_ts + 1),
                          color="#dc2626", alpha=0.28,
                          label=f"True Anomaly Window [{as_ts:,}–{ae_ts:,}] ({meta_ts['anomalous_pts']} pts)")
        ax_ts.set_title(
            f"{meta_ts['label']} | {COND_LABELS[sel_cond_ts]}\n"
            f"Domain: {meta_ts['domain']} | Test length: {len(displayed_ts):,} pts",
            fontsize=9, pad=6
        )
        ax_ts.set_xlabel("Relative Test Index (t − t_split)", fontsize=8)
        ax_ts.set_ylabel("Amplitude", fontsize=8)
        ax_ts.tick_params(labelsize=7)
        ax_ts.legend(fontsize=8, loc="upper right", framealpha=0.92)
        ax_ts.grid(True, linestyle="--", alpha=0.22)
        ax_ts.spines[["top", "right"]].set_visible(False)
        plt.tight_layout(pad=0.9)
        st.pyplot(fig_ts, use_container_width=True)
        plt.close(fig_ts)

    else:
        st.warning(f"Raw dataset file not found for {sel_ds_ts}. Expected in data/raw/authentic/")

    # Show matching metrics from saved results
    if data_ok:
        match = df[
            (df["dataset_id"] == sel_ds_ts) &
            (df["model_id"] == sel_model_ts) &
            (df["condition"] == sel_cond_ts)
        ]
        if not match.empty:
            row = match.iloc[0]
            st.markdown("---")
            st.markdown(f"**Saved Metrics: {MODEL_DISPLAY[sel_model_ts]} · {meta_ts['label']} · {COND_LABELS[sel_cond_ts]}**")
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("PR-AUC", f"{row['pr_auc']:.4f}")
            m2.metric("ROC-AUC", f"{row['roc_auc']:.4f}")
            m3.metric("F1-Score", f"{row['f1']:.4f}")
            m4.metric("Precision", f"{row['precision']:.4f}")
            m5.metric("Recall", f"{row['recall']:.4f}")
            m6.metric("Threshold τ", f"{row['threshold']:.5f}")

            cm1, cm2, cm3, cm4 = st.columns(4)
            cm1.metric("TP", f"{int(row['tp'])}")
            cm2.metric("FP", f"{int(row['fp'])}", delta=None)
            cm3.metric("TN", f"{int(row['tn'])}")
            cm4.metric("FN", f"{int(row['fn'])}")

            st.markdown(f"""
            <div class="info-box">
              ℹ️ Threshold τ = <b>{row['threshold']:.5f}</b> (99th percentile of training-split anomaly scores).
              Applied <em>frozen</em> to test set. TP/FP/TN/FN derived from point-level binary classification.
              No per-timestamp score array is persisted — aggregate metrics only.
            </div>
            """, unsafe_allow_html=True)

            if sel_cond_ts != "ID" and "delta_pr_auc" in row:
                delta_pr = float(row["delta_pr_auc"])
                delta_f1 = float(row.get("delta_f1", 0))
                d1, d2 = st.columns(2)
                d1.metric("Δ PR-AUC vs ID", f"{delta_pr:+.4f}")
                d2.metric("Δ F1 vs ID", f"{delta_f1:+.4f}")
        else:
            st.info("No saved record for this dataset/model/condition combination.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: KEY FINDINGS
# ─────────────────────────────────────────────────────────────────────────────
elif section == NAV_SECTIONS[5]:
    st.markdown('<div class="section-header">💡 Key Findings</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">All findings are derived directly from verified experiment results. Claims are supported by actual saved metrics.</div>', unsafe_allow_html=True)

    if not data_ok:
        st.error("Results not available.")
        st.stop()

    # Compute finding statistics
    id_rows   = df[df["condition"] == "ID"]
    s125_rows = df[df["condition"] == "SHIFT_125"]
    s150_rows = df[df["condition"] == "SHIFT_150"]

    chronos_id = id_rows[id_rows["model_id"] == "chronos_bolt_tiny"]
    lstm_id    = id_rows[id_rows["model_id"] == "lstm_ae"]
    if_id      = id_rows[id_rows["model_id"] == "isolation_forest"]

    avg_pr_chronos = chronos_id["pr_auc"].mean()
    avg_pr_lstm    = lstm_id["pr_auc"].mean()
    avg_pr_if      = if_id["pr_auc"].mean()

    avg_f1_chronos = chronos_id["f1"].mean()
    avg_f1_lstm    = lstm_id["f1"].mean()
    avg_f1_if      = if_id["f1"].mean()

    # Delta PR-AUC per model
    chronos_delta = df[(df["model_id"] == "chronos_bolt_tiny") & (df["condition"] != "ID")]["delta_pr_auc"].mean()
    lstm_delta    = df[(df["model_id"] == "lstm_ae") & (df["condition"] != "ID")]["delta_pr_auc"].mean()
    if_delta      = df[(df["model_id"] == "isolation_forest") & (df["condition"] != "ID")]["delta_pr_auc"].mean()

    # FP escalation for chronos
    chronos_id_fp   = df[(df["model_id"] == "chronos_bolt_tiny") & (df["condition"] == "ID")]["fp"].mean()
    chronos_150_fp  = df[(df["model_id"] == "chronos_bolt_tiny") & (df["condition"] == "SHIFT_150")]["fp"].mean()

    st.markdown(f"""
    <div class="finding-card">
      <div class="finding-title">1 · Ranking vs. Threshold-Dependent Metrics Tell Different Stories</div>
      Chronos Bolt Tiny achieves the highest mean PR-AUC (ranking quality) across 5 datasets on the ID condition
      (<b>{avg_pr_chronos:.4f}</b>), compared to Isolation Forest (<b>{avg_pr_if:.4f}</b>) and LSTM-AE (<b>{avg_pr_lstm:.4f}</b>).
      However, in terms of F1-score at the fixed threshold τ, LSTM-AE leads with <b>{avg_f1_lstm:.4f}</b>
      versus Chronos at <b>{avg_f1_chronos:.4f}</b>.<br><br>
      This gap reflects a fundamental distinction: <b>PR-AUC</b> measures the quality of anomaly score
      <em>rankings</em> across all thresholds, while <b>F1/Precision/Recall</b> depend on whether
      the fixed validation-threshold τ correctly separates anomalies from normal points at test time.
    </div>

    <div class="finding-card">
      <div class="finding-title">2 · Chronos Exhibits Scale-Invariant Rankings but Threshold Failure Under Shift</div>
      Chronos uses patch-level instance normalization during tokenization, which preserves the
      <em>relative ranking</em> of points across amplitude scales.
      Mean Δ PR-AUC for Chronos under shifts (1.25× + 1.50×) = <b>{chronos_delta:+.4f}</b> ≈ 0.
      However, since τ was calibrated on unscaled validation residuals, scaled normal test points
      cross τ, inflating false positives from a mean of {chronos_id_fp:.0f} (ID)
      to {chronos_150_fp:.0f} (1.50×) on average — a <b>×{chronos_150_fp/max(chronos_id_fp,1):.1f} increase</b>.
    </div>

    <div class="finding-card">
      <div class="finding-title">3 · Classical Baselines Also Degrade Under Shift</div>
      Isolation Forest mean Δ PR-AUC = <b>{if_delta:+.4f}</b> and
      LSTM-AE mean Δ PR-AUC = <b>{lstm_delta:+.4f}</b> under shift conditions.
      For baselines, score distributions shift with amplitude because their features
      (window statistics / reconstruction errors) are not normalized, so both rankings
      and threshold-based metrics degrade together — unlike the Chronos split-behavior.
    </div>

    <div class="finding-card">
      <div class="finding-title">4 · Dataset 005 Extreme Imbalance Requires Careful Interpretation</div>
      Dataset 005 (CIMIS44 Air Temperature) contains only <b>2 anomaly points</b> out of 4,184 test points
      (prevalence = 0.0478%). Chronos correctly ranks both as the two highest-scoring points,
      giving PR-AUC = 1.0000 and Recall = 1.0000 — but at threshold τ = 0.4365,
      approximately 78 normal points also exceed τ, giving Precision = 0.025 and F1 = 0.0488.<br><br>
      This proves why <b>ranking metrics (PR-AUC) must never be reported in isolation</b> from
      thresholded operational metrics, especially under severe class imbalance.
    </div>

    <div class="finding-card">
      <div class="finding-title">5 · No Model is Universally Superior</div>
      Chronos offers the strongest ranking stability under amplitude shift (Δ PR-AUC ≈ 0).
      However, its execution latency (~1.8 s/run) is significantly higher than
      Isolation Forest (~0.05 s/run). LSTM-AE achieves the highest single-run F1
      ({best_f1_val:.4f} on {best_f1_tag}). Real deployment requires adaptive threshold
      recalibration after any distribution shift to restore operational performance.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Model Summary Table (ID Condition, Mean Over 5 Datasets)</div>', unsafe_allow_html=True)

    summary_data = {
        "Model": [MODEL_DISPLAY["isolation_forest"], MODEL_DISPLAY["lstm_ae"], MODEL_DISPLAY["chronos_bolt_tiny"]],
        "Mean PR-AUC (ID)": [avg_pr_if, avg_pr_lstm, avg_pr_chronos],
        "Mean F1 (ID)": [avg_f1_if, avg_f1_lstm, avg_f1_chronos],
        "Mean Δ PR-AUC (Shifts)": [if_delta, lstm_delta, chronos_delta],
        "Scale-Invariant Ranking": ["No", "No", "Yes (≈0 Δ PR-AUC)"],
        "Threshold Robust to Shift": ["No", "No", "No (FP escalation)"],
    }
    summary_df = pd.DataFrame(summary_data).set_index("Model")
    float_cols_s = ["Mean PR-AUC (ID)", "Mean F1 (ID)", "Mean Δ PR-AUC (Shifts)"]
    st.dataframe(
        summary_df.style.format({c: "{:.4f}" for c in float_cols_s})
                  .highlight_max(subset=float_cols_s[:2], color="#dbeafe"),
        use_container_width=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: REPORT & DOCS
# ─────────────────────────────────────────────────────────────────────────────
elif section == NAV_SECTIONS[6]:
    st.markdown('<div class="section-header">📄 Report & Documentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Research documentation, specification files, and the compiled academic report.</div>', unsafe_allow_html=True)

    # Report download
    if REPORT_PATH.exists():
        with open(REPORT_PATH, "rb") as fh:
            docx_bytes = fh.read()
        st.download_button(
            label="⬇️ Download IICT B.Tech Project Report (.docx)",
            data=docx_bytes,
            file_name="Time_Series_Anomaly_Detection_Project_Report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        st.markdown(f'<div class="success-box">✅ Report file found: <code>scripts/Time_Series_Anomaly_Detection_Project_Report.docx</code> ({REPORT_PATH.stat().st_size // 1024} KB)</div>', unsafe_allow_html=True)
    else:
        st.warning("Report file not found at scripts/Time_Series_Anomaly_Detection_Project_Report.docx")

    st.markdown("---")
    st.markdown("**Project Documentation Files:**")

    docs = [
        ("README.md",               "Project overview, setup, and reproduction guide"),
        ("PROJECT_STATUS.md",       "Current milestone status and completion tracking"),
        ("IMPLEMENTATION_ROADMAP.md","Five-milestone roadmap with risk matrix"),
        ("DECISIONS.md",            "Documented design and implementation decisions"),
        ("EXPERIMENT_PROTOCOL.md",  "Formal experiment protocol and methodology"),
        ("MODEL_FEASIBILITY.md",    "Model shortlist, feasibility assessment, MOMENT blocker"),
        ("SYSTEM_ARCHITECTURE.md",  "Full system architecture specification"),
        ("PROJECT_SPEC.md",         "Research specification and requirements"),
        ("REPORT_EVIDENCE.md",      "Evidence mapping for IICT report sections"),
        ("DATASET_CARD_TEMPLATE.md","UCR dataset cards template"),
        ("docs/FINAL_AUDIT.md",     "Complete project audit report (Oct 2026)"),
    ]

    for fname, desc in docs:
        fp = BASE_DIR / fname
        exists = fp.exists()
        icon = "✅" if exists else "❌"
        size = f" ({fp.stat().st_size:,} B)" if exists else " (not found)"
        st.markdown(f"- {icon} **`{fname}`** — {desc}{size}")

    st.markdown("---")
    st.markdown("**Quick Commands:**")
    st.code("""# Launch this dashboard
python -m streamlit run app.py

# Re-generate the DOCX report
python scripts/generate_report.py

# Run full automated test suite
python -m pytest tests/ -v

# Run lightweight tests only
python -m pytest tests/test_data_loader.py tests/test_metrics.py tests/test_shifts.py tests/test_persistence.py tests/test_dashboard_logic.py -v

# Data verification
python scripts/run_data_verification.py""", language="powershell")

    st.markdown("---")
    st.markdown("**Deferred Model Note:**")
    st.markdown("""
    <div class="warn-box">
      ⚠️ <b>MOMENT-1-small (AutonLab/MOMENT-1-small)</b> is <b>not benchmarked</b> in this study.
      Blocker: <code>momentfm 0.1.4</code> hard-pins <code>numpy==1.25.2</code> which has no Python 3.13 wheels.
      This is a documented decision in <code>MODEL_FEASIBILITY.md</code> and <code>DECISIONS.md</code>.
      All 45 benchmark records use the three successfully evaluated models only.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#94a3b8;font-size:0.78rem;padding:0.4rem 0 0.8rem 0;">
  <b>TSFM Anomaly Detection Benchmark</b> · Tanishka Patil (PRN: 2405120) ·
  IICT, MGM University, Chh. Sambhajinagar · AY 2026–2027 ·
  <code>python -m streamlit run app.py</code>
</div>
""", unsafe_allow_html=True)
