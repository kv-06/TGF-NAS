"""
HAR Pipeline — Streamlit Frontend
Run:  streamlit run app.py
API:  uvicorn app:app --host 0.0.0.0 --port 8000
"""

import io
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import streamlit as st

# 
API_URL       = "https://kv06-tgf-nas.hf.space"
# API_URL       = "https://kv-06-tgf-nas.hf.space"
# API_URL = "http://localhost:8000"
POLL_INTERVAL = 2

st.set_page_config(
    page_title="HAR Pipeline — TGF-NAS + LAHUP + Quantization",
    page_icon="",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background:linear-gradient(135deg,#1e3a5f,#16213e);
    border-radius:12px; padding:18px 22px; color:#fff;
    text-align:center; border:1px solid #2a5298; margin-bottom:8px;
}
.metric-label { font-size:12px; color:#a0b4cc; margin-bottom:4px; }
.metric-value { font-size:28px; font-weight:700; color:#4fc3f7; }
.metric-sub   { font-size:11px; color:#7fb3cc; margin-top:2px; }
.section-header {
    background:linear-gradient(90deg,#1e3a5f,transparent);
    border-left:4px solid #4fc3f7; padding:7px 14px;
    border-radius:0 6px 6px 0; margin:14px 0 8px 0;
    font-weight:600; color:#e6f3ff;
}
.log-box {
    background:#0d1117; border:1px solid #30363d; border-radius:8px;
    padding:14px; font-family:'Courier New',monospace; font-size:11.5px;
    color:#c9d1d9; height:420px; overflow-y:auto;
    white-space:pre-wrap; word-break:break-all;
}
.status-badge {
    display:inline-block; padding:2px 11px; border-radius:20px;
    font-size:12px; font-weight:600; margin-left:8px;
}
.status-running { background:#1a3a2a; color:#4caf50; border:1px solid #4caf50; }
.status-done    { background:#1a2a4a; color:#4fc3f7; border:1px solid #4fc3f7; }
.status-error   { background:#3a1a1a; color:#f44336; border:1px solid #f44336; }
.status-queued  { background:#2a2a1a; color:#ffc107; border:1px solid #ffc107; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "job_id": None, "status": None, "all_logs": [], "last_line": 0,
    "result": None, "running": False, "parsed": {}, "model_paths": [],
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def mcard(label, value, sub=""):
    return (f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-sub">{sub}</div></div>')


def colorize(line):
    esc = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Cyan bold — key results
    if any(k in line for k in ["Best Architecture Index", "Final Test Accuracy", "TGF-NAS Score:"]):
        return f'<span style="color:#4fc3f7;font-weight:600">{esc}</span>'
    # Green — saves and fine-tune accuracy
    if any(k in line for k in ["Model saved", "[run_pipeline] Saved", "Accuracy after fine-tuning",
                                "Arch Table"]):
        return f'<span style="color:#4caf50">{esc}</span>'
    # Red — errors
    if any(k in line for k in ["ERROR", "Error", "failed", "Traceback"]):
        return f'<span style="color:#f44336">{esc}</span>'
    # Yellow — warnings and accuracy drops
    if any(k in line for k in ["WARNING", "[WARNING]", "Pruned (before FT)",
                                "Accuracy after pruning", "drop:"]):
        return f'<span style="color:#ffc107">{esc}</span>'
    # Purple — section headers
    if any(k in line for k in ["===", "---", "APPLY", "PRUNING SUMMARY", "QUANTIZATION SUMMARY",
                                "QUANT", "TGF-NAS", "run_pipeline", "LAHUP"]):
        return f'<span style="color:#9575cd">{esc}</span>'
    # Teal — epoch lines
    if re.match(r"\s*Epoch\s+\d+", line):
        return f'<span style="color:#80cbc4">{esc}</span>'
    # Orange — probe/scoring stats
    if any(k in line for k in ["Proxy Accuracy", "Architecture Magnitude", "Architecture Consistency",
                                "Training Time", "Trainable Parameters", "Evaluating Model"]):
        return f'<span style="color:#ffb74d">{esc}</span>'
    # Grey-blue — size/sparsity summary lines
    if any(k in line for k in ["Baseline accuracy", "Achieved sparsity", "Model Size",
                                "Auto-detected"]):
        return f'<span style="color:#b0bec5">{esc}</span>'
    return esc


def parse_metrics(logs):
    text = "\n".join(logs)
    m = {}

    def find(pat, cast=float):
        r = re.search(pat, text)
        return cast(r.group(1)) if r else None

    m["final_accuracy"]     = find(r"Final Test Accuracy:\s*([\d.]+)%")
    m["best_arch_idx"]      = find(r"Best Architecture Index:\s*(\d+)", int)
    best_block = re.search(
        r"Best Architecture Index:.*?Proxy Accuracy:\s*([\d.]+)",
        text,
        re.DOTALL
    )
    m["proxy_accuracy"] = float(best_block.group(1)) if best_block else None    
    m["tgfnas_score"]       = find(r"TGF-NAS Score:\s*([-\d.]+)")
    m["alpha"]              = find(r"α=([\d.nan]+)")
    m["beta"]               = find(r"β=([\d.nan]+)")
    m["baseline_accuracy"]  = find(r"Baseline accuracy\s*:\s*([\d.]+)%")
    m["pruned_accuracy"]    = find(r"Accuracy after pruning\s*:\s*([\d.]+)%")
    m["finetuned_accuracy"] = find(r"Accuracy after fine-tuning\s*:\s*([\d.]+)%")
    m["achieved_sparsity"]  = find(r"Achieved sparsity\s*:\s*([\d.]+)%")
    m["size_before"]        = find(r"Model Size before pruning\s*:\s*([\d.]+)")
    m["size_after"]         = find(r"Model Size [Aa]fter pruning\s*:\s*([\d.]+)")

    quant = {}
    for qt in ["FP16", "INT8"]:
        r = re.search(
            rf"{qt}\s+([\d.]+)\s+[\d.]+x\s+([\d.]+)\s+[+-]?[\d.]+\s+([\d.]+)", text)
        if r:
            quant[qt] = {
                "accuracy": float(r.group(2)),
                "size":     float(r.group(1)),
                "speed":    float(r.group(3)),
            }
    if quant:
        m["quantization"] = quant

    epochs = re.findall(
        r"Epoch\s+(\d+)/\d+\s*\|\s*Train Acc:\s*([\d.]+)%\s*\|\s*Test Acc:\s*([\d.]+)%", text)
    if epochs:
        m["epoch_history"] = [{"epoch": int(e[0]), "train_acc": float(e[1]),
                                "test_acc": float(e[2])} for e in epochs]
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap  (matches the user's exact plotting code — same normalisation logic)
# ─────────────────────────────────────────────────────────────────────────────

def build_heatmap(stage_metrics: dict) -> bytes:
    """
    stage_metrics: OrderedDict  {stage_label -> {accuracy, size, speed, sparsity}}
    Produces a PNG matching the reference heatmap code exactly.
    """
    models  = list(stage_metrics.keys())
    metrics = [
        "Accuracy (%) ↑",
        "Model Size (MB) ↓",
        "Inference Speed (ms) ↓",
        "Sparsity (%) ↑",
    ]
    fields  = ["accuracy", "size", "speed", "sparsity"]

    data = np.array([
        [stage_metrics[m].get(f, 0.0) for f in fields]
        for m in models
    ], dtype=float)   # shape: (n_models, n_metrics)

    # ── Normalise — exact logic from user's snippet ──────────────────────────
    norm_data = np.zeros_like(data)
    for col in range(data.shape[1]):
        column  = data[:, col]
        mn, mx  = column.min(), column.max()
        if mx - mn == 0:
            norm = np.zeros_like(column)
        else:
            norm = (column - mn) / (mx - mn)
        if col in (1, 2):     # Model Size ↓ and Inference Speed ↓  →  invert
            norm = 1 - norm
        norm_data[:, col] = norm

    norm_data   = norm_data.T       # (n_metrics, n_models)
    data_display = data.T

    # ── Find where quantisation stages begin ─────────────────────────────────
    quant_labels = {"FP16", "INT8", "quant_fp16", "quant_int8"}
    quant_start  = next(
        (i for i, n in enumerate(models) if n in quant_labels), None)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig_w = max(10, len(models) * 2)
    fig, ax = plt.subplots(figsize=(fig_w, 5), dpi=300)

    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    im   = ax.imshow(norm_data, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, fontsize=10)
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels(metrics, fontsize=10)
    ax.set_title("HAR Dataset – LAHUP & Quantisation Comparison",
                 fontsize=14, fontweight="bold")

    # Separator between pruning and quantisation columns
    if quant_start:
        ax.axvline(x=quant_start - 0.5, color="black", linewidth=2)

    # Cell annotations
    for i in range(norm_data.shape[0]):
        for j in range(norm_data.shape[1]):
            ax.text(j, i, f"{data_display[i, j]:.2f}",
                    ha="center", va="center", fontsize=8, color="black")

    cbar = fig.colorbar(im)
    cbar.set_label("Normalized Performance Score", rotation=90)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def make_heatmap_from_parsed(m: dict, sparsity_param: float) -> bytes | None:
    """
    Use stage_metrics from the API result when available — it has real inference
    speeds for every stage. Fall back to log-parsed values only if result is not
    ready yet (i.e. pipeline still running).
    """
    # Prefer stage_metrics computed by run_pipeline (has real speeds for all stages)
    result = st.session_state.get("result") or {}
    stage_metrics = result.get("stage_metrics")
    if stage_metrics and len(stage_metrics) >= 2:
        return build_heatmap(stage_metrics)

    # Fallback: rebuild from log text (speeds will be 0 for non-quant stages,
    # but this keeps the heatmap appearing while the job is still running)
    sm = {}

    if m.get("final_accuracy") is not None:
        sm["TGF-NAS"] = {
            "accuracy": m["final_accuracy"],
            "size":     m.get("size_before") or 0.0,
            "speed":    0.0,
            "sparsity": 0.0,
        }

    if m.get("pruned_accuracy") is not None:
        sm["LAHUP"] = {
            "accuracy": m["pruned_accuracy"],
            "size":     m.get("size_after") or 0.0,
            "speed":    0.0,
            "sparsity": m.get("achieved_sparsity") or sparsity_param * 100,
        }

    if m.get("finetuned_accuracy") is not None:
        sm["LAHUP+FT"] = {
            "accuracy": m["finetuned_accuracy"],
            "size":     m.get("size_after") or 0.0,
            "speed":    0.0,
            "sparsity": m.get("achieved_sparsity") or sparsity_param * 100,
        }

    for qt, qd in (m.get("quantization") or {}).items():
        sm[qt] = {
            "accuracy": qd["accuracy"],
            "size":     qd["size"],
            "speed":    qd["speed"],
            "sparsity": m.get("achieved_sparsity") or sparsity_param * 100,
        }

    if len(sm) < 2:
        return None
    return build_heatmap(sm)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header(" Pipeline Parameters")

    st.subheader(" Data")
    train_file = st.file_uploader("Training CSV",  type=["csv"])
    test_file  = st.file_uploader("Test CSV",      type=["csv"])

    st.subheader("Dataset fraction")
    dataset_fraction_pct = st.slider(
        "Fraction of dataset to use",
        min_value=5, max_value=100, value=100, step=5,
        format="%d%%",
        help="50% = use 50% of rows from each CSV.  Speeds up experimentation.",
    )
    dataset_fraction = dataset_fraction_pct / 100
    st.caption(f"Using **{dataset_fraction_pct}%** of train & test data")

    st.subheader(" NAS")
    num_architectures = st.slider("Architectures to sample", 5, 100, 20, 5)

    st.subheader(" Pruning")
    sparsity_pct = st.slider("Target Sparsity", 10, 90, 30, 5, format="%d%%")
    sparsity = sparsity_pct / 100

    st.subheader(" Quantization")
    use_fp16 = st.checkbox("FP16", value=True)
    use_int8 = st.checkbox("INT8", value=True)
    quant_list = [q for q, on in [("FP16", use_fp16), ("INT8", use_int8)] if on]

    api_url = API_URL

    st.markdown("---")
    run_btn = st.button(
        " Run Pipeline", type="primary",
        disabled=st.session_state.running,
        use_container_width=True,
    )
    if st.session_state.job_id:
        if st.button(" Reset", use_container_width=True):
            for k, v in DEFAULTS.items():
                st.session_state[k] = v if not isinstance(v, list) else []
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Launch job
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    if not train_file or not test_file:
        st.sidebar.error("Upload both train and test CSVs first.")
    elif not quant_list:
        st.sidebar.error("Select at least one quantization type.")
    else:
        try:
            resp = requests.post(
                f"{api_url}/run",
                files={
                    "train_file": (train_file.name, train_file.getvalue(), "text/csv"),
                    "test_file":  (test_file.name,  test_file.getvalue(),  "text/csv"),
                },
                data={
                    "input_size":        "0",
                    "output_size":       "0",
                    "use_embedding":     "false",
                    "num_architectures": str(num_architectures),
                    "sparsity":          str(sparsity),
                    "quant_types":       ",".join(quant_list),
                    "dataset_fraction":  str(dataset_fraction),
                },
                timeout=30,
            )
            if resp.status_code == 200:
                rj = resp.json()
                for k, v in DEFAULTS.items():
                    st.session_state[k] = v if not isinstance(v, list) else []
                st.session_state.job_id  = rj["job_id"]
                st.session_state.status  = "queued"
                st.session_state.running = True
                st.rerun()
            else:
                st.sidebar.error(f"API {resp.status_code}: {resp.text}")
        except requests.exceptions.ConnectionError:
            st.sidebar.error(f"Cannot reach {api_url}. Is the API running?")


# ─────────────────────────────────────────────────────────────────────────────
# No job yet → landing page
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.job_id:
    st.markdown("""
    <div style="text-align:center;padding:80px 20px;color:#6b7a99">
        <div style="font-size:64px;margin-bottom:16px"></div>
        <h2 style="color:#a0b4cc">HAR Pipeline: TGF-NAS + LAHUP + Quantization</h2>
        <p style="max-width:520px;margin:auto">
            Upload your train and test CSVs in the sidebar, configure the parameters,
            and click <strong>Run Pipeline</strong>.<br><br>
            Results — including the full stage comparison heatmap and model downloads
            — appear live as each stage completes.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Poll API
# ─────────────────────────────────────────────────────────────────────────────
job_id = st.session_state.job_id

if st.session_state.running:
    try:
        lr = requests.get(
            f"{api_url}/logs/{job_id}",
            params={"from_line": st.session_state.last_line},
            timeout=10,
        )
        if lr.status_code == 200:
            lj = lr.json()
            if lj["lines"]:
                st.session_state.all_logs.extend(lj["lines"])
                st.session_state.last_line = lj["total_lines"]
            st.session_state.status = lj["status"]
            if lj["status"] in ("done", "error"):
                st.session_state.running = False
                sr = requests.get(f"{api_url}/status/{job_id}", timeout=10)
                if sr.status_code == 200:
                    sj = sr.json()
                    st.session_state.result      = sj.get("result")
                    st.session_state.model_paths = sj.get("model_paths", [])
    except Exception:
        pass
    st.session_state.parsed = parse_metrics(st.session_state.all_logs)

m      = st.session_state.parsed
status = st.session_state.status or "queued"


# ─────────────────────────────────────────────────────────────────────────────
# Status bar
# ─────────────────────────────────────────────────────────────────────────────
badge = {"running":"status-running","done":"status-done",
         "error":"status-error","queued":"status-queued"}.get(status,"status-queued")
emoji = {"running":"","done":"","error":"","queued":""}.get(status,"")

st.markdown(
    f'**Job:** `{job_id[:8]}…`  '
    f'<span class="status-badge {badge}">{emoji} {status.upper()}</span>',
    unsafe_allow_html=True,
)
# if st.session_state.running:
#     st.progress(0.5)
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# Main layout — left: metrics/heatmap/downloads,  right: logs
# ─────────────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.6])


# ══════════════════════════════════════════════════════════════════════════════
# LEFT COLUMN
# ══════════════════════════════════════════════════════════════════════════════
with col_left:

    # ── NAS results ──────────────────────────────────────────────────────────
    if m.get("final_accuracy") is not None or m.get("best_arch_idx") is not None:
        st.markdown('<div class="section-header"> NAS Results</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            v = f"{m['final_accuracy']:.2f}%" if m.get("final_accuracy") else "…"
            st.markdown(mcard("Final Accuracy", v, "Full training"), unsafe_allow_html=True)
        with c2:
            v = str(m["best_arch_idx"]) if m.get("best_arch_idx") is not None else "…"
            st.markdown(mcard("Best Arch #", v, "Architecture index"), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            v = f"{m['proxy_accuracy']:.4f}" if m.get("proxy_accuracy") is not None else "…"
            st.markdown(mcard("Proxy Acc", v, "Probe phase"), unsafe_allow_html=True)
        with c2:
            v = f"{m['tgfnas_score']:.4f}" if m.get("tgfnas_score") is not None else "…"
            st.markdown(mcard("TGF-NAS Score", v), unsafe_allow_html=True)

    # #  Training curve 
    # if m.get("epoch_history"):
    #     st.markdown('<div class="section-header"> Training Progress</div>', unsafe_allow_html=True)
    #     df_ep = (pd.DataFrame(m["epoch_history"])
    #                .set_index("epoch")[["train_acc", "test_acc"]])
    #     df_ep.columns = ["Train %", "Test %"]
    #     st.line_chart(df_ep)


   

    #  Pruning 
    if m.get("baseline_accuracy") is not None:
        st.markdown('<div class="section-header"> Pruning</div>', unsafe_allow_html=True)
        base = m.get("baseline_accuracy", 0)
        ft   = m.get("finetuned_accuracy") or m.get("pruned_accuracy") or 0
        sb   = m.get("size_before") or 0
        sa   = m.get("size_after")  or 0
        sp   = m.get("achieved_sparsity") or 0
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(mcard("Before Pruning",  f"{base:.2f}%", f"{sb:.3f} MB"), unsafe_allow_html=True)
        with c2:
            st.markdown(mcard("After Fine-tune", f"{ft:.2f}%",   f"{sa:.3f} MB"), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(mcard("Achieved Sparsity", f"{sp:.1f}%"), unsafe_allow_html=True)
        with c2:
            if sb and sa:
                st.markdown(mcard("Size Reduction", f"{sb/sa:.2f}×",
                                   f"{sb:.3f} → {sa:.3f} MB"), unsafe_allow_html=True)

    # ── Quantization table ────────────────────────────────────────────────────
    if m.get("quantization"):
        st.markdown('<div class="section-header"> Quantization</div>', unsafe_allow_html=True)
        rows = [
            {"Method": qt, "Size (MB)": f"{qd['size']:.3f}",
             "Accuracy %": f"{qd['accuracy']:.2f}", "Speed (ms)": f"{qd['speed']:.2f}"}
            for qt, qd in m["quantization"].items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


    # ── Model downloads ───────────────────────────────────────────────────────
    if status == "done" and st.session_state.model_paths:
        st.markdown('<div class="section-header"> Download Models</div>', unsafe_allow_html=True)

        LABEL_MAP = {
            "full_trained": (" Full Trained Model",  "TGF-NAS + full training"),
            "pruned":       (" Pruned Model",         "LAHUP pruning + fine-tune"),
            "quant_fp16":   (" Quantized — FP16",    "Float-16 weights"),
            "quant_int8":   (" Quantized — INT8",    "Int-8 weights"),
        }

        for key in st.session_state.model_paths:
            label, desc = LABEL_MAP.get(key, (key, ""))
            try:
                dl = requests.get(f"{api_url}/download/{job_id}/{key}", timeout=60)
                if dl.status_code == 200:
                    st.download_button(
                        label=f"{label}  —  {desc}",
                        data=dl.content,
                        file_name=f"har_{key}.pth",
                        mime="application/octet-stream",
                        key=f"dl_{key}",
                        use_container_width=True,
                    )
                else:
                    st.warning(f"{key}: {dl.status_code}")
            except Exception as e:
                st.warning(f"Download failed for {key}: {e}")

    if status == "error":
        st.error("Pipeline failed — see the log on the right for the full traceback.")


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — live log
# ══════════════════════════════════════════════════════════════════════════════
with col_right:
    st.markdown('<div class="section-header"> Live Pipeline Log</div>', unsafe_allow_html=True)

    logs     = st.session_state.all_logs
    log_html = (
        "<br>".join(colorize(l) for l in logs)
        if logs else '<span style="color:#555">Waiting for output…</span>'
    )
    st.markdown(f'<div class="log-box">{log_html}</div>', unsafe_allow_html=True)
    st.caption(f"{len(logs)} lines")

    if logs:
        st.download_button(
            " Download Full Log (.txt)",
            data="\n".join(logs),
            file_name="har_pipeline_log.txt",
            mime="text/plain",
            use_container_width=True,
        )

     # ── Heatmap ───────────────────────────────────────────────────────────────
    heatmap_png = make_heatmap_from_parsed(m, sparsity)
    if heatmap_png:
        st.markdown('<div class="section-header"> Stage Comparison Heatmap</div>',
                    unsafe_allow_html=True)
        st.image(heatmap_png, use_container_width=True)
        st.download_button(
            label=" Download Heatmap (.PNG)",
            data=heatmap_png,
            file_name="HAR_LAHUP_HEATMAP.png",
            mime="image/png",
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Auto-refresh while running
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.running:
    time.sleep(POLL_INTERVAL)
    st.rerun()