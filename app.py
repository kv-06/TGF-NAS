"""
HAR Pipeline — Streamlit Frontend
Run: streamlit run app.py
Make sure api.py is running: uvicorn app:app --host 0.0.0.0 --port 8000
"""

import streamlit as st
import requests
import time
import re
import json

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"
POLL_INTERVAL = 2  # seconds between log polls

st.set_page_config(
    page_title="HAR Pipeline — TGF-NAS + LAHUP + Quantization",
    page_icon="🧠",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #16213e);
        border-radius: 12px;
        padding: 20px 24px;
        color: white;
        text-align: center;
        border: 1px solid #2a5298;
    }
    .metric-label { font-size: 13px; color: #a0b4cc; margin-bottom: 4px; }
    .metric-value { font-size: 32px; font-weight: 700; color: #4fc3f7; }
    .metric-sub   { font-size: 12px; color: #7fb3cc; margin-top: 2px; }

    .log-box {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        font-family: 'Courier New', monospace;
        font-size: 12px;
        color: #c9d1d9;
        height: 420px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-break: break-all;
    }

    .section-header {
        background: linear-gradient(90deg, #1e3a5f, transparent);
        border-left: 4px solid #4fc3f7;
        padding: 8px 14px;
        border-radius: 0 6px 6px 0;
        margin: 16px 0 10px 0;
        font-weight: 600;
        color: #e6f3ff;
    }

    .status-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-left: 8px;
    }
    .status-running  { background: #1a3a2a; color: #4caf50; border: 1px solid #4caf50; }
    .status-done     { background: #1a2a4a; color: #4fc3f7; border: 1px solid #4fc3f7; }
    .status-error    { background: #3a1a1a; color: #f44336; border: 1px solid #f44336; }
    .status-queued   { background: #2a2a1a; color: #ffc107; border: 1px solid #ffc107; }

    .highlight-line  { color: #4fc3f7; }
    .success-line    { color: #4caf50; }
    .warning-line    { color: #ffc107; }
    .error-line      { color: #f44336; }
    .summary-box {
        background: #0d1117;
        border: 1px solid #4fc3f7;
        border-radius: 10px;
        padding: 18px 22px;
        margin-top: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────
for key, val in {
    "job_id": None,
    "status": None,
    "all_logs": [],
    "last_line": 0,
    "result": None,
    "running": False,
    "parsed_metrics": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_metrics_from_logs(logs):
    """Extract key metrics from pipeline output logs."""
    metrics = {}
    text = "\n".join(logs)

    # Best architecture index
    m = re.search(r"Best Architecture Index:\s*(\d+)", text)
    if m:
        metrics["best_arch_idx"] = int(m.group(1))

    # Final test accuracy (from full training)
    m = re.search(r"Final Test Accuracy:\s*([\d.]+)%", text)
    if m:
        metrics["final_accuracy"] = float(m.group(1))

    # Proxy accuracy of best arch
    m = re.search(r"Proxy Accuracy:\s*([\d.]+)", text)
    if m:
        metrics["proxy_accuracy"] = float(m.group(1))

    # TGF-NAS score
    m = re.search(r"TGF-NAS Score:\s*([-\d.]+)", text)
    if m:
        metrics["tgfnas_score"] = float(m.group(1))

    # Alpha, Beta
    m = re.search(r"Final weights: α=([\d.]+), β=([\d.]+)", text)
    if m:
        metrics["alpha"] = float(m.group(1))
        metrics["beta"]  = float(m.group(2))

    # Pruning stats
    m = re.search(r"Baseline accuracy\s+:\s+([\d.]+)%", text)
    if m:
        metrics["baseline_accuracy"] = float(m.group(1))

    m = re.search(r"Accuracy after pruning\s*:\s*([\d.]+)%", text)
    if m:
        metrics["pruned_accuracy"] = float(m.group(1))

    m = re.search(r"Accuracy after fine-tuning\s*:\s*([\d.]+)%", text)
    if m:
        metrics["finetuned_accuracy"] = float(m.group(1))

    m = re.search(r"Achieved sparsity\s+:\s+([\d.]+)%", text)
    if m:
        metrics["achieved_sparsity"] = float(m.group(1))

    m = re.search(r"Model Size before pruning\s*:\s*([\d.]+)", text)
    if m:
        metrics["size_before_pruning"] = float(m.group(1))

    m = re.search(r"Model Size After pruning\s*:\s*([\d.]+)", text)
    if m:
        metrics["size_after_pruning"] = float(m.group(1))

    # Quantization results
    quant_results = {}
    for qtype in ["FP16", "INT8"]:
        pattern = rf"{qtype}\s+([\d.]+)\s+([\d.]+x)\s+([\d.]+)\s+([+-]?[\d.]+)\s+([\d.]+)"
        m = re.search(pattern, text)
        if m:
            quant_results[qtype] = {
                "size": float(m.group(1)),
                "reduction": m.group(2),
                "accuracy": float(m.group(3)),
                "drop": float(m.group(4)),
                "speed": float(m.group(5)),
            }
    if quant_results:
        metrics["quantization"] = quant_results

    # Epoch training history (last model only)
    epoch_pattern = r"Epoch\s+(\d+)/(\d+)\s*\|\s*Train Acc:\s*([\d.]+)%\s*\|\s*Test Acc:\s*([\d.]+)%"
    epochs = re.findall(epoch_pattern, text)
    if epochs:
        metrics["epoch_history"] = [
            {"epoch": int(e[0]), "total": int(e[1]),
             "train_acc": float(e[2]), "test_acc": float(e[3])}
            for e in epochs
        ]

    # Arch score table
    arch_pattern = r"(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([-\d.]+)"
    arch_rows = re.findall(arch_pattern, text)
    if arch_rows:
        metrics["arch_table"] = [
            {"arch": int(r[0]), "magnitude": float(r[1]),
             "consistency": float(r[2]), "accuracy": float(r[3]),
             "tgfnas": float(r[4])}
            for r in arch_rows
        ]

    return metrics


def colorize_log_line(line):
    """Apply HTML color highlights to important log lines."""
    if any(k in line for k in ["✅", "Best Architecture", "Final Test Accuracy", "SUMMARY"]):
        return f'<span style="color:#4fc3f7;font-weight:600">{line}</span>'
    if any(k in line for k in ["💾", "saved", "Accuracy after fine"]):
        return f'<span style="color:#4caf50">{line}</span>'
    if any(k in line for k in ["ERROR", "Error", "failed"]):
        return f'<span style="color:#f44336">{line}</span>'
    if any(k in line for k in ["WARNING", "drop:", "Pruned (before FT)"]):
        return f'<span style="color:#ffc107">{line}</span>'
    if any(k in line for k in ["======", "------", "APPLY", "PRUNING", "QUANT", "TGF-NAS"]):
        return f'<span style="color:#9575cd">{line}</span>'
    if line.startswith("Epoch ") or line.startswith("    Epoch "):
        return f'<span style="color:#80cbc4">{line}</span>'
    return line


def render_log_html(logs):
    """Convert log lines to styled HTML for the log box."""
    html_lines = [colorize_log_line(l) for l in logs]
    return "<br>".join(html_lines)


def metric_card(label, value, sub=""):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>
    """


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🧠 HAR Pipeline")
st.markdown("**TGF-NAS** + **LAHUP Pruning** + **Quantization** — Human Activity Recognition")

# ── Sidebar: Parameters ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pipeline Parameters")

    st.subheader("📂 Data")
    train_file = st.file_uploader("Training CSV", type=["csv"], key="train_csv")
    test_file  = st.file_uploader("Test CSV",     type=["csv"], key="test_csv")

    st.subheader("🔢 Model Config")
    input_size  = st.number_input("Input Size (features)", min_value=1,   value=561, step=1)
    output_size = st.number_input("Output Size (classes)", min_value=2,   value=6,   step=1)

    st.subheader("🔍 NAS Config")
    num_architectures = st.slider("Number of Architectures",  min_value=5, max_value=100, value=20, step=5)

    st.subheader("✂️ Pruning")
    sparsity = st.slider("Sparsity", min_value=0.1, max_value=0.9, value=0.3, step=0.05,
                         help="Fraction of weights to prune (e.g. 0.3 = 30%)")

    st.subheader("⚡ Quantization")
    quant_fp16 = st.checkbox("FP16", value=True)
    quant_int8 = st.checkbox("INT8", value=True)
    quant_types_list = []
    if quant_fp16: quant_types_list.append("FP16")
    if quant_int8: quant_types_list.append("INT8")

    st.subheader("🌐 API")
    api_url = st.text_input("API URL", value=API_URL)

    st.markdown("---")
    run_btn = st.button("🚀 Run Pipeline", type="primary",
                        disabled=st.session_state.running,
                        use_container_width=True)

    if st.session_state.job_id:
        stop_btn = st.button("🔴 Reset", use_container_width=True)
        if stop_btn:
            st.session_state.job_id = None
            st.session_state.status = None
            st.session_state.all_logs = []
            st.session_state.last_line = 0
            st.session_state.result = None
            st.session_state.running = False
            st.session_state.parsed_metrics = {}
            st.rerun()


# ── Start Job ─────────────────────────────────────────────────────────────────
if run_btn:
    if not train_file or not test_file:
        st.sidebar.error("Please upload both Train and Test CSV files.")
    elif not quant_types_list:
        st.sidebar.error("Select at least one quantization type.")
    else:
        quant_str = ",".join(quant_types_list)
        files = {
            "train_file": (train_file.name, train_file.getvalue(), "text/csv"),
            "test_file":  (test_file.name,  test_file.getvalue(),  "text/csv"),
        }
        data = {
            "input_size":        str(int(input_size)),
            "output_size":       str(int(output_size)),
            "use_embedding":     "false",
            "num_architectures": str(int(num_architectures)),
            "sparsity":          str(sparsity),
            "quant_types":       quant_str,
        }
        try:
            resp = requests.post(f"{api_url}/run", files=files, data=data, timeout=30)
            if resp.status_code == 200:
                rj = resp.json()
                st.session_state.job_id   = rj["job_id"]
                st.session_state.status   = "queued"
                st.session_state.all_logs = []
                st.session_state.last_line = 0
                st.session_state.result   = None
                st.session_state.running  = True
                st.session_state.parsed_metrics = {}
                st.rerun()
            else:
                st.sidebar.error(f"API Error {resp.status_code}: {resp.text}")
        except requests.exceptions.ConnectionError:
            st.sidebar.error(f"Cannot connect to API at {api_url}. Is it running?")


# ── Main Content ──────────────────────────────────────────────────────────────

if not st.session_state.job_id:
    # Welcome screen
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color: #6b7a99;">
        <div style="font-size: 64px; margin-bottom: 16px;">🚀</div>
        <h3 style="color: #a0b4cc;">Configure your pipeline in the sidebar and click Run</h3>
        <p>Upload your train and test CSV files, set parameters, and launch the full<br>
        TGF-NAS architecture search → LAHUP pruning → quantization pipeline.</p>
    </div>
    """, unsafe_allow_html=True)

    # Show example output
    with st.expander("📋 Example Output Preview (from pipeline-50 run)"):
        example = """Sampling 50 architectures via Ray Tune...
...
✅ Best Architecture Index: 7
   TGF-NAS Score: 1.2341
   Magnitude:     0.000423
   Consistency:   0.8821
   Proxy Accuracy:0.9201

Training for 20 epochs...
Epoch  1/20 | Train Acc: 93.39% | Test Acc: 93.52%
...
Epoch 20/20 | Train Acc: 96.19% | Test Acc: 92.09%

✅ Final Test Accuracy: 92.09%

APPLY LAHUP PRUNING
  Target sparsity : 30.0%
  Baseline accuracy : 92.09%
  Accuracy after pruning : 77.67%  (drop: 14.42%)
  Accuracy after fine-tuning : 92.70%  (recovery: 15.03%)
  Achieved sparsity : 30.00%
  Model Size: 1.61 MB → 1.13 MB

APPLY QUANTIZATION
  FP16  0.80 MB  1.40x  92.70%  +0.00  9.74ms
  INT8  0.41 MB  2.75x  92.50%  +0.20  2.41ms"""
        st.code(example, language="")

else:
    # ── Poll for updates ─────────────────────────────────────────────────────
    job_id = st.session_state.job_id

    if st.session_state.running:
        try:
            log_resp = requests.get(
                f"{api_url}/logs/{job_id}",
                params={"from_line": st.session_state.last_line},
                timeout=10,
            )
            if log_resp.status_code == 200:
                lj = log_resp.json()
                new_lines = lj.get("lines", [])
                if new_lines:
                    st.session_state.all_logs.extend(new_lines)
                    st.session_state.last_line = lj["total_lines"]

                current_status = lj.get("status", "running")
                st.session_state.status = current_status

                if current_status in ("done", "error"):
                    st.session_state.running = False
                    # Fetch final result
                    stat_resp = requests.get(f"{api_url}/status/{job_id}", timeout=10)
                    if stat_resp.status_code == 200:
                        sj = stat_resp.json()
                        st.session_state.result = sj.get("result")
        except Exception:
            pass

        # Parse metrics from current logs
        st.session_state.parsed_metrics = parse_metrics_from_logs(st.session_state.all_logs)

    # ── Status bar ────────────────────────────────────────────────────────────
    status = st.session_state.status or "queued"
    badge_class = {
        "running": "status-running",
        "done":    "status-done",
        "error":   "status-error",
        "queued":  "status-queued",
    }.get(status, "status-queued")

    status_emoji = {"running": "⏳", "done": "✅", "error": "❌", "queued": "🕐"}.get(status, "🕐")
    st.markdown(
        f"**Job:** `{job_id[:8]}...`  "
        f'<span class="status-badge {badge_class}">{status_emoji} {status.upper()}</span>',
        unsafe_allow_html=True,
    )

    if st.session_state.running:
        st.progress(0.5 if status == "running" else 0.0)

    st.markdown("---")

    # ── Two-column layout: metrics left, logs right ───────────────────────────
    col_metrics, col_logs = st.columns([1, 1.6])

    # ── Metrics Panel ─────────────────────────────────────────────────────────
    with col_metrics:
        m = st.session_state.parsed_metrics

        # ── NAS Results
        if any(k in m for k in ["final_accuracy", "best_arch_idx", "tgfnas_score"]):
            st.markdown('<div class="section-header">🔍 NAS Results</div>', unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                val = f"{m['final_accuracy']:.2f}%" if "final_accuracy" in m else "..."
                st.markdown(metric_card("Final Accuracy", val, "Full train"), unsafe_allow_html=True)
            with c2:
                val = str(m.get("best_arch_idx", "..."))
                st.markdown(metric_card("Best Arch #", val, "Architecture index"), unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                val = f"{m['proxy_accuracy']:.4f}" if "proxy_accuracy" in m else "..."
                st.markdown(metric_card("Proxy Acc", val, "Probe phase"), unsafe_allow_html=True)
            with c2:
                val = f"{m['tgfnas_score']:.4f}" if "tgfnas_score" in m else "..."
                st.markdown(metric_card("TGF-NAS Score", val), unsafe_allow_html=True)

            if "alpha" in m:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(metric_card("α (Magnitude)", f"{m['alpha']:.3f}", "Weight"), unsafe_allow_html=True)
                with c2:
                    st.markdown(metric_card("β (Consistency)", f"{m['beta']:.3f}", "Weight"), unsafe_allow_html=True)

        # ── Training progress chart
        if "epoch_history" in m and m["epoch_history"]:
            st.markdown('<div class="section-header">📈 Training Progress</div>', unsafe_allow_html=True)
            import pandas as pd
            ep_df = pd.DataFrame(m["epoch_history"])
            chart_df = ep_df.set_index("epoch")[["train_acc", "test_acc"]]
            chart_df.columns = ["Train Acc %", "Test Acc %"]
            st.line_chart(chart_df)

        # ── Pruning
        if "baseline_accuracy" in m:
            st.markdown('<div class="section-header">✂️ Pruning Results</div>', unsafe_allow_html=True)

            baseline = m.get("baseline_accuracy", 0)
            pruned   = m.get("pruned_accuracy", 0)
            finetuned= m.get("finetuned_accuracy", pruned)
            sparsity_achieved = m.get("achieved_sparsity", 0)
            s_before = m.get("size_before_pruning", 0)
            s_after  = m.get("size_after_pruning", 0)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(metric_card("Before Pruning", f"{baseline:.2f}%", f"{s_before:.2f} MB"), unsafe_allow_html=True)
            with c2:
                st.markdown(metric_card("After Fine-tune", f"{finetuned:.2f}%", f"{s_after:.2f} MB"), unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(metric_card("Achieved Sparsity", f"{sparsity_achieved:.1f}%"), unsafe_allow_html=True)
            with c2:
                if s_before > 0 and s_after > 0:
                    reduction = s_before / s_after
                    st.markdown(metric_card("Size Reduction", f"{reduction:.2f}x", f"{s_before:.2f}→{s_after:.2f} MB"), unsafe_allow_html=True)

        # ── Quantization
        if "quantization" in m:
            st.markdown('<div class="section-header">⚡ Quantization Results</div>', unsafe_allow_html=True)
            quant = m["quantization"]

            import pandas as pd
            rows = []
            for qtype, qr in quant.items():
                rows.append({
                    "Method": qtype,
                    "Size (MB)": f"{qr['size']:.2f}",
                    "Reduction": qr['reduction'],
                    "Accuracy %": f"{qr['accuracy']:.2f}",
                    "Acc Drop": f"{qr['drop']:+.2f}%",
                    "Speed (ms)": f"{qr['speed']:.2f}",
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── Final result from API
        if st.session_state.result:
            st.markdown('<div class="section-header">🎯 Final Result</div>', unsafe_allow_html=True)
            r = st.session_state.result
            st.markdown(f"""
            <div class="summary-box">
                <p><b>Accuracy:</b> {r.get('accuracy', 'N/A'):.2f}%</p>
                <p><b>Sparsity:</b> {r.get('sparsity', 'N/A')}</p>
                <p><b>Quantization:</b> {', '.join(r.get('quantization', []))}</p>
            </div>
            """, unsafe_allow_html=True)

        if status == "error":
            st.error("Pipeline encountered an error. See logs for details.")

    # ── Logs Panel ────────────────────────────────────────────────────────────
    with col_logs:
        st.markdown('<div class="section-header">📜 Pipeline Logs</div>', unsafe_allow_html=True)

        log_lines = st.session_state.all_logs
        log_html  = render_log_html(log_lines) if log_lines else "<span style='color:#555'>Waiting for output...</span>"

        st.markdown(
            f'<div class="log-box" id="log-output">{log_html}</div>',
            unsafe_allow_html=True,
        )

        # Auto-scroll JS
        st.markdown("""
        <script>
            const logBox = document.getElementById('log-output');
            if (logBox) logBox.scrollTop = logBox.scrollHeight;
        </script>
        """, unsafe_allow_html=True)

        st.caption(f"Lines received: {len(log_lines)}")

        # Download logs button
        if log_lines:
            st.download_button(
                "⬇️ Download Logs",
                data="\n".join(log_lines),
                file_name="har_pipeline_output.txt",
                mime="text/plain",
            )

        # Arch score table
        if "arch_table" in m and m["arch_table"]:
            import pandas as pd
            with st.expander("📊 Architecture Score Table"):
                arch_df = pd.DataFrame(m["arch_table"])
                best_idx = m.get("best_arch_idx", -1)
                st.dataframe(
                    arch_df.style.highlight_max(subset=["tgfnas"], color="#1a3a2a"),
                    use_container_width=True, hide_index=True,
                )

    # ── Auto-refresh while running ────────────────────────────────────────────
    if st.session_state.running:
        time.sleep(POLL_INTERVAL)
        st.rerun()