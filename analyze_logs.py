# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

LOGS = Path("logs")

def load_results():
    rows = []
    for summary_path in sorted(LOGS.rglob("benchmark_summary.json")):
        parts = summary_path.relative_to(LOGS).parts
        kernel = parts[0]  # triton / sgmv
        model_or_adapter = parts[2]  # lora-adapter / Qwen3-32B
        label = f"{kernel} / {model_or_adapter}"

        summary = json.loads(summary_path.read_text())
        percentile_path = summary_path.parent / "benchmark_percentile.json"
        percentiles = json.loads(percentile_path.read_text())

        rows.append({"label": label, "summary": summary, "percentiles": percentiles})
    return rows


def write_markdown_table(rows):
    keys = [
        "Output Throughput (tok/s)", "Total Throughput (tok/s)",
        "TTFT (ms)", "TPOT (ms)", "ITL (ms)",
        "Avg Latency (s)", "Req Throughput (req/s)",
    ]
    lines = []
    header = "| Metric | " + " | ".join(r["label"] for r in rows) + " |"
    sep = "|---|" + "|".join("---:" for _ in rows) + "|"
    lines.append(header)
    lines.append(sep)
    for k in keys:
        vals = " | ".join(f"{r['summary'][k]:.2f}" for r in rows)
        lines.append(f"| {k} | {vals} |")

    lines.append("")
    lines.append("### Percentile breakdown")
    lines.append("")

    for metric in ["TTFT (ms)", "ITL (ms)", "TPOT (ms)", "Output (tok/s)"]:
        lines.append(f"#### {metric}")
        lines.append("")
        pct_labels = [p["Percentiles"] for p in rows[0]["percentiles"]]
        header = "| Percentile | " + " | ".join(r["label"] for r in rows) + " |"
        sep = "|---|" + "|".join("---:" for _ in rows) + "|"
        lines.append(header)
        lines.append(sep)
        for i, pct in enumerate(pct_labels):
            vals = " | ".join(f"{r['percentiles'][i][metric]:.2f}" for r in rows)
            lines.append(f"| {pct} | {vals} |")
        lines.append("")

    md = "\n".join(lines)
    Path("logs/results.md").write_text(md)
    print(md)


def make_boxplots(rows):
    metrics = ["TTFT (ms)", "ITL (ms)", "TPOT (ms)", "Output (tok/s)"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 6))

    for ax, metric in zip(axes, metrics):
        box_data = []
        labels = []
        for r in rows:
            values = [p[metric] for p in r["percentiles"]]
            box_data.append(values)
            labels.append(r["label"].replace(" / ", "\n"))

        positions = np.arange(len(box_data))
        bp = ax.boxplot(box_data, positions=positions, widths=0.5, patch_artist=True)
        colors = ["#4C72B0", "#DD8452", "#55A868"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("LoRA Kernel Benchmark: Triton vs SGMV vs Base Model", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig("logs/boxplots.png", dpi=150, bbox_inches="tight")
    print("Saved logs/boxplots.png")


if __name__ == "__main__":
    rows = load_results()
    write_markdown_table(rows)
    make_boxplots(rows)
