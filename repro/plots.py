import csv
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def adaptation_plot(summary, out):
    methods = ["Clean", "Identity", "Random", "Fixed-best (shared)", "Signal-Global",
               "Signal-Pixel Fusion", "Global Oracle", "Pixel Oracle"]
    lookup = {(r["model"], r["method"]): r for r in summary
              if r["condition"] == "six_corruption_macro"}
    widths = [1.6, 1.3, 1.7, 1.65, 1.65, 1.65, 1.8, 1.8, 1.65, 1.65]
    x = np.r_[0, np.cumsum(widths)]
    fig, ax = plt.subplots(figsize=(17, 4))
    ax.set_xlim(0, x[-1])
    ax.set_ylim(8, 0)
    ax.axis("off")

    def cell(c, y, text, span=1, h=1):
        ax.add_patch(
            Rectangle(
                (x[c], y),
                x[c + span] - x[c],
                h,
                facecolor="#e8f0f5" if y < 2 else "#fafafa",
                edgecolor="#c8d4dc",
                lw=0.4,
            )
        )
        ax.text(
            (x[c] + x[c + span]) / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=9,
        )

    cell(0, 0, "Model", h=2)
    cell(1, 0, "Metric ↓", h=2)
    for c, n, s in [
        (2, "Reference", 1),
        (3, "Baselines", 3),
        (6, "Adaptation", 2),
        (8, "Oracles", 2),
    ]:
        cell(c, 0, n, s)
    for c, n in enumerate(
        [
            "Source / Clean",
            "Identity",
            "Random",
            "Fixed-best",
            "Signal-Global",
            "Signal-Pixel",
            "Global",
            "Pixel",
        ],
        2,
    ):
        cell(c, 1, n)
    for mi, m in enumerate(["raft", "acv"]):
        cell(0, 2 + 3 * mi, "RAFT-Stereo" if m == "raft" else "ACVNet", h=3)
        for i, metric in enumerate(["Abs", "1px", "D1"]):
            y = 2 + 3 * mi + i
            cell(1, y, metric + (" (px)" if metric == "Abs" else " (%)"))
            for c, method in enumerate(methods, 2):
                r = lookup[m, method]
                cell(c, y, f"{r[metric]:.2f} ± {r[metric+'_assignment_sd']:.2f}")
    fig.text(0.5, 0.01, "Mean ± standard deviation across three assignments.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ["pdf", "png"]:
        fig.savefig(out / ("adaptation." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)


def evidence_plots(evidence, out):
    path = (
        evidence
        / "sensor_v4_figure_c_layers_dev_20260926_r2/figure_c_proxy_layers_compact.csv"
    )
    if not path.exists():
        return
    with path.open() as f:
        rows = list(csv.DictReader(f))
    # These are distinct estimates, never combined into a composite score.
    metrics = ["AUROC", "AP", "Spearman"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharey="row")
    for col, model in enumerate(["raft", "acv"]):
        rr = [r for r in rows if r.get("model") == model]
        for ax, key in zip(axes[:, col], metrics):
            for prefix, style, label in [
                ("pooled_", "o-", "Condition-pooled"),
                ("within_image_", "s--", "Within-image"),
            ]:
                vals = [
                    (
                        float(r[prefix + key])
                        if r.get(prefix + key) not in [None, "", "NA"]
                        else np.nan
                    )
                    for r in rr
                ]
                ax.plot(range(len(rr)), vals, style, label=label, markersize=4)
            ax.set_ylabel(key)
            ax.set_xticks(range(len(rr)))
            ax.set_xticklabels([r["semantic_id"] for r in rr], fontsize=8)
        axes[0, col].legend(fontsize=8)
        axes[0, col].set_title("RAFT-Stereo" if model == "raft" else "ACVNet")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(out / ("proxy_levels." + ext), dpi=180, bbox_inches="tight")
    plt.close(fig)


def condition_response_plot(evidence, out):
    folder = evidence / "sensor_v4_all_proxy_response_dev_20260926"
    with (folder / "all_proxy_condition_response.csv").open() as f:
        raw = list(csv.DictReader(f))
    with (folder / "condition_task_metrics.csv").open() as f:
        tasks = list(csv.DictReader(f))
    conditions = [
        "clean",
        "brightness",
        "contrast",
        "saturation",
        "gaussian_noise",
        "shot_noise",
        "speckle_noise",
        "six_corruption_macro",
    ]
    labels = [
        "Clean",
        "Bright.",
        "Contr.",
        "Satur.",
        "Gauss.",
        "Shot",
        "Speck.",
        "Macro",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for j, model in enumerate(["raft", "acv"]):
        for metric in ["Abs", "1px", "D1"]:
            values = np.array(
                [
                    float(
                        next(
                            r[metric]
                            for r in tasks
                            if r["model"] == model and r["condition"] == c
                        )
                    )
                    for c in conditions
                ]
            )
            lo, hi = values[:7].min(), values[:7].max()
            scaled = (values - lo) / (hi - lo) if hi > lo else np.zeros_like(values)
            axes[0, j].plot(scaled, "o-", label=metric, markersize=3)
        for proxy in dict.fromkeys(r["proxy"] for r in raw if r["model"] == model):
            values = np.array(
                [
                    float(
                        next(
                            r["response_mean"]
                            for r in raw
                            if r["model"] == model
                            and r["condition"] == c
                            and r["proxy"] == proxy
                        )
                    )
                    for c in conditions
                ]
            )
            lo, hi = values[:7].min(), values[:7].max()
            scaled = (values - lo) / (hi - lo) if hi > lo else np.zeros_like(values)
            axes[1, j].plot(scaled, "o-", label=proxy, markersize=2)
        axes[0, j].set_title("RAFT-Stereo" if model == "raft" else "ACVNet")
        axes[0, j].legend(ncol=3, fontsize=8)
        axes[1, j].legend(
            ncol=2, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18)
        )
        for ax in axes[:, j]:
            ax.set_ylabel("Min–max scaled response")
            ax.set_xticks(range(8))
            ax.set_xticklabels(labels, rotation=20)
            ax.grid(alpha=0.2)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(out / ("condition_response." + ext), dpi=180, bbox_inches="tight")
    plt.close(fig)
