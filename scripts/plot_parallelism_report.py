"""Figures and tables of the DP vs DP x CP report (reports/context_parallel/).

Reads the benchmark lines written by benchmark_parallelism.py and the training
histories of compare_parallelism_training.py, copies them to the report folder
and writes the figures and a tables.md with every number.

    python scripts/plot_parallelism_report.py
"""
import glob
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORT = "reports/context_parallel"
BENCH = {"1node": "results/cp_bench/bench_1node.jsonl", "scaling": "results/cp_bench/bench_scaling.jsonl"}
COMPARE = "results/cp_compare"
GPU_MEM_GB = 64

# chart chrome and the categorical slots, in their fixed order (one color per configuration)
SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CONFIG_COLORS = {
    "DP": "#2a78d6",
    "DP + ckpt": "#eb6834",
    "CP2 Ulysses": "#1baf7a",
    "CP4 Ulysses": "#eda100",
    "CP4 all-gather": "#e87ba4",
    "CP8 Ulysses": "#008300",
}
CP_COLORS = {1: "#2a78d6", 2: "#eb6834", 4: "#1baf7a", 8: "#eda100"}  # scaling: one line per cp size

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.labelsize": 9,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8.5,
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "font.family": "sans-serif", "text.color": INK,
})
MARKER = dict(marker="o", markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.5)


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def label(r):
    if r["cp"] in (0, 1):
        return "DP + ckpt" if r["grad_ckpt"] else "DP"
    name = f"CP{r['cp']} {'Ulysses' if r['backend'] == 'ulysses' else 'all-gather'}"
    return name + (" + ckpt" if r["grad_ckpt"] else "")


def dedup(records):
    """One record per configuration: the successful one if any, else the first failure."""
    out = {}
    for r in records:
        key = (r["tag"], r["nodes"], r["cp"], r.get("backend"), r["grad_ckpt"], r["patch_size"], r["global_batch"])
        if key not in out or (r["status"] == "ok" and out[key]["status"] != "ok"):
            out[key] = r
    return list(out.values())


def fmt(r, field, f="{:.3g}"):
    return f.format(r[field]) if r["status"] == "ok" else r["status"]


def table(records, title):
    rows = [f"### {title}", "",
            "| patch | tokens/car | GPUs | config | cars/group | ckpt | step (s) | cars/s | tokens/s/GPU | TFLOP/s/GPU | MFU | peak mem (GB) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        ok = r["status"] == "ok"
        rows.append("| {} | {:.0f} | {} | {}{} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["patch_size"], r["mean_tokens_per_car"], r["world"], label(r), " (legacy path)" if r["cp"] == 0 else "",
            r["cars_per_group"], "yes" if r["grad_ckpt"] else "no",
            f"{r['step_time_s']:.3f}" if ok else f"**{r['status']}**", f"{r['cars_per_s']:.2f}" if ok else "-",
            f"{r['tokens_per_s_per_gpu'] / 1e3:.0f}k" if ok else "-", f"{r['model_tflops_per_gpu']:.0f}" if ok else "-",
            f"{100 * r['mfu']:.0f}%" if ok else "-", f"{r['peak_mem_alloc_gb']:.1f}" if ok else "-"))
    return "\n".join(rows) + "\n"


def fig_long_sequences(records, path):
    """Peak memory and throughput per GPU vs tokens per car, one line per configuration (1 node)."""
    fig, (ax_mem, ax_tp) = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    ticks = {}
    for k, (name, color) in enumerate(CONFIG_COLORS.items()):
        rs = sorted([r for r in records if label(r) == name], key=lambda r: r["mean_tokens_per_car"])
        if not rs:
            continue
        ok = [r for r in rs if r["status"] == "ok"]
        x = [r["mean_tokens_per_car"] for r in ok]
        ax_mem.plot(x, [r["peak_mem_alloc_gb"] for r in ok], color=color, label=name, **MARKER)
        ax_tp.plot(x, [r["model_tflops_per_gpu"] for r in ok], color=color, label=name, **MARKER)
        for r in rs:
            ticks[r["mean_tokens_per_car"]] = r["patch_size"]
            if r["status"] == "OOM":  # nudged sideways so that coinciding failures stay visible
                ax_mem.plot(r["mean_tokens_per_car"] * (1 + 0.04 * (k - 2)), GPU_MEM_GB, marker="X", markersize=9,
                            color=color, markeredgecolor=SURFACE, markeredgewidth=1.5, linestyle="none")
    ax_mem.axhline(GPU_MEM_GB, color=MUTED, linewidth=1)
    ax_mem.text(0.02, GPU_MEM_GB - 1.5, "H100 memory: 64 GB  (✖ = out of memory)", color=INK2, fontsize=8,
                va="top", transform=ax_mem.get_yaxis_transform())
    ax_mem.set_ylim(0, GPU_MEM_GB + 4)
    for ax in (ax_mem, ax_tp):
        ax.set_xscale("log")
        ax.set_xticks(sorted(ticks), labels=[f"p={ticks[t]}\n{t / 1e3:.0f}k" for t in sorted(ticks)])
        ax.minorticks_off()
        ax.set_xlabel("patch size and tokens per car")
    ax_mem.set_ylabel("peak memory per GPU (GB)")
    ax_mem.set_title("Memory per GPU, one car per group")
    ax_tp.set_ylabel("model TFLOP/s per GPU")
    ax_tp.set_title("Throughput per GPU")
    handles, labels = ax_mem.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_scaling(records, path):
    """Step time vs number of GPUs at a fixed global batch of 8 cars."""
    tags = sorted({r["tag"] for r in records if r["tag"].endswith("_b8")}, key=lambda t: -int(t.split("_")[0][1:]))
    fig, axes = plt.subplots(1, len(tags), figsize=(5 * len(tags), 3.8), constrained_layout=True, squeeze=False)
    for ax, tag in zip(axes[0], tags):
        rs = [r for r in records if r["tag"] == tag and r["status"] == "ok" and (r["backend"] in (None, "ulysses"))]
        best = {}
        for cp in sorted({r["cp"] for r in rs}):
            pts = sorted([r for r in rs if r["cp"] == cp], key=lambda r: r["world"])
            name = "DP" if cp <= 1 else f"DP x CP{cp}"
            ax.plot([r["world"] for r in pts], [r["step_time_s"] for r in pts], color=CP_COLORS.get(cp, INK2),
                    label=name, **MARKER)
            for r in pts:
                best[r["world"]] = min(best.get(r["world"], np.inf), r["step_time_s"])
        if best:
            w0 = min(best)
            ws = np.array(sorted(best))
            ax.plot(ws, best[w0] * w0 / ws, color=MUTED, linewidth=1, linestyle="--", label="ideal")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([4, 8, 16, 32], labels=["4", "8", "16", "32"])
        ax.set_xlabel("GPUs (global batch 8 cars)")
        ax.set_ylabel("step time (s)")
        p = tag.split("_")[0][1:]
        ax.set_title(f"Strong scaling, patch size {p}")
        ax.legend(loc="best")
    fig.savefig(path, dpi=150)
    plt.close(fig)


# the short training runs, by results/cp_compare/<run> folder: label and color (fixed slot order)
RUNS = {
    "cp1_ulysses": ("DP", "#2a78d6"),
    "cp1_ulysses_replicate": ("DP, second run", "#eb6834"),
    "cp2_ulysses": ("CP2 Ulysses", "#1baf7a"),
    "cp4_ulysses": ("CP4 Ulysses", "#eda100"),
    "cp4_allgather": ("CP4 all-gather", "#e87ba4"),
    "cp0_ulysses": ("DP, legacy data path", "#008300"),
}


def fig_training(histories, path, window=50):
    """Loss curves of the short training runs, and their step-by-step difference to DP
    (the legacy-path run draws other batches, so it is only in the left panel)."""
    fig, (ax, ax_d) = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
    ref = np.asarray(histories["cp1_ulysses"]["loss_history"]) if "cp1_ulysses" in histories else None
    kernel = np.ones(window) / window
    for run, (name, color) in RUNS.items():
        if run not in histories:
            continue
        loss = np.asarray(histories[run]["loss_history"])
        ax.plot(np.arange(window, len(loss) + 1), np.convolve(loss, kernel, mode="valid"), color=color,
                label=name, linewidth=1.6)
        if ref is not None and run not in ("cp1_ulysses", "cp0_ulysses"):
            n = min(len(ref), len(loss))
            rel = np.convolve(np.abs(loss[:n] - ref[:n]) / ref[:n], kernel, mode="valid")
            ax_d.plot(np.arange(window, n + 1), 100 * rel, color=color, label=name, linewidth=1.6)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel(f"training loss (normalised MSE, {window}-step mean)")
    ax.set_title("Training loss")
    ax.legend(loc="upper right")
    ax_d.set_yscale("log")
    ax_d.set_xlabel("step")
    ax_d.set_ylabel(f"|loss - loss DP| / loss DP (%, {window}-step mean)")
    ax_d.set_title("Step-by-step difference to DP (same batches)")
    ax_d.legend(loc="lower right")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(os.path.join(REPORT, "data"), exist_ok=True)
    tables = []
    one_node = dedup(load_jsonl(BENCH["1node"]))
    scaling = dedup(load_jsonl(BENCH["scaling"]))
    for name, path in BENCH.items():
        if os.path.exists(path):
            shutil.copy(path, os.path.join(REPORT, "data", os.path.basename(path)))
    if one_node:
        tables.append(table([r for r in one_node if r["tag"].startswith("p16")], "1 node, production model (p=16), global batch 8"))
        long_seq = [r for r in one_node if r["tag"].endswith("_min")]
        tables.append(table(sorted(long_seq, key=lambda r: -r["patch_size"]), "1 node, longer sequences, one car per DP/CP group (1 car per GPU with DP, 1/2 with CP2, 1/4 with CP4)"))
        fig_long_sequences(long_seq, os.path.join(REPORT, "long_sequences.png"))
    if scaling:
        tables.append(table(sorted(scaling, key=lambda r: (r["tag"], r["world"], r["cp"])), "Strong scaling, global batch 8"))
        fig_scaling(scaling, os.path.join(REPORT, "strong_scaling.png"))
    histories = {}
    for path in sorted(glob.glob(os.path.join(COMPARE, "*", "history.json"))):
        run = os.path.basename(os.path.dirname(path))
        with open(path) as f:
            h = json.load(f)
        if run not in RUNS or h.get("steps", 0) < 100:
            continue  # smoke tests
        histories[run] = h
        shutil.copy(path, os.path.join(REPORT, "data", f"history_{run}.json"))
    if histories:
        fig_training(histories, os.path.join(REPORT, "training_curves.png"))
        ref = np.asarray(histories["cp1_ulysses"]["loss_history"]) if "cp1_ulysses" in histories else None
        rows = ["### Short training runs (1 node each, p=16, global batch 8, same seed)", "",
                "| run | steps | step (s) | final loss (mean of last 100) | mean rel. loss difference to DP, first 100 steps / all | val MSE (EMA, 50 cars) |",
                "|---|---|---|---|---|---|"]
        for run, (name, _) in RUNS.items():
            if run not in histories:
                continue
            h = histories[run]
            loss = np.asarray(h["loss_history"])
            diff = "-"
            if ref is not None and run not in ("cp1_ulysses", "cp0_ulysses"):
                n = min(len(ref), len(loss))
                rel = np.abs(loss[:n] - ref[:n]) / ref[:n]
                diff = f"{100 * rel[:100].mean():.3f}% / {100 * rel.mean():.2f}%"
            rows.append(f"| {name} | {h['steps']} | {h['step_time_s']:.3f} | {loss[-100:].mean():.5f} | {diff} | "
                        f"{h['val_mse_ema']:.5f} |")
        tables.append("\n".join(rows) + "\n")
    with open(os.path.join(REPORT, "tables.md"), "w") as f:
        f.write("\n".join(tables))
    print("\n".join(tables))


if __name__ == "__main__":
    main()
