#!/usr/bin/env python3
"""Plot GPU time (%) for force kernels vs system size N, split by pin mode."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

CSV = Path(__file__).parent / "results/pin_scaling/gpu_pct.csv"
OUT = Path(__file__).parent / "results/pin_scaling/gpu_pct.pdf"

df = pd.read_csv(CSV)
df["N"] = df["N"].astype(int)
df = df.sort_values("N")

fig, (ax_naive, ax_cll) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

MARKERS = {"force_ss_k": "o", "force_pin_k": "s", "force_pin_cll_k": "^"}
COLORS  = {"force_ss_k": "#1f77b4", "force_pin_k": "#d62728", "force_pin_cll_k": "#2ca02c"}
LABELS  = {"force_ss_k": r"force\_ss\_k  (O(N²))",
           "force_pin_k": r"force\_pin\_k  (O(N·N$_p$))",
           "force_pin_cll_k": r"force\_pin\_cll\_k  (O(N))"}

for ax, mode, pin_kernel in [
    (ax_naive, "naive", "force_pin_k"),
    (ax_cll,   "cll",   "force_pin_cll_k"),
]:
    sub = df[df["mode"] == mode]

    for kernel in ["force_ss_k", pin_kernel]:
        kdf = sub[sub["kernel"] == kernel].sort_values("N")
        ax.plot(kdf["N"], kdf["gpu_pct"],
                marker=MARKERS[kernel], color=COLORS[kernel],
                linewidth=1.8, markersize=6,
                label=LABELS[kernel])

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_xlabel("N (skyrmions)", fontsize=11)
    ax.set_ylabel("GPU time (%)", fontsize=11)
    ax.set_title(f"Pin mode: {mode}", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.grid(True, which="major", linestyle="--", alpha=0.5)
    ax.grid(True, which="minor", linestyle=":", alpha=0.3)
    ax.legend(fontsize=9, loc="center right")
    ax.set_xscale("log", base=2)

fig.suptitle("GPU time share per kernel vs system size", fontsize=13, y=1.01)
# fig.tight_layout()
# fig.savefig(OUT, bbox_inches="tight")
# print(f"Saved: {OUT}")
plt.show()
