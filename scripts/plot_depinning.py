#!/usr/bin/env python3
"""Plot vx_avg vs F_D from skyrmion simulation summary files."""

import sys
import re
import numpy as np
import matplotlib.pyplot as plt


def parse_summary(path):
    data = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) == 2:
                data[parts[0]] = float(parts[1])
    return data


def main(files):
    records = []
    for path in files:
        d = parse_summary(path)
        if "F_D" not in d or "vx_avg" not in d:
            print(f"Warning: missing F_D or vx_avg in {path}", file=sys.stderr)
            continue
        records.append((d["F_D"], d["vx_avg"]))

    if not records:
        sys.exit("No valid data found.")

    records.sort(key=lambda r: r[0])
    F_D, vx_avg = zip(*records)

    fig, ax = plt.subplots()
    ax.plot(F_D, vx_avg, "o-", markersize=4)
    ax.set_xlabel(r"$F_D$")
    ax.set_ylabel(r"$\langle v_x \rangle$")
    ax.set_title("Depinning curve")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    # plt.savefig("depinning_vx.png", dpi=150)
    # print("Saved depinning_vx.png")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: plot_depinning.py <summary1.dat> [summary2.dat ...]")
    main(sys.argv[1:])
