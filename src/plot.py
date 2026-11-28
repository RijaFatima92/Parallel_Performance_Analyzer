"""Reads results/metrics.csv and writes figures/*.png (title, axis labels w/ units, legend)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "metrics.csv")
FIG = os.path.join(ROOT, "figures")

COL = {"time": "T_P", "speedup": "speedup", "eff": "efficiency", "overhead": "overhead"}
TITLES = {"time": ("Execution time vs. p", "Parallel time T_P (seconds)"),
          "speedup": ("Speedup vs. p", "Speedup S = T_S / T_P (dimensionless)"),
          "eff": ("Efficiency vs. p", "Efficiency E = S / p (dimensionless)"),
          "overhead": ("Total overhead T_o vs. p", "Total overhead T_o (seconds)")}


def draw(ax, m, which):
    ps = sorted(m.p.unique())
    for n in sorted(m.n.unique()):
        d = m[m.n == n].sort_values("p")
        ax.plot(d.p, d[COL[which]], "o-", label=f"n = {n:.0e}")
    if which == "speedup":
        ax.plot(ps, ps, "k--", label="Ideal linear speedup")
    if which == "eff":
        ax.axhline(1.0, color="gray", ls=":", label="Ideal E = 1")
    ax.set_title(TITLES[which][0]); ax.set_ylabel(TITLES[which][1])
    ax.set_xlabel("Number of processes p (count)")
    ax.set_xticks(ps); ax.grid(alpha=.3); ax.legend()


def main():
    os.makedirs(FIG, exist_ok=True)
    m = pd.read_csv(CSV)
    names = [("time", "1_time_vs_p.png"), ("speedup", "2_speedup_vs_p.png"),
             ("eff", "3_efficiency_vs_p.png"), ("overhead", "4_overhead_vs_p.png")]
    for which, fname in names:
        fig, ax = plt.subplots(figsize=(6, 4))
        draw(ax, m, which)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (which, _) in zip(axes.ravel(), names):
        draw(ax, m, which)
    fig.suptitle("Monte-Carlo pi: Parallel Performance Dashboard", fontsize=14)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "5_dashboard.png"), dpi=150)
    plt.close(fig)
    print(f"Saved figures -> {FIG}")


if __name__ == "__main__":
    main()
