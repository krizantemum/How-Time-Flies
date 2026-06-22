"""
Generate publication-quality figures for the How-Love-Flies report,
reading the ACTUAL pipeline output under csvFiles/clustering/.

Figures produced (300 dpi PNG, in this same folder):
  fig1_mood_scatter.png     - K-Means year clusters on (valence, energy) + centroids
  fig2_mood_elbow.png       - WSSSE vs K with Kneedle chord + chosen K
  fig3_genre_elbow.png      - WSSSE vs K for genre-mix clustering + chosen K
  fig4_genre_timeline.png   - year-by-year genre-mix cluster timeline
"""

import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CLU = os.path.join(ROOT, "csvFiles", "clustering")

# Same color-blind-friendly palette the Scala code uses (so the report matches
# the program's own PNGs). RGB -> matplotlib hex.
PALETTE = [
    "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00",
    "#FFD700", "#A65628", "#F781BF", "#40E0D0", "#666666",
]


def _read_single_csv(folder):
    """Spark writes one part-*.csv per folder; read it (tab-separated)."""
    path = glob.glob(os.path.join(folder, "part-*.csv"))[0]
    return pd.read_csv(path, sep="\t")


# ----------------------------------------------------------------------------
# Figure 1 - mood K-Means scatter (valence x energy), colored by cluster
# ----------------------------------------------------------------------------
def fig1_mood_scatter():
    df = _read_single_csv(os.path.join(CLU, "year_mood", "year_clusters"))
    df = df.sort_values("year").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 8))

    # one scatter call per cluster so the legend is clean
    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]
        ax.scatter(sub["avg_valence"], sub["avg_energy"],
                   s=160, color=PALETTE[c % len(PALETTE)],
                   edgecolor="black", linewidth=0.8, zorder=3,
                   label=f"Cluster {c}")
        # plot the cluster centroid (mean of the member years) as a star
        ax.scatter(sub["avg_valence"].mean(), sub["avg_energy"].mean(),
                   marker="*", s=520, color=PALETTE[c % len(PALETTE)],
                   edgecolor="black", linewidth=1.1, zorder=4)

    # faint chronological trajectory so the reader sees time flows 1990 -> 2022
    ax.plot(df["avg_valence"], df["avg_energy"], color="0.6",
            linewidth=1.0, alpha=0.6, zorder=1)

    # year labels
    for _, r in df.iterrows():
        ax.annotate(int(r["year"]),
                    (r["avg_valence"], r["avg_energy"]),
                    textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=7.5, color="0.25")

    ax.set_xlabel("Mean valence (positivity)", fontsize=12)
    ax.set_ylabel("Mean energy", fontsize=12)
    ax.set_title("K-Means clustering of years on (valence, energy)\n"
                 "1990–2022, K=5 auto-selected by elbow; ★ = cluster centroid",
                 fontsize=13)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(title="Mood era", loc="upper right", framealpha=0.95)
    fig.tight_layout()
    out = os.path.join(HERE, "fig1_mood_scatter.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("wrote", out)


# ----------------------------------------------------------------------------
# elbow helper - replicate the Scala pickElbowK (max distance from chord)
# ----------------------------------------------------------------------------
def _kneedle_pick(ks, wssse):
    x1, y1 = ks[0], wssse[0]
    x2, y2 = ks[-1], wssse[-1]
    num = np.abs((y2 - y1) * ks - (x2 - x1) * wssse + (x2 * y1 - x1 * y2))
    return ks[int(np.argmax(num))]


def _elbow_fig(folder, title, out_name):
    elbow = _read_single_csv(os.path.join(folder, "elbow")).sort_values("k")
    ks = elbow["k"].to_numpy(dtype=float)
    w = elbow["wssse"].to_numpy(dtype=float)
    best = int(_kneedle_pick(ks, w))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(ks, w, "o-", color="#377EB8", linewidth=2, markersize=7,
            label="WSSSE (training cost)")
    # the chord between first and last point that the heuristic measures from
    ax.plot([ks[0], ks[-1]], [w[0], w[-1]], "--", color="0.5",
            label="chord (first→last K)")
    # mark the chosen elbow
    by = w[list(ks).index(best)]
    ax.scatter([best], [by], s=260, facecolor="none",
               edgecolor="#E41A1C", linewidth=2.5, zorder=5)
    ax.annotate(f"elbow: K={best}", (best, by),
                textcoords="offset points", xytext=(14, 14),
                fontsize=11, color="#E41A1C", fontweight="bold")

    ax.set_xlabel("Number of clusters K", fontsize=12)
    ax.set_ylabel("WSSSE  (within-cluster sum of squared errors)", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(ks.astype(int))
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(HERE, out_name)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("wrote", out)


# ----------------------------------------------------------------------------
# Figure 4 - genre-mix cluster timeline (one colored cell per year)
# ----------------------------------------------------------------------------
def fig4_genre_timeline():
    df = _read_single_csv(os.path.join(CLU, "year_genre", "year_clusters"))
    df = df.sort_values("year").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(13, 3.2))
    for i, r in df.iterrows():
        c = int(r["cluster"])
        ax.add_patch(plt.Rectangle((i, 0), 1, 1,
                     facecolor=PALETTE[c % len(PALETTE)],
                     edgecolor="black", linewidth=0.6))
        ax.text(i + 0.5, 0.5, str(c), ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")
        ax.text(i + 0.5, -0.18, int(r["year"]), ha="center", va="top",
                fontsize=7.5, rotation=90)

    ax.set_xlim(0, len(df))
    ax.set_ylim(-0.6, 1.05)
    ax.axis("off")
    ax.set_title("Genre-mix K-Means clusters per year (1990–2022, K=5)\n"
                 "cells sharing a color = years with a similar genre composition",
                 fontsize=12)

    handles = [Line2D([0], [0], marker="s", color="w",
               markerfacecolor=PALETTE[c % len(PALETTE)], markersize=11,
               markeredgecolor="black", label=f"Cluster {c}")
               for c in sorted(df["cluster"].unique())]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.05), ncol=len(handles), frameon=False)
    fig.tight_layout()
    out = os.path.join(HERE, "fig4_genre_timeline.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig1_mood_scatter()
    _elbow_fig(os.path.join(CLU, "year_mood"),
               "Elbow method for mood clustering (valence × energy)",
               "fig2_mood_elbow.png")
    _elbow_fig(os.path.join(CLU, "year_genre"),
               "Elbow method for genre-mix clustering",
               "fig3_genre_elbow.png")
    fig4_genre_timeline()
    print("All figures written to", HERE)
