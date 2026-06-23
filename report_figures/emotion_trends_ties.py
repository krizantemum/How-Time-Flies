"""
Tie-aware BERT lyric-emotion trends, 1990-2022.

The main pipeline's dominant-emotion rule breaks ties arbitrarily: a song with
joy=2, love=2 (or joy=2, anger=2, love=2) is credited to whichever emotion
comes first in the list. This script recomputes the per-year emotion
composition from the SAME per-song chunk counts saved in
checkpoints/finalResult (so BERT is NOT re-run), but when several emotions tie
for a song's maximum it gives each tied emotion an equal fractional vote
(1 / number-tied).

Because every song still contributes a total weight of exactly 1, each year's
five shares still sum to 1 and stay directly comparable to
emotion_trends_results.csv (the single-winner version).

Outputs (in this folder):
  emotion_trends_ties_results.csv   - year x emotion shares (tie-aware)
  figX_emotion_trends_ties.png      - tie-aware composition vs. single-winner

Run:  python report_figures/emotion_trends_ties.py
"""

import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FINAL = os.path.join(ROOT, "checkpoints", "finalResult")

EMOTIONS = ["joy", "sadness", "anger", "love", "fear"]
COLORS = {"joy": "#f4d35e", "love": "#f72585", "anger": "#f95738",
          "fear": "#b388ff", "sadness": "#2a5a8c"}
YEAR_LO, YEAR_HI = 1990, 2022


def load_final():
    files = glob.glob(os.path.join(FINAL, "*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet under {FINAL} - run the BERT job first.")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[(df["year"] >= YEAR_LO) & (df["year"] <= YEAR_HI)]
    print(f"Loaded {len(df):,} songs, years {df['year'].min()}-{df['year'].max()}.")
    return df


def tie_aware_weights(df):
    """Return a (len(df) x 5) weight matrix: for each song, equal fractional
    weight 1/k across the k emotions tied for that song's max chunk count.
    Songs with no classified chunks (all zero) get zero weight everywhere."""
    counts = df[EMOTIONS].to_numpy(dtype=float)
    maxes = counts.max(axis=1, keepdims=True)
    winners = (counts == maxes) & (maxes > 0)        # boolean mask of tied maxima
    k = winners.sum(axis=1, keepdims=True)           # how many emotions tied
    weights = np.divide(winners, k, out=np.zeros_like(counts), where=k > 0)  # 1/k each
    return weights


def yearly_shares(df, weights):
    """Year-by-year shares from the tie-aware weights (each year sums to 1)."""
    w = pd.DataFrame(weights, columns=EMOTIONS)
    w["year"] = df["year"].to_numpy()
    by_year = w.groupby("year")[EMOTIONS].sum()
    shares = by_year.div(by_year.sum(axis=1), axis=0).sort_index()
    assert np.allclose(shares.sum(axis=1), 1.0)
    return shares


def make_figure(shares):
    years = shares.index.to_numpy()
    order = ["joy", "love", "anger", "fear", "sadness"]

    fig, ax = plt.subplots(figsize=(12, 6.2))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # Draw bands manually so each gets an edge. Thin bands (love, fear) get a
    # bright outline so they don't blend into their neighbours; the rest get a
    # subtle dark separator.
    lower = np.zeros_like(years, dtype=float)
    for e in order:
        upper = lower + shares[e].to_numpy()
        ax.fill_between(years, lower, upper, color=COLORS[e], alpha=0.95,
                        linewidth=0, zorder=2)
        if e in ("love", "fear"):
            ax.plot(years, upper, color="white", lw=1.6, alpha=0.95, zorder=4)
            ax.plot(years, lower, color="white", lw=1.6, alpha=0.95, zorder=4)
        else:
            ax.plot(years, upper, color="#0d1117", lw=1.0, alpha=0.6, zorder=3)
        lower = upper

    ax.set_xlim(years.min(), years.max())
    ax.set_ylim(0, 1)
    ax.set_title("Tie-aware Lyric Emotion Composition, 1990–2022",
                 color="#f0f6fc", fontsize=15, pad=14, loc="left", weight="bold")
    ax.text(0.0, 1.005, "ties split equally across the tied emotions · shares sum to 1",
            transform=ax.transAxes, color="#8b949e", fontsize=10)
    ax.set_xlabel("Year", color="#c9d1d9")
    ax.set_ylabel("Share of songs (tie-aware)", color="#c9d1d9")
    ax.set_xticks(years)
    ax.set_xticklabels([int(y) for y in years], rotation=60, ha="right", fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors="#8b949e")
    # right-side band labels
    lower = 0.0
    for e in order:
        mid = lower + shares[e].iloc[-1] / 2
        ax.text(years.max() + 0.3, mid, e, color=COLORS[e], va="center",
                ha="left", fontsize=11, weight="bold")
        lower += shares[e].iloc[-1]
    fig.subplots_adjust(right=0.9)
    fig.tight_layout()
    out = os.path.join(HERE, "figX_emotion_trends_ties.png")
    fig.savefig(out, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {out}")


def make_individual(shares):
    """One artistic line chart per emotion, each on its own y-scale so the
    trend is readable. 'joy' is titled 'happiness'."""
    years = shares.index.to_numpy()
    titles = {"joy": "happiness", "sadness": "sadness", "anger": "anger",
              "love": "love", "fear": "fear"}
    for e, nice in titles.items():
        y = shares[e].to_numpy()
        c = COLORS[e]
        fig, ax = plt.subplots(figsize=(11, 5.2))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")

        # soft fill under the line + glow + bright core
        ax.fill_between(years, y, y.min(), color=c, alpha=0.18, zorder=1)
        for lw, a in [(12, 0.05), (7, 0.09), (4, 0.16)]:
            ax.plot(years, y, color=c, lw=lw, alpha=a,
                    solid_capstyle="round", zorder=2)
        ax.plot(years, y, color=c, lw=2.0, alpha=0.98, zorder=3)
        ax.scatter(years, y, color=c, s=28, edgecolor="white",
                   linewidth=0.6, zorder=4)

        pad = (y.max() - y.min()) * 0.15 + 1e-6
        ax.set_xlim(years.min(), years.max())
        ax.set_ylim(y.min() - pad, y.max() + pad)
        ax.set_title(f"How Time Flies — {nice.capitalize()} in Lyrics, 1990–2022",
                     color="#f0f6fc", fontsize=15, pad=14, loc="left", weight="bold")
        ax.set_xlabel("Year", color="#c9d1d9")
        ax.set_ylabel(f"{nice} share of songs", color="#c9d1d9")
        ax.set_xticks(years)
        ax.set_xticklabels([int(yr) for yr in years], rotation=60, ha="right",
                           fontsize=8)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v*100:.1f}%")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors="#8b949e")
        ax.grid(True, axis="y", linestyle=":", color="#30363d", alpha=0.6)

        fig.tight_layout()
        out = os.path.join(HERE, f"figX_emotion_{e}.png")
        fig.savefig(out, dpi=300, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"wrote {out}")


def main():
    df = load_final()

    # how often do ties actually happen?
    counts = df[EMOTIONS].to_numpy(float)
    maxes = counts.max(axis=1, keepdims=True)
    k = ((counts == maxes) & (maxes > 0)).sum(axis=1)
    n_tie = int((k > 1).sum())
    n_empty = int((maxes.ravel() == 0).sum())
    print(f"songs with a tie: {n_tie:,} ({n_tie / len(df) * 100:.1f}%)  | "
          f"unclassified (all-zero): {n_empty:,}")
    print("tie multiplicity counts:",
          {int(m): int((k == m).sum()) for m in sorted(set(k)) if m > 1})

    weights = tie_aware_weights(df)
    shares = yearly_shares(df, weights)

    out_csv = os.path.join(HERE, "emotion_trends_ties_results.csv")
    shares.add_prefix("tieaware_").to_csv(out_csv)
    print(f"wrote {out_csv}")

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n=== Tie-aware yearly shares ===")
    print(shares.to_string())

    make_figure(shares)
    make_individual(shares)


if __name__ == "__main__":
    main()
