"""
Artistic "painted" stacked-area of the five BERT lyric emotions, 1990-2022.

A companion beauty figure to fig8: it shows how much vertical space each
emotion (joy, sadness, anger, love, fear) occupies each year -- i.e. its share
of songs whose dominant emotion is that class. Because shares are normalized
within each year, every year fills the full height, so the bands read as a
shifting composition rather than as raw counts.

Reads the precomputed song-level shares from emotion_trends_results.csv (so it
does NOT need the Spark parquet checkpoint). Boundaries are spline-smoothed for
an organic, painted look; colours echo the warm->cool palette of
valence_artistic.py.

Output:
  figX_emotion_area_artistic.png

Run:  python report_figures/emotion_area_artistic.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "emotion_trends_results.csv")

# Stacked bottom -> top: warm emotions low, cool emotions high, so the picture
# fades from fiery to cold the same way the valence figure does.
ORDER = ["joy", "love", "anger", "fear", "sadness"]
COLORS = {
    "joy":     "#f4d35e",  # warm gold
    "love":    "#f72585",  # vivid magenta -- pops between gold and red
    "anger":   "#f95738",  # ember red
    "fear":    "#b388ff",  # bright lavender -- pops between red and blue
    "sadness": "#2a5a8c",  # deep blue
}


def load_shares():
    df = pd.read_csv(CSV).sort_values("year").reset_index(drop=True)
    years = df["year"].to_numpy(float)
    shares = {e: df[f"song_{e}"].to_numpy(float) for e in ORDER}
    return years, shares


def _smooth(x, ys, k=24):
    """Spline-smooth one boundary on a fine grid; linear fallback w/o SciPy."""
    xs = np.linspace(x.min(), x.max(), len(x) * k)
    out = np.interp(xs, x, ys)
    try:
        from scipy.interpolate import make_interp_spline
        out = make_interp_spline(x, ys, k=3)(xs)
    except Exception:
        pass
    return xs, np.clip(out, 0, None)


def make_figure(years, shares):
    # cumulative stacked boundaries, smoothed, then renormalized to fill 0..1
    cum = np.cumsum([shares[e] for e in ORDER], axis=0)  # top edge of each band
    xs = None
    smooth_cum = []
    for row in cum:
        xs, s = _smooth(years, row)
        smooth_cum.append(s)
    smooth_cum = np.array(smooth_cum)
    smooth_cum /= smooth_cum[-1]  # force the top band to land exactly at 1.0

    fig, ax = plt.subplots(figsize=(12, 6.4))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    lower = np.zeros_like(xs)
    for i, e in enumerate(ORDER):
        upper = smooth_cum[i]
        ax.fill_between(xs, lower, upper, color=COLORS[e], alpha=0.95,
                        linewidth=0, zorder=2)
        # crisp dark separator along each band's upper edge so thin layers
        # (love, fear) stay distinct from their neighbours
        ax.plot(xs, upper, color="#0d1117", alpha=0.7, lw=1.4, zorder=3)
        ax.plot(xs, upper, color="white", alpha=0.18, lw=0.6, zorder=4)
        # band label placed at its mid-height on the right side
        ymid = (lower[-1] + upper[-1]) / 2
        ax.text(years.max() + 0.4, ymid, e, va="center", ha="left",
                color=COLORS[e], fontsize=11, weight="bold")
        lower = upper

    ax.set_xlim(years.min(), years.max())
    ax.set_ylim(0, 1)

    ax.set_title("How Time Flies — Lyric Emotion Composition, 1990–2022",
                 color="#f0f6fc", fontsize=16, pad=16, loc="left", weight="bold")
    ax.text(0.0, 1.005,
            "share of songs by dominant BERT emotion · each year fills the full height",
            transform=ax.transAxes, color="#8b949e", fontsize=10, ha="left")
    ax.set_xlabel("Year", color="#c9d1d9", fontsize=11)
    ax.set_ylabel("Share of songs", color="#c9d1d9", fontsize=11)

    # every year on the x-axis, rotated to stay readable
    ax.set_xticks(years)
    ax.set_xticklabels([int(y) for y in years], rotation=60, ha="right", fontsize=8)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{int(v*100)}%" for v in np.linspace(0, 1, 6)])

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors="#8b949e")

    # leave room on the right for the band labels
    fig.subplots_adjust(right=0.9)
    fig.tight_layout()
    out = os.path.join(HERE, "figX_emotion_area_artistic.png")
    fig.savefig(out, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    yrs, sh = load_shares()
    make_figure(yrs, sh)
