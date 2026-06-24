"""
Year-by-year evolution of the five BERT lyric emotions (joy, sadness, anger,
love, fear), from the pipeline's final per-song output checkpoints/finalResult.

Two complementary views are produced for each year 1990-2022:
  * song-level   - share of songs whose DOMINANT emotion is each class, computed
                   tie-aware: when a song's top chunk-count is tied between k
                   emotions, each tied emotion gets an equal 1/k vote instead of
                   the whole song going to the first emotion in the list. Every
                   song still contributes total weight 1, so each year's shares
                   sum to 1. (The old single-winner rule biased ties toward joy.)
  * chunk-level  - share of all 8-line lyric chunks classified as each class
                   (sum of the per-song joy..fear counts). Finer-grained; uses
                   every chunk, not just each song's winner.

Because shares are normalized within each year, the corpus growing from ~2.7k
to ~29k songs/year does not bias them (it only tightens later years' variance).

Consistent with trend_tests.py, each song-level emotion share is also tested
for a monotonic trend with Mann-Kendall + Benjamini-Hochberg FDR.

Outputs (in this folder):
  emotion_trends_results.csv   - year x emotion shares (song- and chunk-level)
  fig8_emotion_trends.png      - stacked area + per-emotion line chart
"""

import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from trend_tests import mann_kendall, benjamini_hochberg  # verified core

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FINAL = os.path.join(ROOT, "checkpoints", "finalResult")

EMOTIONS = ["joy", "sadness", "anger", "love", "fear"]
COLORS = {"joy": "#FFB000", "sadness": "#377EB8", "anger": "#E41A1C",
          "love": "#F781BF", "fear": "#7E4FA2"}


def load_final():
    files = glob.glob(os.path.join(FINAL, "*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet under {FINAL} - run the BERT job first.")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"Loaded {len(df):,} songs from {len(files)} part files, "
          f"years {df['year'].min()}-{df['year'].max()}.")
    return df


def tie_aware_weights(df):
    """(len(df) x 5) weight matrix: each song spreads weight 1 equally over the
    k emotions tied for its max chunk-count (1/k each). Songs with no classified
    chunks (all-zero) get zero weight. This removes the joy bias of the
    single-winner argmax, which awarded the whole song to the first tied emotion
    in [joy, sadness, anger, love, fear]."""
    counts = df[EMOTIONS].to_numpy(dtype=float)
    maxes = counts.max(axis=1, keepdims=True)
    winners = (counts == maxes) & (maxes > 0)        # tied maxima
    k = winners.sum(axis=1, keepdims=True)           # how many tied
    return np.divide(winners, k, out=np.zeros_like(counts), where=k > 0)


def shares(df):
    """Return (song_level, chunk_level) DataFrames indexed by year, columns =
    EMOTIONS, each row summing to 1."""
    # song-level: tie-aware 1/k vote per song, summed per year
    w = pd.DataFrame(tie_aware_weights(df), columns=EMOTIONS)
    w["year"] = df["year"].to_numpy()
    song = w.groupby("year")[EMOTIONS].sum()
    song = song.div(song.sum(axis=1), axis=0).sort_index()

    # chunk-level: sum the per-song chunk counts per year
    chunk = df.groupby("year")[EMOTIONS].sum()
    chunk = chunk.div(chunk.sum(axis=1), axis=0).sort_index()

    # sanity: every year's shares sum to 1
    assert np.allclose(song.sum(axis=1), 1.0) and np.allclose(chunk.sum(axis=1), 1.0)
    return song, chunk


def trend_table(song):
    """Mann-Kendall per emotion on the song-level yearly share + BH-FDR."""
    rows = []
    for e in EMOTIONS:
        mk = mann_kendall(song[e].to_numpy())
        rows.append({"emotion": e, "tau": mk["tau"], "Z": mk["Z"], "p_raw": mk["p"],
                     "share_1990": song[e].iloc[0], "share_2022": song[e].iloc[-1],
                     "direction": "increasing" if mk["tau"] > 0 else "decreasing"})
    res = pd.DataFrame(rows)
    res["p_bh"], res["sig_fdr"] = benjamini_hochberg(res["p_raw"].to_numpy(), 0.05)
    return res


def analyze():
    df = load_final()
    song, chunk = shares(df)

    overall = pd.Series(tie_aware_weights(df).sum(axis=0), index=EMOTIONS)
    print("\nOverall dominant-emotion weight (tie-aware 1/k):")
    print((overall.to_frame("songs")
           .assign(pct=lambda d: (d.songs / d.songs.sum() * 100).round(1))).to_string())

    res = trend_table(song)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    print("\n=== Mann-Kendall trend of each emotion's song-level yearly share ===")
    print(res.to_string(index=False))
    print(f"\n{int(res['sig_fdr'].sum())}/5 emotions show a significant monotonic "
          f"trend (FDR alpha=0.05).")

    # persist tidy table
    out = song.add_prefix("song_").join(chunk.add_prefix("chunk_"))
    out.to_csv(os.path.join(HERE, "emotion_trends_results.csv"))
    res.to_csv(os.path.join(HERE, "emotion_trend_tests.csv"), index=False)
    print(f"wrote {os.path.join(HERE, 'emotion_trends_results.csv')}")

    make_figure(song, res)


def make_figure(song, res):
    years = song.index.to_numpy()
    sig = dict(zip(res["emotion"], zip(res["tau"], res["p_bh"], res["sig_fdr"])))

    def lbl(e):
        tau, p_bh, is_sig = sig[e]
        return f"{e}  (τ={tau:+.2f}, p={p_bh:.1e}{'*' if is_sig else ''})"

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 11), sharex=True)

    # Panel A - stacked area: composition of dominant emotions over time.
    ax1.stackplot(years, *[song[e] for e in EMOTIONS],
                  labels=EMOTIONS, colors=[COLORS[e] for e in EMOTIONS], alpha=0.9)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("share of songs", fontsize=11)
    ax1.set_title("Year-by-year evolution of BERT lyric emotions (1990–2022)\n"
                  "A: composition (stacked)   B: joy (dominant)   "
                  "C: sadness/anger/love/fear — with Mann–Kendall trend", fontsize=12)
    ax1.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), title="emotion")
    ax1.margins(x=0)

    # Panel B - joy on its own scale (its ~58% share compresses the rest).
    ax2.plot(years, song["joy"], "-o", color=COLORS["joy"], markersize=3.5,
             linewidth=1.8, label=lbl("joy"))
    ax2.set_ylabel("share of songs", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
               title="* = significant (FDR)")
    ax2.margins(x=0)

    # Panel C - the four lower-share emotions together on one scale.
    for e in ("sadness", "anger", "love", "fear"):
        ax3.plot(years, song[e], "-o", color=COLORS[e], markersize=3.5,
                 linewidth=1.8, label=lbl(e))
    ax3.set_ylabel("share of songs", fontsize=11)
    ax3.set_xlabel("Year", fontsize=11)
    ax3.grid(True, linestyle=":", alpha=0.5)
    ax3.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
               title="* = significant (FDR)")
    ax3.margins(x=0)

    fig.tight_layout()
    out = os.path.join(HERE, "fig8_emotion_trends.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    analyze()
