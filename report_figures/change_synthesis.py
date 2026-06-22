"""
Cross-analysis synthesis: WHICH YEARS changed, according to EVERY analysis.

Aggregates the break/boundary years produced across the whole project into one
consensus timeline, so turning points flagged by multiple independent methods
stand out. Sources:

  * K-Means mood clusters     - boundary years (cluster label changes)
  * K-Means genre clusters    - boundary years
  * (valence,energy) change-points + Pettitt        (change_points_results.csv)
  * per-feature change-points + Pettitt             (change_points_features_*.csv)
  * the 5 BERT emotions       - change-points run here on their yearly shares
                                (emotion_trends_results.csv) via the verified core

Outputs (this folder):
  change_synthesis_by_year.csv    - year, vote_count, smoothed(+/-1), which signals
  fig9_change_synthesis.png       - raster (signal x year) + consensus bar
"""

import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from change_points import detect_change_points, pettitt  # verified core

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "csvFiles")
YEARS = list(range(1990, 2023))

# group -> color, for the raster
GROUP_COLOR = {
    "Audio: mood / eras": "#1b9e77",
    "Audio features":     "#7570b3",
    "Genre":              "#d95f02",
    "Lyrics: themes":     "#e7298a",
    "Lyrics: emotions":   "#377eb8",
}


def _read_tsv(folder_glob):
    part = [p for p in glob.glob(folder_glob) if not os.path.basename(p).startswith(".")][0]
    return pd.read_csv(part, sep="\t")


def _cluster_boundaries(folder):
    df = _read_tsv(os.path.join(DATA, "clustering", folder, "year_clusters", "part-*.csv"))
    df = df.sort_values("year").reset_index(drop=True)
    cl, yr = df["cluster"].to_numpy(), df["year"].to_numpy()
    return [int(yr[i]) for i in range(1, len(cl)) if cl[i] != cl[i - 1]]


def _parse_years(s):
    if pd.isna(s) or str(s).strip() == "":
        return []
    return [int(float(x)) for x in str(s).split(";") if str(x).strip() != ""]


def collect_signals():
    """Return list of (name, group, [years])."""
    sig = []

    # --- audio mood / eras ---
    sig.append(("mood K-Means clusters", "Audio: mood / eras",
                _cluster_boundaries("year_mood")))
    cp = pd.read_csv(os.path.join(HERE, "change_points_results.csv"))
    for _, r in cp.iterrows():
        ys = _parse_years(r["break_years"])
        if r["method"] == "optimal_partition_bic":
            sig.append(("valence×energy change-points", "Audio: mood / eras", ys))
        elif r["method"] == "pettitt":
            sig.append((f"{r['target']} (Pettitt)", "Audio: mood / eras", ys))
        # kmeans_cluster_boundaries row is redundant with mood clusters above

    # --- genre ---
    sig.append(("genre K-Means clusters", "Genre", _cluster_boundaries("year_genre")))

    # --- per audio/lyric feature change-points ---
    feat = pd.read_csv(os.path.join(HERE, "change_points_features_results.csv"))
    theme = {"loneliness", "love", "lust"}
    for _, r in feat.iterrows():
        name = r["feature"]
        group = "Lyrics: themes" if name in theme else "Audio features"
        # use reliable BIC breaks; if BIC is trend-dominated, fall back to Pettitt
        ys = _parse_years(r["bic_break_years"]) if r["bic_reliable"] \
            else [int(r["pettitt_break_year"])]
        disp = "love words" if name == "love" else name
        sig.append((f"{disp}{'' if r['bic_reliable'] else ' (Pettitt)'}", group, ys))

    # --- BERT emotions: detect change-points on their yearly shares here ---
    em = pd.read_csv(os.path.join(HERE, "emotion_trends_results.csv"))
    em = em.sort_values("year").reset_index(drop=True)
    yr = em["year"].to_numpy()
    for e in ["joy", "sadness", "anger", "love", "fear"]:
        series = em[f"song_{e}"].to_numpy()
        det = detect_change_points(series, 8)
        ys = [int(yr[s]) for s in det["breaks"]] if det["best_m"] < 8 \
            else [int(yr[pettitt(series)["loc"]])]
        sig.append((f"{e} (emotion)", "Lyrics: emotions", ys))

    return sig


def synthesize():
    signals = collect_signals()

    # exact-year vote counts + a +/-1 smoothed count (detection has ~1y jitter)
    votes = {y: 0 for y in YEARS}
    who = {y: [] for y in YEARS}
    for name, _grp, ys in signals:
        for y in ys:
            if y in votes:
                votes[y] += 1
                who[y].append(name)
    smooth = {y: votes.get(y - 1, 0) + votes[y] + votes.get(y + 1, 0) for y in YEARS}

    rows = [{"year": y, "votes": votes[y], "votes_pm1": smooth[y],
             "signals": "; ".join(who[y])} for y in YEARS]
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(HERE, "change_synthesis_by_year.csv"), index=False)

    print(f"{len(signals)} signals aggregated across all analyses.\n")
    print("=== Years ranked by number of independent analyses flagging a change ===")
    top = out.sort_values(["votes", "votes_pm1"], ascending=False).head(12)
    for _, r in top.iterrows():
        if r["votes"] > 0:
            print(f"  {r['year']}: {r['votes']} direct (±1y: {r['votes_pm1']})  <- {r['signals']}")
    print(f"\nwrote {os.path.join(HERE, 'change_synthesis_by_year.csv')}")

    make_figure(signals, votes, smooth)
    return out


def make_figure(signals, votes, smooth):
    # order rows by group then name; first group on top
    order = list(GROUP_COLOR)
    signals = sorted(signals, key=lambda s: (order.index(s[1]), s[0]))
    names = [s[0] for s in signals]
    n = len(signals)

    fig, (axr, axb) = plt.subplots(
        2, 1, figsize=(13, 9), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})

    # consensus peaks (smoothed) for guide lines
    peak_years = [y for y in YEARS if smooth[y] >= 5]
    for y in peak_years:
        for ax in (axr, axb):
            ax.axvspan(y - 0.5, y + 0.5, color="gold", alpha=0.18, zorder=0)

    # raster
    for row, (name, grp, ys) in enumerate(signals):
        y_pos = n - 1 - row
        axr.scatter(ys, [y_pos] * len(ys), s=130, marker="s",
                    color=GROUP_COLOR[grp], edgecolor="black", linewidth=0.5, zorder=3)
    axr.set_yticks(range(n))
    axr.set_yticklabels(names[::-1], fontsize=8.5)
    axr.set_ylim(-0.5, n - 0.5)
    axr.grid(True, axis="x", linestyle=":", alpha=0.4)
    axr.set_title("Which years changed — every analysis's break years, and their consensus\n"
                  "each square = a detected change/boundary; gold bands = years flagged "
                  "by ≥5 analyses (±1y)", fontsize=12)
    # group legend
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=9,
               markerfacecolor=c, markeredgecolor="black", label=g)
               for g, c in GROUP_COLOR.items()]
    axr.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.005, 1.0),
               fontsize=8.5, title="signal group")

    # consensus bar
    axb.bar(YEARS, [votes[y] for y in YEARS], color="#555555", label="direct votes")
    axb.plot(YEARS, [smooth[y] for y in YEARS], "-o", color="#E41A1C",
             markersize=3, linewidth=1.4, label="±1-year smoothed")
    axb.set_ylabel("# analyses", fontsize=10)
    axb.set_xlabel("Year", fontsize=11)
    axb.grid(True, axis="y", linestyle=":", alpha=0.4)
    axb.legend(loc="upper left", fontsize=8.5)
    axb.set_xticks(range(1990, 2023, 2))
    axb.set_xlim(1989, 2023)
    plt.setp(axb.get_xticklabels(), rotation=45, ha="right")

    out = os.path.join(HERE, "fig9_change_synthesis.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    synthesize()
