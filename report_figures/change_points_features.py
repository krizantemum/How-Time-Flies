"""
Fix 3 (extension) - Change-point detection on additional per-year series:
danceability, duration, liveness, loneliness, and love / lust word usage.

Separate, standalone analysis. It REUSES the change-point machinery from
change_points.py (exact optimal-partitioning DP + modified-BIC model
selection, and Pettitt's test) - all already validated there by a brute-force
optimality oracle - and applies it, univariately, to each series.

Note: for a univariate series the optimal partition and the BIC selection are
invariant to affine rescaling of the values (an additive/multiplicative change
shifts every segment's SSE by the same factor), so detection is run on the raw
yearly means; the figure therefore shows interpretable units.

Data sources (each a Spark-written, tab-separated folder under csvFiles/):
  danceabilityData/avg_danceability        durationData/avg_duration_sec
  livenessData/avg_liveness                lonely/avg_val           (LonelinessAnalyzer)
  loveLustYear/avg_love_per_song, avg_lust_per_song                 (LoveAndLustAnalyzer)

Outputs (in this folder):
  change_points_features_results.csv
  fig7_change_points_features.png
"""

import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reuse the verified core (import works because this script lives in the same
# folder, which Python puts on sys.path[0] when run directly).
from change_points import detect_change_points, pettitt, run_self_tests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "csvFiles")

# Search up to 8 change-points. If BIC still prefers the ceiling, the series
# has no well-defined number of abrupt shifts (it is trend/noise dominated and
# the piecewise-constant model just staircases it); we flag that and defer to
# the Pettitt single-break test, which does not over-segment.
CAP = 8

# (display label, folder under csvFiles/, value column, unit for the y-axis)
FEATURES = [
    ("danceability", "danceabilityData", "avg_danceability", ""),
    ("duration",     "durationData",     "avg_duration_sec", "s"),
    ("liveness",     "livenessData",     "avg_liveness",     ""),
    ("loneliness",   "lonely",           "avg_val",          "words/song"),
    ("love",         "loveLustYear",     "avg_love_per_song", "words/song"),
    ("lust",         "loveLustYear",     "avg_lust_per_song", "words/song"),
]


def load_series(folder, value_col):
    part = [p for p in glob.glob(os.path.join(DATA, folder, "part-*.csv"))
            if not os.path.basename(p).startswith(".")][0]
    df = pd.read_csv(part, sep="\t")
    if value_col not in df.columns:
        raise KeyError(f"{value_col!r} not in {folder} (have {list(df.columns)})")
    df = df[["year", value_col]].dropna().sort_values("year").reset_index(drop=True)
    return df["year"].to_numpy(), df[value_col].to_numpy(dtype=float)


def analyze():
    rows = []
    panels = []
    for label, folder, col, unit in FEATURES:
        years, vals = load_series(folder, col)
        det = detect_change_points(vals, CAP)
        breaks = [int(years[s]) for s in det["breaks"]]
        # BIC hitting the search ceiling => no well-defined number of breaks.
        trend_dominated = det["best_m"] >= CAP
        pet = pettitt(vals)
        pet_year = int(years[pet["loc"]])
        pet_sig = pet["p"] < 0.05

        rows.append({
            "feature": label,
            "n_years": len(years),
            "bic_n_change_points": det["best_m"],
            "bic_break_years": ";".join(map(str, breaks)),
            "bic_reliable": not trend_dominated,
            "pettitt_break_year": pet_year,
            "pettitt_K": pet["K"],
            "pettitt_p": pet["p"],
            "pettitt_significant_0.05": pet_sig,
        })
        panels.append((label, unit, years, vals, breaks, pet_year, pet["p"],
                       trend_dominated))

        flag = "  [TREND-DOMINATED: BIC at ceiling, count unreliable]" \
            if trend_dominated else ""
        print(f"[{label}] BIC m={det['best_m']} breaks={breaks} | "
              f"Pettitt {pet_year} (p={pet['p']:.3g}, "
              f"{'sig' if pet_sig else 'n.s.'}){flag}")

    res = pd.DataFrame(rows)
    out_csv = os.path.join(HERE, "change_points_features_results.csv")
    res.to_csv(out_csv, index=False)
    print("\n" + res.to_string(index=False))
    print(f"\nwrote {out_csv}")

    make_figure(panels)
    return res


def make_figure(panels):
    fig, axes = plt.subplots(3, 2, figsize=(13, 9))
    axes = axes.ravel()
    for ax, (label, unit, years, vals, breaks, pet_year, pet_p, trend) in \
            zip(axes, panels):
        ax.plot(years, vals, "-o", color="#377EB8", markersize=3.5, linewidth=1.6)
        # BIC change-points: dashed when a reliable count, faint/dotted-grey when
        # the count is trend-dominated (BIC hit the ceiling) and not to be trusted.
        bic_style = dict(color="0.55", linestyle=(0, (1, 2)), linewidth=1.0) if trend \
            else dict(color="black", linestyle="--", linewidth=1.4)
        for i, by in enumerate(breaks):
            ax.axvline(by - 0.5, label=("BIC change-points" if i == 0 else None),
                       **bic_style)
        # Pettitt single break (significance-tested) - the robust primary result.
        sig = "*" if pet_p < 0.05 else ""
        ax.axvline(pet_year - 0.5, color="#E41A1C", linestyle=":", linewidth=2.0,
                   label=f"Pettitt break {pet_year} (p={pet_p:.1g}{sig})")
        unit_txt = f" ({unit})" if unit else ""
        if trend:
            ttl = f"{label}{unit_txt} — trend-dominated (no reliable break count)"
        else:
            ttl = f"{label}{unit_txt} — {len(breaks)} change-point(s): " \
                  f"{', '.join(map(str, breaks)) if breaks else 'none'}"
        ax.set_title(ttl, fontsize=10.5)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(fontsize=7.5, loc="best")
        ax.set_xlabel("Year", fontsize=9)

    fig.suptitle("Change-point detection on additional per-year series\n"
                 "red dotted = Pettitt single break (primary); black dashed = "
                 "modified-BIC breaks; grey dotted = BIC count unreliable (trend-dominated)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(HERE, "fig7_change_points_features.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    run_self_tests()   # re-verify the shared core before using it
    analyze()
