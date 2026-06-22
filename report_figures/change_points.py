"""
Fix 3 - Change-point detection on the year-level mood trajectory.

This is a SEPARATE, standalone analysis. It does not modify or replace the
K-Means year clustering; it complements it. The argument for the report:
two independent methods - unsupervised clustering (YearMoodClusterer) and
statistical change-point detection (here) - are run on the same year-level
(valence, energy) trajectory. Where they agree on a boundary year, the "era"
is corroborated and is not an artifact of either method.

Methods (implemented from first principles in NumPy, auditable for the report):
  * Optimal partitioning (exact dynamic program) with an L2 / Gaussian
    mean-shift cost, for 0..M change-points. Number of change-points selected
    by BIC. This is the exact version of PELT/ruptures' model="l2" - exact is
    fine because n = 33.
  * Pettitt's test - a non-parametric test for a single dominant change-point,
    returning its location and an approximate p-value (complements the
    Mann-Kendall trend test in trend_tests.py).

Verification (run_self_tests, runs every time before any result):
  * the DP optimum is checked against a brute-force search over all
    segmentations on small series (exact-optimality oracle);
  * known single/double mean-shifts are recovered; BIC picks 0 on pure noise;
  * Pettitt is checked on a clear shift vs flat noise.

Outputs (in this folder):
  change_points_results.csv     - detected break years + Pettitt per feature
  fig6_change_points.png        - valence & energy over time, with detected
                                  change-points and the K-Means cluster bands
"""

import glob
import itertools
import math
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "csvFiles")
MAX_CP = 5  # search 0..5 change-points (>= the 4-ish boundaries clustering implies)

PALETTE = [
    "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00",
    "#FFD700", "#A65628", "#F781BF", "#40E0D0", "#666666",
]


# ---------------------------------------------------------------------------
# Core: optimal partitioning with an L2 (Gaussian mean-shift) cost
# ---------------------------------------------------------------------------
def _sse_factory(x):
    """Return seg_sse(a, b) = total within-segment SSE of x[a:b] across all
    columns, in O(1) via prefix sums. x is (n, d)."""
    x = np.atleast_2d(x.T).T if x.ndim == 1 else x
    n, _ = x.shape
    p1 = np.vstack([np.zeros(x.shape[1]), np.cumsum(x, axis=0)])       # sum
    p2 = np.vstack([np.zeros(x.shape[1]), np.cumsum(x ** 2, axis=0)])  # sum of squares

    def seg_sse(a, b):
        m = b - a
        s1 = p1[b] - p1[a]
        s2 = p2[b] - p2[a]
        return float(np.sum(s2 - (s1 ** 2) / m))  # SSE about the segment mean

    return seg_sse, n


def optimal_partition(x, max_cp):
    """
    Exact optimal partitioning. Returns, for each m in 0..max_cp:
      total_sse[m]      - minimal total within-segment SSE using exactly m
                          change-points (m+1 segments)
      breakpoints[m]    - list of segment-start indices (the change-point
                          positions; a value s means a new segment starts at s)
    """
    seg_sse, n = _sse_factory(np.asarray(x, dtype=float))
    max_cp = min(max_cp, n - 1)

    INF = float("inf")
    # C[j][b] = min cost to partition x[0:b] using exactly j change-points.
    C = [[INF] * (n + 1) for _ in range(max_cp + 1)]
    prev = [[-1] * (n + 1) for _ in range(max_cp + 1)]
    for b in range(1, n + 1):
        C[0][b] = seg_sse(0, b)
    for j in range(1, max_cp + 1):
        for b in range(j + 1, n + 1):           # need >= j+1 points for j+1 segments
            for s in range(j, b):               # last segment is [s, b), each prior >= 1
                v = C[j - 1][s] + seg_sse(s, b)
                if v < C[j][b]:
                    C[j][b] = v
                    prev[j][b] = s

    total_sse, breakpoints = [], []
    for m in range(max_cp + 1):
        total_sse.append(C[m][n])
        bks, b, j = [], n, m
        while j > 0:
            s = prev[j][b]
            bks.append(s)
            b, j = s, j - 1
        breakpoints.append(sorted(bks))
    return total_sse, breakpoints


def select_by_bic(total_sse, n, d):
    """
    Pick the number of change-points by a modified BIC for a piecewise-constant
    Gaussian mean model with a pooled variance. Crucially the change-point
    LOCATIONS are counted as free parameters (~2 d.f. per change-point), which
    is what stops vanilla BIC from over-segmenting noise on small n
    (modified BIC, Zhang & Siegmund 2007). Up to constants in m,
        BIC(m) = n*d*log(SSE_m/(n*d)) + ((m+1)*d + m + 1)*log(n*d).
    Returns (best_m, bic_array).
    """
    nd = n * d
    bic = []
    for m, sse in enumerate(total_sse):
        # (m+1)*d segment means + m change-point locations + 1 variance
        k = (m + 1) * d + m + 1
        bic.append(nd * math.log(max(sse, 1e-12) / nd) + k * math.log(nd))
    return int(np.argmin(bic)), np.array(bic)


def detect_change_points(x, max_cp=MAX_CP):
    """Full pipeline: returns best_m, its breakpoints, all breakpoints, BIC."""
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    d = 1 if x.ndim == 1 else x.shape[1]
    total_sse, breakpoints = optimal_partition(x, max_cp)
    best_m, bic = select_by_bic(total_sse, n, d)
    return {"best_m": best_m, "breaks": breakpoints[best_m],
            "all_breaks": breakpoints, "bic": bic, "total_sse": total_sse}


# ---------------------------------------------------------------------------
# Pettitt's test - single change-point with an approximate p-value
# ---------------------------------------------------------------------------
def pettitt(x):
    x = np.asarray(x, dtype=float)
    n = x.size
    # U_t = sum_{i<=t} sum_{j>t} sign(x_i - x_j); built via a running rank sum.
    u = np.empty(n - 1)
    for t in range(1, n):
        diff = np.sign(x[:t][:, None] - x[t:][None, :])
        u[t - 1] = diff.sum()
    k_idx = int(np.argmax(np.abs(u)))           # 0-based position t-1
    k = abs(u[k_idx])
    p = 2.0 * math.exp(-6.0 * k * k / (n ** 3 + n ** 2))
    return {"loc": k_idx + 1, "K": float(k), "p": min(1.0, p)}


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def _brute_force_sse(x, m):
    """Minimal total SSE over ALL segmentations with exactly m change-points
    (segments non-empty). Reference oracle for the DP. O(C(n-1, m)) - small n only."""
    seg_sse, n = _sse_factory(np.asarray(x, dtype=float))
    best = float("inf")
    for combo in itertools.combinations(range(1, n), m):
        bounds = [0, *combo, n]
        total = sum(seg_sse(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1))
        best = min(best, total)
    return best


def run_self_tests():
    print("=== self-tests ===")
    rng = np.random.default_rng(7)

    # 1. DP optimum == brute-force optimum on small random series, for m=1,2,3.
    for _ in range(60):
        x = rng.normal(size=(9, rng.integers(1, 3)))
        total_sse, _ = optimal_partition(x, 3)
        for m in (1, 2, 3):
            assert abs(total_sse[m] - _brute_force_sse(x, m)) < 1e-9
    print("[PASS] DP total-SSE matches brute-force optimum (m=1,2,3)")

    # 2. Single clean mean-shift at index 16 is recovered and BIC selects m=1.
    x = np.r_[rng.normal(0, 0.05, 16), rng.normal(5, 0.05, 17)]
    det = detect_change_points(x, MAX_CP)
    assert det["best_m"] == 1 and det["breaks"] == [16]
    print("[PASS] single mean-shift recovered; BIC picks m=1 at the true break")

    # 3. Two shifts recovered when constrained to m=2.
    x = np.r_[rng.normal(0, 0.05, 10), rng.normal(4, 0.05, 10), rng.normal(-2, 0.05, 13)]
    _, bks = optimal_partition(x, 3)
    assert bks[2] == [10, 20]
    print("[PASS] two mean-shifts recovered at the true breaks")

    # 4. Stationary noise -> BIC resists over-segmentation. A single draw can
    #    occasionally yield a spurious break, so validate the property
    #    statistically: the large majority of null series select 0 breaks.
    ms = np.array([detect_change_points(rng.normal(0, 1.0, 40), MAX_CP)["best_m"]
                   for _ in range(200)])
    assert (ms == 0).mean() >= 0.75 and ms.mean() < 0.5
    print(f"[PASS] BIC resists over-segmentation on noise "
          f"({(ms == 0).mean():.0%} of null series pick m=0)")

    # 5. Pettitt: clear shift -> small p at right place; flat noise -> large p.
    shift = np.r_[rng.normal(0, 0.1, 20), rng.normal(3, 0.1, 20)]
    rp = pettitt(shift)
    assert rp["p"] < 0.01 and abs(rp["loc"] - 20) <= 1
    assert pettitt(rng.normal(0, 1, 40))["p"] > 0.2
    print("[PASS] Pettitt: detects clear shift (p<0.01), null on noise (p>0.2)")

    # 6. Optional ruptures cross-check, if installed.
    try:
        import ruptures as rpt
        for _ in range(20):
            x = np.r_[rng.normal(0, .3, 15), rng.normal(3, .3, 18)].reshape(-1, 1)
            mine = optimal_partition(x, 1)[1][1]
            theirs = rpt.Dynp(model="l2", min_size=1, jump=1).fit(x).predict(n_bkps=1)
            assert mine == [theirs[0]]
        print("[PASS] breakpoints match ruptures.Dynp(model='l2')")
    except ImportError:
        print("[skip] ruptures not installed - DP validated by brute force above")

    print("=== all self-tests passed ===\n")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _read_feature(feature):
    folder = os.path.join(DATA, f"{feature}Data")
    part = [p for p in glob.glob(os.path.join(folder, "part-*.csv"))
            if not os.path.basename(p).startswith(".")][0]
    df = pd.read_csv(part, sep="\t").sort_values("year").reset_index(drop=True)
    col = next(c for c in df.columns if c.startswith("avg_"))
    return df.rename(columns={col: feature})[["year", feature]]


def _read_mood_clusters():
    folder = os.path.join(DATA, "clustering", "year_mood", "year_clusters")
    part = [p for p in glob.glob(os.path.join(folder, "part-*.csv"))
            if not os.path.basename(p).startswith(".")][0]
    return pd.read_csv(part, sep="\t").sort_values("year").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyze():
    val = _read_feature("valence")
    eng = _read_feature("energy")
    df = val.merge(eng, on="year").sort_values("year").reset_index(drop=True)
    years = df["year"].to_numpy()
    n = len(years)

    # Standardize each feature (z-score) so the bivariate L2 cost weights
    # valence and energy equally - mirrors the StandardScaler the clusterer uses.
    z = df[["valence", "energy"]].to_numpy(dtype=float)
    z = (z - z.mean(axis=0)) / z.std(axis=0, ddof=0)

    det = detect_change_points(z, MAX_CP)
    break_years = [int(years[s]) for s in det["breaks"]]
    print("=== Change-point detection on standardized (valence, energy) ===")
    print(f"BIC-selected change-points: {det['best_m']}  -> break year(s): {break_years}")
    print(f"BIC by m (0..{MAX_CP}): " +
          ", ".join(f"{m}:{b:.1f}" for m, b in enumerate(det['bic'])))
    for m in range(1, MAX_CP + 1):
        ys = [int(years[s]) for s in det["all_breaks"][m]]
        print(f"   if m={m}: breaks at {ys}")

    # Pettitt per single feature (significance-tested single break).
    pet = {f: pettitt(df[f].to_numpy()) for f in ("valence", "energy")}
    for f, r in pet.items():
        print(f"Pettitt[{f}]: break at {int(years[r['loc']])} "
              f"(K={r['K']:.0f}, p={r['p']:.3g})")

    # Compare to the K-Means mood-cluster boundaries (years where the cluster
    # label changes between consecutive years).
    clusters = _read_mood_clusters()
    cl = clusters["cluster"].to_numpy()
    cyears = clusters["year"].to_numpy()
    cluster_boundaries = [int(cyears[i]) for i in range(1, len(cl)) if cl[i] != cl[i - 1]]
    print(f"\nK-Means cluster boundary years (unchanged, for comparison): "
          f"{cluster_boundaries}")
    agree = sorted(set(by for by in break_years
                       for cb in cluster_boundaries if abs(by - cb) <= 1))
    print(f"Change-points within +/-1 yr of a cluster boundary: {agree}")

    # Persist a tidy results table.
    rows = [{"method": "optimal_partition_bic", "target": "valence+energy",
             "n_change_points": det["best_m"],
             "break_years": ";".join(map(str, break_years)), "p_value": ""}]
    for f, r in pet.items():
        rows.append({"method": "pettitt", "target": f, "n_change_points": 1,
                     "break_years": str(int(years[r["loc"]])), "p_value": r["p"]})
    rows.append({"method": "kmeans_cluster_boundaries", "target": "valence+energy",
                 "n_change_points": len(cluster_boundaries),
                 "break_years": ";".join(map(str, cluster_boundaries)), "p_value": ""})
    out_csv = os.path.join(HERE, "change_points_results.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")

    make_figure(df, years, break_years, clusters)
    return det, pet, cluster_boundaries


def make_figure(df, years, break_years, clusters):
    """Valence & energy over time; K-Means cluster bands as background;
    detected change-points as vertical lines (the corroboration figure)."""
    cl = clusters["cluster"].to_numpy()
    cyears = clusters["year"].to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, feat, color in zip(axes, ("valence", "energy"), ("#1b9e77", "#d95f02")):
        # cluster bands behind the line
        for i in range(len(cyears)):
            lo = cyears[i] - 0.5
            ax.axvspan(lo, lo + 1, color=PALETTE[int(cl[i]) % len(PALETTE)], alpha=0.12)
        ax.plot(df["year"], df[feat], "-o", color=color, markersize=4,
                linewidth=1.8, label=f"mean {feat}")
        for by in break_years:
            ax.axvline(by - 0.5, color="black", linestyle="--", linewidth=1.6)
        ax.set_ylabel(f"mean {feat}", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(loc="best", fontsize=9)

    cp_txt = ", ".join(map(str, break_years))
    axes[0].set_title(
        "Change-point detection vs K-Means eras (valence × energy)\n"
        f"dashed = detected change-point(s) at {cp_txt}; "
        "color bands = K-Means mood clusters", fontsize=12)
    axes[1].set_xlabel("Year", fontsize=11)
    fig.tight_layout()
    out = os.path.join(HERE, "fig6_change_points.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    run_self_tests()
    analyze()
