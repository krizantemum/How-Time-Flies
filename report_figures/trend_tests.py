"""
Fix 2 - Statistical trend testing of the per-year audio-feature series.

For each of the 10 features that the Spark pipeline writes to
csvFiles/<feature>Data/, this script tests whether the yearly mean has a
statistically significant monotonic trend over 1990-2022, and by how much:

  * Mann-Kendall test  -> S, Z, Kendall's tau, two-sided p-value
                          (non-parametric, no normality assumption)
  * Sen's slope + 95% CI -> trend magnitude in feature-units per year
  * Benjamini-Hochberg  -> FDR correction across the 10 simultaneous tests

The core estimators are implemented from first principles in NumPy so the
script is dependency-light and the math is auditable for the report. SciPy /
statsmodels, if installed, are used ONLY as an independent oracle inside the
self-tests (run_self_tests) - they are never required to produce results.

Outputs (in this folder):
  trend_test_results.csv        - one row per feature, full statistics
  fig5_feature_trend_tests.png  - Kendall's tau per feature, flagged by FDR

Run:  python report_figures/trend_tests.py
"""

import glob
import math
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FEATURE_GLOB = os.path.join(ROOT, "csvFiles", "*Data")
ALPHA = 0.05
Z_95 = 1.959963984540054  # standard-normal 0.975 quantile


# ---------------------------------------------------------------------------
# Core statistics (pure NumPy)
# ---------------------------------------------------------------------------
def _normal_sf(z):
    """Upper-tail survival function of N(0,1) via the stdlib error function."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _tie_correction(values):
    """Sum of t(t-1)(2t+5) over tied value-groups; 0 when all values distinct."""
    _, counts = np.unique(values, return_counts=True)
    tied = counts[counts > 1]
    return float(np.sum(tied * (tied - 1) * (2 * tied + 5)))


def mann_kendall(values):
    """
    Mann-Kendall monotonic-trend test on a time-ordered 1-D series.

    S    = sum_{i<j} sign(x_j - x_i)
    Var  = [n(n-1)(2n+5) - sum t(t-1)(2t+5)] / 18   (tie-corrected)
    Z    = (S-1)/sqrt(Var) if S>0; (S+1)/sqrt(Var) if S<0; else 0  (continuity)
    p    = 2 * P(N(0,1) > |Z|)
    tau  = S / sqrt((n0-n1)(n0-n2))   (Kendall tau-b; n2=0 for distinct time)
    """
    x = np.asarray(values, dtype=float)
    n = x.size
    if n < 3:
        raise ValueError("Mann-Kendall needs at least 3 points")

    # S via the sign of every pairwise difference (upper triangle).
    diff = x[None, :] - x[:, None]
    s = float(np.sum(np.sign(diff[np.triu_indices(n, k=1)])))

    var_s = (n * (n - 1) * (2 * n + 5) - _tie_correction(x)) / 18.0

    if s > 0:
        z = (s - 1) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / math.sqrt(var_s)
    else:
        z = 0.0
    p = 2.0 * _normal_sf(abs(z))

    n0 = n * (n - 1) / 2.0
    _, counts = np.unique(x, return_counts=True)
    n1 = float(np.sum(counts * (counts - 1) / 2.0))  # ties in values
    tau = s / math.sqrt((n0 - n1) * n0)              # n2 = 0 (years distinct)

    return {"n": n, "S": s, "var_S": var_s, "Z": z, "p": p, "tau": tau}


def sens_slope(times, values, alpha=0.05):
    """
    Sen's (Theil-Sen) slope: median of all pairwise slopes, plus the
    rank-based 100(1-alpha)% confidence interval (Gilbert 1987), expressed
    in feature-units per unit time.
    """
    t = np.asarray(times, dtype=float)
    x = np.asarray(values, dtype=float)
    i, j = np.triu_indices(t.size, k=1)
    slopes = np.sort((x[j] - x[i]) / (t[j] - t[i]))
    med = float(np.median(slopes))

    n = t.size
    var_s = (n * (n - 1) * (2 * n + 5) - _tie_correction(x)) / 18.0
    c = Z_95 * math.sqrt(var_s) if abs(alpha - 0.05) < 1e-9 else \
        (-_normal_quantile(alpha / 2)) * math.sqrt(var_s)
    n_slopes = slopes.size
    ranks = np.arange(1, n_slopes + 1)
    lo = float(np.interp((n_slopes - c) / 2.0, ranks, slopes))
    hi = float(np.interp((n_slopes + c) / 2.0 + 1, ranks, slopes))
    lo, hi = min(lo, med), max(hi, med)  # guard interpolation/clamp edges
    return {"slope": med, "ci_lo": lo, "ci_hi": hi}


def _normal_quantile(p):
    """Inverse standard-normal CDF (Acklam's rational approximation)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= phigh:
        q = p - 0.5
        r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def benjamini_hochberg(pvals, alpha=0.05):
    """BH step-up FDR: returns (adjusted_p, reject) preserving input order."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * m / (np.arange(1, m + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]      # enforce monotonicity
    adj = np.clip(adj, 0, 1)
    out = np.empty(m)
    out[order] = adj
    return out, out <= alpha


# ---------------------------------------------------------------------------
# Verification - run every time before any result is produced
# ---------------------------------------------------------------------------
def run_self_tests():
    print("=== self-tests ===")

    # 1. Normal helpers against known values.
    assert abs(_normal_sf(0.0) - 0.5) < 1e-12
    assert abs((1 - _normal_sf(1.959963985)) - 0.975) < 1e-6
    assert abs(_normal_quantile(0.975) - Z_95) < 1e-4
    print("[PASS] normal CDF / quantile helpers")

    # 2. Perfectly increasing series -> S = n(n-1)/2, tau = 1, tiny p.
    inc = np.arange(1, 11, dtype=float)
    r = mann_kendall(inc)
    assert r["S"] == 10 * 9 / 2 and abs(r["tau"] - 1.0) < 1e-12 and r["p"] < 1e-3
    dec = inc[::-1]
    assert abs(mann_kendall(dec)["tau"] + 1.0) < 1e-12
    print("[PASS] monotonic known-answer (tau = +1 / -1)")

    # 3. Hand-computed small example: x = [1, 3, 2, 4]
    #    signs: (1,3)+ (1,2)+ (1,4)+ (3,2)- (3,4)+ (2,4)+  => S = +4, tau = 4/6
    r3 = mann_kendall([1, 3, 2, 4])
    assert r3["S"] == 4.0 and abs(r3["tau"] - 2.0 / 3.0) < 1e-12
    print("[PASS] hand-computed S on [1,3,2,4] == +4, tau == 2/3")

    # 4. Sen's slope recovers an exact line and lo <= slope <= hi.
    t = np.arange(2000, 2010, dtype=float)
    s = sens_slope(t, 2.5 * t - 17.0)
    assert abs(s["slope"] - 2.5) < 1e-9 and s["ci_lo"] <= s["slope"] <= s["ci_hi"]
    print("[PASS] Sen's slope recovers exact linear slope")

    # 5. BH: textbook monotonicity + p_adj >= p_raw, and a known vector.
    praw = np.array([0.001, 0.008, 0.039, 0.041, 0.9])
    padj, rej = benjamini_hochberg(praw, 0.05)
    assert np.all(padj >= praw - 1e-12) and np.all(np.diff(padj[np.argsort(praw)]) >= -1e-12)
    print("[PASS] Benjamini-Hochberg monotonicity / p_adj >= p_raw")

    # 6. Independent oracle cross-check (SciPy / statsmodels) if available.
    try:
        from scipy.stats import kendalltau, theilslopes
        rng = np.random.default_rng(0)
        for _ in range(200):
            x = rng.normal(size=12) + np.linspace(0, rng.normal(), 12)
            tt = np.arange(12.0)
            mine = mann_kendall(x)
            tau_sp, _ = kendalltau(tt, x)
            assert abs(mine["tau"] - tau_sp) < 1e-9, (mine["tau"], tau_sp)
            slope_sp = theilslopes(x, tt)[0]
            assert abs(sens_slope(tt, x)["slope"] - slope_sp) < 1e-9
        print("[PASS] tau matches scipy.kendalltau; slope matches scipy.theilslopes (200 cases)")
    except ImportError:
        print("[skip] scipy not installed - relying on known-answer tests above")

    try:
        from statsmodels.stats.multitest import multipletests
        praw = np.array([0.001, 0.02, 0.03, 0.2, 0.5, 0.5])
        mine_adj, _ = benjamini_hochberg(praw)
        sm_adj = multipletests(praw, method="fdr_bh")[1]
        assert np.allclose(mine_adj, sm_adj, atol=1e-12)
        print("[PASS] BH adjusted-p matches statsmodels.multipletests")
    except ImportError:
        print("[skip] statsmodels not installed - BH validated by known-answer test")

    print("=== all self-tests passed ===\n")


# ---------------------------------------------------------------------------
# Data loading + analysis
# ---------------------------------------------------------------------------
def load_feature_series():
    """Return {feature_name: DataFrame[year, value, num_songs]} for every
    csvFiles/<feature>Data folder, sorted by year."""
    out = {}
    for folder in sorted(glob.glob(FEATURE_GLOB)):
        parts = [p for p in glob.glob(os.path.join(folder, "part-*.csv"))
                 if not os.path.basename(p).startswith(".")]
        if not parts:
            continue
        df = pd.read_csv(parts[0], sep="\t").sort_values("year").reset_index(drop=True)
        value_col = next(c for c in df.columns if c.startswith("avg_"))
        feat = value_col[len("avg_"):]
        out[feat] = df.rename(columns={value_col: "value"})[["year", "value", "num_songs"]]
    return out


def analyze():
    series = load_feature_series()
    print(f"Loaded {len(series)} feature series: {', '.join(series)}\n")

    rows = []
    for feat, df in series.items():
        years = df["year"].to_numpy(dtype=float)
        vals = df["value"].to_numpy(dtype=float)
        # data sanity: full 1990-2022 window, strictly increasing distinct years
        assert df["year"].is_monotonic_increasing and df["year"].is_unique
        mk = mann_kendall(vals)
        sen = sens_slope(years, vals)
        span = years.max() - years.min()
        base = vals[0]
        rows.append({
            "feature": feat,
            "n": mk["n"],
            "tau": mk["tau"],
            "S": mk["S"],
            "Z": mk["Z"],
            "p_raw": mk["p"],
            "sen_slope_per_yr": sen["slope"],
            "ci_lo": sen["ci_lo"],
            "ci_hi": sen["ci_hi"],
            "change_per_decade": sen["slope"] * 10,
            "total_change": sen["slope"] * span,
            # Percent-change is only meaningful on a ratio scale with a positive
            # baseline. Loudness (dB) is interval-scaled with a negative base, so
            # report it as NaN there and cite the absolute dB change instead.
            "pct_change_vs_1990": (sen["slope"] * span / base * 100) if base > 0 else np.nan,
            "direction": "increasing" if mk["tau"] > 0 else "decreasing",
        })

    res = pd.DataFrame(rows)
    res["p_bh"], res["significant_fdr"] = benjamini_hochberg(res["p_raw"].to_numpy(), ALPHA)
    res = res.sort_values("p_bh").reset_index(drop=True)

    out_csv = os.path.join(HERE, "trend_test_results.csv")
    res.to_csv(out_csv, index=False)

    # Console report
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    print("=== Mann-Kendall trend test (FDR-corrected), sorted by significance ===")
    print(res[["feature", "tau", "Z", "p_raw", "p_bh", "significant_fdr",
               "sen_slope_per_yr", "change_per_decade", "pct_change_vs_1990",
               "direction"]].to_string(index=False))
    n_sig = int(res["significant_fdr"].sum())
    print(f"\n{n_sig}/{len(res)} features show a significant monotonic trend "
          f"at FDR alpha={ALPHA}.")
    print(f"wrote {out_csv}")

    make_figure(res)
    return res


def make_figure(res):
    """Lollipop of Kendall's tau per feature, flagged by FDR significance."""
    d = res.sort_values("tau").reset_index(drop=True)
    colors = ["#377EB8" if s else "#BBBBBB" for s in d["significant_fdr"]]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hlines(d["feature"], 0, d["tau"], color=colors, linewidth=2, zorder=1)
    ax.scatter(d["tau"], d["feature"], color=colors, s=90,
               edgecolor="black", linewidth=0.7, zorder=2)
    ax.axvline(0, color="black", linewidth=1)

    for _, r in d.iterrows():
        off = 0.03 if r["tau"] >= 0 else -0.03
        ha = "left" if r["tau"] >= 0 else "right"
        star = " *" if r["significant_fdr"] else ""
        ax.annotate(f"p={r['p_bh']:.1e}{star}", (r["tau"], r["feature"]),
                    textcoords="offset points",
                    xytext=(8 if r["tau"] >= 0 else -8, 0),
                    ha=ha, va="center", fontsize=8, color="0.3")

    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("Kendall's τ  (monotonic trend strength, 1990–2022)", fontsize=12)
    ax.set_title("Mann-Kendall trend test per audio feature\n"
                 "blue = significant after Benjamini–Hochberg FDR (α=0.05); "
                 "grey = not significant", fontsize=12)
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    out = os.path.join(HERE, "fig5_feature_trend_tests.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    run_self_tests()
    analyze()
