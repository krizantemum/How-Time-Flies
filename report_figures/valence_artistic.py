"""
Artistic time-series of mean musical valence, 1990-2022.

A companion "beauty" figure to the analytical fig5: it shows the same yearly
avg_valence series the Spark pipeline writes to csvFiles/valenceData/, but
rendered for the eye rather than the referee -- a glowing line over a
warm-to-cool gradient (warm = brighter/happier years, cool = sombre years),
point sizes scaled by how many songs back each year.

Output:
  figX_valence_artistic.png

Run:  python report_figures/valence_artistic.py
"""

import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import PathPatch
from matplotlib.path import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VALENCE_DIR = os.path.join(ROOT, "csvFiles", "valenceData")


def load_valence():
    """Return the yearly valence series sorted by year."""
    parts = [p for p in glob.glob(os.path.join(VALENCE_DIR, "part-*.csv"))
             if not os.path.basename(p).startswith(".")]
    if not parts:
        raise FileNotFoundError(f"no part-*.csv in {VALENCE_DIR}")
    df = pd.read_csv(parts[0], sep="\t").sort_values("year").reset_index(drop=True)
    return df.rename(columns={"avg_valence": "value"})[["year", "value", "num_songs"]]


def _smooth(x, y, k=24):
    """Dense Catmull-Rom-ish smoothing via piecewise cubic on a fine grid."""
    xs = np.linspace(x.min(), x.max(), len(x) * k)
    # cubic interpolation through the points for a flowing, organic curve
    from numpy.polynomial import polynomial as P  # noqa: F401  (kept light)
    ys = np.interp(xs, x, y)  # base
    try:
        from scipy.interpolate import make_interp_spline
        ys = make_interp_spline(x, y, k=3)(xs)
    except Exception:
        pass  # graceful: straight interp if SciPy absent
    return xs, ys


def make_figure(df):
    years = df["year"].to_numpy(float)
    vals = df["value"].to_numpy(float)
    songs = df["num_songs"].to_numpy(float)

    # warm (joyful) -> cool (sombre) palette
    palette = LinearSegmentedColormap.from_list(
        "valence", ["#1b2a4a", "#3b6ea5", "#6fb1c7", "#f4d35e", "#ee964b", "#f95738"]
    )

    xs, ys = _smooth(years, vals)
    vmin, vmax = vals.min() - 0.01, vals.max() + 0.01

    fig, ax = plt.subplots(figsize=(11, 6.2))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # --- gradient fill under the curve, clipped to the area below the line ---
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    grad = np.tile(grad, (1, 2))
    im = ax.imshow(
        grad, aspect="auto", origin="lower",
        extent=[xs.min(), xs.max(), vmin, vmax],
        cmap=palette, alpha=0.85, zorder=1,
    )
    # build the clip path: area between the curve and the bottom axis
    verts = [(xs[0], vmin)] + list(zip(xs, ys)) + [(xs[-1], vmin), (xs[0], vmin)]
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
    clip = PathPatch(Path(verts, codes), transform=ax.transData)
    im.set_clip_path(clip)

    # --- glow: stack translucent thick lines, thin bright core on top ---
    for lw, a in [(14, 0.05), (9, 0.08), (5, 0.16)]:
        ax.plot(xs, ys, color="#ffd27d", lw=lw, alpha=a, solid_capstyle="round", zorder=3)
    ax.plot(xs, ys, color="#fff4d6", lw=2.0, alpha=0.95, zorder=4)

    # --- yearly points, coloured by valence, sized by corpus support ---
    sizes = 30 + 220 * (songs - songs.min()) / (np.ptp(songs) + 1e-9)
    ax.scatter(years, vals, c=vals, cmap=palette, vmin=vmin, vmax=vmax,
               s=sizes, edgecolor="white", linewidth=0.6, zorder=5)

    # --- styling: minimal, editorial ---
    ax.set_xlim(years.min(), years.max())
    ax.set_ylim(vmin, vmax)
    ax.set_title("How Time Flies — Mean Musical Valence, 1990–2022",
                 color="#f0f6fc", fontsize=16, pad=16, loc="left", weight="bold")
    ax.text(0.0, 1.005,
            "warmer = more positive years · point size ∝ songs that year",
            transform=ax.transAxes, color="#8b949e", fontsize=10, ha="left")
    ax.set_xlabel("Year", color="#c9d1d9", fontsize=11)
    ax.set_ylabel("Valence  (0 = sombre  →  1 = upbeat)", color="#c9d1d9", fontsize=11)

    # every year on the x-axis, rotated so all 33 labels stay readable
    ax.set_xticks(years)
    ax.set_xticklabels([int(y) for y in years], rotation=60, ha="right", fontsize=8)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors="#8b949e")
    ax.grid(True, axis="y", linestyle=":", color="#30363d", alpha=0.6)

    fig.tight_layout()
    out = os.path.join(HERE, "figX_valence_artistic.png")
    fig.savefig(out, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    make_figure(load_valence())
