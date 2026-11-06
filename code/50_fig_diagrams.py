"""Step 50 - Figure 1 (conceptual architecture), Figure 2 (research model) and Figure 3 (data processing and analytical design).

Outputs: outputs/figures/fig_architecture, fig_framework and fig_method_flow (png and pdf)
All counts in Figure 3 are read from the data files rather than typed in.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

import figstyle as fs
from common import FIGURES, ROOT, load_sample

RAW = ROOT / "data" / "raw" / "tiktok_market_raw_scrape_20250528.csv"


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def architecture():
    """Figure 1 - funnel from the spatial triad (macro) through its three moments and cue families to the observable cues (micro)."""
    from matplotlib.patches import Polygon
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    ax.set_xlim(0, 100); ax.set_ylim(8, 101); ax.axis("off")

    # funnel bands: (top y, bottom y, half-width at top, half-width at bottom), narrowing from theory to cues
    bands = [(100, 86, 48, 44), (84, 62, 44, 40), (60, 42, 40, 36), (40, 12, 36, 32)]
    for (y0, y1, h0, h1), fc in zip(bands, (fs.MID, fs.LIGHT, fs.LIGHT, fs.LIGHT)):
        ax.add_patch(Polygon([(50 - h0, y0), (50 + h0, y0), (50 + h1, y1), (50 - h1, y1)], closed=True,
                             fc=fc, ec="none", zorder=0))

    ax.text(50, 95.4, "Lefebvre's (1991) spatial triad", ha="center", va="center", fontsize=8.6, fontweight="bold", color=fs.INK)
    ax.text(50, 90.4, "space is produced where conceived, lived and perceived space meet",
            ha="center", va="center", fontsize=7.0, fontstyle="italic", color=fs.INK)

    w = 21.5
    xs = (50 - 1.5 * w - 3, 50 - w / 2, 50 + w / 2 + 3)
    mid = [x + w / 2 for x in xs]
    for m in mid:
        fs.arrow(ax, (m, 86), (m, 81.4))
    moments = [("Representations of space", "conceived space", "verbal signs, codes", "and knowledge"),
               ("Representational space", "lived space", "images, symbols", "and atmosphere"),
               ("Spatial practice", "perceived space", "routes, networks", "and relations")]
    for x, t in zip(xs, moments):
        fs.box(ax, x, 64.4, w, 17, list(t), size=6.8, head_size=7.2, italic_second=True)
    for x in (xs[0] + w, xs[1] + w):
        fs.arrow(ax, (x + 0.3, 72.9), (x + 2.7, 72.9), style="<|-|>", ms=5, color=fs.GREY)

    families = ["Embedded narrative", "Sensory enhancement", "Interaction guidance"]
    for x, m, t in zip(xs, mid, families):
        fs.arrow(ax, (m, 64.4), (m, 56.4))
        fs.box(ax, x, 46, w, 10.4, [t], size=7.2, fc=fs.INK)
        ax.texts[-1].set_color("white")

    cues = [("Caption length", "Semantic richness", "Video duration"),
            ("On-screen text stickers", "Visual quality", "Vertical framing"),
            ("Hashtag count", "Location tag (POI)", "Ad label")]
    for x, m, t in zip(xs, mid, cues):
        fs.arrow(ax, (m, 46), (m, 37.4))
        fs.box(ax, x, 15, w, 22.4, list(t), size=6.8, bold_first=False)

    for (y0, y1, h0, _), lab in zip(bands, ("Theory", "Moments", "Cue families", "Observable cues")):
        ax.text(50 - h0 - 1.6, (y0 + y1) / 2, lab, rotation=90, ha="center", va="center", fontsize=7.2, fontweight="bold", color=fs.GREY)
    fs.arrow(ax, (98.5, 99), (98.5, 13), color=fs.GREY, lw=0.7)
    ax.text(96.6, 56, "from macro to micro", rotation=270, ha="center", va="center", fontsize=6.8, fontstyle="italic", color=fs.GREY)
    save(fig, "fig_architecture")


def framework():
    fig, ax = plt.subplots(figsize=(7.0, 4.9))
    ax.set_xlim(0, 106); ax.set_ylim(-9, 59); ax.axis("off")

    rows = [(41, "Representations of space", "(conceived space)", "Embedded narrative (H1)",
             ["Semantic richness", "Caption length", "Video duration"]),
            (24.5, "Representational space", "(lived space)", "Sensory enhancement (H2)",
             ["On-screen text", "Visual quality, framing", "Loudness, emoji"]),
            (8, "Spatial practice", "(perceived space)", "Interaction guidance (H3)",
             ["Hashtags", "Location tag", "Ad label"])]
    h = 14.5
    for y, t1, t2, fam, cues in rows:
        fs.box(ax, 0.6, y, 24.4, h, [t1, t2], fc=fs.LIGHT, italic_second=True, size=7.2, head_size=7.8)
        fs.arrow(ax, (25, y + h / 2), (28, y + h / 2), lw=0.8)
        fs.box(ax, 28, y, 31, h, [fam] + cues, size=7.2, head_size=7.8, align="left", pad_x=1.6)
        ax.plot([59, 62], [y + h / 2] * 2, color=fs.INK, lw=0.8, zorder=2)
    ax.plot([62, 62], [12.2, 48.2], color=fs.INK, lw=0.8, zorder=2)

    fs.box(ax, 26.8, 5.4, 34.2, 52.6, [], fc="none", ls=(0, (3, 2)), lw=0.8, ec=fs.GREY, rounding=1.2, zorder=1)
    fs.tag(ax, 43, 5.4, "H6, H7")

    fs.box(ax, 71, 43, 29, 10.5, ["Video exposure", "plays", "H3a-b"], size=7.2, head_size=7.8, italic_second=True)
    fs.box(ax, 71, 24.5, 29, 11.5, ["Affective conversion", "likes, comments (per play)", "H1b-c, H2a, H3c"], size=7.2, head_size=7.8, italic_second=True)
    fs.box(ax, 71, 6, 29, 12.5, ["Instrumental conversion", "saves, shares (per play)", "H1a, H1c, H2a-b, H3c"], size=7.2, head_size=7.8, italic_second=True)
    for yy in (48.2, 30.2, 12.2):
        fs.arrow(ax, (62, yy), (71, yy))
    fs.arrow(ax, (85.5, 43), (85.5, 36.0))
    ax.plot([101, 103, 103, 101], [48.2, 48.2, 12.2, 12.2], color=fs.INK, lw=0.8)
    fs.tag(ax, 104.2, 30.2, "H4", size=6.8)
    fs.arrow(ax, (85.5, 24.5), (85.5, 18.5), style="<|-|>")
    fs.tag(ax, 85.5, 21.5, "H5", size=6.8)

    fs.box(ax, 28, -8.2, 72, 5.4, ["Controls: followers, following, days since posting, caption language, posting year"],
           size=7.2, bold_first=False, ls=(0, (3, 2)), lw=0.8, ec=fs.GREY)
    fs.arrow(ax, (62, -2.8), (62, 4.2), ls=(0, (3, 2)), lw=0.8)
    save(fig, "fig_framework")


def method_flow():
    raw = pd.read_csv(RAW, usecols=["video_info.video_id"], low_memory=False)
    n_raw = len(raw)
    n_unique = raw["video_info.video_id"].nunique()
    feat = pd.read_csv(ROOT / "data" / "processed" / "02_features.csv", low_memory=False)
    d1 = feat[~feat.days_since_post.between(0, 3000)]
    ok = feat[feat.days_since_post.between(0, 3000)]
    d2 = ok[ok.flag_zero_duration.astype(bool)]
    ok = ok[~ok.flag_zero_duration.astype(bool)]
    d3 = ok[ok.plays <= 0]
    n_final = len(load_sample())
    assert n_unique - len(d1) - len(d2) - len(d3) == n_final, "exclusion counts do not add up"
    n_excl = len(d1) + len(d2) + len(d3)

    fig, ax = plt.subplots(figsize=(6.6, 7.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 108); ax.axis("off")

    def stage(y0, y1, label):
        fs.box(ax, 1, y0, 98, y1 - y0, [], fc="none", ls=(0, (4, 2)), lw=0.7, ec=fs.GREY, rounding=1.0, zorder=0)
        ax.text(3, y1 - 2.2, label + " ", ha="left", va="center", fontsize=7.8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=fs.GREY, lw=0.6))

    stage(87, 107.5, "Stage 1: Collection")
    fs.box(ax, 8, 90, 50, 12, ["Keyword-based search of TikTok", "cafes, bookstores, museums, parks, markets", "(Chinese and English terms)"], size=7.2)
    fs.box(ax, 62, 90, 34, 12, ["Harvest of public metadata", "28 May 2025", f"{n_raw:,} records"], size=7.2)
    fs.arrow(ax, (58, 96), (62, 96))

    stage(47, 84, "Stage 2: Cleaning")
    fs.box(ax, 8, 69, 50, 9, ["De-duplication by video ID", f"{n_unique:,} unique videos"], size=7.2)
    fs.box(ax, 62, 69, 34, 9, [f"Excluded: n = {n_raw - n_unique}", "duplicate records"], size=7.0, ec=fs.GREY)
    fs.arrow(ax, (58, 73.5), (62, 73.5), ls=(0, (2, 2)))
    fs.box(ax, 8, 53.5, 50, 10, ["Validity screening", f"{n_unique - n_excl:,} videos"], size=7.2)
    fs.box(ax, 62, 50.5, 34, 15, [f"Excluded: n = {n_excl}", f"invalid posting date ({len(d1)})",
                                  f"zero duration ({len(d2)})", f"zero plays ({len(d3)})"], size=6.9, ec=fs.GREY)
    fs.arrow(ax, (58, 58.5), (62, 58.5), ls=(0, (2, 2)))
    fs.arrow(ax, (33, 90), (33, 78)); fs.arrow(ax, (33, 69), (33, 63.5))
    fs.box(ax, 8, 40.5, 84, 6.0, [f"Analytic sample: N = {n_final:,} videos"], size=8.0, lw=1.4, fc=fs.LIGHT)
    fs.arrow(ax, (33, 53.5), (33, 46.5))

    stage(22, 39, "Stage 3: Feature construction")
    fs.box(ax, 8, 24.3, 84, 11.2, ["Cue families from platform fields (no video-frame processing)",
                                   "narrative: semantic richness, caption length, duration   |   sensory: on-screen text, quality, framing",
                                   "guidance: hashtags, location tag, ad label   |   controls   |   log1p and z-standardisation"], size=6.6)
    fs.arrow(ax, (50, 40.5), (50, 35.5))

    stage(0.5, 19.5, "Stage 4: Analysis")
    for x, w, t in [(4, 29.5, ["Study 1", "NB2 funnel models", "exposure, like, save conversion"]),
                    (35.5, 29, ["Study 2", "Decoupling tests", "exposure vs. conversion;", "affective vs. instrumental"]),
                    (66.5, 29.5, ["Study 3", "Complementarity, fsQCA", "combinations of cues;", "permutation placebo"])]:
        fs.box(ax, x, 2.2, w, 9.8, t, size=6.6)
    ax.plot([50, 50], [24.3, 15.2], color=fs.INK, lw=0.8)
    ax.plot([19, 81], [15.2, 15.2], color=fs.INK, lw=0.8)
    for x in (19, 50, 81):
        fs.arrow(ax, (x, 15.2), (x, 12))
    save(fig, "fig_method_flow")


if __name__ == "__main__":
    fs.apply()
    architecture()
    framework()
    method_flow()
    print("Saved fig_architecture, fig_framework and fig_method_flow")
