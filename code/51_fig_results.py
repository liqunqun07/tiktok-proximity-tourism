"""Step 51 - Result figures (journal style: serif type, black/grey markers).

Inputs : outputs/tables/table_nb_funnel.csv, table_engagement_types.csv, table_complementarity.csv
Outputs: outputs/figures/fig_funnel_coefficients, fig_decoupling, fig_engagement_types, fig_complementarity,
         fig_save_landscape, fig_robustness (.png and .pdf)
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figstyle as fs
from common import CUES, DIMENSION, FIGURES, LABELS, TABLES, load_sample

MARK = {"Embedded narrative": "o", "Sensory enhancement": "s", "Interaction guidance": "^"}
FAMILY = {"Representations of space": "Embedded narrative", "Representational space": "Sensory enhancement",
          "Spatial practice": "Interaction guidance", "Control": "Control"}
STAGES = [("reach", "Video exposure (plays)"),
          ("like_conversion", "Like conversion (likes / plays)"),
          ("save_conversion", "Save conversion (saves / plays)")]


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def dim_axis(ax, labels, show_labels=True):
    y = np.arange(len(labels))[::-1]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    if not show_labels:
        ax.tick_params(labelleft=False)
    ax.axvline(0, color=fs.GREY, lw=0.7, zorder=0)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    return y


def dim_separators(ax, y):
    """Thin lines between the three cue families."""
    for i in (2, 7):  # after video duration, after emoji
        ax.axhline(y[i] - 0.5, color=fs.MID, lw=0.6, zorder=0)


def funnel_figure() -> None:
    t = pd.read_csv(TABLES / "table_nb_funnel.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.8))
    for k, (ax, (model, title)) in enumerate(zip(axes, STAGES)):
        d = t[t.model == model].set_index("variable").loc[CUES]
        y = dim_axis(ax, [LABELS[c] for c in CUES], show_labels=(k == 0))
        dim_separators(ax, y)
        for yi, (cue, r) in zip(y, d.iterrows()):
            sig = r.p < .05
            m = MARK[FAMILY[DIMENSION[cue]]]
            ax.plot([r.ci_low, r.ci_high], [yi, yi], color=fs.INK, lw=0.9, zorder=2)
            ax.plot(r.b, yi, m, ms=4.6, mfc=fs.INK if sig else "white", mec=fs.INK, mew=0.9, zorder=3)
        ax.set_title(title, fontsize=7.6)
        ax.set_xlabel("NB2 coefficient (95% CI)", fontsize=7.4)
        ax.tick_params(labelsize=7)
    handles = [plt.Line2D([], [], marker=m, ls="", mfc=fs.INK, mec=fs.INK, ms=4.6, label=k) for k, m in MARK.items()]
    handles.append(plt.Line2D([], [], marker="o", ls="", mfc="white", mec=fs.INK, ms=4.6, label="p ≥ .05"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7.2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig_funnel_coefficients")


def engagement_figure() -> None:
    t = pd.read_csv(TABLES / "table_engagement_types.csv").set_index("variable").loc[CUES]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})
    y = dim_axis(a1, [LABELS[c] for c in CUES])
    dim_axis(a2, [LABELS[c] for c in CUES], show_labels=False)
    for a in (a1, a2):
        dim_separators(a, y)
    for yi, cue in zip(y, CUES):
        r = t.loc[cue]
        a1.plot([r.b_affective, r.b_instrumental], [yi, yi], color=fs.GREY, lw=0.8, zorder=1)
    a1.scatter(t.b_affective, y, marker="o", s=22, facecolor="white", edgecolor=fs.INK, linewidth=0.9, zorder=3, label="Affective (likes, comments)")
    a1.scatter(t.b_instrumental, y, marker="s", s=22, color=fs.INK, zorder=3, label="Instrumental (saves, shares)")
    a1.set_xlabel("Mean NB2 coefficient", fontsize=7.4)
    a1.set_title("(a) Conversion by engagement type", fontsize=7.6)
    a1.legend(frameon=False, fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, handletextpad=0.2, columnspacing=0.8)
    lo, hi = t.difference - 1.96 * t.se_difference, t.difference + 1.96 * t.se_difference
    for yi, cue in zip(y, CUES):
        r = t.loc[cue]
        sig = r.p_difference < .05
        a2.plot([lo[cue], hi[cue]], [yi, yi], color=fs.INK, lw=0.9, zorder=2)
        a2.plot(r.difference, yi, "D", ms=4.2, mfc=fs.INK if sig else "white", mec=fs.INK, mew=0.9, zorder=3)
    a2.set_xlabel("Instrumental minus affective (95% CI)", fontsize=7.4)
    a2.set_title("(b) Contrast (H5)", fontsize=7.6)
    for a in (a1, a2):
        a.tick_params(labelsize=7)
    fig.tight_layout()
    save(fig, "fig_engagement_types")


def complementarity_figure() -> None:
    t = pd.read_csv(TABLES / "table_complementarity.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.1), gridspec_kw={"width_ratios": [1, 1.05]})
    ks = np.arange(0, 4)
    for label, part, mk, fc in [("Like conversion", "like_by_k", "o", "white"), ("Save conversion", "save_by_k", "s", fs.INK)]:
        d = t[t.part == part].sort_values("term")
        irr = np.r_[1.0, d.irr.values]; lo = np.r_[1.0, d.ci_low.values]; hi = np.r_[1.0, d.ci_high.values]
        a1.errorbar(ks, irr, yerr=[irr - lo, hi - irr], fmt="-" + mk, color=fs.INK, mfc=fc, mec=fs.INK, ms=4.8, lw=0.9, capsize=2, label=label)
    s1 = t[(t.part == "save_by_k") & (t.term == "k=1")].irr.iloc[0]
    a1.plot(ks, s1 ** ks, ls=":", color=fs.GREY, lw=1.0, label="Additive expectation (save)")
    a1.axhline(1, color=fs.MID, lw=0.7, zorder=0)
    a1.set_xticks(ks); a1.set_xlabel("Number of triad dimensions with a strong cue (k)", fontsize=7.4)
    a1.set_ylabel("IRR relative to k = 0", fontsize=7.4)
    a1.set_title("(a) Conversion by k", fontsize=7.6)
    a1.legend(frameon=False, fontsize=6.6, loc="upper left")
    a1.tick_params(labelsize=7)

    d = t[t.part == "save_interactions"].set_index("term").loc[["N:S", "N:G", "S:G", "N:S:G"]]
    labs = ["Narrative × sensory", "Narrative × guidance", "Sensory × guidance", "Narrative × sensory × guidance"]
    y = np.arange(len(labs))[::-1]
    a2.axvline(1, color=fs.GREY, lw=0.7)
    for yi, (_, r) in zip(y, d.iterrows()):
        a2.plot([r.ci_low, r.ci_high], [yi, yi], color=fs.INK, lw=0.9)
        a2.plot(r.irr, yi, "s", ms=4.6, mfc=fs.INK if r.p < .05 else "white", mec=fs.INK, mew=0.9)
    a2.set_yticks(y); a2.set_yticklabels(labs, fontsize=7)
    a2.set_xlabel("Interaction IRR, save conversion (95% CI)", fontsize=7.4)
    a2.set_title("(b) Interactions among strong cues", fontsize=7.6)
    a2.tick_params(labelsize=7)
    fig.tight_layout()
    save(fig, "fig_complementarity")


def decoupling_figure() -> None:
    """Reach coefficient (x) against save-conversion coefficient (y) for every cue."""
    t = pd.read_csv(TABLES / "table_nb_funnel.csv")
    r = t[t.model == "reach"].set_index("variable").loc[CUES]
    s = t[t.model == "save_conversion"].set_index("variable").loc[CUES]
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.axhline(0, color=fs.GREY, lw=0.7, zorder=0)
    ax.axvline(0, color=fs.GREY, lw=0.7, zorder=0)
    for c in CUES:
        if c == "is_ad":
            ax.plot(r.loc[c, "b"], -0.54, "^", ms=5.5, mfc=fs.INK, mec=fs.INK, zorder=3)
            continue
        sig = (r.loc[c, "p"] < .05) or (s.loc[c, "p"] < .05)
        ax.plot([r.loc[c, "ci_low"], r.loc[c, "ci_high"]], [s.loc[c, "b"]] * 2, color=fs.MID, lw=0.8, zorder=1)
        ax.plot([r.loc[c, "b"]] * 2, [s.loc[c, "ci_low"], s.loc[c, "ci_high"]], color=fs.MID, lw=0.8, zorder=1)
        ax.plot(r.loc[c, "b"], s.loc[c, "b"], MARK[FAMILY[DIMENSION[c]]], ms=5.5, mfc=fs.INK if sig else "white",
                mec=fs.INK, mew=0.9, zorder=3)
    off = {"semantic_richness": (6, 6), "text_length": (8, 14), "video_duration": (6, -9), "has_text_overlay": (-6, 7),
           "video_quality_score": (-10, 16), "is_vertical": (6, 3), "loudness": (-12, -14), "emoji_count": (-6, 7),
           "hashtag_count": (6, 6), "has_poi": (6, -10), "is_ad": (-6, -10)}
    for c in CUES:
        if c == "is_ad":
            continue
        dx, dy = off[c]
        ax.annotate(LABELS[c], (r.loc[c, "b"], s.loc[c, "b"]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=6.8, ha="left" if dx > 0 else "right", zorder=4,
                    arrowprops=dict(arrowstyle="-", lw=0.4, color=fs.GREY, shrinkA=0, shrinkB=3))
    ax.set_ylim(-0.66, 0.27)
    ax.annotate(f"Ad label: save conversion {s.loc['is_ad', 'b']:.2f},\nexposure {r.loc['is_ad', 'b']:+.2f} (off scale)",
                (r.loc["is_ad", "b"], -0.54), xytext=(r.loc["is_ad", "b"] - 0.015, -0.60), fontsize=6.8, ha="right", zorder=5,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color=fs.INK))
    xl, yl = ax.get_xlim(), ax.get_ylim()
    qk = dict(transform=ax.transAxes, fontsize=7, style="italic", color=fs.GREY)
    ax.text(0.99, 0.985, "Exposure up, saves up", ha="right", va="top", **qk)
    ax.text(0.012, 0.985, "Exposure down, saves up", ha="left", va="top", **qk)
    ax.text(0.012, 0.015, "Exposure down, saves down", ha="left", va="bottom", **qk)
    ax.text(0.99, 0.015, "Exposure up, saves down", ha="right", va="bottom", **qk)
    ax.set_xlabel("Coefficient on video exposure (95% CI)", fontsize=7.6)
    ax.set_ylabel("Coefficient on save conversion (95% CI)", fontsize=7.6)
    ax.tick_params(labelsize=7)
    handles = [plt.Line2D([], [], marker=m, ls="", mfc=fs.INK, mec=fs.INK, ms=5, label=k) for k, m in MARK.items()]
    handles.append(plt.Line2D([], [], marker="o", ls="", mfc="white", mec=fs.INK, ms=5, label="Both p ≥ .05"))
    ax.legend(handles=handles, frameon=False, fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
    fig.tight_layout()
    save(fig, "fig_decoupling")


def landscape_figure() -> None:
    """(a) Median save rate over richness x visual-quality quintiles; (b) distribution of save rate by k."""
    d = load_sample()
    d["save_rate"] = d.collects / d.plays
    d["rq"] = pd.qcut(d.semantic_richness.rank(method="first"), 5, labels=False)
    d["qq"] = pd.qcut(d.video_quality_score.rank(method="first"), 5, labels=False)
    grid = d.groupby(["qq", "rq"]).save_rate.median().unstack() * 100
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.3), gridspec_kw={"width_ratios": [1.05, 1]})
    im = a1.imshow(grid.values, origin="lower", cmap="Greys", aspect="auto", vmin=grid.values.min() * 0.9, vmax=grid.values.max() * 1.15)
    for i in range(5):
        for j in range(5):
            v = grid.values[i, j]
            a1.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.2,
                    color="white" if v > grid.values.min() + 0.6 * (grid.values.max() - grid.values.min()) else fs.INK)
    a1.set_xticks(range(5)); a1.set_yticks(range(5))
    a1.set_xticklabels(["Q1\nlow", "Q2", "Q3", "Q4", "Q5\nhigh"], fontsize=7)
    a1.set_yticklabels(["Q1 low", "Q2", "Q3", "Q4", "Q5 high"], fontsize=7)
    a1.set_xlabel("Semantic richness (quintile)", fontsize=7.4)
    a1.set_ylabel("Visual quality (quintile)", fontsize=7.4)
    a1.set_title("(a) Median save conversion (% of plays)", fontsize=7.6)
    for sp in a1.spines.values():
        sp.set_visible(True)

    d["N"] = d.semantic_richness > d.semantic_richness.median()
    d["S"] = d.video_quality_score > d.video_quality_score.median()
    d["k"] = d.N.astype(int) + d.S.astype(int) + d.has_poi.astype(int)
    pos = d[d.save_rate > 0]
    data = [np.log10(pos.save_rate[pos.k == k].values * 100) for k in range(4)]
    parts = a2.violinplot(data, positions=range(4), widths=0.8, showextrema=False)
    for b in parts["bodies"]:
        b.set_facecolor(fs.MID); b.set_edgecolor(fs.INK); b.set_linewidth(0.7); b.set_alpha(1)
    q = [np.percentile(x, [25, 50, 75]) for x in data]
    for k, (lo, md, hi) in enumerate(q):
        a2.plot([k, k], [lo, hi], color=fs.INK, lw=2.2, solid_capstyle="butt", zorder=3)
        a2.plot(k, md, "o", mfc="white", mec=fs.INK, ms=4.5, zorder=4)
        a2.text(k, hi + 0.06, f"{(10 ** md):.2f}", fontsize=6.8, va="bottom", ha="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
    a2.set_xlim(-0.6, 3.6)
    a2.set_xticks(range(4)); a2.set_xticklabels([f"k = {k}\n(n = {int((d.k == k).sum()):,})" for k in range(4)], fontsize=7)
    ticks = [-2, -1, 0, 1]
    a2.set_yticks(ticks); a2.set_yticklabels([f"{10 ** t:g}" for t in ticks], fontsize=7)
    a2.set_ylabel("Save conversion (% of plays, log scale)", fontsize=7.4)
    a2.set_title("(b) Distribution by number of strong dimensions", fontsize=7.6)
    fig.tight_layout()
    save(fig, "fig_save_landscape")


def robustness_figure() -> None:
    rb = pd.read_csv(TABLES / "table_robustness.csv")
    qr = pd.read_csv(TABLES / "table_qca_robustness.csv")
    specs = ["main", "english", "non_english", "trim_top1", "mature_90d", "ols_lograte", "overlay_t4", "overlay_t9"]
    names = ["Main", "English", "Non-English", "Trim top 1%", "≥ 90 days", "OLS log-rate", "Text: stickers only", "Text: cover title only"]
    vars_ = ["semantic_richness", "has_text_overlay", "video_quality_score", "is_vertical", "has_poi", "is_ad"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 4.0), gridspec_kw={"width_ratios": [1.05, 1]})
    m = rb[rb.model == "save_conversion"]
    y0 = np.arange(len(vars_))[::-1] * 1.0
    for i, v in enumerate(vars_):
        for j, sp in enumerate(specs):
            r = m[(m.variable == v) & (m.spec == sp)]
            if r.empty:
                continue
            r = r.iloc[0]
            y = y0[i] + (j - 3.5) * 0.1
            main = sp == "main"
            a1.plot([r.b - 1.96 * r.se, r.b + 1.96 * r.se], [y, y], color=fs.INK if main else fs.GREY, lw=1.0 if main else 0.6, zorder=2)
            a1.plot(r.b, y, "o", ms=3.6 if main else 2.6, mfc=fs.INK if (main or r.p < .05) else "white",
                    mec=fs.INK if main else fs.GREY, mew=0.6, zorder=3)
    a1.axvline(0, color=fs.GREY, lw=0.7, zorder=0)
    a1.set_yticks(y0); a1.set_yticklabels([LABELS[v] for v in vars_], fontsize=7)
    for i in range(len(vars_) - 1):
        a1.axhline(y0[i] - 0.5, color=fs.MID, lw=0.5, zorder=0)
    a1.set_xlabel("Save-conversion coefficient (95% CI)", fontsize=7.4)
    a1.set_title("(a) Eight regression specifications", fontsize=7.6)
    a1.tick_params(labelsize=7)

    q = qr[qr.outcome == "SAVECONV"].reset_index(drop=True)
    lab = q.spec.str.replace("Main (n >= 50, cons >= .80, PRI >= .65)", "Main", regex=False)
    y = np.arange(len(q))[::-1]
    a2.axvline(q.sol_cov.iloc[0], color=fs.MID, lw=0.7, ls=":", zorder=0)
    for yi, (_, r) in zip(y, q.iterrows()):
        a2.plot([0, r.sol_cov], [yi, yi], color=fs.GREY, lw=0.7, zorder=1)
        a2.plot(r.sol_cov, yi, "s" if r.RICH_in_all_paths else "o", ms=3.0 + 1.4 * r.n_paths,
                mfc=fs.INK if r.spec.startswith("Main") else "white", mec=fs.INK, mew=0.8, zorder=3)
        a2.text(0.395, yi, f"{r.sol_cons:.3f}", fontsize=6.6, va="center", ha="right")
    a2.set_yticks(y); a2.set_yticklabels(lab, fontsize=6.4)
    a2.set_xlim(0, 0.4); a2.set_ylim(-0.8, len(q) + 0.3)
    a2.text(0.395, len(q) - 0.2, "Consistency", fontsize=6.6, ha="right", va="bottom", style="italic")
    a2.set_xlabel("Solution coverage, high save conversion", fontsize=7.4)
    a2.set_title("(b) Eighteen fsQCA specifications", fontsize=7.6)
    a2.tick_params(labelsize=7)
    h = [plt.Line2D([], [], marker="s", ls="", mfc="white", mec=fs.INK, ms=4.5, label="Richness in all paths"),
         plt.Line2D([], [], marker="o", ls="", mfc="white", mec=fs.INK, ms=4.5, label="Richness not in all paths")]
    a2.legend(handles=h, frameon=False, fontsize=6.4, loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=2)
    fig.tight_layout()
    save(fig, "fig_robustness")


if __name__ == "__main__":
    fs.apply()
    funnel_figure()
    decoupling_figure()
    engagement_figure()
    complementarity_figure()
    landscape_figure()
    robustness_figure()
    print("Saved result figures")
