"""Stage 30: fsQCA of video cue configurations across three engagement-funnel outcomes.

Input : data/processed/02_features.csv via common.load_sample(); fsQCA engine in qcalib.py
Output: outputs/tables/table_qca_{anchors,necessity,truth_tables,solutions,permutation,
        robustness}.csv
        outputs/figures/fig_qca_configurations.{png,pdf}
Run   : .venv/bin/python code/30_qca.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import qcalib as qca
import figstyle
from common import FIGURES, SEED, TABLES, load_sample

figstyle.apply()
rng = np.random.default_rng(SEED)

COND = {"RICH": "semantic_richness", "OVL": "has_text_overlay", "QUAL": "video_quality_score",
        "POI": "has_poi", "FOL": "log_follower_count"}
FUZZY = {"RICH", "QUAL", "FOL"}
NAMES = list(COND)
COND_LABEL = {"RICH": "Semantic richness", "OVL": "On-screen text", "QUAL": "Visual quality",
              "POI": "Location tag", "FOL": "Followers"}
OUTS = ["REACH", "LIKECONV", "SAVECONV", "~LIKECONV", "~SAVECONV"]
MAIN_OUTS = OUTS[:3]
OUT_LABEL = {"REACH": "Exposure (plays)", "LIKECONV": "Like conversion", "SAVECONV": "Save conversion",
             "~LIKECONV": "Low like conversion", "~SAVECONV": "Low save conversion"}
OUT_COLOR = {k: "#111111" for k in ["REACH", "LIKECONV", "SAVECONV", "~LIKECONV", "~SAVECONV"]}
# Directional expectations for intermediate solutions (none for negated outcomes).
DE = {"REACH": dict(FOL=1, OVL=1, POI=1), "LIKECONV": dict(OVL=1, FOL=1),
      "SAVECONV": dict(RICH=1, QUAL=1, POI=1)}
MAIN = dict(n_cut=50, incl=0.80, pri_cut=0.65)
B_PERM, B_BOOT = 200, 500


def de(outcome, names):
    return [DE.get(outcome, {}).get(n) for n in names]


def raw_outcomes(d):
    return {"REACH": np.log(d.plays),
            "LIKECONV": np.log((d.likes + 1) / (d.plays + 1)),
            "SAVECONV": np.log((d.collects + 1) / (d.plays + 1))}


def calibrate_count(x, groups, anchors):
    """Direct calibration for integer counts: crossover set half a unit above the median,
    so captions at the typical count are not scored as (barely) in the set."""
    x = np.asarray(x, float); out = np.empty(len(x))
    for k in np.unique(groups):
        idx = groups == k
        lo, mid, hi = np.percentile(x[idx], anchors)
        out[idx] = qca._direct(x[idx], lo, mid + 0.5, max(hi, mid + 1.5))
    return out


def make_data(d, anchors=(5, 50, 95), pooled=False, cond=COND):
    """Calibrate conditions and outcomes within posting-year group (or pooled)."""
    g = np.zeros(len(d)) if pooled else d.cal_group.values
    M = pd.DataFrame({k: (calibrate_count(d[v].values, g, anchors) if k == "RICH" else
                          qca.calibrate_grouped(d[v].values, g, anchors)) if k in FUZZY
                      else d[v].astype(float).values for k, v in cond.items()})
    Y = {k: qca.calibrate_grouped(v.values, g, anchors) for k, v in raw_outcomes(d).items()}
    Y["~LIKECONV"], Y["~SAVECONV"] = 1 - Y["LIKECONV"], 1 - Y["SAVECONV"]
    return M, Y


def shuffle_within(y, groups):
    out = y.copy()
    for idx in groups:
        out[idx] = y[rng.permutation(idx)]
    return out


def spec_run(label, d, names=NAMES, anchors=(5, 50, 95), pooled=False, cond=COND,
             n_cut=MAIN["n_cut"], incl=MAIN["incl"], pri_cut=MAIN["pri_cut"]):
    M, Y = make_data(d, anchors, pooled, cond)
    rows = []
    for o in OUTS:
        r = qca.solve(M[names], Y[o], names, n_cut, incl, pri_cut, de(o, names))
        im = r["intermediate"]
        terms = [t["term"] for t in im["fit"]["terms"]] if im else []

        def in_all(c):
            return bool(terms) and c in names and all(t[names.index(c)] == 1 for t in terms)

        rows.append(dict(spec=label, N=len(d), n_cut=n_cut, outcome=o, n_sufficient_rows=len(r["on_idx"]),
                         n_paths=len(terms), sol_cons=im["fit"]["sol_cons"] if im else np.nan,
                         sol_cov=im["fit"]["sol_cov"] if im else np.nan,
                         paths=" + ".join(t["label"] for t in im["fit"]["terms"]) if im else "",
                         RICH_in_all_paths=in_all("RICH"), OVL_in_all_paths=in_all("OVL"),
                         FOL_absent_from_some_path=bool(terms) and "FOL" in names
                         and any(t[names.index("FOL")] != 1 for t in terms)))
    return rows


# ---------------------------------------------------------------- figures
# Path order and IDs shared with manuscript Table 7 (S = save, L = like, E = exposure).
PATH_ORDER = {"SAVECONV": ["RICH * OVL * FOL", "RICH * QUAL * POI * FOL", "RICH * OVL * QUAL * POI", "OVL * QUAL * POI * FOL"],
              "LIKECONV": ["~RICH * OVL * FOL", "OVL * ~QUAL * FOL", "RICH * OVL * ~POI"]}
PATH_PREFIX = {"SAVECONV": "S", "LIKECONV": "L", "REACH": "E"}


def config_chart(sols, fname):
    """Ragin-style chart: big = core, small = peripheral; filled = present, crossed = absent."""
    rank = lambda o, p: PATH_ORDER.get(o, []).index(p["label"]) if p["label"] in PATH_ORDER.get(o, []) else 99
    blocks = [(o, dict(sols[o], paths=sorted(sols[o]["paths"], key=lambda p: rank(o, p))))
              for o in ["SAVECONV", "LIKECONV", "REACH"] if o in sols and sols[o]["paths"]]
    xs, x, heads = [], 0.0, []
    for o, s in blocks:
        start = x
        for _ in s["paths"]:
            xs.append(x); x += 1
        heads.append((o, start, x - 1)); x += 0.7
    ny = len(NAMES); metrics = ["Consistency", "PRI", "Raw coverage", "Unique coverage"]
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * x, 4.6))
    k = 0
    for o, s in blocks:
        col = OUT_COLOR[o]
        for p in s["paths"]:
            cx = xs[k]; k += 1
            pid = PATH_PREFIX[o] + str(s["paths"].index(p) + 1)
            ax.text(cx, ny - 0.42, pid, ha="center", va="center", fontsize=8, color=col)
            for j, n in enumerate(NAMES):
                y = ny - 1 - j; v, core = p["term"][j], j in p["core"]
                size = 210 if core else 70
                if v == 1:
                    ax.scatter(cx, y, s=size, color=col, zorder=3, linewidths=0)
                elif v == 0:
                    ax.scatter(cx, y, s=size, facecolor="white", edgecolor=col, linewidths=1.3, zorder=3)
                    ax.scatter(cx, y, s=size * 0.45, marker="x", color=col, linewidths=1.2, zorder=4)
            for i, m in enumerate([p["cons"], p["PRI"], p["raw_cov"], p["uniq_cov"]]):
                ax.text(cx, -1.0 - 0.55 * i, f"{m:.3f}".lstrip("0"), ha="center", va="center", fontsize=7.6)
    for o, a, b in heads:
        s = sols[o]
        ax.text((a + b) / 2, ny - 0.05, OUT_LABEL[o], ha="center", va="bottom", fontsize=8.8,
                fontweight="bold", color=OUT_COLOR[o])
        ax.plot([a - 0.35, b + 0.35], [ny - 0.15] * 2, color=OUT_COLOR[o], lw=1.4)
        ax.text((a + b) / 2, -1.0 - 0.55 * 4.1,
                f"Sol. cons {s['sol_cons']:.3f}\nSol. cov {s['sol_cov']:.3f}".replace(" 0.", " ."),
                ha="center", va="top", fontsize=7.4)
    for j, n in enumerate(NAMES):
        ax.text(-0.7, ny - 1 - j, COND_LABEL[n], ha="right", va="center", fontsize=8.6)
        ax.axhline(ny - 1 - j, color="#e5e9f0", lw=0.6, zorder=0)
    for i, m in enumerate(metrics):
        ax.text(-0.7, -1.0 - 0.55 * i, m, ha="right", va="center", fontsize=8, color="#4c566a")
    ax.axhline(-0.55, color="#8f98a8", lw=0.6)
    ax.set_xlim(-0.7, x - 0.7); ax.set_ylim(-4.6, ny + 0.6); ax.axis("off")
    ink = "#2e3440"
    h = [Line2D([], [], marker="o", ls="", color=ink, markersize=10, label="Core condition present"),
         Line2D([], [], marker="o", ls="", color=ink, markersize=6, label="Peripheral condition present"),
         Line2D([], [], marker="$\\otimes$", ls="", color=ink, markersize=10, label="Condition absent"),
         Line2D([], [], ls="", label="Blank: don't care")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=7.8, frameon=False)
    save(fig, fname)


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------- main
def main():
    plt.rcParams.update({"font.size": 9})
    TABLES.mkdir(parents=True, exist_ok=True)
    d = load_sample()
    d["cal_group"] = np.where(d.year.astype(int) <= 2020, "<=2020", d.year)
    M, Y = make_data(d)
    groups = [np.where(d.cal_group.values == g)[0] for g in np.unique(d.cal_group)]

    # Calibration anchors.
    raw = {**{n: d[COND[n]] for n in sorted(FUZZY)}, **raw_outcomes(d)}
    anch = []
    for v, x in raw.items():
        for g in sorted(d.cal_group.unique()):
            xv = x[d.cal_group == g]; a = np.percentile(xv, [5, 50, 95])
            anch.append(dict(set=v, source=COND.get(v, v), year_group=g, n=len(xv), full_non=a[0], crossover=a[1],
                             full_in=a[2], share_at_crossover=float((xv == a[1]).mean())))
    # Row order must not depend on set/dict iteration (hash order varies across
    # processes), so sort by fixed keys: source, then year group chronologically.
    anchors = pd.DataFrame(anch)
    year_order = {g: i for i, g in enumerate(["<=2020", "2021", "2022", "2023", "2024", "2025"])}
    anchors["_g"] = anchors.year_group.map(year_order)
    anchors.sort_values(["source", "_g"], kind="mergesort").drop(columns="_g") \
        .to_csv(TABLES / "table_qca_anchors.csv", index=False)

    # Necessity.
    nec = pd.concat([qca.necessity(M, Y[o], NAMES).assign(outcome=o) for o in OUTS])
    nec[["outcome", "cond", "consistency", "coverage", "RoN"]].to_csv(TABLES / "table_qca_necessity.csv", index=False)

    # Truth tables and solutions.
    main_res, tts, sol_rows, chart = {}, [], [], {}
    for o in OUTS:
        r = qca.solve(M[NAMES], Y[o], NAMES, MAIN["n_cut"], MAIN["incl"], MAIN["pri_cut"], de(o, NAMES))
        main_res[o] = r
        tt = r["tt"].copy()
        for j, n in enumerate(NAMES):
            tt[n] = tt.bits.map(lambda b: b[j])
        tt["OUT"] = np.where(tt.row.isin(r["on_idx"]), "1", np.where(tt.row.isin(r["rem_idx"]), "?", "0"))
        tts.append(tt.drop(columns="bits").assign(outcome=o))
    pd.concat(tts)[["outcome", "row", *NAMES, "n", "incl", "PRI", "OUT"]].sort_values(
        ["outcome", "n"], ascending=[True, False]).to_csv(TABLES / "table_qca_truth_tables.csv", index=False)

    # Bootstrap CIs for intermediate paths and solution.
    boot = {}
    for o in OUTS:
        im = main_res[o]["intermediate"]
        if not im:
            continue
        T = np.stack([qca.term_membership(t["term"], M, NAMES) for t in im["fit"]["terms"]]); y = Y[o]
        cs, cv, sc, sv = np.empty((B_BOOT, len(T))), np.empty((B_BOOT, len(T))), np.empty(B_BOOT), np.empty(B_BOOT)
        for b in range(B_BOOT):
            idx = rng.integers(0, len(y), len(y)); yb, Tb = y[idx], T[:, idx]
            mn = np.minimum(Tb, yb).sum(1); cs[b], cv[b] = mn / Tb.sum(1), mn / yb.sum()
            U = Tb.max(0); sc[b], sv[b] = qca.cons(U, yb), qca.cov(U, yb)
        boot[o] = dict(cons=np.percentile(cs, [2.5, 97.5], 0), cov=np.percentile(cv, [2.5, 97.5], 0),
                       sol_cons=np.percentile(sc, [2.5, 97.5]), sol_cov=np.percentile(sv, [2.5, 97.5]))

    for o in OUTS:
        r = main_res[o]
        pars = r["parsimonious"]["terms"] if r["parsimonious"] else []
        chart[o] = dict(paths=[], sol_cons=np.nan, sol_cov=np.nan)
        for typ in ("complex", "parsimonious", "intermediate"):
            s = r[typ]
            if not s:
                sol_rows.append(dict(outcome=o, solution=typ, path="(no sufficient rows)"))
                continue
            f = s["fit"]; bo = boot.get(o) if typ == "intermediate" else None
            for i, t in enumerate(f["terms"]):
                mem = qca.term_membership(t["term"], M, NAMES)
                core = qca.core_peripheral(t["term"], pars) if typ == "intermediate" else set()
                marks = {n: ("" if t["term"][j] == "-" else ("core" if j in core or typ == "parsimonious"
                         else "peripheral") + ("+" if t["term"][j] == 1 else "-")) for j, n in enumerate(NAMES)}
                row = dict(outcome=o, solution=typ, path=t["label"], **marks, cons=t["cons"], PRI=t["PRI"],
                           raw_cov=t["raw_cov"], uniq_cov=t["uniq_cov"], n_cases=int((mem > 0.5).sum()),
                           n_cases_consistent=int(((mem > 0.5) & (Y[o] > 0.5)).sum()),
                           sol_cons=f["sol_cons"], sol_PRI=f["sol_pri"], sol_cov=f["sol_cov"], n_models=f["n_models"])
                if bo:
                    row.update(cons_lo=bo["cons"][0, i], cons_hi=bo["cons"][1, i], cov_lo=bo["cov"][0, i],
                               cov_hi=bo["cov"][1, i], sol_cons_lo=bo["sol_cons"][0], sol_cons_hi=bo["sol_cons"][1],
                               sol_cov_lo=bo["sol_cov"][0], sol_cov_hi=bo["sol_cov"][1])
                sol_rows.append(row)
                if typ == "intermediate":
                    chart[o]["paths"].append(dict(t, core=core))
            if typ == "intermediate":
                chart[o].update(sol_cons=f["sol_cons"], sol_cov=f["sol_cov"])
    sols = pd.DataFrame(sol_rows)
    sols.to_csv(TABLES / "table_qca_solutions.csv", index=False)
    config_chart(chart, "fig_qca_configurations")

    # Whole-pipeline permutation placebo: shuffle outcome within year group.
    K = len(NAMES); A = M[NAMES].values.astype(np.float32)
    R = np.ones((len(d), 2 ** K), dtype=np.float32)
    for idx in range(2 ** K):
        for j in range(K):
            R[:, idx] = np.minimum(R[:, idx], A[:, j] if (idx >> (K - 1 - j)) & 1 else 1 - A[:, j])
    msum = R.sum(0); ok = (R > 0.5).sum(0) >= MAIN["n_cut"]

    def row_stats(y):
        y = y.astype(np.float32); mn = np.minimum(R, y[:, None]).sum(0)
        both = np.minimum(np.minimum(R, y[:, None]), (1 - y)[:, None]).sum(0)
        inc, pr = mn / np.maximum(msum, 1e-9), (mn - both) / np.maximum(msum - both, 1e-9)
        on = (inc >= MAIN["incl"]) & (pr >= MAIN["pri_cut"]) & ok
        return int(on.sum()), float(np.max(np.where(ok, inc, 0)))

    def sol_cov(y, n_on):
        if n_on == 0:
            return 0.0
        r = qca.solve(M[NAMES], y, NAMES, MAIN["n_cut"], MAIN["incl"], MAIN["pri_cut"], [None] * K)
        return r["complex"]["fit"]["sol_cov"] if r["complex"] else 0.0

    perm = []
    for o in OUTS:
        n0, m0 = row_stats(Y[o]); c0 = sol_cov(Y[o], n0)
        null = []
        for _ in range(B_PERM):
            ys = shuffle_within(Y[o], groups); n, m = row_stats(ys); null.append((n, m, sol_cov(ys, n)))
        nl = np.array(null)
        p = lambda obs, col: ((nl[:, col] >= obs).sum() + 1) / (B_PERM + 1)
        perm.append(dict(outcome=o, obs_n_sufficient_rows=n0, null_mean_n_rows=nl[:, 0].mean(), null_max_n_rows=nl[:, 0].max(),
                         p_n_rows=p(n0, 0), obs_max_row_consistency=m0, null_mean_max_cons=nl[:, 1].mean(),
                         null_q95_max_cons=np.percentile(nl[:, 1], 95), p_max_cons=p(m0, 1),
                         obs_complex_sol_cov=c0, null_mean_sol_cov=nl[:, 2].mean(), p_sol_cov=p(c0, 2), reps=B_PERM))
    pd.DataFrame(perm).to_csv(TABLES / "table_qca_permutation.csv", index=False)

    # Robustness specifications.
    nonen = d[d.lang_group != "English"]; en = d[d.lang_group == "English"]
    scaled = lambda s: max(10, int(round(MAIN["n_cut"] * len(s) / len(d))))  # frequency cut scaled to subsample size
    specs = [("Main (n >= 50, cons >= .80, PRI >= .65)", d, {}),
             ("Consistency >= .75", d, dict(incl=0.75)), ("Consistency >= .85", d, dict(incl=0.85)),
             ("PRI >= .60", d, dict(pri_cut=0.60)), ("PRI >= .70", d, dict(pri_cut=0.70)),
             ("Frequency >= 20", d, dict(n_cut=20)), ("Frequency >= 100", d, dict(n_cut=100)),
             ("Anchors 10/50/90", d, dict(anchors=(10, 50, 90))), ("Anchors 20/50/80", d, dict(anchors=(20, 50, 80))),
             ("Pooled calibration (not by year)", d, dict(pooled=True)),
             *[(f"Drop {COND_LABEL[n].lower()}", d, dict(names=[m for m in NAMES if m != n])) for n in NAMES],
             (f"English only (n >= {scaled(en)})", en, dict(n_cut=scaled(en))),
             (f"Non-English only (n >= {scaled(nonen)})", nonen, dict(n_cut=scaled(nonen))),
             ("On-screen text = stickers only", d, dict(cond={**COND, "OVL": "has_overlay_type4"}))]
    rob = [row for lab, s, kw in specs for row in spec_run(lab, s, **kw)]
    pd.DataFrame(rob).to_csv(TABLES / "table_qca_robustness.csv", index=False)

    im = sols[(sols.solution == "intermediate") & sols.path.ne("(no sufficient rows)")]
    print(im[["outcome", "path", "cons", "PRI", "raw_cov", "uniq_cov", "sol_cons", "sol_cov"]].round(3).to_string(index=False))
    print(pd.DataFrame(perm)[["outcome", "obs_n_sufficient_rows", "p_n_rows", "p_max_cons", "p_sol_cov"]].to_string(index=False))


if __name__ == "__main__":
    main()
