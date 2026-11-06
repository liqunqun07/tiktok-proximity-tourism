"""Step 25 - Complementarity of the three triad dimensions (regression bridge to fsQCA).

Input : data/processed/02_features.csv (via common.load_sample)
Outputs (outputs/tables/):
  table_complementarity.csv       conversion rates and NB2 IRRs by number of dimensions with a strong cue (k = 0-3),
                                  convexity test, and the narrative x sensory x guidance interaction model
  table_complementarity_alt.csv   same with on-screen text as the sensory indicator

A video has a strong cue in a dimension when
  narrative : semantic richness above the sample median
  sensory   : visual quality above the sample median (alt: on-screen text present)
  guidance  : location tag present
k counts the dimensions with a strong cue. If the dimensions only add, the log-rate of save conversion
rises linearly in k; if they combine (Lefebvre's dialectical claim), the rise is convex:
b(k=3) - 3 * b(k=1) > 0 (Wald test with HC1 covariance).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from common import CONTROLS, TABLES, load_sample, zname

warnings.filterwarnings("ignore")
CTRL = " + ".join(zname(c) for c in CONTROLS) + " + C(lang_group) + C(year)"


def fit(df, outcome, rhs):
    m = smf.negativebinomial(f"{outcome} ~ {rhs} + {CTRL}", data=df, offset=df.log_plays)
    return m.fit(cov_type="HC1", disp=0, maxiter=500, method="lbfgs")


def run(df, sensory: str) -> pd.DataFrame:
    d = df.copy()
    d["N"] = (d.semantic_richness > d.semantic_richness.median()).astype(int)
    d["S"] = (d.video_quality_score > d.video_quality_score.median()).astype(int) if sensory == "quality" else d.has_text_overlay
    d["G"] = d.has_poi
    d["k"] = d.N + d.S + d.G
    rows = []
    for k, g in d.groupby("k"):
        rows.append(dict(part="descriptive", term=f"k={k}", n=len(g), like_rate=g.likes.sum() / g.plays.sum(),
                         save_rate=g.collects.sum() / g.plays.sum(),
                         median_save_rate=(g.collects / g.plays).median()))
    for y, label in [("likes", "like"), ("collects", "save")]:
        r = fit(d, y, "C(k)")
        for k in (1, 2, 3):
            t = f"C(k)[T.{k}]"
            rows.append(dict(part=f"{label}_by_k", term=f"k={k}", b=r.params[t], irr=np.exp(r.params[t]),
                             ci_low=np.exp(r.conf_int().loc[t, 0]), ci_high=np.exp(r.conf_int().loc[t, 1]), p=r.pvalues[t]))
        # convexity: effect of all three dimensions exceeds three times the single-dimension effect
        t1, t3 = "C(k)[T.1]", "C(k)[T.3]"
        w = np.zeros(len(r.params)); idx = list(r.params.index)
        w[idx.index(t3)] = 1; w[idx.index(t1)] = -3
        est = float(w @ r.params.values); se = float(np.sqrt(w @ r.cov_params().values @ w))
        rows.append(dict(part=f"{label}_convexity", term="b(k=3) - 3*b(k=1)", b=est, se=se, p=2 * stats.norm.sf(abs(est / se))))
        r2 = fit(d, y, "N*S + N*G + S*G + N:S:G")
        for t in ["N", "S", "G", "N:S", "N:G", "S:G", "N:S:G"]:
            rows.append(dict(part=f"{label}_interactions", term=t, b=r2.params[t], irr=np.exp(r2.params[t]),
                             ci_low=np.exp(r2.conf_int().loc[t, 0]), ci_high=np.exp(r2.conf_int().loc[t, 1]), p=r2.pvalues[t]))
    return pd.DataFrame(rows)


def main() -> None:
    df = load_sample()
    main_ = run(df, "quality")
    main_.to_csv(TABLES / "table_complementarity.csv", index=False)
    run(df, "overlay").to_csv(TABLES / "table_complementarity_alt.csv", index=False)
    pd.set_option("display.width", 200)
    print(main_.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
