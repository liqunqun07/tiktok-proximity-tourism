"""Step 10 - Descriptives and negative binomial funnel models.

Input : data/processed/02_features.csv (via common.load_sample)
Outputs (outputs/tables/):
  table_descriptives.csv      descriptive statistics of cues, controls, outcomes
  table_correlations.csv      Spearman correlations among cues and controls
  table_vif.csv               variance inflation factors
  table_nb_funnel.csv         NB2 coefficients, IRRs, HC1 SEs, 95% CIs, p-values
  table_nb_fit.csv            model N, alpha, log-likelihood, McFadden pseudo-R2
  table_nb_blocks.csv         hierarchical entry of the three triad dimensions (LR tests)

Funnel stages (all NB2 with estimated dispersion, HC1-robust inference):
  reach            plays
  like_conversion  likes,    offset log(plays)
  save_conversion  collects, offset log(plays)
  relative_save    collects, offset log(likes): the save-vs-like contrast (H4);
                   its coefficients equal the difference between save and like
                   conversion in log-rate terms.
  comment_conversion / share_conversion: supplementary, offset log(plays).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

from common import (CONTROLS, CUES, DIMENSION, GUIDANCE, LABELS, NARRATIVE, SENSORY,
                    TABLES, load_sample, rhs, zname)

warnings.filterwarnings("ignore", category=RuntimeWarning)

MODELS = {
    "reach": ("plays", None),
    "like_conversion": ("likes", "log_plays"),
    "save_conversion": ("collects", "log_plays"),
    "relative_save": ("collects", "log_likes"),
    "comment_conversion": ("comments", "log_plays"),
    "share_conversion": ("shares", "log_plays"),
}


def fit_nb(df: pd.DataFrame, outcome: str, offset: str | None, formula_rhs: str):
    data = df.loc[df.likes > 0] if offset == "log_likes" else df
    model = smf.negativebinomial(f"{outcome} ~ {formula_rhs}", data=data,
                                 offset=data[offset] if offset else None)
    res = model.fit(cov_type="HC1", disp=0, maxiter=500, method="bfgs")
    # BFGS can diverge to an exploding alpha on the zero-heavy comment counts; L-BFGS converges there.
    if not res.mle_retvals.get("converged", False) or res.params["alpha"] > 100:
        res = model.fit(cov_type="HC1", disp=0, maxiter=1000, method="lbfgs")
    assert res.mle_retvals.get("converged", False), f"NB fit did not converge: {outcome}"
    return res


def null_llf(df: pd.DataFrame, outcome: str, offset: str | None) -> float:
    data = df.loc[df.likes > 0] if offset == "log_likes" else df
    return smf.negativebinomial(f"{outcome} ~ 1", data=data,
                                offset=data[offset] if offset else None).fit(disp=0, maxiter=5000, method="nm").llf


def descriptives(df: pd.DataFrame) -> None:
    raw = CUES + CONTROLS + ["plays", "likes", "comments", "shares", "collects"]
    d = df[raw].describe(percentiles=[.5]).T[["mean", "std", "min", "50%", "max"]]
    d.insert(0, "dimension", [DIMENSION.get(c, "Outcome") for c in raw])
    d.insert(0, "label", [LABELS.get(c, c.capitalize()) for c in raw])
    rates = pd.DataFrame({
        "label": ["Like conversion (likes/plays)", "Save conversion (collects/plays)"],
        "dimension": "Outcome",
        "mean": [(df.likes / df.plays).mean(), (df.collects / df.plays).mean()],
        "std": [(df.likes / df.plays).std(), (df.collects / df.plays).std()],
        "min": [(df.likes / df.plays).min(), (df.collects / df.plays).min()],
        "50%": [(df.likes / df.plays).median(), (df.collects / df.plays).median()],
        "max": [(df.likes / df.plays).max(), (df.collects / df.plays).max()],
    }, index=["like_rate", "save_rate"])
    pd.concat([d, rates]).rename(columns={"50%": "median"}).to_csv(TABLES / "table_descriptives.csv")

    terms = [zname(c) for c in CUES + CONTROLS]
    df[terms].corr(method="spearman").round(3).to_csv(TABLES / "table_correlations.csv")
    X = sm.add_constant(df[terms].astype(float)).values
    pd.DataFrame({"variable": CUES + CONTROLS,
                  "vif": [variance_inflation_factor(X, i + 1) for i in range(len(terms))]}
                 ).to_csv(TABLES / "table_vif.csv", index=False)


def main() -> None:
    df = load_sample()
    print(f"Analytic sample N = {len(df)}")
    descriptives(df)

    coef_rows, fit_rows = [], []
    full_rhs = rhs()
    for name, (y, off) in MODELS.items():
        res = fit_nb(df, y, off, full_rhs)
        ci = res.conf_int()
        for col in CUES + CONTROLS:
            t = zname(col)
            coef_rows.append(dict(model=name, variable=col, label=LABELS[col], dimension=DIMENSION[col],
                                  b=res.params[t], irr=np.exp(res.params[t]), se=res.bse[t],
                                  ci_low=ci.loc[t, 0], ci_high=ci.loc[t, 1], p=res.pvalues[t]))
        ll0 = null_llf(df, y, off)
        fit_rows.append(dict(model=name, outcome=y, offset=off or "", n=int(res.nobs),
                             alpha=res.params["alpha"], llf=res.llf, aic=res.aic,
                             pseudo_r2=1 - res.llf / ll0))
        print(f"  {name:<19} n={int(res.nobs)} alpha={res.params['alpha']:.3f}", flush=True)
    coefs = pd.DataFrame(coef_rows)
    coefs["stars"] = pd.cut(coefs.p, [-1, .001, .01, .05, 1], labels=["***", "**", "*", ""]).astype(str)
    coefs.to_csv(TABLES / "table_nb_funnel.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(TABLES / "table_nb_fit.csv", index=False)

    # Hierarchical entry: controls -> + narrative -> + sensory -> + guidance.
    blocks = [("Controls", []), ("+ Embedded narrative", NARRATIVE),
              ("+ Sensory enhancement", NARRATIVE + SENSORY),
              ("+ Interaction guidance", NARRATIVE + SENSORY + GUIDANCE)]
    block_rows = []
    for name in ["reach", "like_conversion", "save_conversion"]:
        y, off = MODELS[name]
        ll0 = null_llf(df, y, off)
        prev = None
        for label, cues in blocks:
            res = fit_nb(df, y, off, rhs(cues))
            row = dict(model=name, block=label, k=len(res.params), llf=res.llf, pseudo_r2=1 - res.llf / ll0)
            if prev is not None:
                lr = 2 * (res.llf - prev.llf)
                df_diff = len(res.params) - len(prev.params)
                row.update(lr_chi2=lr, df=df_diff, lr_p=stats.chi2.sf(lr, df_diff),
                           delta_pseudo_r2=row["pseudo_r2"] - (1 - prev.llf / ll0))
            block_rows.append(row)
            prev = res
    pd.DataFrame(block_rows).to_csv(TABLES / "table_nb_blocks.csv", index=False)

    view = coefs.assign(cell=coefs.b.map("{:+.3f}".format) + coefs.stars)
    print(view.pivot(index="variable", columns="model", values="cell")
          .reindex(CUES + CONTROLS)[list(MODELS)].to_string())


if __name__ == "__main__":
    main()
