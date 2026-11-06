"""Step 20 - Robustness of the funnel estimates.

Input : data/processed/02_features.csv (via common.load_sample)
Output: outputs/tables/table_robustness.csv   long format, one row per spec x model x cue
        outputs/tables/table_robustness_summary.csv   sign/significance agreement with the main model

Specifications (reach, like conversion, save conversion, relative save):
  main         full sample, NB2 + HC1 (same as step 10)
  english      English captions only
  non_english  non-English captions only
  trim_top1    excluding the top 1% of accounts by followers
  mature_90d   videos online for at least 90 days (counters closer to saturation)
  ols_lograte  OLS on log((y+1)/(plays+1)) (log plays for exposure), HC1
  overlay_t4   on-screen text replaced by sticker text only (type 4)
  overlay_t9   on-screen text replaced by cover-title text only (type 9)
"""
from __future__ import annotations

import importlib
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import CUES, TABLES, load_sample, rhs, zname

warnings.filterwarnings("ignore")
fit_nb = importlib.import_module("10_nb_funnel").fit_nb

FOCAL = {
    "reach": ("plays", None),
    "like_conversion": ("likes", "log_plays"),
    "save_conversion": ("collects", "log_plays"),
    "relative_save": ("collects", "log_likes"),
}


def ols_lograte(df, y, off, formula_rhs):
    data = df.loc[df.likes > 0] if off == "log_likes" else df
    if off is None:
        dv = np.log(data[y])
    else:
        base = data["likes"] if off == "log_likes" else data["plays"]
        dv = np.log((data[y] + 1) / (base + 1))
    return smf.ols(f"dv ~ {formula_rhs}", data=data.assign(dv=dv)).fit(cov_type="HC1")


def main() -> None:
    df = load_sample()
    english = df.lang_group == "English"
    specs = {
        "main": (df, rhs(), "nb"),
        "english": (df[english], rhs(lang=False), "nb"),
        "non_english": (df[~english], rhs(), "nb"),
        "trim_top1": (df[df.log_follower_count < df.log_follower_count.quantile(.99)], rhs(), "nb"),
        "mature_90d": (df[df.days_since_post >= 90], rhs(), "nb"),
        "ols_lograte": (df, rhs(), "ols"),
        "overlay_t4": (df.assign(has_text_overlay=df.has_overlay_type4), rhs(), "nb"),
        "overlay_t9": (df.assign(has_text_overlay=df.has_overlay_type9), rhs(), "nb"),
    }
    rows = []
    for spec, (data, formula_rhs, est) in specs.items():
        for model, (y, off) in FOCAL.items():
            res = fit_nb(data, y, off, formula_rhs) if est == "nb" else ols_lograte(data, y, off, formula_rhs)
            for col in CUES:
                t = zname(col)
                rows.append(dict(spec=spec, model=model, variable=col, n=int(res.nobs),
                                 b=res.params[t], se=res.bse[t], p=res.pvalues[t]))
        print(f"  {spec} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "table_robustness.csv", index=False)

    main_ = out[out.spec == "main"].set_index(["model", "variable"])
    alt = out[out.spec != "main"].join(main_[["b", "p"]], on=["model", "variable"], rsuffix="_main")
    alt["same_sign_sig"] = (np.sign(alt.b) == np.sign(alt.b_main)) & (alt.p < .05)
    summary = (alt.groupby(["model", "variable"])
               .agg(main_b=("b_main", "first"), main_p=("p_main", "first"),
                    n_specs=("spec", "size"), n_agree=("same_sign_sig", "sum"))
               .reset_index())
    summary.to_csv(TABLES / "table_robustness_summary.csv", index=False)
    summary["cell"] = summary.main_b.map("{:+.3f}".format) + np.where(summary.main_p < .05, "*", " ") \
        + " " + summary.n_agree.astype(str) + "/" + summary.n_specs.astype(str)
    print(summary.pivot(index="variable", columns="model", values="cell").reindex(CUES)[list(FOCAL)].to_string())


if __name__ == "__main__":
    main()
