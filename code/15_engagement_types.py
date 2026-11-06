"""Step 15 - Affective versus instrumental conversion (H5).

Input : data/processed/02_features.csv (via common.load_sample)
Output: outputs/tables/table_engagement_types.csv

Fits the four conversion models of step 10 (likes, comments = affective;
collects, shares = instrumental; NB2, offset log(plays)) on the same videos and
combines them with a seemingly-unrelated sandwich covariance (Stata ``suest``
logic: cross-products of each model's influence functions). For each cue it
tests the contrast
    instrumental - affective = (b_collects + b_shares)/2 - (b_likes + b_comments)/2
which is the H5 test, plus the like-vs-save pair as a check on step 10.
"""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

from common import CUES, DIMENSION, LABELS, TABLES, load_sample, rhs, zname

fit_nb = importlib.import_module("10_nb_funnel").fit_nb
OUTCOMES = {"likes": "affective", "comments": "affective", "collects": "instrumental", "shares": "instrumental"}


def influence(res) -> np.ndarray:
    """Per-observation influence functions, n x k: score_i @ (-H)^-1."""
    m = res.model
    return m.score_obs(res.params) @ np.linalg.inv(-m.hessian(res.params))


def main() -> None:
    df = load_sample()
    n = len(df)
    fits, infl = {}, {}
    for y in OUTCOMES:
        res = fit_nb(df, y, "log_plays", rhs())
        fits[y], infl[y] = res, influence(res)
        print(f"  {y} fitted", flush=True)

    rows = []
    for col in CUES:
        t = zname(col)
        idx = {y: list(fits[y].params.index).index(t) for y in OUTCOMES}
        b = {y: fits[y].params[t] for y in OUTCOMES}
        # Influence of each coefficient across the four equations (n x 4) -> joint HC1 covariance.
        IF = np.column_stack([infl[y][:, idx[y]] for y in OUTCOMES])
        V = IF.T @ IF * n / (n - 1)
        w_contrast = np.array([-.5, -.5, .5, .5])
        w_pair = np.array([-1, 0, 1, 0])
        aff = (b["likes"] + b["comments"]) / 2
        ins = (b["collects"] + b["shares"]) / 2
        se_c = float(np.sqrt(w_contrast @ V @ w_contrast))
        se_p = float(np.sqrt(w_pair @ V @ w_pair))
        diff_pair = b["collects"] - b["likes"]
        rows.append(dict(
            variable=col, label=LABELS[col], dimension=DIMENSION[col],
            **{f"b_{y}": b[y] for y in OUTCOMES},
            b_affective=aff, b_instrumental=ins,
            difference=ins - aff, se_difference=se_c,
            p_difference=2 * stats.norm.sf(abs((ins - aff) / se_c)),
            save_minus_like=diff_pair, se_save_minus_like=se_p,
            p_save_minus_like=2 * stats.norm.sf(abs(diff_pair / se_p)),
        ))
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "table_engagement_types.csv", index=False)
    print(out[["variable", "b_affective", "b_instrumental", "difference", "p_difference",
               "save_minus_like", "p_save_minus_like"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
