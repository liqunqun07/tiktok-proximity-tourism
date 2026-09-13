"""
负二项回归离散参数(alpha)设定的稳健性检验
背景：04_regression_negbin.py 中的主模型将负二项分布的离散参数alpha固定为1.0，
这是一个未经检验的简化设定。更规范的做法(Cameron & Trivedi, 2013已在文中引用)
是将alpha作为待估参数，与回归系数一同通过最大似然联合估计(NB2 MLE)。
本脚本对五个因变量分别比较：
  (a) 固定alpha=1.0的GLM负二项回归(与主模型一致)
  (b) MLE同时估计alpha的负二项回归(以(a)的系数为起始值，避免优化不收敛)
并对比两者的AIC(拟合优度)与H1-H4核心变量的系数方向/显著性判定是否一致。

输入：../data/02_features.csv
输出：
  ../outputs/table20_alpha_mle_fit.csv          5个模型：固定alpha vs MLE估计alpha 的AIC与alpha_hat对比
  ../outputs/table21_alpha_mle_vs_fixed_coefs.csv  核心变量在两种设定下的系数/p值/方向对比(完整版)
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

df = pd.read_csv(DATA_DIR / "02_features.csv", low_memory=False)
df = df[(df["days_since_post"] >= 0) & (df["days_since_post"] <= 3000)].copy()
df = df[~df["flag_zero_duration"].astype(bool)].copy()
for c in ["video_quality_score", "loudness"]:
    df[c] = df[c].fillna(df[c].mean())

def lang_group(x):
    if x == "en": return "English"
    if x in ("zh-Hans", "zh-Hant", "zh"): return "Chinese"
    return "Other"
df["lang_group"] = df["language"].apply(lang_group)

cont_vars = ["text_length", "video_duration", "music_duration", "hashtag_count",
             "semantic_richness", "video_quality_score", "loudness", "emoji_count",
             "log_follower_count", "days_since_post"]
for c in cont_vars:
    df[c + "_z"] = (df[c] - df[c].mean()) / df[c].std()

FEATURES_Z = [c + "_z" for c in cont_vars] + ["has_music", "is_vertical", "has_poi", "is_ad", "is_question"]
formula_rhs = " + ".join(FEATURES_Z) + " + C(lang_group)"
CORE_VARS = ["text_length_z", "video_duration_z", "hashtag_count_z", "semantic_richness_z",
             "has_music", "video_quality_score_z", "is_vertical", "loudness_z",
             "has_poi", "is_ad", "is_question"]
DVS = ["likes", "comments", "shares", "plays", "collects"]

fit_rows, coef_rows = [], []
for dv in DVS:
    f = f"{dv} ~ {formula_rhs}"
    fixed = smf.glm(formula=f, data=df, family=sm.families.NegativeBinomial(alpha=1.0)).fit()
    start = np.append(fixed.params.values, 1.0)
    mle = smf.negativebinomial(formula=f, data=df).fit(
        start_params=start, method="bfgs", disp=0, maxiter=500
    )
    converged = bool(mle.mle_retvals.get("converged"))
    fit_rows.append({
        "DV": dv, "alpha_hat": round(mle.params["alpha"], 4),
        "AIC_fixed_alpha1": round(fixed.aic, 1), "AIC_mle_alpha": round(mle.aic, 1),
        "AIC_improvement": round(fixed.aic - mle.aic, 1), "mle_converged": converged,
    })
    print(f"{dv}: alpha_hat={mle.params['alpha']:.3f}  AIC {fixed.aic:.1f} -> {mle.aic:.1f}  converged={converged}")
    for v in CORE_VARS:
        fb, fp = fixed.params[v], fixed.pvalues[v]
        mb, mp = mle.params[v], mle.pvalues[v]
        coef_rows.append({
            "DV": dv, "variable": v,
            "fixed_alpha1_coef": round(fb, 4), "fixed_alpha1_p": round(fp, 4),
            "mle_alpha_coef": round(mb, 4), "mle_alpha_p": round(mp, 4),
            "sign_agree": (fb > 0) == (mb > 0),
            "sig_agree_at_05": (fp < 0.05) == (mp < 0.05),
        })

fit_df = pd.DataFrame(fit_rows)
coef_df = pd.DataFrame(coef_rows)
fit_df.to_csv(OUT_DIR / "table20_alpha_mle_fit.csv", index=False)
coef_df.to_csv(OUT_DIR / "table21_alpha_mle_vs_fixed_coefs.csv", index=False)

print("\n方向一致率:", round(coef_df.sign_agree.mean(), 4))
print("显著性判定一致率:", round(coef_df.sig_agree_at_05.mean(), 4))
print("已保存 table20_alpha_mle_fit.csv, table21_alpha_mle_vs_fixed_coefs.csv")
