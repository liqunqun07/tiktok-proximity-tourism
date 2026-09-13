"""
公平性调节效应 —— 图表生成
输出：
  fig_equity_forest.png            交互系数森林图（5个DV × video_quality_score:small_creator）
  fig_equity_margins.png           预测边际效应图（清晰度对点赞的效应，按创作者规模分组）
  fig_equity_jn.png                Johnson-Neyman 显著性区间图（likes模型）
  fig_equity_income_robustness.png 收入分组稳健性对比图
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

df = pd.read_csv(DATA_DIR / "02_features.csv", low_memory=False)
df = df[(df["days_since_post"] >= 0) & (df["days_since_post"] <= 3000)].copy()
df = df[~df["flag_zero_duration"].astype(bool)].copy()
df["creator_scale"] = pd.qcut(df["log_follower_count"], 3, labels=["Small", "Mid", "Large"])
df["small_creator"] = (df["creator_scale"] == "Small").astype(int)
df["lang_group"] = df["language"].apply(
    lambda x: "English" if x == "en" else ("Chinese" if x in ("zh-Hans", "zh-Hant", "zh") else "Other")
)
for c in ["video_quality_score", "loudness", "hashtag_count", "text_length",
          "video_duration", "semantic_richness", "days_since_post"]:
    df[c + "_z"] = (df[c] - df[c].mean()) / df[c].std()
df["video_quality_score_z"] = df["video_quality_score_z"].fillna(df["video_quality_score_z"].mean())
df["loudness_z"] = df["loudness_z"].fillna(df["loudness_z"].mean())

DVS = ["likes", "comments", "shares", "plays", "collects"]

# =========================================================
# Figure: Forest plot of interaction coefficients (video_quality_score_z : small_creator)
# =========================================================
forest_rows = []
for dv in DVS:
    rhs = ["video_quality_score_z*small_creator", "loudness_z", "has_music", "has_poi",
           "hashtag_count_z", "text_length_z", "video_duration_z", "semantic_richness_z",
           "is_ad", "is_question", "log_follower_count", "days_since_post_z", "C(lang_group)"]
    formula = f"{dv} ~ " + " + ".join(rhs)
    m = smf.glm(formula, data=df, family=sm.families.NegativeBinomial(alpha=1.0)).fit()
    coef = m.params["video_quality_score_z:small_creator"]
    se = m.bse["video_quality_score_z:small_creator"]
    forest_rows.append({"DV": dv, "coef": coef, "lo": coef - 1.96 * se, "hi": coef + 1.96 * se})
forest_df = pd.DataFrame(forest_rows)

fig, ax = plt.subplots(figsize=(7, 4))
y_pos = np.arange(len(forest_df))
ax.errorbar(forest_df["coef"], y_pos,
            xerr=[forest_df["coef"] - forest_df["lo"], forest_df["hi"] - forest_df["coef"]],
            fmt="o", color="#4C72B0", capsize=4)
ax.axvline(0, color="grey", linestyle="--", linewidth=1)
ax.set_yticks(y_pos)
ax.set_yticklabels([d.capitalize() for d in forest_df["DV"]])
ax.set_xlabel("Interaction coefficient: Video quality score × Small creator (95% CI)")
ax.set_title("Fig. Interaction effect across five engagement outcomes")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_equity_forest.png", dpi=200)
plt.close(fig)
print("Saved fig_equity_forest.png")
print(forest_df.round(4))

# =========================================================
# Figure: Predicted margins (video_quality_score effect on likes, by creator scale)
# =========================================================
rhs = ["video_quality_score_z*small_creator", "video_quality_score_z*large_creator" if "large_creator" in df.columns else "loudness_z",
       "has_music", "has_poi", "hashtag_count_z", "log_follower_count", "days_since_post_z", "C(lang_group)"]
df["large_creator"] = (df["creator_scale"] == "Large").astype(int)
formula = ("likes ~ video_quality_score_z*small_creator + video_quality_score_z*large_creator + "
           "has_music + has_poi + hashtag_count_z + log_follower_count + days_since_post_z + C(lang_group)")
m = smf.glm(formula, data=df, family=sm.families.NegativeBinomial(alpha=1.0)).fit()

qs_range = np.linspace(-2, 2, 50)
fig, ax = plt.subplots(figsize=(7, 5))
for scale, small_v, large_v, color in [("Small", 1, 0, "#4C72B0"), ("Mid", 0, 0, "#55A868"), ("Large", 0, 1, "#DD8452")]:
    lin = (m.params["Intercept"] + m.params["video_quality_score_z"] * qs_range
           + m.params["small_creator"] * small_v + m.params["large_creator"] * large_v
           + m.params["video_quality_score_z:small_creator"] * qs_range * small_v
           + m.params["video_quality_score_z:large_creator"] * qs_range * large_v
           + m.params["log_follower_count"] * df["log_follower_count"].mean())
    pred = np.exp(lin)
    ax.plot(qs_range, pred, label=f"{scale} creators", color=color, linewidth=2)
ax.set_xlabel("Video clarity score (standardized)")
ax.set_ylabel("Predicted likes (at mean of controls)")
ax.set_title("Fig. Predicted engagement by clarity score, across creator scales")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_equity_margins.png", dpi=200)
plt.close(fig)
print("Saved fig_equity_margins.png")

# =========================================================
# Figure: Johnson-Neyman plot (likes model, continuous log_follower_count moderator)
# =========================================================
formula_jn = ("likes ~ video_quality_score_z*log_follower_count + loudness_z + has_music + "
              "has_poi + hashtag_count_z + days_since_post_z + C(lang_group)")
m_jn = smf.glm(formula_jn, data=df, family=sm.families.NegativeBinomial(alpha=1.0)).fit()
b1 = m_jn.params["video_quality_score_z"]
b3 = m_jn.params["video_quality_score_z:log_follower_count"]
cov = m_jn.cov_params()
var_b1 = cov.loc["video_quality_score_z", "video_quality_score_z"]
var_b3 = cov.loc["video_quality_score_z:log_follower_count", "video_quality_score_z:log_follower_count"]
cov_b1b3 = cov.loc["video_quality_score_z", "video_quality_score_z:log_follower_count"]
t_crit = stats.t.ppf(0.975, m_jn.df_resid)

w_range = np.linspace(df["log_follower_count"].min(), df["log_follower_count"].max(), 200)
theta = b1 + b3 * w_range
se_theta = np.sqrt(var_b1 + (w_range**2) * var_b3 + 2 * w_range * cov_b1b3)
ci_lo = theta - t_crit * se_theta
ci_hi = theta + t_crit * se_theta

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(w_range, theta, color="#4C72B0", linewidth=2, label="Conditional effect of clarity score")
ax.fill_between(w_range, ci_lo, ci_hi, color="#4C72B0", alpha=0.2, label="95% CI")
ax.axhline(0, color="grey", linestyle="--", linewidth=1)
sig_mask = (ci_lo > 0) | (ci_hi < 0)
if sig_mask.any():
    for lo_idx, hi_idx in zip(np.where(np.diff(sig_mask.astype(int)) == 1)[0],
                                np.where(np.diff(sig_mask.astype(int)) == -1)[0]):
        ax.axvspan(w_range[lo_idx], w_range[hi_idx], color="red", alpha=0.08)
ax.set_xlabel("log(follower count)")
ax.set_ylabel("Conditional effect of clarity score on log(likes)")
ax.set_title("Fig. Johnson-Neyman plot: clarity score effect by follower count (likes model)")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_equity_jn.png", dpi=200)
plt.close(fig)
print("Saved fig_equity_jn.png")

# =========================================================
# Figure: Income-group robustness comparison
# =========================================================
INCOME_MAP = {
    "US": "High income", "GB": "High income", "CA": "High income", "AU": "High income",
    "JP": "High income", "SG": "High income", "DE": "High income", "FR": "High income",
    "KR": "High income", "NL": "High income", "IT": "High income", "ES": "High income",
    "SA": "High income", "AE": "High income", "TW": "High income",
    "PL": "Upper-middle income", "BR": "Upper-middle income", "MX": "Upper-middle income",
    "TR": "Upper-middle income", "TH": "Upper-middle income", "MY": "Upper-middle income",
    "PH": "Upper-middle income",
    "ID": "Lower-middle income", "VN": "Lower-middle income", "EG": "Lower-middle income",
    "PK": "Lower-middle income",
    "MM": "Low income", "NP": "Low income", "KH": "Low income",
}
raw = pd.read_csv(DATA_DIR / "论文集市数据0528.csv", usecols=["video_info.video_id", "location"], low_memory=False)
raw = raw.rename(columns={"video_info.video_id": "video_id"})
df2 = df.merge(raw, left_on="video_info.video_id", right_on="video_id", how="left")
df2["income_group"] = df2["location"].map(INCOME_MAP)
order = ["High income", "Upper-middle income", "Lower-middle income", "Low income"]

fig, ax = plt.subplots(figsize=(7, 5))
for grp, color in zip(order, ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]):
    sub = df2[df2["income_group"] == grp]
    if len(sub) < 50:
        continue
    m_g = smf.glm("likes ~ video_quality_score_z + loudness_z + has_music + has_poi + "
                   "hashtag_count_z + log_follower_count + days_since_post_z",
                   data=sub, family=sm.families.NegativeBinomial(alpha=1.0)).fit()
    coef = m_g.params["video_quality_score_z"]
    se = m_g.bse["video_quality_score_z"]
    ax.errorbar([coef], [grp], xerr=[[coef - (coef - 1.96*se)], [(coef + 1.96*se) - coef]],
                fmt="o", color=color, capsize=4, markersize=8)
ax.axvline(0, color="grey", linestyle="--", linewidth=1)
ax.set_xlabel("Coefficient of clarity score on log(likes) (95% CI)")
ax.set_title("Fig. Robustness of the polish-penalty across income-group contexts")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_equity_income_robustness.png", dpi=200)
plt.close(fig)
print("Saved fig_equity_income_robustness.png")
