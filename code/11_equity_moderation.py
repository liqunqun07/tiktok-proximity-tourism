"""
公平性调节效应分析（Equity Moderation Analysis）
——为"Does Rawness Level the Playing Field?"新论文提供全部核心实证结果

研究问题：内容策略（尤其是技术专业性/感官包装、互动引导）对用户互动的效应，
是否因创作者规模（粉丝数代理的"小/中/大"账号）而系统性不同？
即：粗粝内容/低成本策略是否真的具有"拉平"小商贩与大账号差距的公平化效应？

输入：../data/02_features.csv, ../data/论文集市数据0528.csv（补充 author_info.video_count, location）
输出（全部落盘至 ../outputs/）：
  table_equity_01_creator_scale_desc.csv     创作者规模分组描述性统计 + ANOVA
  table_equity_02_income_group_desc.csv      收入分组样本分布
  table_equity_03_interaction_negbin.csv     交互效应负二项回归主表（5个DV）
  table_equity_04_simple_slopes.csv          简单斜率分析（小/中/大账号条件效应）
  table_equity_05_income_robustness.csv      收入分组稳健性检验
  table_equity_06_jn_intervals.csv           Johnson-Neyman 显著性区间
  equity_moderation_summary.txt              完整回归输出文本
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

log_lines = []
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log_lines.append(s)

# ---------- 1. 数据加载与合并 ----------
df = pd.read_csv(DATA_DIR / "02_features.csv", low_memory=False)
raw = pd.read_csv(
    DATA_DIR / "论文集市数据0528.csv",
    usecols=["video_info.video_id", "author_info.video_count", "location"],
    low_memory=False,
)
raw = raw.rename(columns={
    "video_info.video_id": "video_id",
    "author_info.video_count": "author_video_count",
})
df = df.merge(raw, left_on="video_info.video_id", right_on="video_id", how="left")

df = df[(df["days_since_post"] >= 0) & (df["days_since_post"] <= 3000)].copy()
df = df[~df["flag_zero_duration"].astype(bool)].copy()
log(f"分析样本量: {len(df)}")

# ---------- 2. 创作者规模变量 ----------
# 主指标：粉丝数(log)三分位；稳健替代指标：发布视频总数三分位（代理"偶发/常规/高产"创作者）
df["creator_scale"] = pd.qcut(
    df["log_follower_count"], 3, labels=["Small", "Mid", "Large"]
)
df["small_creator"] = (df["creator_scale"] == "Small").astype(int)
df["large_creator"] = (df["creator_scale"] == "Large").astype(int)

df["postfreq_scale"] = pd.qcut(
    df["author_video_count"].rank(method="first"), 3,
    labels=["Occasional", "Regular", "Prolific"],
)

# ---------- 3. 国家收入分组（World Bank FY2027 classification, GNI per capita, Atlas method） ----------
# 来源: World Bank Country and Lending Groups
# https://datahelpdesk.worldbank.org/knowledgebase/articles/906519
INCOME_MAP = {
    # High income
    "US": "High income", "GB": "High income", "CA": "High income", "AU": "High income",
    "JP": "High income", "SG": "High income", "DE": "High income", "FR": "High income",
    "KR": "High income", "NL": "High income", "IT": "High income", "ES": "High income",
    "SA": "High income", "AE": "High income", "TW": "High income",
    # Upper-middle income
    "PL": "Upper-middle income", "BR": "Upper-middle income", "MX": "Upper-middle income",
    "TR": "Upper-middle income", "TH": "Upper-middle income", "MY": "Upper-middle income",
    "PH": "Upper-middle income",
    # Lower-middle income
    "ID": "Lower-middle income", "VN": "Lower-middle income", "EG": "Lower-middle income",
    "PK": "Lower-middle income",
    # Low income
    "MM": "Low income", "NP": "Low income", "KH": "Low income",
}
df["income_group"] = df["location"].map(INCOME_MAP)
df["non_high_income"] = np.where(df["income_group"].isna(), np.nan,
                                  (df["income_group"] != "High income").astype(float))

# ---------- 4. z标准化 ----------
for c in ["video_quality_score", "loudness", "text_length", "video_duration",
          "hashtag_count", "semantic_richness", "emoji_count",
          "log_follower_count", "days_since_post"]:
    df[c + "_z"] = (df[c] - df[c].mean()) / df[c].std()
df["video_quality_score_z"] = df["video_quality_score_z"].fillna(df["video_quality_score_z"].mean())
df["loudness_z"] = df["loudness_z"].fillna(df["loudness_z"].mean())

DVS = ["likes", "comments", "shares", "plays", "collects"]
CORE_MODERATED = ["video_quality_score_z", "loudness_z", "has_music", "has_poi", "hashtag_count_z"]

# =========================================================
# Table 1: 创作者规模分组描述性统计 + 单因素方差分析(ANOVA)
# =========================================================
desc_rows = []
for var in ["video_quality_score", "loudness", "has_music", "has_poi", "hashtag_count",
            "likes", "comments", "shares", "plays", "collects"]:
    groups = [df.loc[df["creator_scale"] == g, var].dropna() for g in ["Small", "Mid", "Large"]]
    fstat, pval = stats.f_oneway(*groups)
    row = {"variable": var}
    for g, gs in zip(["Small", "Mid", "Large"], groups):
        row[f"mean_{g}"] = gs.mean()
    row["F"] = fstat
    row["p"] = pval
    desc_rows.append(row)
tbl1 = pd.DataFrame(desc_rows)
tbl1.to_csv(OUT_DIR / "table_equity_01_creator_scale_desc.csv", index=False)
log("\n=== Table equity-1: 创作者规模分组描述性统计(ANOVA) ===")
log(tbl1.round(4).to_string())
log("\n各分组样本量:", df["creator_scale"].value_counts().to_dict())

# =========================================================
# Table 2: 收入分组样本分布
# =========================================================
tbl2 = df["income_group"].value_counts(dropna=False).rename_axis("income_group").reset_index(name="n")
tbl2.to_csv(OUT_DIR / "table_equity_02_income_group_desc.csv", index=False)
log("\n=== Table equity-2: 国家收入分组样本分布 ===")
log(tbl2.to_string())

# =========================================================
# Table 3: 交互效应负二项回归（核心表）
# NB2: log mu_i = beta0 + beta1*X_i + beta2*Small_i + beta3*(X_i * Small_i) + controls
# =========================================================
interaction_rows = []
full_blocks = []

for dv in DVS:
    log_dv = f"log_{dv}"
    df[log_dv] = np.log1p(df[dv])  # 用于稳健性对照，正式表用GLM NegativeBinomial
    rhs_terms = []
    for mv in ["video_quality_score_z", "loudness_z", "has_music", "has_poi"]:
        rhs_terms.append(f"{mv}*small_creator")
    rhs_terms += ["hashtag_count_z", "text_length_z", "video_duration_z",
                  "semantic_richness_z", "is_ad", "is_question",
                  "log_follower_count", "days_since_post_z", "C(lang_group)"]
    df["lang_group"] = df["language"].apply(
        lambda x: "English" if x == "en" else ("Chinese" if x in ("zh-Hans", "zh-Hant", "zh") else "Other")
    )
    formula = f"{dv} ~ " + " + ".join(rhs_terms)
    model = smf.glm(formula, data=df, family=sm.families.NegativeBinomial(alpha=1.0)).fit()
    full_blocks.append(f"\n--- {dv}: 交互效应负二项回归 (NB2, alpha=1) ---\n" + model.summary().as_text())

    for mv in ["video_quality_score_z", "loudness_z", "has_music", "has_poi"]:
        main_name = mv
        inter_name = f"{mv}:small_creator"
        interaction_rows.append({
            "DV": dv, "moderated_var": mv,
            "main_coef": model.params.get(main_name, np.nan),
            "main_p": model.pvalues.get(main_name, np.nan),
            "interaction_coef": model.params.get(inter_name, np.nan),
            "interaction_p": model.pvalues.get(inter_name, np.nan),
        })

tbl3 = pd.DataFrame(interaction_rows)
tbl3.to_csv(OUT_DIR / "table_equity_03_interaction_negbin.csv", index=False)
log("\n=== Table equity-3: 交互效应负二项回归结果(核心表) ===")
log(tbl3.round(4).to_string())

with open(OUT_DIR / "equity_moderation_summary.txt", "w") as f:
    f.write("\n".join(log_lines) + "\n\n" + "\n".join(full_blocks))

# =========================================================
# Table 4: 简单斜率分析 (Simple Slope Test)
# 条件效应: d E[y]/dX | Scale=s  近似为 beta1 + beta3 * Scale_dummy(s)
# 对 Small / Mid / Large 三组分别重新估计子样本回归，报告条件斜率
# =========================================================
slope_rows = []
for dv in DVS:
    for scale in ["Small", "Mid", "Large"]:
        sub = df[df["creator_scale"] == scale]
        rhs = ["video_quality_score_z", "loudness_z", "has_music", "has_poi",
               "hashtag_count_z", "log_follower_count", "days_since_post_z", "C(lang_group)"]
        formula = f"{dv} ~ " + " + ".join(rhs)
        m = smf.glm(formula, data=sub, family=sm.families.NegativeBinomial(alpha=1.0)).fit()
        for mv in ["video_quality_score_z", "loudness_z", "has_music", "has_poi"]:
            slope_rows.append({
                "DV": dv, "creator_scale": scale, "moderated_var": mv,
                "conditional_slope": m.params.get(mv, np.nan),
                "se": m.bse.get(mv, np.nan),
                "p": m.pvalues.get(mv, np.nan),
                "n": len(sub),
            })
tbl4 = pd.DataFrame(slope_rows)
tbl4.to_csv(OUT_DIR / "table_equity_04_simple_slopes.csv", index=False)
log("\n=== Table equity-4: 简单斜率分析(按创作者规模分组的条件效应) ===")
log(tbl4.round(4).to_string())

# =========================================================
# Table 5: 收入分组稳健性检验
# =========================================================
income_rows = []
sub_income = df.dropna(subset=["non_high_income"]).copy()
for dv in DVS:
    rhs = []
    for mv in ["video_quality_score_z", "loudness_z"]:
        rhs.append(f"{mv}*non_high_income")
    rhs += ["has_music", "has_poi", "hashtag_count_z", "log_follower_count",
            "days_since_post_z", "C(lang_group)"]
    formula = f"{dv} ~ " + " + ".join(rhs)
    m = smf.glm(formula, data=sub_income, family=sm.families.NegativeBinomial(alpha=1.0)).fit()
    for mv in ["video_quality_score_z", "loudness_z"]:
        income_rows.append({
            "DV": dv, "moderated_var": mv,
            "main_coef": m.params.get(mv, np.nan),
            "main_p": m.pvalues.get(mv, np.nan),
            "interaction_coef": m.params.get(f"{mv}:non_high_income", np.nan),
            "interaction_p": m.pvalues.get(f"{mv}:non_high_income", np.nan),
        })
tbl5 = pd.DataFrame(income_rows)
tbl5.to_csv(OUT_DIR / "table_equity_05_income_robustness.csv", index=False)
log(f"\n=== Table equity-5: 收入分组稳健性检验 (N={len(sub_income)}) ===")
log(tbl5.round(4).to_string())

# =========================================================
# Table 6: Johnson-Neyman 显著性区间
# 连续调节变量版本: log_follower_count (而非三分位), 检验 video_quality_score_z 效应
# 在何种粉丝数取值范围内显著为负/为正
# 条件效应 theta(W) = b1 + b3*W；SE(theta) via delta method；
# JN点: 求解 |theta(W)/SE(theta(W))| = t_crit
# =========================================================
jn_rows = []
for dv in DVS:
    rhs = ["video_quality_score_z*log_follower_count", "loudness_z", "has_music",
           "has_poi", "hashtag_count_z", "days_since_post_z", "C(lang_group)"]
    formula = f"{dv} ~ " + " + ".join(rhs)
    m = smf.glm(formula, data=df, family=sm.families.NegativeBinomial(alpha=1.0)).fit()
    b1 = m.params["video_quality_score_z"]
    b3 = m.params["video_quality_score_z:log_follower_count"]
    cov = m.cov_params()
    var_b1 = cov.loc["video_quality_score_z", "video_quality_score_z"]
    var_b3 = cov.loc["video_quality_score_z:log_follower_count", "video_quality_score_z:log_follower_count"]
    cov_b1b3 = cov.loc["video_quality_score_z", "video_quality_score_z:log_follower_count"]
    t_crit = stats.t.ppf(0.975, m.df_resid)

    # theta(W) = b1 + b3 W ; Var(theta) = var_b1 + W^2 var_b3 + 2W cov_b1b3
    # 求解 theta(W)^2 = t_crit^2 * Var(theta(W))  ->  关于W的二次方程
    qa = b3**2 - (t_crit**2) * var_b3
    qb = 2 * b1 * b3 - (t_crit**2) * 2 * cov_b1b3
    qc = b1**2 - (t_crit**2) * var_b1
    disc = qb**2 - 4 * qa * qc
    roots = []
    if qa != 0 and disc >= 0:
        roots = sorted([(-qb + np.sqrt(disc)) / (2 * qa), (-qb - np.sqrt(disc)) / (2 * qa)])
    jn_rows.append({
        "DV": dv, "b1_quality": b1, "b3_quality_x_logfollower": b3,
        "JN_lower_logfollower": roots[0] if len(roots) == 2 else np.nan,
        "JN_upper_logfollower": roots[1] if len(roots) == 2 else np.nan,
        "sample_min_logfollower": df["log_follower_count"].min(),
        "sample_max_logfollower": df["log_follower_count"].max(),
    })
tbl6 = pd.DataFrame(jn_rows)
tbl6.to_csv(OUT_DIR / "table_equity_06_jn_intervals.csv", index=False)
log("\n=== Table equity-6: Johnson-Neyman 显著性区间(连续粉丝数调节) ===")
log(tbl6.round(4).to_string())

with open(OUT_DIR / "equity_moderation_summary.txt", "w") as f:
    f.write("\n".join(log_lines) + "\n\n" + "\n".join(full_blocks))

log("\n全部公平性调节效应分析完成，结果已写入 outputs/table_equity_0*.csv 与 equity_moderation_summary.txt")
