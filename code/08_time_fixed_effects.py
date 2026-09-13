"""
时间固定效应稳健性检验
背景：样本发文时间分布严重不均（2023-2025年占99%以上），主模型仅用连续变量
days_since_post 线性控制时间效应，可能无法充分吸收平台/用户基数随时间的非线性变化
（如算法调整、内容生态演化等年度层面的整体偏移），导致核心效应被时间趋势混淆。
本脚本在主模型基础上加入"发文年份"固定效应（哑变量），重新估计五个负二项回归模型，
对比H1-H4核心变量在有/无年份固定效应下的系数方向与显著性是否保持稳定。

输入：../data/02_features.csv
输出：
  ../outputs/table17_year_fe_robustness.csv   核心变量：基准模型 vs 年份固定效应模型 系数对比
  ../outputs/table18_year_distribution.csv    样本按年份分布
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_DIR / "02_features.csv", low_memory=False)
df["post_datetime"] = pd.to_datetime(df["post_datetime"], errors="coerce")
df = df[(df["days_since_post"] >= 0) & (df["days_since_post"] <= 3000)].copy()
df = df[~df["flag_zero_duration"].astype(bool)].copy()

# 早期年份样本量极小(2018=3, 2019=27, 2020=147), 单独作为哑变量会导致估计不稳定,
# 故将2020年及以前合并为"<=2020"一档,其余按年份单独设组
df["post_year"] = df["post_datetime"].dt.year
df["year_group"] = df["post_year"].apply(lambda y: "<=2020" if y <= 2020 else str(int(y)))
year_counts = df["year_group"].value_counts().sort_index()
year_counts.rename_axis("year_group").reset_index(name="n").to_csv(
    OUT_DIR / "table18_year_distribution.csv", index=False
)
print("样本按年份分布:\n", year_counts)

def lang_group(x):
    if x == "en":
        return "English"
    if x in ("zh-Hans", "zh-Hant", "zh"):
        return "Chinese"
    return "Other"
df["lang_group"] = df["language"].apply(lang_group)

FEATURES = [
    "text_length", "video_duration", "music_duration", "hashtag_count", "semantic_richness",
    "has_music", "video_quality_score", "is_vertical", "loudness",
    "has_poi", "is_ad", "is_question", "emoji_count",
    "log_follower_count", "days_since_post",
]
DVS = ["likes", "comments", "shares", "plays", "collects"]
CORE_VARS = [  # H1-H4 核心变量，用于前后对比
    "text_length_z", "video_duration_z", "hashtag_count_z", "semantic_richness_z",
    "has_music", "video_quality_score_z", "is_vertical", "loudness_z",
    "has_poi", "is_ad", "is_question",
]

reg_df = df.copy()
for c in ["video_quality_score", "loudness"]:
    reg_df[c] = reg_df[c].fillna(reg_df[c].mean())

cont_vars = ["text_length", "video_duration", "music_duration", "hashtag_count",
             "semantic_richness", "video_quality_score", "loudness", "emoji_count",
             "log_follower_count", "days_since_post"]
for c in cont_vars:
    reg_df[c + "_z"] = (reg_df[c] - reg_df[c].mean()) / reg_df[c].std()

Z_FEATURES = [c + "_z" if c in cont_vars else c for c in FEATURES]

baseline_rhs = " + ".join(Z_FEATURES) + " + C(lang_group)"
fe_rhs = baseline_rhs + " + C(year_group)"

rows = []
for dv in DVS:
    m_base = smf.glm(f"{dv} ~ {baseline_rhs}", data=reg_df,
                      family=sm.families.NegativeBinomial(alpha=1.0)).fit()
    m_fe = smf.glm(f"{dv} ~ {fe_rhs}", data=reg_df,
                    family=sm.families.NegativeBinomial(alpha=1.0)).fit()
    print(f"{dv}: baseline AIC={m_base.aic:.1f}, +yearFE AIC={m_fe.aic:.1f}")
    for var in CORE_VARS:
        b_coef, b_p = m_base.params.get(var, np.nan), m_base.pvalues.get(var, np.nan)
        f_coef, f_p = m_fe.params.get(var, np.nan), m_fe.pvalues.get(var, np.nan)
        rows.append({
            "DV": dv, "variable": var,
            "baseline_coef": round(b_coef, 4), "baseline_p": round(b_p, 4),
            "yearFE_coef": round(f_coef, 4), "yearFE_p": round(f_p, 4),
            "sign_agree": (b_coef > 0) == (f_coef > 0),
            "sig_agree_at_05": (b_p < 0.05) == (f_p < 0.05),
        })

out = pd.DataFrame(rows)
out.to_csv(OUT_DIR / "table17_year_fe_robustness.csv", index=False)
print(f"\n方向一致率: {out['sign_agree'].mean():.1%}  显著性判定一致率: {out['sig_agree_at_05'].mean():.1%}")
print("已保存 table17_year_fe_robustness.csv, table18_year_distribution.csv")
