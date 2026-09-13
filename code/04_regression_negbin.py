"""
负二项回归（Negative Binomial Regression）
输入：../data/02_features.csv
输出：
  ../outputs/table7_regression_results.csv    5个模型(likes/comments/shares/plays/collects)系数汇总表
  ../outputs/table8_vif.csv                   多重共线性(VIF)检验
  ../outputs/regression_summary.txt           完整回归输出(含标准误、p值、伪R方等)

因变量为计数数据且严重过度离散(见03描述性统计脚本输出的var/mean远大于1)，
故不用OLS/泊松回归，改用负二项回归(NB2)，这是社交媒体互动数据分析的标准做法。
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_DIR / "02_features.csv", low_memory=False)

# 剔除1970年代等明显时间戳错误的记录(单条采集错误，非系统性缺失)
df = df[(df["days_since_post"] >= 0) & (df["days_since_post"] <= 3000)].copy()
# 剔除时长<=0的异常记录
df = df[~df["flag_zero_duration"].astype(bool)].copy()
print(f"回归样本量(剔除时间戳异常与零时长记录后): {len(df)}")
print("注意: is_original(是否原创)字段在原始爬取数据中恒为0(该字段未被平台/采集脚本正确回填)，无区分度，已从模型中剔除，作为数据局限在论文中说明。")

# 语言归并为 English / Chinese / Other，避免过多哑变量
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

# 用于回归的数据，缺失值用均值填补并加标记（video_quality_score/loudness有少量缺失）
reg_df = df.copy()
for c in ["video_quality_score", "loudness"]:
    reg_df[c] = reg_df[c].fillna(reg_df[c].mean())

# 标准化连续变量（除0/1哑变量外），便于跨变量比较回归系数大小
cont_vars = ["text_length", "video_duration", "music_duration", "hashtag_count",
             "semantic_richness", "video_quality_score", "loudness", "emoji_count",
             "log_follower_count", "days_since_post"]
for c in cont_vars:
    reg_df[c + "_z"] = (reg_df[c] - reg_df[c].mean()) / reg_df[c].std()

Z_FEATURES = [c + "_z" if c in cont_vars else c for c in FEATURES]

# ---------- VIF 多重共线性检验 ----------
X_vif = sm.add_constant(reg_df[Z_FEATURES].astype(float))
vif_data = pd.DataFrame({
    "variable": X_vif.columns,
    "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])],
})
vif_data.to_csv(OUT_DIR / "table8_vif.csv", index=False)
print("\nVIF检验结果 (经验法则: VIF>10提示严重共线性):")
print(vif_data.round(2))

# ---------- 负二项回归：5个因变量分别建模 ----------
formula_rhs = " + ".join(Z_FEATURES) + " + C(lang_group)"
results_summary = []
full_text_blocks = []

for dv in DVS:
    formula = f"{dv} ~ {formula_rhs}"
    print(f"\n拟合模型: {dv} ~ 特征变量 (负二项回归)")
    model = smf.glm(
        formula=formula, data=reg_df,
        family=sm.families.NegativeBinomial(alpha=1.0),
    )
    res = model.fit()
    full_text_blocks.append(f"===== DV = {dv} =====\n" + res.summary().as_text() + "\n\n")

    for name, coef, se, p in zip(res.params.index, res.params, res.bse, res.pvalues):
        results_summary.append({
            "DV": dv, "variable": name, "coef": coef, "se": se, "p_value": p,
            "sig": "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "",
        })

results_df = pd.DataFrame(results_summary)
results_df.to_csv(OUT_DIR / "table7_regression_results.csv", index=False)
with open(OUT_DIR / "regression_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(full_text_blocks))

print(f"\n已保存: table7_regression_results.csv (系数汇总), regression_summary.txt (完整模型输出)")

# ---------- 简要结果预览：每个DV下显著(p<0.05)的自变量 ----------
print("\n各模型显著变量(p<0.05)预览:")
for dv in DVS:
    sub = results_df[(results_df["DV"] == dv) & (results_df["p_value"] < 0.05) & (results_df["variable"] != "Intercept")]
    sig_vars = ", ".join(f"{r.variable}({r.coef:.2f})" for r in sub.itertuples())
    print(f"  {dv}: {sig_vars if sig_vars else '(无显著变量)'}")
