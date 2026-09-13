"""
稳健性检验
输入：../data/02_features.csv
输出：../outputs/table9_robustness_subsamples.csv

做两组稳健性检验：
  (a) 仅保留 has_poi==1 （明确标注了具体地点/POI）的子样本重新估计 likes 模型，
      检验主效应在"更严格意义上的实体市集打卡视频"子集中是否依然成立
  (b) 按语言分组(English vs Non-English)分别估计 likes 模型，检验跨语言稳健性
"""
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

cont_vars = ["text_length", "video_duration", "music_duration", "hashtag_count",
             "semantic_richness", "video_quality_score", "loudness", "emoji_count",
             "log_follower_count", "days_since_post"]
for c in cont_vars:
    df[c + "_z"] = (df[c] - df[c].mean()) / df[c].std()

FEATURES_Z = [c + "_z" for c in cont_vars] + ["has_music", "is_vertical", "has_poi", "is_ad", "is_question"]
formula = "likes ~ " + " + ".join(FEATURES_Z)

def fit_nb(sub_df, label):
    if len(sub_df) < 50:
        print(f"[{label}] 样本量过小({len(sub_df)})，跳过")
        return []
    f = formula
    if "has_poi" in f and sub_df["has_poi"].nunique() < 2:
        f = f.replace(" + has_poi", "")
    model = smf.glm(formula=f, data=sub_df, family=sm.families.NegativeBinomial(alpha=1.0))
    try:
        res = model.fit(maxiter=200)
    except Exception as e:
        print(f"[{label}] 拟合失败(未收敛): {e}，跳过该子样本")
        return []
    rows = []
    for name, coef, se, pv in zip(res.params.index, res.params, res.bse, res.pvalues):
        rows.append({"subsample": label, "n": len(sub_df), "variable": name,
                     "coef": coef, "se": se, "p_value": pv})
    print(f"[{label}] N={len(sub_df)} 拟合完成")
    return rows

all_rows = []
all_rows += fit_nb(df, "Full sample")
all_rows += fit_nb(df[df["has_poi"] == 1], "POI-tagged only (strict marketplace check-in)")
all_rows += fit_nb(df[df["language"] == "en"], "English only")
all_rows += fit_nb(df[df["language"] != "en"], "Non-English")

out = pd.DataFrame(all_rows)
out.to_csv(OUT_DIR / "table9_robustness_subsamples.csv", index=False)
print(f"\n已保存: table9_robustness_subsamples.csv")
