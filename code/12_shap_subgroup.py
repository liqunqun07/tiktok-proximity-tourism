"""
SHAP 子样本对比分析（小账号 vs 大账号）
——为公平性调节效应提供机器学习端的交叉验证证据

方法：分别对 Small / Large 创作者子样本独立训练 XGBoost 回归模型（DV=log1p(likes)），
比较两个子样本中 SHAP 特征重要性排序、以及 video_quality_score/loudness 的
SHAP 方向是否与负二项交互效应结果一致。

输出：
  table_equity_07_shap_subgroup.csv   两个子样本SHAP重要性对比表
  fig_equity_shap_subgroup.png        对比条形图
"""
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

df = pd.read_csv(DATA_DIR / "02_features.csv", low_memory=False)
df = df[(df["days_since_post"] >= 0) & (df["days_since_post"] <= 3000)].copy()
df = df[~df["flag_zero_duration"].astype(bool)].copy()
df["creator_scale"] = pd.qcut(df["log_follower_count"], 3, labels=["Small", "Mid", "Large"])
df["lang_group"] = df["language"].apply(
    lambda x: "English" if x == "en" else ("Chinese" if x in ("zh-Hans", "zh-Hant", "zh") else "Other")
)
df = pd.get_dummies(df, columns=["lang_group"], prefix="lang")

FEATURES = [
    "text_length", "video_duration", "music_duration", "hashtag_count", "semantic_richness",
    "has_music", "video_quality_score", "is_vertical", "loudness",
    "has_poi", "is_ad", "is_question", "emoji_count", "days_since_post",
    "lang_English", "lang_Chinese",
]
for c in ["video_quality_score", "loudness"]:
    df[c] = df[c].fillna(df[c].mean())

df["y"] = np.log1p(df["likes"])

results = []
shap_records = {}
for scale in ["Small", "Large"]:
    sub = df[df["creator_scale"] == scale].copy()
    X = sub[FEATURES].astype(float)
    y = sub["y"]
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        base_score=0.5,
    )
    model.fit(X, y)
    # 用XGBoost原生的pred_contribs计算SHAP值，避免shap库对新版xgboost booster dump格式的解析兼容问题
    dmat = xgb.DMatrix(X, feature_names=FEATURES)
    contribs = model.get_booster().predict(dmat, pred_contribs=True)
    shap_values = contribs[:, :-1]  # 最后一列是base value，去掉
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_records[scale] = dict(zip(FEATURES, mean_abs_shap))
    for f, v in zip(FEATURES, mean_abs_shap):
        results.append({"creator_scale": scale, "feature": f, "mean_abs_shap": v, "n": len(sub)})

tbl = pd.DataFrame(results)
tbl_wide = tbl.pivot(index="feature", columns="creator_scale", values="mean_abs_shap").reset_index()
tbl_wide.to_csv(OUT_DIR / "table_equity_07_shap_subgroup.csv", index=False)
print(tbl_wide.sort_values("Large", ascending=False).round(4).to_string())

# ---- Figure: SHAP importance comparison bar chart ----
plot_feats = ["video_quality_score", "loudness", "has_music", "has_poi", "hashtag_count",
              "text_length", "days_since_post"]
small_vals = [shap_records["Small"][f] for f in plot_feats]
large_vals = [shap_records["Large"][f] for f in plot_feats]
x = np.arange(len(plot_feats))
w = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(x - w/2, small_vals, height=w, label="Small creators", color="#4C72B0")
ax.barh(x + w/2, large_vals, height=w, label="Large creators", color="#DD8452")
ax.set_yticks(x)
ax.set_yticklabels(plot_feats)
ax.set_xlabel("Mean |SHAP value| (log-likes model)")
ax.set_title("SHAP feature importance: small vs. large creators")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_equity_shap_subgroup.png", dpi=200)
print("\nSaved:", OUT_DIR / "fig_equity_shap_subgroup.png")
