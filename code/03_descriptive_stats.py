"""
描述性统计与探索性分析
输入：../data/02_features.csv
输出：
  ../outputs/table1_descriptive_stats.csv        连续变量描述性统计
  ../outputs/table2_categorical_breakdown.csv     分类变量频数分布
  ../outputs/table3_correlation_matrix.csv        相关系数矩阵
  ../outputs/table4_engagement_by_language.csv    分语言互动量对比
  ../outputs/table5_engagement_by_poi.csv         分POI类别互动量对比
  ../outputs/table6_temporal_trend.csv            发布量随时间趋势
  ../outputs/fig1_engagement_distribution.png     互动量分布直方图(log scale)
  ../outputs/fig2_correlation_heatmap.png         相关性热力图
  ../outputs/fig3_temporal_trend.png              发文量时间趋势图
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_DIR / "02_features.csv", low_memory=False)
df["post_datetime"] = pd.to_datetime(df["post_datetime"], errors="coerce")

CONTINUOUS = [
    "text_length", "video_duration", "music_duration", "hashtag_count", "semantic_richness",
    "video_quality_score", "aspect_ratio", "loudness", "audio_peak",
    "emoji_count", "log_follower_count", "days_since_post",
    "likes", "comments", "shares", "plays", "collects",
]
CATEGORICAL = ["has_music", "is_vertical", "has_poi", "is_ad", "is_original", "is_question", "language"]

# ---------- 表1：连续变量描述性统计 ----------
desc = df[CONTINUOUS].describe(percentiles=[.25, .5, .75, .9]).T
desc["missing_pct"] = (df[CONTINUOUS].isna().mean() * 100).round(2)
desc.to_csv(OUT_DIR / "table1_descriptive_stats.csv")
print("表1 已保存: 连续变量描述性统计")
print(desc[["count", "mean", "std", "min", "50%", "max", "missing_pct"]].round(2))

# ---------- 表2：分类变量频数 ----------
cat_rows = []
for c in CATEGORICAL:
    vc = df[c].value_counts(dropna=False)
    for k, v in vc.items():
        cat_rows.append({"variable": c, "value": k, "count": v, "pct": round(v / len(df) * 100, 2)})
cat_df = pd.DataFrame(cat_rows)
cat_df.to_csv(OUT_DIR / "table2_categorical_breakdown.csv", index=False)
print("\n表2 已保存: 分类变量频数分布")

# ---------- 表3：相关系数矩阵 ----------
corr = df[CONTINUOUS].corr(method="spearman")
corr.to_csv(OUT_DIR / "table3_correlation_matrix.csv")
print("\n表3 已保存: Spearman相关系数矩阵")

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(CONTINUOUS)))
ax.set_yticks(range(len(CONTINUOUS)))
ax.set_xticklabels(CONTINUOUS, rotation=90, fontsize=8)
ax.set_yticklabels(CONTINUOUS, fontsize=8)
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Spearman Correlation Matrix (Feature & Engagement Variables)")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig2_correlation_heatmap.png", dpi=150)
plt.close(fig)
print("图2 已保存: 相关性热力图")

# ---------- 表4：分语言互动量对比（proximity tourism研究常见的跨文化/跨语言分组）----------
top_langs = df["language"].value_counts().head(8).index.tolist()
lang_stats = df[df["language"].isin(top_langs)].groupby("language")[
    ["likes", "comments", "shares", "plays", "collects"]
].agg(["mean", "median", "count"])
lang_stats.to_csv(OUT_DIR / "table4_engagement_by_language.csv")
print("\n表4 已保存: 分语言互动量对比 (Top 8 语言)")

# ---------- 表5：分POI类别互动量对比 ----------
top_poi = df["poi_category"].value_counts().head(10).index.tolist()
poi_stats = df[df["poi_category"].isin(top_poi)].groupby("poi_category")[
    ["likes", "comments", "shares", "plays", "collects"]
].agg(["mean", "median", "count"])
poi_stats.to_csv(OUT_DIR / "table5_engagement_by_poi.csv")
print("表5 已保存: 分POI类别互动量对比 (Top 10 类别)")

# ---------- 表6 / 图3：发文量随时间趋势 ----------
df["post_month"] = df["post_datetime"].dt.to_period("M").astype(str)
trend = df.groupby("post_month").agg(
    video_count=("video_info.video_id", "count"),
    avg_likes=("likes", "mean"),
    avg_plays=("plays", "mean"),
).reset_index()
trend.to_csv(OUT_DIR / "table6_temporal_trend.csv", index=False)
print("\n表6 已保存: 发文量/互动量月度趋势")

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.bar(trend["post_month"], trend["video_count"], color="steelblue", alpha=0.6, label="Video count")
ax1.set_ylabel("Video count")
ax1.tick_params(axis="x", rotation=90, labelsize=7)
ax2 = ax1.twinx()
ax2.plot(trend["post_month"], trend["avg_likes"], color="firebrick", marker="o", markersize=3, label="Avg likes")
ax2.set_ylabel("Average likes")
fig.suptitle("Monthly Posting Volume and Average Engagement")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig3_temporal_trend.png", dpi=150)
plt.close(fig)
print("图3 已保存: 发文量时间趋势图")

# ---------- 图1：互动量分布(log scale) ----------
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, col in zip(axes, ["likes", "comments", "shares", "plays", "collects"]):
    data = np.log1p(df[col])
    ax.hist(data, bins=40, color="slategray")
    ax.set_title(f"log(1+{col})")
fig.suptitle("Distribution of Engagement Metrics (log-transformed)")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig1_engagement_distribution.png", dpi=150)
plt.close(fig)
print("图1 已保存: 互动量分布直方图")

# ---------- 过度离散检验（判断是否需要负二项回归而非普通泊松/OLS）----------
print("\n过度离散检验 (方差/均值比值，远大于1说明需要负二项回归):")
for col in ["likes", "comments", "shares", "plays", "collects"]:
    mean_v = df[col].mean()
    var_v = df[col].var()
    print(f"  {col}: mean={mean_v:.1f}, var={var_v:.1f}, var/mean={var_v/mean_v:.1f}")

print("\n全部描述性分析完成，结果已写入 outputs/ 文件夹")
