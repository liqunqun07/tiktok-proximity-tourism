"""
特征工程脚本
输入：../data/01_cleaned.csv
输出：../data/02_features.csv

构建"空间三元论"三维度的可操作化变量（全部来自文本/元数据字段，不涉及视频画面处理）：
  嵌入叙事 embedded narrative   : 文本长度、视频时长、音乐时长、语义丰富度(TF-IDF)、话题标签数
  感官增强 sensory enhancement  : 是否配乐、视频清晰度、宽高比(是否竖屏)、音量响度(Loudness)
  互动引导 interaction guidance : 是否有定位、是否广告、是否原创、文本是否疑问句、文案表情符号数
  控制变量 controls             : 粉丝数(log)、发布至今天数、语言、POI大类
"""
import re
import ast
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IN_PATH = DATA_DIR / "01_cleaned.csv"
OUT_PATH = DATA_DIR / "02_features.csv"

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "]+", flags=re.UNICODE
)

def count_emoji(text):
    if not isinstance(text, str):
        return 0
    return len(EMOJI_PATTERN.findall(text))

def parse_loudness(vol_str):
    """从 video_info.volume_info 形如 "{'Loudness': -19.6, 'Peak': 0.49545}" 中解析响度"""
    if not isinstance(vol_str, str) or not vol_str.strip():
        return np.nan
    try:
        d = ast.literal_eval(vol_str)
        return float(d.get("Loudness", np.nan))
    except (ValueError, SyntaxError):
        return np.nan

def parse_peak(vol_str):
    if not isinstance(vol_str, str) or not vol_str.strip():
        return np.nan
    try:
        d = ast.literal_eval(vol_str)
        return float(d.get("Peak", np.nan))
    except (ValueError, SyntaxError):
        return np.nan

def main():
    df = pd.read_csv(IN_PATH, low_memory=False)
    print(f"读入 {len(df)} 行")

    df["post_datetime"] = pd.to_datetime(df["post_datetime"], errors="coerce")

    # ---------- 嵌入叙事 embedded narrative ----------
    df["text_length"] = df["description"].fillna("").apply(len)
    df["video_duration"] = df["video_info.duration"]
    df["music_duration"] = df["music_info.duration"].fillna(0)
    df["hashtag_count"] = df["hashtags_string"].fillna("").apply(
        lambda s: len([h for h in str(s).split(",") if h.strip()]) if s else 0
    )

    # 语义丰富度：对 description 做 TF-IDF，取每条文本高权重词(权重前25%)的数量作为"语义丰富度"代理指标
    corpus = df["description"].fillna("").astype(str).tolist()
    vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)
    tfidf_arr = tfidf_matrix.toarray()
    # 每行的75分位阈值，超过阈值的词数视为"高权重词数量"
    def high_weight_count(row):
        nonzero = row[row > 0]
        if len(nonzero) == 0:
            return 0
        thresh = np.percentile(nonzero, 75)
        return int((row >= thresh).sum())
    df["semantic_richness"] = [high_weight_count(tfidf_arr[i]) for i in range(tfidf_arr.shape[0])]

    # ---------- 感官增强 sensory enhancement ----------
    df["has_music"] = (df["music_duration"] > 0).astype(int)
    df["video_quality_score"] = pd.to_numeric(df["video_info.video_quality"], errors="coerce")
    df.loc[df["video_quality_score"] <= 0, "video_quality_score"] = np.nan  # 0分视为缺失而非真实低质量
    df["aspect_ratio"] = df["video_info.height"] / df["video_info.width"].replace(0, np.nan)
    df["is_vertical"] = (df["aspect_ratio"] > 1).astype(int)  # 竖屏=1 (16:9竖版是短视频常见形态)
    df["loudness"] = df["video_info.volume_info"].apply(parse_loudness)
    df["audio_peak"] = df["video_info.volume_info"].apply(parse_peak)

    # ---------- 互动引导 interaction guidance ----------
    df["has_poi"] = df["poi.name"].notna().astype(int)
    df["is_ad"] = df["flags.is_ad"].fillna(False).astype(bool).astype(int)
    df["is_original"] = df["flags.original_item"].fillna(False).astype(bool).astype(int)
    df["is_question"] = df["description"].fillna("").str.contains(r"\?").astype(int)
    df["emoji_count"] = df["description"].apply(count_emoji)

    # ---------- 控制变量 ----------
    df["log_follower_count"] = np.log1p(df["author_info.follower_count"].fillna(0))
    df["log_following_count"] = np.log1p(df["author_info.following_count"].fillna(0))
    heart = df["author_info.heart_count"].fillna(0)
    heart_bad = (heart < 0).sum()
    if heart_bad:
        print(f"注意: author_info.heart_count 存在 {heart_bad} 条负值(疑似整数溢出的采集错误)，已裁剪为0")
    heart = heart.clip(lower=0)
    df["log_author_heart_count"] = np.log1p(heart)
    max_date = df["post_datetime"].max()
    df["days_since_post"] = (max_date - df["post_datetime"]).dt.days
    df["language"] = df["text_language"].fillna("unknown")
    df["poi_category"] = df["poi.tt_type_name_super"].fillna("no_poi")

    # ---------- 因变量 ----------
    for c, new in [
        ("interaction_stats.digg_count", "likes"),
        ("interaction_stats.comment_count", "comments"),
        ("interaction_stats.share_count", "shares"),
        ("interaction_stats.play_count", "plays"),
        ("interaction_stats.collect_count", "collects"),
    ]:
        df[new] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    keep = [
        "video_info.video_id", "post_datetime", "language", "poi_category",
        # 嵌入叙事
        "text_length", "video_duration", "music_duration", "hashtag_count", "semantic_richness",
        # 感官增强
        "has_music", "video_quality_score", "aspect_ratio", "is_vertical", "loudness", "audio_peak",
        # 互动引导
        "has_poi", "is_ad", "is_original", "is_question", "emoji_count",
        # 控制变量
        "log_follower_count", "log_following_count", "log_author_heart_count", "days_since_post",
        # 因变量
        "likes", "comments", "shares", "plays", "collects",
        # 异常值标记
        "flag_zero_duration", "flag_zero_play", "flag_inconsistent_engagement",
    ]
    out = df[keep].copy()
    out.to_csv(OUT_PATH, index=False)
    print(f"已保存特征数据: {OUT_PATH}，共 {len(out)} 行, {len(keep)} 列")
    print("\n数值变量缺失率:")
    print((out.isna().mean() * 100).round(1).sort_values(ascending=False).head(15))

if __name__ == "__main__":
    main()
