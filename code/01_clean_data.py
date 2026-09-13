"""
数据清洗脚本
输入：../data/论文集市数据0528.csv （TikTok "集市" 场景原始爬取数据）
输出：../data/01_cleaned.csv

处理内容：
1. 按 video_info.video_id 去重（保留首次出现的记录）
2. 时间字段解析（create_time 时间戳 -> datetime）
3. 关键字段缺失标记（不删行，用哑变量标注缺失，保留最大样本量）
4. 异常值检查（如互动数为负、时长为0等明显错误记录）
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IN_PATH = DATA_DIR / "论文集市数据0528.csv"
OUT_PATH = DATA_DIR / "01_cleaned.csv"

KEEP_COLS = [
    "video_info.video_id", "description", "create_time",
    "video_info.duration", "video_info.height", "video_info.width",
    "video_info.ratio", "video_info.video_quality", "video_info.volume_info",
    "music_info.duration", "music_info.title",
    "author_info.author_id", "author_info.follower_count",
    "author_info.following_count", "author_info.heart_count", "author_info.video_count",
    "interaction_stats.digg_count", "interaction_stats.share_count",
    "interaction_stats.comment_count", "interaction_stats.play_count",
    "interaction_stats.collect_count",
    "hashtags_string", "location", "text_language",
    "flags.original_item", "flags.is_ad",
    "poi.name", "poi.type", "poi.category", "poi.city", "poi.country",
    "poi.tt_type_name_tiny", "poi.tt_type_name_medium", "poi.tt_type_name_super",
]

def main():
    print(f"读取原始数据: {IN_PATH}")
    df = pd.read_csv(IN_PATH, low_memory=False)
    print(f"原始行数: {len(df)}")

    missing_cols = [c for c in KEEP_COLS if c not in df.columns]
    if missing_cols:
        print("警告，以下字段在原始数据中不存在，将跳过:", missing_cols)
    cols = [c for c in KEEP_COLS if c in df.columns]
    df = df[cols].copy()

    # 去重
    before = len(df)
    df = df.drop_duplicates(subset=["video_info.video_id"], keep="first")
    print(f"去重: {before} -> {len(df)} (去掉 {before - len(df)} 条重复视频)")

    # 时间解析
    df["create_time"] = pd.to_numeric(df["create_time"], errors="coerce")
    df["post_datetime"] = pd.to_datetime(df["create_time"], unit="s", errors="coerce")

    # 数值字段强制转换
    numeric_cols = [
        "video_info.duration", "video_info.height", "video_info.width",
        "video_info.video_quality", "music_info.duration",
        "author_info.follower_count", "author_info.following_count",
        "author_info.heart_count", "author_info.video_count",
        "interaction_stats.digg_count", "interaction_stats.share_count",
        "interaction_stats.comment_count", "interaction_stats.play_count",
        "interaction_stats.collect_count",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 异常值标记（不删除，只标注，供后续分析决定是否排除）
    df["flag_zero_duration"] = df["video_info.duration"].fillna(0) <= 0
    df["flag_zero_play"] = df["interaction_stats.play_count"].fillna(0) <= 0
    # 播放量为0但点赞/评论>0，属数据异常（可能是采集时播放数未回填）
    df["flag_inconsistent_engagement"] = (
        (df["interaction_stats.play_count"].fillna(0) == 0)
        & (df["interaction_stats.digg_count"].fillna(0) > 0)
    )

    print("异常值统计:")
    print("  duration<=0 :", df["flag_zero_duration"].sum())
    print("  play_count<=0 :", df["flag_zero_play"].sum())
    print("  play=0但digg>0(疑似异常) :", df["flag_inconsistent_engagement"].sum())

    df.to_csv(OUT_PATH, index=False)
    print(f"已保存: {OUT_PATH} ，共 {len(df)} 行")

if __name__ == "__main__":
    main()
