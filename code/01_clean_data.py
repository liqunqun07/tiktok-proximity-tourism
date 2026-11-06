"""Step 01 - Clean the raw TikTok scrape.

Input : data/raw/tiktok_market_raw_scrape_20250528.csv
Output: data/processed/01_cleaned.csv

Operations
1. De-duplicate on ``video_info.video_id`` (keep first occurrence).
2. Parse the ``create_time`` Unix timestamp into a datetime column.
3. Coerce numeric fields; flag (not drop) anomalous records so that later
   stages decide exclusions with explicit rules.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
IN_PATH = DATA_DIR / "raw" / "tiktok_market_raw_scrape_20250528.csv"
OUT_PATH = DATA_DIR / "processed" / "01_cleaned.csv"

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
    "hashtags_string", "video_stickers", "location", "text_language",
    "flags.is_ad",
    "poi.name", "poi.type", "poi.category", "poi.city", "poi.country",
    "poi.tt_type_name_tiny", "poi.tt_type_name_medium", "poi.tt_type_name_super",
]

NUMERIC_COLS = [
    "video_info.duration", "video_info.height", "video_info.width",
    "video_info.video_quality", "music_info.duration",
    "author_info.follower_count", "author_info.following_count",
    "author_info.heart_count", "author_info.video_count",
    "interaction_stats.digg_count", "interaction_stats.share_count",
    "interaction_stats.comment_count", "interaction_stats.play_count",
    "interaction_stats.collect_count",
]


def main() -> None:
    print(f"Reading raw data: {IN_PATH}")
    df = pd.read_csv(IN_PATH, low_memory=False)
    print(f"Raw rows: {len(df)}")

    missing_cols = [c for c in KEEP_COLS if c not in df.columns]
    if missing_cols:
        print("Warning: fields absent from the raw data will be skipped:", missing_cols)
    df = df[[c for c in KEEP_COLS if c in df.columns]].copy()

    before = len(df)
    df = df.drop_duplicates(subset=["video_info.video_id"], keep="first")
    print(f"Deduplicated: {before} -> {len(df)} (removed {before - len(df)} duplicate videos)")

    df["create_time"] = pd.to_numeric(df["create_time"], errors="coerce")
    df["post_datetime"] = pd.to_datetime(df["create_time"], unit="s", errors="coerce")

    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Anomaly flags (rows are kept; downstream steps apply explicit exclusion rules).
    df["flag_zero_duration"] = df["video_info.duration"].fillna(0) <= 0
    df["flag_zero_play"] = df["interaction_stats.play_count"].fillna(0) <= 0
    # Plays registered as zero while likes/comments exist: likely an unfilled counter at harvest time.
    df["flag_inconsistent_engagement"] = (
        (df["interaction_stats.play_count"].fillna(0) == 0)
        & (df["interaction_stats.digg_count"].fillna(0) > 0)
    )

    print("Anomaly counts:")
    print("  duration<=0 :", int(df["flag_zero_duration"].sum()))
    print("  play_count<=0 :", int(df["flag_zero_play"].sum()))
    print("  play=0 with likes>0 (suspect):", int(df["flag_inconsistent_engagement"].sum()))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH} ({len(df)} rows)")


if __name__ == "__main__":
    main()
