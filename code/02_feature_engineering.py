"""Step 02 - Feature engineering on the cleaned scrape.

Input : data/processed/01_cleaned.csv
Output: data/processed/02_features.csv

Builds the analysis variables directly from platform fields (no video-frame
processing), grouped by the three dimensions of Lefebvre's spatial triad:

* Representations of space / embedded narrative: caption length, semantic
  richness, video duration.
* Representational space / sensory enhancement: text overlay, platform
  quality score, vertical framing, loudness, emoji count.
* Spatial practice / interaction guidance: hashtag count, location tag,
  ad label.

Controls: follower and following counts, days since posting, caption
language, posting year. Outcomes: plays, likes, comments, shares, collects.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "processed" / "01_cleaned.csv"
OUT_PATH = ROOT / "data" / "processed" / "02_features.csv"

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "]+", flags=re.UNICODE
)


def count_emoji(text) -> int:
    if not isinstance(text, str):
        return 0
    return len(EMOJI_PATTERN.findall(text))


def parse_loudness(vol_str) -> float:
    """Loudness (LUFS) from a ``video_info.volume_info`` string like
    "{'Loudness': -19.6, 'Peak': 0.49545}"."""
    if not isinstance(vol_str, str) or not vol_str.strip():
        return np.nan
    try:
        return float(ast.literal_eval(vol_str).get("Loudness", np.nan))
    except (ValueError, SyntaxError):
        return np.nan


def overlay_types(sticker_str) -> set:
    """Sticker types carrying non-empty text (4 = on-screen text, 9 = title/cover text)."""
    if not isinstance(sticker_str, str) or not sticker_str.strip():
        return set()
    try:
        items = ast.literal_eval(sticker_str)
    except (ValueError, SyntaxError):
        return set()
    return {x.get("stickerType") for x in items
            if isinstance(x, dict) and any(str(t).strip() for t in (x.get("stickerText") or []))}


def main() -> None:
    df = pd.read_csv(IN_PATH, low_memory=False)
    print(f"Loaded {len(df)} rows")
    df["post_datetime"] = pd.to_datetime(df["post_datetime"], errors="coerce")
    caption = df["description"].fillna("").astype(str)

    # ---------- Representations of space / embedded narrative ----------
    df["text_length"] = caption.str.len()
    df["video_duration"] = df["video_info.duration"]
    # Semantic richness: number of high-weight TF-IDF terms (>= caption's 75th percentile weight).
    tfidf = TfidfVectorizer(max_features=3000, stop_words="english").fit_transform(caption.tolist()).tocsr()
    richness = np.zeros(tfidf.shape[0], dtype=int)
    for i in range(tfidf.shape[0]):
        w = tfidf.data[tfidf.indptr[i]:tfidf.indptr[i + 1]]
        if len(w):
            richness[i] = int((w >= np.percentile(w, 75)).sum())
    df["semantic_richness"] = richness

    # ---------- Representational space / sensory enhancement ----------
    types = df["video_stickers"].apply(overlay_types)
    df["has_text_overlay"] = types.apply(bool).astype(int)
    df["has_overlay_type4"] = types.apply(lambda t: 4 in t).astype(int)
    df["has_overlay_type9"] = types.apply(lambda t: 9 in t).astype(int)
    df["video_quality_score"] = pd.to_numeric(df["video_info.video_quality"], errors="coerce")
    df.loc[df["video_quality_score"] <= 0, "video_quality_score"] = np.nan  # 0 = missing, not a real low score
    df["is_vertical"] = (df["video_info.height"] / df["video_info.width"].replace(0, np.nan) > 1).astype(int)
    df["loudness"] = df["video_info.volume_info"].apply(parse_loudness)
    df["emoji_count"] = df["description"].apply(count_emoji)

    # ---------- Spatial practice / interaction guidance ----------
    df["hashtag_count"] = df["hashtags_string"].fillna("").apply(
        lambda s: len([h for h in str(s).split(",") if h.strip()]))
    df["has_poi"] = df["poi.name"].notna().astype(int)
    df["is_ad"] = df["flags.is_ad"].fillna(False).astype(bool).astype(int)

    # ---------- Controls ----------
    df["log_follower_count"] = np.log1p(df["author_info.follower_count"].fillna(0))
    df["log_following_count"] = np.log1p(df["author_info.following_count"].fillna(0))
    df["days_since_post"] = (df["post_datetime"].max() - df["post_datetime"]).dt.days
    df["post_year"] = df["post_datetime"].dt.year
    df["language"] = df["text_language"].fillna("unknown")
    df["poi_category"] = df["poi.tt_type_name_super"].fillna("no_poi")

    # ---------- Outcomes ----------
    for c, new in [
        ("interaction_stats.play_count", "plays"),
        ("interaction_stats.digg_count", "likes"),
        ("interaction_stats.comment_count", "comments"),
        ("interaction_stats.share_count", "shares"),
        ("interaction_stats.collect_count", "collects"),
    ]:
        df[new] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    keep = [
        "video_info.video_id", "post_datetime", "post_year", "language", "poi_category",
        "text_length", "semantic_richness", "video_duration",
        "has_text_overlay", "has_overlay_type4", "has_overlay_type9",
        "video_quality_score", "is_vertical", "loudness", "emoji_count",
        "hashtag_count", "has_poi", "is_ad",
        "log_follower_count", "log_following_count", "days_since_post",
        "plays", "likes", "comments", "shares", "collects",
        "flag_zero_duration", "flag_zero_play", "flag_inconsistent_engagement",
    ]
    out = df[keep].copy()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved {OUT_PATH} ({len(out)} rows, {len(keep)} cols)")
    print("On-screen text share:", round(out["has_text_overlay"].mean(), 3))


if __name__ == "__main__":
    main()
