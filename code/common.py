"""Shared analytic sample and variable definitions for all analysis stages."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "processed" / "02_features.csv"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
SEED = 20261101

# Cue variables by spatial-triad dimension (order = table order).
NARRATIVE = ["text_length", "semantic_richness", "video_duration"]
SENSORY = ["has_text_overlay", "video_quality_score", "is_vertical", "loudness", "emoji_count"]
GUIDANCE = ["hashtag_count", "has_poi", "is_ad"]
CUES = NARRATIVE + SENSORY + GUIDANCE
CONTROLS = ["log_follower_count", "log_following_count", "days_since_post"]
LOG_FIRST = {"text_length", "semantic_richness", "video_duration", "emoji_count", "hashtag_count"}

LABELS = {
    "text_length": "Caption length",
    "semantic_richness": "Semantic richness",
    "video_duration": "Video duration",
    "has_text_overlay": "On-screen text",
    "video_quality_score": "Visual quality",
    "is_vertical": "Vertical framing",
    "loudness": "Loudness",
    "emoji_count": "Emoji count",
    "hashtag_count": "Hashtag count",
    "has_poi": "Location tag",
    "is_ad": "Ad label",
    "log_follower_count": "Followers (log)",
    "log_following_count": "Following (log)",
    "days_since_post": "Days since posting",
}
DIMENSION = {**{c: "Representations of space" for c in NARRATIVE},
             **{c: "Representational space" for c in SENSORY},
             **{c: "Spatial practice" for c in GUIDANCE},
             **{c: "Control" for c in CONTROLS}}


def lang_group(code: str) -> str:
    if code == "en":
        return "English"
    if code in {"zh", "zh-Hans", "zh-Hant"}:
        return "Chinese"
    return "Other"


def zname(col: str) -> str:
    """Regression term for a variable: continuous -> standardized ``<col>_z``; binary -> itself."""
    return col if col.startswith(("has_", "is_")) else f"{col}_z"


def load_sample() -> pd.DataFrame:
    """Analytic sample (N = 22,924): valid posting window, non-zero duration, at least one play.

    Continuous cues are log1p-transformed when right-skewed and then z-standardized;
    missing quality/loudness are mean-imputed (< 1% of rows).
    """
    df = pd.read_csv(FEATURES, low_memory=False)
    df = df.loc[df.days_since_post.between(0, 3000)
                & ~df.flag_zero_duration.astype(bool)
                & (df.plays > 0)].reset_index(drop=True)
    df["lang_group"] = df.language.map(lang_group)
    df["year"] = df.post_year.astype(int).astype(str)
    for col in ["video_quality_score", "loudness"]:
        df[col] = df[col].fillna(df[col].mean())
    for col in CUES + CONTROLS:
        if zname(col) != col:
            x = np.log1p(df[col].clip(lower=0)) if col in LOG_FIRST else df[col]
            df[zname(col)] = (x - x.mean()) / x.std()
    df["log_plays"] = np.log(df.plays)
    df["log_likes"] = np.log(df.likes.clip(lower=1))
    return df


def rhs(cues=CUES, lang=True, year=True) -> str:
    terms = [zname(c) for c in list(cues) + CONTROLS]
    if lang:
        terms.append("C(lang_group)")
    if year:
        terms.append("C(year)")
    return " + ".join(terms)
