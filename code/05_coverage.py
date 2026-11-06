"""Sample composition for Appendix Tables A3 and A4.

The metadata store no search-term field (the seed links came from TikTok web
search and are not part of the package), so per-search-term contribution and
saturation curves cannot be computed. The data do support a composition
argument: which everyday-place categories the sample contains, cross-checked
over three independent sources (POI type, caption keywords, hashtag stems),
the presence of genre tags that creators attach to nearby-place content, and
the posting-year and caption-language composition.

Output:
  outputs/tables/tableA3_coverage.csv   source, category, videos, share

A video can match a category through several sources at once, so shares are
computed against N within each row and do not sum to 100 within a source.

Run:  python code/05_coverage.py
"""
from __future__ import annotations

import re

import pandas as pd

from common import ROOT, TABLES, load_sample

CLEANED = ROOT / "data" / "processed" / "01_cleaned.csv"

# Hashtag stems that carry no place information (actor name, fashion brand,
# marketing); excluded before stem matching.
TAG_STOPLIST = {"parkbogum", "gallerydept", "marketing"}

# Place-type patterns applied to lowercased captions and POI type names.
# Each entry is one everyday-place category; English and Chinese terms that
# the bilingual captions actually use are listed together.
PLACE_PATTERNS = {
    "Cafes and coffee": r"caf[eé]|coffee|espresso|latte|matcha|barista|咖啡",
    "Books and bookstores": r"\bbook|书店",
    "Museums and galleries": r"museum|gallery|exhibit|博物馆|美术馆|展览",
    "Parks and outdoors": r"\bparks?\b|garden|beach|island|hiking|trail|\blake\b|nature|公园|花园|户外",
    "Libraries": r"librar|图书馆",
    "Markets": r"\bmarkets?\b|bazaar|夜市|集市|市场",
}

# Same place types matched through hashtag stems (prefix match on the
# normalized tag, after the stoplist).
PLACE_TAG_STEMS = {
    "Cafes and coffee": ("cafe", "coffee"),
    "Books and bookstores": ("book",),
    "Museums and galleries": ("museum", "gallery", "exhibit"),
    "Parks and outdoors": ("park", "nature", "hike"),
    "Libraries": ("librar",),
    "Markets": ("market", "bazaar"),
}

# Platform discovery-genre tags for nearby/novelty place content.
GENRE_TAG_STEMS = {
    "Hidden-gem tags": ("hiddengem",),
    "Date-idea tags": ("dateidea",),
    "Places-to-visit tags": ("placestovisit",),
    "Things-to-do tags": ("thingstodo",),
    "Weekend tags": ("weekend",),
    "Support-local tags": ("supportlocal", "shoplocal", "eatlocal"),
    "City-walk tags": ("citywalk",),
}


def parse_tags(raw: object) -> list[str]:
    """Split ``hashtags_string`` into a normalized list of lowercase tags."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    return [t.strip().lower().lstrip("#") for t in raw.split(",") if t.strip()]


def main() -> None:
    sample = load_sample()  # analytic sample (N = 22,924), adds lang_group and year
    n = len(sample)
    print(f"Analytic sample: {n} videos")
    if n != 22924:
        print(f"Warning: sample size {n} differs from the manuscript N = 22,924")

    # Caption, hashtag and POI fields live in the cleaned file; merge on the
    # video id (both files derive the id from the same cleaned column).
    clean = pd.read_csv(
        CLEANED,
        usecols=["video_info.video_id", "description", "hashtags_string",
                 "poi.tt_type_name_tiny", "poi.tt_type_name_medium"],
        low_memory=False,
    )
    df = sample.merge(clean, on="video_info.video_id", how="left", validate="one_to_one")

    df["tags"] = df["hashtags_string"].map(parse_tags)
    tag_sets = [set(tags) - TAG_STOPLIST for tags in df["tags"]]
    caption = df["description"].fillna("").str.lower()
    poi_types = (df["poi.tt_type_name_tiny"].fillna("") + " "
                 + df["poi.tt_type_name_medium"].fillna("")).str.lower()

    rows: list[dict] = []

    def add(source: str, category: str, mask: pd.Series) -> None:
        videos = int(mask.sum())
        rows.append({"source": source, "category": category,
                     "videos": videos, "share": round(videos / n * 100, 2)})

    def tag_mask(stems: tuple[str, ...]) -> pd.Series:
        """Videos carrying at least one tag that starts with any given stem."""
        return pd.Series([any(tag.startswith(s) for tag in tags for s in stems)
                          for tags in tag_sets], index=df.index)

    add("Sample", "Analytic sample", pd.Series(True, index=df.index))

    # Place-type composition from three independent sources.
    for name, pattern in PLACE_PATTERNS.items():
        add("POI type", name, poi_types.str.contains(pattern, regex=True))
    for name, pattern in PLACE_PATTERNS.items():
        add("Caption keywords", name, caption.str.contains(pattern, regex=True))
    for name, stems in PLACE_TAG_STEMS.items():
        add("Hashtag families", name, tag_mask(stems))

    # Discovery-genre tags carried by the sample itself.
    for name, stems in GENRE_TAG_STEMS.items():
        add("Genre hashtags", name, tag_mask(stems))

    # Posting-year and caption-language composition.
    for year in sorted(df["year"].unique()):
        add("Posting year", year, df["year"] == year)
    for lang in ["English", "Chinese", "Other"]:
        add("Caption language", lang, df["lang_group"] == lang)

    coverage = pd.DataFrame(rows, columns=["source", "category", "videos", "share"])
    out_coverage = TABLES / "tableA3_coverage.csv"
    coverage.to_csv(out_coverage, index=False)
    print(f"Wrote {out_coverage} ({len(coverage)} rows)")


if __name__ == "__main__":
    main()
