"""Raw TikTok metadata collection (reference only; NOT executed by main.py).

This script documents how the raw scrape ``data/raw/tiktok_market_raw_scrape_20250528.csv``
was collected in May 2025 via the public TikTok web endpoint (using the
``pyktok`` helper library). It is kept purely for provenance:

* The scrape required live access to TikTok's web API in May 2025 and cannot be
  re-executed retroactively; the harvested records are frozen in the raw CSV.
* ``main.py`` therefore starts the pipeline from this raw CSV, and this script
  is never invoked by it.

Usage (manual, requires network access and ``pip install pyktok pandas``):
    python code/00_raw_api_collection_reference.py <seed_links.csv> <output_dir>

``<seed_links.csv>`` must contain a ``link`` column with TikTok video URLs.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pprint import pprint

import pandas as pd


def get_full_metadata(url: str, retries: int = 3):
    """Fetch full public metadata for one TikTok video URL (with retries)."""
    import pyktok as pyk

    for attempt in range(retries):
        try:
            tt_json = pyk.alt_get_tiktok_json(url)
            if not tt_json or "__DEFAULT_SCOPE__" not in tt_json:
                print(f"Invalid JSON response for URL: {url}")
                return None
            data1 = tt_json.get("__DEFAULT_SCOPE__", {}).get("webapp.video-detail", {})
            if not data1:
                print(f"No video-detail data found for URL: {url}")
                return None
            share_meta = data1.get("shareMeta", {})
            data = data1.get("itemInfo", {}).get("itemStruct", {})
            if not data:
                print(f"No itemStruct data found for URL: {url}")
                return None
            extracted = extract_tiktok_data(data)
            extracted["shareMeta_title"] = share_meta.get("title", "")
            extracted["shareMeta_desc"] = share_meta.get("desc", "")
            extracted["url"] = url
            return extracted
        except Exception as exc:  # noqa: BLE001 - collection script, log and retry
            print(f"Attempt {attempt + 1} failed for URL: {url}: {exc}")
            traceback.print_exc()
            if attempt < retries - 1:
                time.sleep(1)
            continue
    return None


def extract_tiktok_data(data: dict) -> dict:
    """Flatten one TikTok ``itemStruct`` payload into a flat record."""
    extracted = {
        "id": str(data.get("id")),
        "description": data.get("desc"),
        "create_time": data.get("createTime"),
        "schedule_time": data.get("scheduleTime", 0),
        "create_time_change": pd.to_datetime(data.get("createTime"), unit="s") if data.get("createTime") else None,
        "schedule_time_change": pd.to_datetime(data.get("scheduleTime", 0), unit="s") if data.get("scheduleTime") else None,
        # Video info
        "video_info.video_id": str(data.get("video", {}).get("id")),
        "video_info.height": data.get("video", {}).get("height"),
        "video_info.width": data.get("video", {}).get("width"),
        "video_info.duration": data.get("video", {}).get("duration"),
        "video_info.ratio": data.get("video", {}).get("ratio"),
        "video_info.cover_url": data.get("video", {}).get("cover"),
        "video_info.origin_cover_url": data.get("video", {}).get("originCover"),
        "video_info.dynamic_cover_url": data.get("video", {}).get("dynamicCover"),
        "video_info.play_url": data.get("video", {}).get("playAddr"),
        "video_info.download_url": data.get("video", {}).get("downloadAddr"),
        "video_info.volume_info": str(data.get("video", {}).get("volumeInfo")),
        "video_info.video_quality": data.get("video", {}).get("VQScore"),
        # Author info
        "author_info.author_id": str(data.get("author", {}).get("id")),
        "author_info.unique_id": data.get("author", {}).get("uniqueId"),
        "author_info.nickname": data.get("author", {}).get("nickname"),
        "author_info.avatar_urls": data.get("author", {}).get("avatarLarger"),
        "author_info.signature": data.get("author", {}).get("signature"),
        "author_info.follower_count": data.get("authorStats", {}).get("followerCount", 0),
        "author_info.following_count": data.get("authorStats", {}).get("followingCount", 0),
        "author_info.heart_count": data.get("authorStats", {}).get("heartCount", 0),
        "author_info.video_count": data.get("authorStats", {}).get("videoCount", 0),
        # Music info
        "music_info.music_id": str(data.get("music", {}).get("id")),
        "music_info.title": data.get("music", {}).get("title"),
        "music_info.play_url": data.get("music", {}).get("playUrl"),
        "music_info.author_name": data.get("music", {}).get("authorName"),
        "music_info.duration": data.get("music", {}).get("duration"),
        # Interaction stats
        "interaction_stats.digg_count": data.get("stats", {}).get("diggCount", 0),
        "interaction_stats.share_count": data.get("stats", {}).get("shareCount", 0),
        "interaction_stats.comment_count": data.get("stats", {}).get("commentCount", 0),
        "interaction_stats.play_count": data.get("stats", {}).get("playCount", 0),
        "interaction_stats.collect_count": data.get("stats", {}).get("collectCount", 0),
        # Hashtags / location / language
        "hashtags_string": ", ".join(tag["hashtagName"] for tag in data.get("textExtra", []) if "hashtagName" in tag),
        "video_stickers": str(data.get("stickersOnItem", [])),
        "location": str(data.get("locationCreated", "")),
        "text_language": data.get("textLanguage", "en"),
        "suggested_words": str(data.get("suggestedWords", [])),
        "diversification_labels": str(data.get("diversificationLabels", [])),
        # Flags
        "flags.original_item": data.get("originalItem", False),
        "flags.official_item": data.get("officalItem", False),
        "flags.for_friend": data.get("forFriend", False),
        "flags.stitch_enabled": data.get("stitchEnabled", True),
        "flags.is_ad": data.get("isAd", False),
        "flags.secret": data.get("secret", False),
        "flags.private_item": data.get("privateItem", False),
        "flags.duet_enabled": data.get("duetEnabled", True),
        # Additional info
        "additional_info.diversification_id": data.get("diversificationId"),
        "additional_info.channel_tags": str(data.get("channelTags", [])),
        "additional_info.is_aigc": data.get("IsAigc", False),
        "additional_info.category_type": data.get("CategoryType"),
    }
    poi = data.get("poi", {})
    extracted.update({
        "poi.name": poi.get("name", ""),
        "poi.address": poi.get("address", ""),
        "poi.city": poi.get("city", ""),
        "poi.province": poi.get("province", ""),
        "poi.country": poi.get("country", ""),
        "poi.id": poi.get("id", ""),
        "poi.father_poi_id": poi.get("fatherPoiId", ""),
        "poi.father_poi_name": poi.get("fatherPoiName", ""),
        "poi.type": poi.get("type", ""),
        "poi.category": poi.get("category", ""),
        "poi.city_code": poi.get("cityCode", ""),
        "poi.country_code": poi.get("countryCode", ""),
        "poi.tt_type_code": poi.get("ttTypeCode", ""),
        "poi.type_code": poi.get("typeCode", ""),
        "poi.tt_type_name_tiny": poi.get("ttTypeNameTiny", ""),
        "poi.tt_type_name_medium": poi.get("ttTypeNameMedium", ""),
        "poi.tt_type_name_super": poi.get("ttTypeNameSuper", ""),
    })
    return extracted


def process_single_url(url: str, download_folder: str):
    """Collect metadata for one URL and (optionally) download the video file."""
    import pyktok as pyk

    try:
        metadata = get_full_metadata(url)
        if not metadata:
            return None, url
        video_id = metadata["id"]
        video_path = os.path.join(download_folder, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            try:
                pyk.save_tiktok(url, True, video_path)
                metadata["video_downloaded"] = True
                print(f"Downloaded video: {video_id}")
            except Exception as exc:  # noqa: BLE001 - download failures are non-fatal
                metadata["video_downloaded"] = False
                print(f"Failed to download video {video_id}: {exc}")
        else:
            metadata["video_downloaded"] = False
            print(f"Video already exists: {video_id}")
        metadata["video_path"] = video_path
        return metadata, None
    except Exception as exc:  # noqa: BLE001
        print(f"Error processing {url}: {exc}")
        traceback.print_exc()
        return None, url


def process_urls(url_list: list[str], download_folder: str, max_workers: int = 3):
    """Collect metadata for a list of URLs with a small thread pool."""
    results, failed_urls = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_url, url, download_folder): url for url in url_list}
        for future in as_completed(futures):
            url = futures[future]
            try:
                result, failed_url = future.result()
                if result:
                    results.append(result)
                if failed_url:
                    failed_urls.append(failed_url)
            except Exception as exc:  # noqa: BLE001
                failed_urls.append(url)
                print(f"Exception processing {url}: {exc}")
    return results, failed_urls


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    seed_links_csv, output_dir = sys.argv[1], sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)
    download_folder = os.path.join(output_dir, "downloaded_videos")
    os.makedirs(download_folder, exist_ok=True)
    output_file = os.path.join(output_dir, "raw_scrape.csv")

    df_input = pd.read_csv(seed_links_csv)
    url_list = df_input["link"].dropna().str.strip().unique().tolist()
    processed_links: list[str] = []
    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file)
        processed_links = df_existing["url"].dropna().str.strip().unique().tolist()
        print(f"Already processed: {len(processed_links)} URLs")
    todo = list(set(url_list) - set(processed_links))
    print(f"URLs remaining to process: {len(todo)}")
    if not todo:
        print("All URLs have been processed. Exiting.")
        return

    all_results: list[dict] = []
    all_failed: list[str] = []
    batch_size = 20
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        print(f"\nProcessing batch {i // batch_size + 1}: {len(batch)} URLs")
        batch_results, batch_failed = process_urls(batch, download_folder)
        all_results.extend(batch_results)
        all_failed.extend(batch_failed)
        if batch_results:
            result_df = pd.json_normalize(batch_results)
            id_columns = ["id", "poi.id", "video_info.video_id", "author_info.author_id", "music_info.music_id"]
            for col in id_columns:
                if col in result_df.columns:
                    result_df[col] = result_df[col].astype(str) + "\t"
            result_df.to_csv(output_file, mode="a" if os.path.exists(output_file) else "w",
                             index=False, header=not os.path.exists(output_file), float_format="%.0f")
            print(f"Saved batch {i // batch_size + 1} with {len(batch_results)} items")

    if all_failed:
        pd.DataFrame({"failed_urls": all_failed}).to_csv(os.path.join(output_dir, "failed_urls.csv"), index=False)
        print(f"\nTotal failed: {len(all_failed)} URLs. Saved to failed_urls.csv")

    downloaded = sum(1 for r in all_results if r.get("video_downloaded", False))
    print("\nProcessing Summary:")
    print(f"- Successfully processed: {len(all_results)} videos")
    print(f"- Downloaded videos: {downloaded}")
    print(f"- Already existed: {len(all_results) - downloaded}")
    print(f"- Failed URLs: {len(all_failed)}")
    if all_results:
        print("\nSample metadata:")
        pprint(all_results[0])


if __name__ == "__main__":
    main()
