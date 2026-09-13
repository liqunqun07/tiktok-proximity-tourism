import os
import pandas as pd
import pyktok as pyk
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import traceback
from pprint import pprint

# Configuration
output_file = "/Users/kikili/Desktop/代码/论文数据/集市/论文集市数据0528.csv"
folder_path = "/Users/kikili/Desktop/代码/论文数据/集市"
input_path = folder_path + '/0527打卡需要提取.csv'
download_folder = os.path.join(folder_path, "downloaded_videos2")
os.makedirs(download_folder, exist_ok=True)


# Function to extract all metadata including shareMeta
def get_full_metadata(url, retries=3):
    for attempt in range(retries):
        try:
            tt_json = pyk.alt_get_tiktok_json(url)
            if not tt_json or '__DEFAULT_SCOPE__' not in tt_json:
                print(f"Invalid JSON response for URL: {url}")
                return None

            data1 = tt_json.get('__DEFAULT_SCOPE__', {}).get('webapp.video-detail', {})
            if not data1:
                print(f"No video-detail data found for URL: {url}")
                return None

            # Get shareMeta data
            share_meta = data1.get('shareMeta', {})
            title = share_meta.get('title', '')
            desc_content = share_meta.get('desc', '')

            # Get itemStruct data
            data = data1.get('itemInfo', {}).get('itemStruct', {})
            if not data:
                print(f"No itemStruct data found for URL: {url}")
                return None

            # Extract all metadata
            extracted = extract_tiktok_data(data)

            # Add shareMeta fields
            extracted['shareMeta_title'] = title
            extracted['shareMeta_desc'] = desc_content
            extracted['url'] = url

            return extracted

        except Exception as e:
            print(f"Attempt {attempt + 1} failed for URL {url}: {str(e)}")
            traceback.print_exc()
            if attempt < retries - 1:
                time.sleep(1)  # Wait before retrying
            continue
    return None


def extract_tiktok_data(data):
    extracted_data = {
        'id': str(data.get('id')),
        'description': data.get('desc'),
        'create_time': data.get('createTime'),
        'schedule_time': data.get('scheduleTime', 0),
        'create_time_change': pd.to_datetime(data.get('createTime'), unit='s') if data.get('createTime') else None,
        'schedule_time_change': pd.to_datetime(data.get('scheduleTime', 0), unit='s') if data.get('scheduleTime') else None,

        # Video info
        'video_info.video_id': str(data.get('video', {}).get('id')),
        'video_info.height': data.get('video', {}).get('height'),
        'video_info.width': data.get('video', {}).get('width'),
        'video_info.duration': data.get('video', {}).get('duration'),
        'video_info.ratio': data.get('video', {}).get('ratio'),
        'video_info.cover_url': data.get('video', {}).get('cover'),
        'video_info.origin_cover_url': data.get('video', {}).get('originCover'),
        'video_info.dynamic_cover_url': data.get('video', {}).get('dynamicCover'),
        'video_info.play_url': data.get('video', {}).get('playAddr'),
        'video_info.download_url': data.get('video', {}).get('downloadAddr'),
        'video_info.volume_info': str(data.get('video', {}).get('volumeInfo')),
        'video_info.video_quality': data.get('video', {}).get('VQScore'),

        # Author info
        'author_info.author_id': str(data.get('author', {}).get('id')),
        'author_info.unique_id': data.get('author', {}).get('uniqueId'),
        'author_info.nickname': data.get('author', {}).get('nickname'),
        'author_info.avatar_urls': data.get('author', {}).get('avatarLarger'),
        'author_info.signature': data.get('author', {}).get('signature'),
        'author_info.follower_count': data.get('authorStats', {}).get('followerCount', 0),
        'author_info.following_count': data.get('authorStats', {}).get('followingCount', 0),
        'author_info.heart_count': data.get('authorStats', {}).get('heartCount', 0),
        'author_info.video_count': data.get('authorStats', {}).get('videoCount', 0),

        # Music info
        'music_info.music_id': str(data.get('music', {}).get('id')),
        'music_info.title': data.get('music', {}).get('title'),
        'music_info.play_url': data.get('music', {}).get('playUrl'),
        'music_info.author_name': data.get('music', {}).get('authorName'),
        'music_info.duration': data.get('music', {}).get('duration'),

        # Interaction stats
        'interaction_stats.digg_count': data.get('stats', {}).get('diggCount', 0),
        'interaction_stats.share_count': data.get('stats', {}).get('shareCount', 0),
        'interaction_stats.comment_count': data.get('stats', {}).get('commentCount', 0),
        'interaction_stats.play_count': data.get('stats', {}).get('playCount', 0),
        'interaction_stats.collect_count': data.get('stats', {}).get('collectCount', 0),

        # Hashtags
        'hashtags_string': ', '.join(
            tag['hashtagName'] for tag in data.get('textExtra', []) if 'hashtagName' in tag
        ),

        # Stickers
        'video_stickers': str(data.get('stickersOnItem', [])),

        # Location
        'location': str(data.get('locationCreated', '')),

        # Language
        'text_language': data.get('textLanguage', 'en'),

        # Suggested words
        'suggested_words': str(data.get('suggestedWords', [])),

        # Diversification labels
        'diversification_labels': str(data.get('diversificationLabels', [])),

        # Flags
        'flags.original_item': data.get('originalItem', False),
        'flags.official_item': data.get('officalItem', False),
        'flags.for_friend': data.get('forFriend', False),
        'flags.stitch_enabled': data.get('stitchEnabled', True),
        'flags.is_ad': data.get('isAd', False),
        'flags.secret': data.get('secret', False),
        'flags.private_item': data.get('privateItem', False),
        'flags.duet_enabled': data.get('duetEnabled', True),

        # Additional info
        'additional_info.diversification_id': data.get('diversificationId'),
        'additional_info.channel_tags': str(data.get('channelTags', [])),
        'additional_info.is_aigc': data.get('IsAigc', False),
        'additional_info.category_type': data.get('CategoryType')
    }

    # Add POI info if available
    poi = data.get('poi', {})
    extracted_data.update({
        'poi.name': poi.get('name', ''),
        'poi.address': poi.get('address', ''),
        'poi.city': poi.get('city', ''),
        'poi.province': poi.get('province', ''),
        'poi.country': poi.get('country', ''),
        'poi.id': poi.get('id', ''),
        'poi.father_poi_id': poi.get('fatherPoiId', ''),
        'poi.father_poi_name': poi.get('fatherPoiName', ''),
        'poi.type': poi.get('type', ''),
        'poi.category': poi.get('category', ''),
        'poi.city_code': poi.get('cityCode', ''),
        'poi.country_code': poi.get('countryCode', ''),
        'poi.tt_type_code': poi.get('ttTypeCode', ''),
        'poi.type_code': poi.get('typeCode', ''),
        'poi.tt_type_name_tiny': poi.get('ttTypeNameTiny', ''),
        'poi.tt_type_name_medium': poi.get('ttTypeNameMedium', ''),
        'poi.tt_type_name_super': poi.get('ttTypeNameSuper', '')
    })

    return extracted_data



# Function to process a single URL (metadata + download)
def process_single_url(url):
    try:
        # Get metadata
        metadata = get_full_metadata(url)
        if not metadata:
            return None, url

        # Download video
        video_id = metadata['id']
        video_filename = f"{video_id}.mp4"
        video_path = os.path.join(download_folder, video_filename)

        if not os.path.exists(video_path):
            try:
                pyk.save_tiktok(url, True, video_path)
                metadata['video_downloaded'] = True
                print(f"Downloaded video: {video_id}")
            except Exception as e:
                metadata['video_downloaded'] = False
                print(f"Failed to download video {video_id}: {str(e)}")
        else:
            metadata['video_downloaded'] = False
            print(f"Video already exists: {video_id}")

        metadata['video_path'] = video_path
        return metadata, None

    except Exception as e:
        print(f"Error processing {url}: {str(e)}")
        traceback.print_exc()
        return None, url


# Main processing function
def process_urls(url_list, max_workers=3):
    results = []
    failed_urls = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(process_single_url, url): url for url in url_list}

        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                result, failed_url = future.result()
                if result:
                    results.append(result)
                if failed_url:
                    failed_urls.append(failed_url)
            except Exception as e:
                failed_urls.append(url)
                print(f"Exception processing {url}: {str(e)}")

    return results, failed_urls


if __name__ == "__main__":
    # 1. 读取待提取的链接
    df_input = pd.read_csv(input_path)
    url_list = df_input['link'].dropna().str.strip().unique().tolist()

    # 2. 如果已有输出文件，读取已处理过的链接
    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file)
        processed_links = df_existing['url'].dropna().str.strip().unique().tolist()
        print(f"Already processed: {len(processed_links)} URLs")
    else:
        processed_links = []

    # 3. 比较差异：找出未处理的链接
    url_to_process = list(set(url_list) - set(processed_links))
    print(f"URLs remaining to process: {len(url_to_process)}")

    # 4. 如果没有待处理链接，直接退出
    if not url_to_process:
        print("All URLs have been processed. Exiting.")
        exit()

    # 5. 分批处理并保存每20条
    all_results = []
    all_failed_urls = []
    batch_size = 20

    for i in range(0, len(url_to_process), batch_size):
        batch_urls = url_to_process[i:i + batch_size]
        print(f"\nProcessing batch {i // batch_size + 1}: {len(batch_urls)} URLs")

        batch_results, batch_failed = process_urls(batch_urls)
        all_results.extend(batch_results)
        all_failed_urls.extend(batch_failed)

        if batch_results:
            result_df = pd.json_normalize(batch_results)

            id_columns = ['id', 'poi.id', 'video_info.video_id', 'author_info.author_id', 'music_info.music_id']
            for col in id_columns:
                if col in result_df.columns:
                    result_df[col] = result_df[col].astype(str) + '\t'

            if os.path.exists(output_file):
                result_df.to_csv(output_file, mode='a', index=False, header=False, float_format='%.0f')
            else:
                result_df.to_csv(output_file, index=False, float_format='%.0f')

            print(f"Saved batch {i // batch_size + 1} with {len(batch_results)} items")

    # 6. 最后保存失败链接
    if all_failed_urls:
        failed_df = pd.DataFrame({'failed_urls': all_failed_urls})
        failed_output = os.path.join(folder_path, 'failed_urls.csv')
        failed_df.to_csv(failed_output, index=False)
        print(f"\nTotal failed: {len(all_failed_urls)} URLs. Saved to {failed_output}")

    # 7. 打印总结
    downloaded_count = sum(1 for r in all_results if r.get('video_downloaded', False))
    print(f"\nProcessing Summary:")
    print(f"- Successfully processed: {len(all_results)} videos")
    print(f"- Downloaded videos: {downloaded_count}")
    print(f"- Already existed: {len(all_results) - downloaded_count}")
    print(f"- Failed URLs: {len(all_failed_urls)}")
    print(f"\nVideos saved in: {download_folder}")

    # 打印一个样例
    if all_results:
        print("\nSample metadata:")
        pprint(all_results[0])


