"""
变量维度划分（空间三元论操作化，与 02_feature_engineering.py 输出的 02_features.csv 字段一一对应）
被 04_regression_negbin.py 的假设分组、06_ml_shap.py 的SHAP分组共用，
保证回归分析和机器学习分析用的是同一套变量口径，结果才能互相对照。
"""

FEATURE_GROUPS = {
    "嵌入叙事 Embedded Narrative": {
        "color": "#4C72B0",
        "features": [
            "text_length",        # 文本长度
            "video_duration",     # 视频时长
            "music_duration",     # 音乐时长
            "hashtag_count",      # 标签个数
            "semantic_richness",  # 语义丰富度(TF-IDF高权重词数)
        ],
    },
    "感官增强 Sensory Enhancement": {
        "color": "#DD8452",
        "features": [
            "has_music",            # 是否配乐
            "video_quality_score",  # 视频清晰度得分
            "is_vertical",           # 是否竖屏
            "loudness",              # 音量响度(dB)
        ],
    },
    "互动引导 Interaction Guidance": {
        "color": "#55A868",
        "features": [
            "has_poi",       # 是否有定位
            "is_ad",         # 是否广告
            "is_question",   # 文案是否疑问句
            "emoji_count",   # 文案表情符号数
        ],
    },
    "控制变量 Controls": {
        "color": "#8C8C8C",
        "features": [
            "log_follower_count",  # 粉丝数(log)
            "days_since_post",     # 发布至今天数
            # 语言在建模脚本里用 one-hot / C(lang_group) 单独处理，不在此列出
        ],
    },
}

# 对应的假设标签，用于结果表的"假设"列
HYPOTHESIS_MAP = {
    "text_length": "H1", "video_duration": "H1", "music_duration": "H1",
    "hashtag_count": "H1", "semantic_richness": "H1",
    "has_music": "H2", "video_quality_score": "H2", "is_vertical": "H2", "loudness": "H2",
    "has_poi": "H3", "is_ad": "H4", "is_question": "H3", "emoji_count": "H3",
    "log_follower_count": "control", "days_since_post": "control",
}

DVS = ["likes", "comments", "shares", "plays", "collects"]

def all_features():
    feats = []
    for g in FEATURE_GROUPS.values():
        feats += g["features"]
    return feats

def feature_to_group():
    mapping = {}
    for gname, g in FEATURE_GROUPS.items():
        for f in g["features"]:
            mapping[f] = gname
    return mapping

def feature_to_color():
    mapping = {}
    for gname, g in FEATURE_GROUPS.items():
        for f in g["features"]:
            mapping[f] = g["color"]
    return mapping
