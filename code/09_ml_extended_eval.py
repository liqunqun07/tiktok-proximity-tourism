"""
机器学习模型评估指标补充
为证明XGBoost模型的预测表现并非偶然拟合，本脚本在原有 R2/RMSE(单次80/20划分)基础上补充：
  1. MAE(测试集，对数尺度)
  2. 5折交叉验证 R2(均值±标准差)，检验单次train/test划分结果的稳定性
  3. 基线模型对比：仅用log_follower_count(粉丝数)一个特征的XGBoost模型，
     用于证明内容特征(嵌入叙事/感官增强/互动引导)相对"账号体量决定一切"这一朴素基线的增量解释力

输入：../data/02_features.csv, ../outputs/table10_ml_performance.csv(用于核对一致性)
输出：../outputs/table19_ml_extended_metrics.csv
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dimensions import all_features

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

FEATURES = all_features()
DVS = ["likes", "comments", "shares", "plays", "collects"]

df = pd.read_csv(DATA_DIR / "02_features.csv", low_memory=False)
df = df[(df["days_since_post"] >= 0) & (df["days_since_post"] <= 3000)].copy()
df = df[~df["flag_zero_duration"].astype(bool)].copy()
for c in ["video_quality_score", "loudness"]:
    df[c] = df[c].fillna(df[c].mean())

def lang_group(x):
    if x == "en": return "English"
    if x in ("zh-Hans", "zh-Hant", "zh"): return "Chinese"
    return "Other"
df["lang_group"] = df["language"].apply(lang_group)
lang_dummies = pd.get_dummies(df["lang_group"], prefix="lang", drop_first=True)
X_full = pd.concat([df[FEATURES + ["log_follower_count", "days_since_post"]].drop(
    columns=["log_follower_count", "days_since_post"]).assign(
        log_follower_count=df["log_follower_count"], days_since_post=df["days_since_post"]
    ), lang_dummies], axis=1)
X_baseline = df[["log_follower_count"]]

def make_model():
    return xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=4,
    )

rows = []
for dv in DVS:
    y = np.log1p(df[dv].values)

    # ---- 全特征模型：单次划分 + MAE ----
    X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42)
    model = make_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2_full = r2_score(y_test, y_pred)
    rmse_full = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae_full = mean_absolute_error(y_test, y_pred)

    # ---- 全特征模型：5折交叉验证 R2 ----
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(make_model(), X_full, y, cv=kf, scoring="r2", n_jobs=1)

    # ---- 基线模型：仅粉丝数(log) ----
    Xb_train, Xb_test, yb_train, yb_test = train_test_split(X_baseline, y, test_size=0.2, random_state=42)
    base_model = make_model()
    base_model.fit(Xb_train, yb_train)
    yb_pred = base_model.predict(Xb_test)
    r2_base = r2_score(yb_test, yb_pred)
    rmse_base = float(np.sqrt(mean_squared_error(yb_test, yb_pred)))
    mae_base = mean_absolute_error(yb_test, yb_pred)

    rows.append({
        "DV": dv,
        "R2_full_test": round(r2_full, 4),
        "RMSE_full_test": round(rmse_full, 4),
        "MAE_full_test": round(mae_full, 4),
        "R2_5fold_mean": round(cv_scores.mean(), 4),
        "R2_5fold_std": round(cv_scores.std(), 4),
        "R2_baseline_followers_only": round(r2_base, 4),
        "RMSE_baseline_followers_only": round(rmse_base, 4),
        "MAE_baseline_followers_only": round(mae_base, 4),
        "R2_gain_over_baseline": round(r2_full - r2_base, 4),
    })
    print(f"{dv}: R2_full={r2_full:.3f} (5fold {cv_scores.mean():.3f}±{cv_scores.std():.3f}), "
          f"R2_baseline(仅粉丝数)={r2_base:.3f}, 增量={r2_full - r2_base:.3f}, MAE_full={mae_full:.3f}")

out = pd.DataFrame(rows)
out.to_csv(OUT_DIR / "table19_ml_extended_metrics.csv", index=False)
print("\n已保存 table19_ml_extended_metrics.csv")
