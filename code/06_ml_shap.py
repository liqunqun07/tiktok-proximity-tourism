"""
机器学习 + SHAP 可解释性分析（负二项回归的补充稳健性检验，不替代假设检验）
输入：../data/02_features.csv
输出：
  ../outputs/table10_ml_performance.csv          5个模型的样本外拟合优度(R2/RMSE)
  ../outputs/table11_shap_importance.csv         每个特征在每个模型下的平均|SHAP|重要性 + 所属维度 + 所属假设
  ../outputs/table12_regression_vs_shap_sign.csv 回归系数方向 与 SHAP方向 的对照表(核心结果)
  ../outputs/fig4_shap_summary_{dv}.png          5张SHAP summary plot(按维度上色)
  ../outputs/fig5_shap_importance_by_dimension.png 按理论维度汇总的重要性条形图
  ../outputs/ml_progress.json                    实时进度(供本地看板轮询)

方法说明：
- 因变量做 log1p 变换后用 XGBoost 回归(平方误差)拟合，避免树模型直接拟合极端右偏计数数据不稳定
- 80/20 训练测试集划分，SHAP值在测试集上计算，避免过拟合导致重要性虚高
- SHAP在这里只用于"重要性排序"和"方向核验"，不产出p值，不替代第4步的负二项回归假设检验
"""
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
def rmse_score(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dimensions import FEATURE_GROUPS, HYPOTHESIS_MAP, DVS, all_features, feature_to_group, feature_to_color
from progress import Progress

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

FEATURES = all_features()  # 13个特征，按三个理论维度+控制变量分组，定义见 dimensions.py
G2COLOR = feature_to_color()
F2GROUP = feature_to_group()

step_names = ["加载与清洗数据", "读取回归结果(用于方向对照)"]
for dv in DVS:
    step_names += [f"训练 {dv} 模型", f"计算 {dv} SHAP值", f"生成 {dv} 图表"]
step_names += ["按维度汇总重要性", "生成回归vs SHAP方向对照表", "全部完成"]
prog = Progress(step_names)
step_i = 0

try:
    # ---------- Step 0: 加载数据 ----------
    prog.start_step(step_i, "读取 02_features.csv")
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
    prog.done_step(step_i, f"N={len(df)}, 特征数={X_full.shape[1]}")
    step_i += 1

    # ---------- Step 1: 读取回归系数(用于最后的方向对照) ----------
    prog.start_step(step_i, "读取 table7_regression_results.csv")
    reg_path = OUT_DIR / "table7_regression_results.csv"
    reg_results = pd.read_csv(reg_path) if reg_path.exists() else None
    prog.done_step(step_i, "OK" if reg_results is not None else "未找到，将跳过方向对照")
    step_i += 1

    perf_rows = []
    shap_rows = []

    for dv in DVS:
        # ---- 训练模型 ----
        prog.start_step(step_i, f"{dv}: 划分训练/测试集并训练XGBoost")
        y = np.log1p(df[dv].values)
        X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42)
        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=4,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = rmse_score(y_test, y_pred)
        perf_rows.append({"DV": dv, "n_train": len(X_train), "n_test": len(X_test),
                           "R2_test_log_scale": round(r2, 4), "RMSE_test_log_scale": round(rmse, 4)})
        prog.done_step(step_i, f"R2(log scale)={r2:.3f}, RMSE={rmse:.3f}")
        step_i += 1

        # ---- SHAP ----
        # 用XGBoost原生的pred_contribs计算SHAP值，避免shap库对新版xgboost booster dump格式的解析兼容问题
        prog.start_step(step_i, f"{dv}: 计算SHAP值(测试集, n={len(X_test)})")
        dtest = xgb.DMatrix(X_test, feature_names=list(X_full.columns))
        contribs = model.get_booster().predict(dtest, pred_contribs=True)
        shap_values = contribs[:, :-1]  # 最后一列是base value，不是特征贡献
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        for feat, val in zip(X_full.columns, mean_abs_shap):
            mean_signed = shap_values[:, list(X_full.columns).index(feat)].mean()
            shap_rows.append({
                "DV": dv, "feature": feat,
                "dimension": F2GROUP.get(feat, "语言/其他 Language & Other"),
                "hypothesis": HYPOTHESIS_MAP.get(feat, "-"),
                "mean_abs_shap": val, "mean_signed_shap": mean_signed,
            })
        prog.done_step(step_i, "SHAP值计算完成")
        step_i += 1

        # ---- 画图 ----
        prog.start_step(step_i, f"{dv}: 绘制SHAP summary plot")
        fig = plt.figure(figsize=(9, 6))
        shap.summary_plot(shap_values, X_test, show=False, max_display=15)
        plt.title(f"SHAP Summary — DV = {dv}")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"fig4_shap_summary_{dv}.png", dpi=150)
        plt.close(fig)
        prog.done_step(step_i, "已保存图")
        step_i += 1

    pd.DataFrame(perf_rows).to_csv(OUT_DIR / "table10_ml_performance.csv", index=False)
    shap_df = pd.DataFrame(shap_rows)
    shap_df.to_csv(OUT_DIR / "table11_shap_importance.csv", index=False)

    # ---------- 按维度汇总重要性 ----------
    prog.start_step(step_i, "按理论维度汇总SHAP重要性")
    dim_summary = shap_df.groupby(["DV", "dimension"])["mean_abs_shap"].sum().reset_index()
    fig, axes = plt.subplots(1, 5, figsize=(24, 5), sharey=False)
    dims_order = list(FEATURE_GROUPS.keys()) + ["语言/其他 Language & Other"]
    colors_map = {**{k: v["color"] for k, v in FEATURE_GROUPS.items()}, "语言/其他 Language & Other": "#C44E52"}
    def en_label(d):
        # dimensions.py 中维度 key 的格式为 "中文 英文"，图表只显示英文部分，避免无中文字体时方块显示
        return d.split(" ", 1)[1] if " " in d else d

    for ax, dv in zip(axes, DVS):
        sub = dim_summary[dim_summary["DV"] == dv].set_index("dimension").reindex(dims_order).fillna(0)
        ax.barh([en_label(d) for d in sub.index], sub["mean_abs_shap"], color=[colors_map[d] for d in sub.index])
        ax.set_title(dv)
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle("Sum of |SHAP| by Theoretical Dimension, per Dependent Variable")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_shap_importance_by_dimension.png", dpi=150)
    plt.close(fig)
    prog.done_step(step_i, "已保存 fig5")
    step_i += 1

    # ---------- 回归 vs SHAP 方向对照表 ----------
    prog.start_step(step_i, "生成回归系数方向 vs SHAP方向对照表")
    if reg_results is not None:
        rows = []
        for dv in DVS:
            for feat in FEATURES:
                reg_row = reg_results[(reg_results["DV"] == dv) & (reg_results["variable"] == feat)]
                shap_row = shap_df[(shap_df["DV"] == dv) & (shap_df["feature"] == feat)]
                if reg_row.empty or shap_row.empty:
                    continue
                reg_coef = reg_row.iloc[0]["coef"]
                reg_p = reg_row.iloc[0]["p_value"]
                shap_signed = shap_row.iloc[0]["mean_signed_shap"]
                rows.append({
                    "DV": dv, "feature": feat, "hypothesis": HYPOTHESIS_MAP.get(feat, "-"),
                    "regression_coef": round(reg_coef, 4), "regression_p": round(reg_p, 4),
                    "regression_significant": reg_p < 0.05,
                    "shap_mean_signed": round(shap_signed, 4),
                    "direction_agree": (reg_coef > 0) == (shap_signed > 0),
                })
        pd.DataFrame(rows).to_csv(OUT_DIR / "table12_regression_vs_shap_sign.csv", index=False)
        agree_rate = pd.DataFrame(rows)["direction_agree"].mean()
        prog.done_step(step_i, f"方向一致率 = {agree_rate:.1%}")
    else:
        prog.done_step(step_i, "跳过(未找到回归结果表)")
    step_i += 1

    prog.start_step(step_i, "汇总完成")
    prog.done_step(step_i, "全部输出已写入 outputs/")
    prog.finish()
    print("ML+SHAP 分析全部完成")

except Exception as e:
    prog.fail(step_i, e)
    raise
