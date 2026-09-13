# TikTok 邻近旅游短视频研究 · 数据与代码 / TikTok Proximity Tourism Short-Video Research · Data and Code

> 论文 / Paper: *The Sharper the Edit, the Colder the Reception? An Authenticity-Threat Perspective on Proximity Tourism Short Videos on TikTok*（投稿至 SAGE Open）

---

## 目录 / Table of Contents

| # | 中文 | English |
|---|------|---------|
| 1 | 数据说明 / Data Description | ↓ |
| 2 | 代码说明 / Code Description | ↓ |
| 3 | 复现方法 / Reproduction | ↓ |
| 4 | 关键结果速览 / Key Results | ↓ |
| 5 | 引用与许可 / Citation & License | ↓ |

---

## 1. 数据说明 / Data Description

### `data/01_cleaned.csv` — 清洗后数据 / Cleaned data
- **说明**：原始爬取数据经去重、时间解析、异常标记后的清洗版本。
- **规模**：23,010 行（唯一视频）。
- **English**: Deduplicated, time-parsed, anomaly-flagged version of the raw scraped data. 23,010 rows (unique videos).

### `data/02_features.csv` — 特征工程后数据 / Feature-engineered data
- **说明**：在清洗数据基础上构建全部自变量、因变量与控制变量（共 32 列），是所有后续分析的统一输入。
- **English**: All independent / dependent / control variables (32 columns) built on top of the cleaned data — the single shared input for every downstream analysis.

### `data/TikTok变量说明.xlsx` — 变量中英对照字典 / Variable dictionary
- **说明**：所有字段的中文-英文-含义对照表。
- **English**: Chinese–English field mapping with definitions for every variable.

> **数据局限说明 / Data limitation**：`is_original`（是否原创）字段在原始爬取中恒为 0，无法用于建模，已从模型中剔除。该局限已在论文中说明。

---

## 2. 代码说明 / Code Description

所有脚本均在 `code/` 目录下，共享 `dimensions.py`（变量维度划分）与 `progress.py`（进度记录工具）。

| 脚本 / Script | 中文用途 / Purpose (CN) | English purpose |
|---|---|---|
| `00原始采集脚本_参考用.py` | 原始 pyktok 爬虫脚本，仅供参考，无需重跑 | Original pyktok scraper — reference only, not required to re-run |
| `01_clean_data.py` | 数据清洗（去重 + 时间解析 + 异常标记） | Data cleaning |
| `02_feature_engineering.py` | 特征构建：将 Lefebvre 空间三元论操作化为三维度平台元数据 | Feature engineering: operationalizing Lefebvre's spatial triad into three theoretical dimensions |
| `03_descriptive_stats.py` | 描述性统计、相关性、时间趋势、过度离散检验 | Descriptive stats, correlations, time trends, overdispersion tests |
| `04_regression_negbin.py` | **主模型**：负二项回归（5 个因变量） | **Main model**: negative binomial regression (5 DVs) |
| `05_robustness_checks.py` | 稳健性检验（POI 子样本、分语言子样本） | Robustness checks (POI subsample, by-language subsample) |
| `06_ml_shap.py` | 机器学习 + SHAP 可解释性（补充稳健性检验） | ML + SHAP explainability (supplementary robustness) |
| `07_extended_diagnostics.py` | 扩展诊断：变量编码方向、零膨胀、二次项、ZINB vs NB | Extended diagnostics: coding direction, zero-inflation, quadratic terms, ZINB vs NB |
| `08_time_fixed_effects.py` | 时间固定效应稳健性检验（发文年份哑变量） | Time fixed-effects robustness (posting-year dummies) |
| `09_ml_extended_eval.py` | ML 补充评估：MAE、5 折交叉验证、基线对比 | Extended ML evaluation: MAE, 5-fold CV, baseline comparison |
| `10_nb_dispersion_mle.py` | 负二项离散参数 α 的 MLE 设定稳健性（Appendix A.2） | NB dispersion parameter α robustness: fixed vs. MLE (Appendix A.2) |

---

## 3. 复现方法 / Reproduction

**环境 / Environment**：Python 3.10，依赖见 `requirements.txt`（或直接使用仓库自带的 `.venv`）。

```bash
cd code
../.venv/bin/python3 01_clean_data.py
../.venv/bin/python3 02_feature_engineering.py
../.venv/bin/python3 03_descriptive_stats.py
../.venv/bin/python3 04_regression_negbin.py
../.venv/bin/python3 05_robustness_checks.py
```

后续脚本（`06`–`10`）为补充/稳健性检验，按需运行；全部输出落盘至 `outputs/`。

---

## 4. 关键结果速览 / Key Results

- **有效样本**：唯一视频 23,010 条；回归建模剔除时间戳异常/零时长后为 22,925 条。
- **5 个互动指标**（likes / comments / shares / plays / collects）全部严重过度离散（方差/均值比高达万至百万级），故采用负二项回归，而非 OLS / 泊松回归。
- **VIF 最大 4.5**，无严重多重共线性问题。
- **核心发现**：技术精致度代理变量（清晰度、响度、音乐叠加）与用户互动呈**一致负相关**——即"精致反噬"（polish backfire）效应；广告标记内容互动更低；粉丝数为最强正向预测因子，标签数驱动互动，地理标签呈差异化效应。
- **机器学习验证**：XGBoost 样本外 R² 达 0.35–0.43（对数尺度），回归系数与 SHAP 方向一致率达 64%。

---

## 5. 引用与许可 / Citation & License

**建议引用 / Suggested citation**（待发表后更新）：
> Li, Q. (2026). The Sharper the Edit, the Colder the Reception? An Authenticity-Threat Perspective on Proximity Tourism Short Videos on TikTok. *SAGE Open*.

**许可 / License**：本仓库代码与数据按 **CC BY 4.0** 提供，可供自由引用、复现与二次分析；原始 TikTok 数据受平台条款约束，引用时请遵守。/ Code and data are provided under CC BY 4.0 for reproducibility and secondary analysis; original TikTok data remain subject to the platform's terms of use.